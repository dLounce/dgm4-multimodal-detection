"""The detector: LoRA-CLIP backbone + co-attention fusion head.

A frozen CLIP-ViT-L/14 provides image and text token features; only small LoRA
adapters on its attention projections are trained, so the backbone stays cheap
to fine-tune. The fusion head (see fusion.py) does the cross-modal reasoning
and the actual prediction.

build_detector() assembles the whole thing. The training objective combines
binary BCE, multi-label BCE, and the contrastive term below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from fusion import Fusion


def build_clip_lora(clip_path, dtype=torch.bfloat16):
    """Frozen CLIP with trainable LoRA adapters on the attention projections."""
    clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to("cuda", dtype)
    for p in clip.parameters():
        p.requires_grad_(False)

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none",
    )
    clip = get_peft_model(clip, cfg)

    for p in clip.parameters():           # keep adapters in fp32 for stable AdamW
        if p.requires_grad:
            p.data = p.data.float()
    return clip


class Detector(nn.Module):
    def __init__(self, clip_lora, fusion):
        super().__init__()
        self.clip, self.fusion = clip_lora, fusion

    def forward(self, pv, ids, am):
        out = self.clip(pixel_values=pv, input_ids=ids, attention_mask=am)
        ob, om, fi, ft = self.fusion(
            out.vision_model_output.last_hidden_state,
            out.text_model_output.last_hidden_state,
            am == 0,                      # attention_mask 0 -> PAD
        )
        # last two embeddings feed the contrastive loss
        return ob, om, fi, ft, out.image_embeds, out.text_embeds


def build_detector(clip_path, fusion_layers=2, dtype=torch.bfloat16):
    """Assemble the full model: LoRA-CLIP backbone + co-attention fusion head."""
    clip_lora = build_clip_lora(clip_path, dtype)
    fusion = Fusion(layers=fusion_layers)
    return Detector(clip_lora, fusion)


def contrastive(img_emb, txt_emb, binary, temp=0.07):
    """Image-text contrastive (InfoNCE) loss over authentic samples only.

    Manipulated pairs carry a deliberately mismatched caption, so pulling their
    image and text embeddings together would be wrong. Restricting the
    objective to real pairs preserves CLIP's alignment while the rest of the
    model learns to pull real and fake apart.
    """
    auth = (binary == 0)
    if auth.sum() < 2:
        return img_emb.new_zeros(())
    i = F.normalize(img_emb.float(), dim=-1)
    t = F.normalize(txt_emb.float(), dim=-1)
    logits = (i @ t.T) / temp
    tgt = torch.arange(len(i), device=i.device)
    li = F.cross_entropy(logits, tgt, reduction="none")
    lt = F.cross_entropy(logits.T, tgt, reduction="none")
    return (li[auth].mean() + lt[auth].mean()) / 2