"""Localization heads for DGM4 grounding.

Two sub-tasks share one tiny scoring head (grounding_head):

    image / box .. crop each MTCNN face, embed it with CLIP, and score which
                   face is the tampered one. The predicted box is the
                   highest-scoring face.
    text / token . score each fused text token; the flagged tokens are the
                   predicted tampered words.

This module provides the datasets, collate factories, and token-labelling
needed to train and evaluate both. The training loops and feature caching live
in the eval script.
"""

import json

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from data import ROOT, resolve_path
from metrics import box_iou


def grounding_head(dim=768, hidden=256, dropout=0.3):
    """Small MLP scoring a feature vector for manipulation (one logit).

    Used over per-face CLIP embeddings for box grounding (dropout 0.3) and over
    fused text-token features for token grounding (dropout 0.2).
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


# ----------------------------- face / box grounding -----------------------------
class FaceCropDS(Dataset):
    """One sample per MTCNN-detected face; label 1 if it is the tampered face.

    A face counts as tampered when its box matches the annotated fake_image_box
    (IoU > 0.99 — both come from the same detector, so a real match is near
    exact).
    """
    def __init__(self, split, root=ROOT, max_imgs=None):
        items = json.load(open(f"{root}/metadata/{split}.json"))
        self.root = root
        self.s = []
        c = 0
        for e in items:
            if "face" not in e["fake_cls"] or not e["fake_image_box"]:
                continue
            gt, path = e["fake_image_box"], resolve_path(e["image"], root)
            for b in e["mtcnn_boxes"]:
                label = 1.0 if box_iou([b], [gt])[0] > 0.99 else 0.0
                self.s.append((path, b, label, e["id"]))
            c += 1
            if max_imgs and c >= max_imgs:
                break

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        p, b, l, iid = self.s[i]
        return {
            "image": Image.open(p).convert("RGB").crop(tuple(b)),
            "label": l, "box": b, "img_id": iid,
        }


def make_crop_collate(processor):
    def collate(batch):
        return {
            "pixel_values": processor(images=[b["image"] for b in batch],
                                      return_tensors="pt")["pixel_values"],
            "label": torch.tensor([b["label"] for b in batch]),
            "box": [b["box"] for b in batch],
            "img_id": [b["img_id"] for b in batch],
        }
    return collate


# ----------------------------- text / token grounding -----------------------------
def token_label_valid(tokenizer, text, fake_pos, max_len=77):
    """Per-token (label, valid) for one caption.

    Each subword is mapped back to the whitespace word it covers and marked
    tampered if that word index is in fake_pos. valid is 0 for special tokens
    (BOS/EOS/PAD) so they're excluded from loss and metrics.
    """
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    spans, i = [], 0                          # char span of each whitespace word
    for w in text.split():
        s = text.index(w, i); e = s + len(w); i = e
        spans.append((s, e))
    fake = set(fake_pos)
    lab, val = [], []
    for ts, te in enc["offset_mapping"]:
        if ts == te:                          # special token
            lab.append(0.0); val.append(0.0); continue
        mid = (ts + te) / 2
        w = next((k for k, (ws, we) in enumerate(spans) if ws <= mid < we), None)
        lab.append(1.0 if (w is not None and w in fake) else 0.0)
        val.append(1.0)
    return lab, val


class TextDS(Dataset):
    """Captions carrying a text manipulation (those with fake_text_pos set)."""
    def __init__(self, split, root=ROOT, max_n=None):
        items = json.load(open(f"{root}/metadata/{split}.json"))
        self.root = root
        self.s = [e for e in items if "text" in e["fake_cls"] and e["fake_text_pos"]]
        if max_n:
            self.s = self.s[:max_n]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        e = self.s[i]
        return {
            "image": Image.open(resolve_path(e["image"], self.root)).convert("RGB"),
            "text": e["text"],
            "fake_pos": e["fake_text_pos"],
        }


def make_text_collate(processor):
    def collate(batch):
        enc = processor(text=[b["text"] for b in batch], images=[b["image"] for b in batch],
                        return_tensors="pt", padding=True, truncation=True)
        L = enc["input_ids"].shape[1]
        labs, vals = [], []
        for b in batch:
            lab, val = token_label_valid(processor.tokenizer, b["text"], b["fake_pos"])
            labs.append(lab[:L] + [0.0] * (L - len(lab)))
            vals.append(val[:L] + [0.0] * (L - len(val)))
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_label": torch.tensor(labs),
            "token_valid": torch.tensor(vals),
        }
    return collate