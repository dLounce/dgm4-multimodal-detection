"""DGM4 dataset loading.

DGM4 (Detecting and Grounding Multi-Modal Media Manipulation) is a
human-centric news dataset of ~230k image-text pairs built on VisualNews:
77,426 pristine and 152,574 manipulated. Manipulations are face swap,
face attribute edit, text swap, and text attribute edit, applied singly or
in combination. Each sample carries a binary real/fake label, the
manipulation class, a tampered image bounding box, and the token indices of
the tampered words.

Dataset and paper:
    R. Shao, T. Wu, Z. Liu. "Detecting and Grounding Multi-Modal Media
    Manipulation." CVPR 2023. https://arxiv.org/abs/2304.02556
    Extended version (TPAMI 2024): https://arxiv.org/abs/2309.14203
    Official code & data: https://github.com/rshaojimmy/MultiModal-DeepFake

Set DGM4_ROOT to point at the dataset. It defaults to the Kaggle mirror
(kaggle.com/datasets/shubhamkumar9812/dgm4-dataset) so the notebooks run
without changes.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.environ.get(
    "DGM4_ROOT", "/kaggle/input/datasets/shubhamkumar9812/dgm4-dataset"
)

# The four atomic manipulations. Combined classes (e.g. "face_swap&text_swap")
# are just the union of these, which is what makes the target multi-label.
ATOMIC = ["face_swap", "face_attribute", "text_swap", "text_attribute"]


def resolve_path(rel, root=ROOT):
    # JSON stores paths as "DGM4/<split>/<method>/<file>". The Kaggle mirror
    # unzips with the method folder nested twice (manipulation/infoswap/
    # infoswap/X); the official release doesn't. Try the doubled layout first,
    # fall back to the flat one.
    rel = rel.replace("DGM4/", "", 1)
    parts = rel.split("/")
    if len(parts) >= 3:
        doubled = os.path.join(root, *parts[:2], parts[1], *parts[2:])
        if os.path.exists(doubled):
            return doubled
    return os.path.join(root, *parts)


class DGM4(Dataset):
    def __init__(self, root=ROOT, split="train"):
        self.root = root
        self.items = json.load(open(f"{root}/metadata/{split}.json"))

    def __len__(self):
        return len(self.items)

    def _labels(self, fake_cls):
        parts = set(fake_cls.split("&"))
        binary = 0.0 if fake_cls == "orig" else 1.0
        multi = torch.tensor([1.0 if a in parts else 0.0 for a in ATOMIC])
        return binary, multi

    def __getitem__(self, i):
        e = self.items[i]
        binary, multi = self._labels(e["fake_cls"])
        path = resolve_path(e["image"], self.root)
        box = e.get("fake_image_box") or [0, 0, 0, 0]
        return {
            "id": e["id"],
            "image_path": path,
            "image": Image.open(path).convert("RGB"),
            "text": e["text"],
            "binary": torch.tensor(binary),
            "multi": multi,
            "bbox": torch.tensor(box, dtype=torch.float),
            "has_box": torch.tensor(float(bool(e.get("fake_image_box")))),
            "text_pos": e.get("fake_text_pos", []),
            "fake_cls": e["fake_cls"],
        }


def make_collate(processor):
    """Batch raw samples through a CLIP processor into model-ready tensors.

    The processor handles tokenization and image preprocessing; everything
    else (labels, boxes, token positions) is stacked or kept as a list.
    """
    def collate(batch):
        enc = processor(
            text=[b["text"] for b in batch],
            images=[b["image"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "binary": torch.stack([b["binary"] for b in batch]),
            "multi": torch.stack([b["multi"] for b in batch]),
            "bbox": torch.stack([b["bbox"] for b in batch]),
            "has_box": torch.stack([b["has_box"] for b in batch]),
            "text_pos": [b["text_pos"] for b in batch],
            "id": [b["id"] for b in batch],
            "fake_cls": [b["fake_cls"] for b in batch],
        }
    return collate