"""Evaluate a trained DGM4 detector on a data split.

Reports the detection and manipulation-type metrics from the DGM4 protocol.

    python scripts/evaluate.py \
        --clip-path models/clip-vit-large-patch14 \
        --checkpoint checkpoints/detector_best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from data import DGM4, make_collate
from detector import build_detector
from metrics import binary_metrics, multilabel_metrics


@torch.no_grad()
def run(model, loader):
    model.eval()
    pb, pm, gb, gm = [], [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ob, om, *_ = model(b["pixel_values"].cuda(),
                               b["input_ids"].cuda(),
                               b["attention_mask"].cuda())
        pb.append(torch.sigmoid(ob.float()).cpu())
        pm.append(torch.sigmoid(om.float()).cpu())
        gb.append(b["binary"])
        gm.append(b["multi"])
    return (binary_metrics(torch.cat(gb).numpy(), torch.cat(pb).numpy()),
            multilabel_metrics(torch.cat(gm).numpy(), torch.cat(pm).numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-path", required=True, help="local CLIP-ViT-L/14 directory")
    ap.add_argument("--checkpoint", required=True, help="trained detector .pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    proc = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    model = build_detector(args.clip_path).cuda()
    # checkpoint holds only the trainable params; the frozen CLIP base is
    # restored by build_detector, hence strict=False.
    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    loader = DataLoader(DGM4(split=args.split), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=make_collate(proc), pin_memory=True)
    bm, mm = run(model, loader)

    print(f"=== DGM4 {args.split} ===")
    print("detection :", {k: round(float(v), 4) for k, v in bm.items()})
    print("type      :", {k: round(mm[k], 4) for k in ["mAP", "CF1", "OF1"]})
    print("AP per class [fs, fa, ts, ta]:", mm["AP_per_class"])


if __name__ == "__main__":
    main()