"""Train the DGM4 detector (LoRA-CLIP + co-attention fusion).

Objective: binary BCE + multi-label BCE + contrastive (on authentic pairs).
Only the LoRA adapters and fusion heads train; the CLIP backbone stays frozen.
The best checkpoint by val AUC+mAP is saved.

    python scripts/train.py --clip-path models/clip-vit-large-patch14
"""

import argparse
import json
import os
import sys
import time

# src/ uses flat imports, so put it on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from data import ATOMIC, DGM4, make_collate
from detector import build_detector, contrastive
from metrics import binary_metrics, multilabel_metrics


def multilabel_pos_weight(root):
    """Per-attribute neg/pos ratio, for the multi-label BCE — read from labels
    only (no images), so it's cheap."""
    items = json.load(open(f"{root}/metadata/train.json"))
    y = torch.tensor([
        [1.0 if a in set(e["fake_cls"].split("&")) else 0.0 for a in ATOMIC]
        for e in items
    ])
    return (y == 0).sum(0) / (y.sum(0) + 1e-6)


@torch.no_grad()
def evaluate(model, loader):
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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.5, help="contrastive loss weight")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/detector_best.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    proc = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    collate = make_collate(proc)
    model = build_detector(args.clip_path).cuda()

    tr_loader = DataLoader(DGM4(split="train"), batch_size=args.batch_size, shuffle=True,
                           drop_last=True, num_workers=args.workers,
                           collate_fn=collate, pin_memory=True)
    va_loader = DataLoader(DGM4(split="val"), batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(tr_loader)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    bce = nn.BCEWithLogitsLoss()
    bce_m = nn.BCEWithLogitsLoss(pos_weight=multilabel_pos_weight(DGM4().root).cuda())

    best = -1
    for ep in range(args.epochs):
        model.train()
        t = time.time()
        run = 0.0
        for step, b in enumerate(tr_loader):
            pv, ids, am = b["pixel_values"].cuda(), b["input_ids"].cuda(), b["attention_mask"].cuda()
            yb, ym = b["binary"].cuda(), b["multi"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ob, om, fi, ft, ie, te = model(pv, ids, am)
                loss = bce(ob.float(), yb) + bce_m(om.float(), ym) + args.lam * contrastive(ie, te, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"  ep{ep} {step}/{len(tr_loader)} loss {loss.item():.3f} "
                      f"({time.time()-t:.0f}s)", end="\r")

        bm, mm = evaluate(model, va_loader)
        sc = bm["AUC"] + mm["mAP"]
        star = ""
        if sc > best:
            best = sc
            star = " *BEST"
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items() if k in trainable},
                       args.out)
        print(f"\nep{ep} | loss {run/len(tr_loader):.3f} | "
              f"val AUC {bm['AUC']:.3f} EER {bm['EER']:.3f} ACC {bm['ACC']:.3f} | "
              f"mAP {mm['mAP']:.3f} CF1 {mm['CF1']:.3f} OF1 {mm['OF1']:.3f} | "
              f"{time.time()-t:.0f}s{star}")

    print("best val AUC+mAP:", round(best, 4))


if __name__ == "__main__":
    main()