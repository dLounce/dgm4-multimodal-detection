"""Train the DGM4 grounding heads on top of a trained detector.

Two small heads, both from localize.grounding_head:

    box   - scores each MTCNN face crop (CLIP image_embeds); the top-scoring
            crop is the predicted tampered region.
    token - scores each fused text token; the flagged tokens are the predicted
            tampered words.

The detector (backbone + fusion) is frozen and loaded from a checkpoint.

    python scripts/train_grounding.py \
        --clip-path models/clip-vit-large-patch14 \
        --checkpoint checkpoints/detector_best.pt --task both
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from detector import build_detector
from localize import (FaceCropDS, TextDS, grounding_head, make_crop_collate,
                      make_text_collate, token_label_valid)
from metrics import box_iou


# ----------------------------- box head -----------------------------
@torch.no_grad()
def extract_crop_feats(model, proc, split, max_imgs=None, bs=256, workers=8):
    """Embed every face crop with the (trained) CLIP backbone."""
    ds = FaceCropDS(split, max_imgs=max_imgs)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=make_crop_collate(proc), pin_memory=True)
    # CLIP needs token ids even for image_embeds; the text is ignored.
    dummy = proc.tokenizer("a photo", return_tensors="pt")
    feats, labels = [], []
    model.clip.eval()
    for b in dl:
        n = b["pixel_values"].shape[0]
        ids = dummy["input_ids"].expand(n, -1).cuda()
        am = dummy["attention_mask"].expand(n, -1).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.clip(pixel_values=b["pixel_values"].cuda(), input_ids=ids, attention_mask=am)
        feats.append(out.image_embeds.float().cpu())
        labels.append(b["label"])
    return ds, torch.cat(feats), torch.cat(labels)


def box_iou_recall(ds, scores):
    """Per image: highest-scoring face is the prediction; IoU against GT box.

    Grouped by image path — the dataset's `id` field collides across images.
    """
    groups = defaultdict(list)
    for i, (path, _box, _lbl, _id) in enumerate(ds.s):
        groups[path].append(i)
    ious, nfaces = [], []
    for _path, idxs in groups.items():
        gt = [j for j in idxs if ds.s[j][2] == 1.0]
        if not gt:
            continue
        pred_box = ds.s[idxs[int(scores[idxs].argmax())]][1]
        ious.append(float(box_iou([pred_box], [ds.s[gt[0]][1]])[0]))
        nfaces.append(len(idxs))
    ious = np.array(ious)
    return {
        "images": len(ious),
        "avg_faces": round(float(np.mean(nfaces)), 2),
        "Recall@0.5": round(float((ious >= 0.5).mean()), 4),
        "Recall@0.75": round(float((ious >= 0.75).mean()), 4),
        "mean_IoU": round(float(ious.mean()), 4),
    }


def train_box_head(model, proc, args):
    print("extracting train crops ...")
    _, Xtr, ytr = extract_crop_feats(model, proc, "train", max_imgs=args.box_max_imgs)
    print("extracting test crops ...")
    ds_te, Xte, yte = extract_crop_feats(model, proc, "test")
    print(f"train crops {tuple(Xtr.shape)} pos={int(ytr.sum())} | "
          f"test crops {tuple(Xte.shape)} pos={int(yte.sum())}")

    head = grounding_head(dropout=0.3).cuda()
    Xtr, ytr = Xtr.cuda(), ytr.cuda()
    pw = ((ytr == 0).sum() / ytr.sum()).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.box_lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    n, bs = len(Xtr), args.box_batch_size
    for ep in range(args.box_epochs):
        head.train()
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = bce(head(Xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        s_te = torch.sigmoid(head(Xte.cuda()).squeeze(-1)).cpu()
    acc = ((s_te >= 0.5).float() == yte).float().mean().item()
    print(f"per-crop test acc: {acc:.4f}")
    print("box grounding (test):", box_iou_recall(ds_te, s_te.numpy()))

    torch.save(head.state_dict(), args.box_out)
    print("saved", args.box_out)


# ----------------------------- token head -----------------------------
def token_pos_weight(proc, n=5000):
    pos = neg = 0.0
    for e in TextDS("train").s[:n]:
        lab, val = token_label_valid(proc.tokenizer, e["text"], e["fake_text_pos"])
        pos += sum(lab)
        neg += sum(v - l for l, v in zip(lab, val))
    return torch.tensor(neg / max(pos, 1.0))


@torch.no_grad()
def eval_tokens(model, head, loader):
    head.eval()
    tp = fp = fn = 0
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, ft, _, _ = model(b["pixel_values"].cuda(), b["input_ids"].cuda(),
                                      b["attention_mask"].cuda())
        pred = (torch.sigmoid(head(ft.float()).squeeze(-1)) >= 0.5).cpu()
        lab, val = b["token_label"].bool(), b["token_valid"].bool()
        tp += int((pred & lab & val).sum())
        fp += int((pred & ~lab & val).sum())
        fn += int((~pred & lab & val).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / (P + R), 4) if P + R else 0.0}


def train_token_head(model, proc, args):
    collate = make_text_collate(proc)
    mk = lambda sp, sh: DataLoader(TextDS(sp), batch_size=args.token_batch_size, shuffle=sh,
                                   drop_last=sh, num_workers=8, collate_fn=collate, pin_memory=True)
    tr_loader, va_loader, te_loader = mk("train", True), mk("val", False), mk("test", False)

    head = grounding_head(dropout=0.2).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.token_lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=token_pos_weight(proc).cuda())

    best = -1
    for ep in range(args.token_epochs):
        head.train()
        run = 0.0
        for step, b in enumerate(tr_loader):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, _, ft, _, _ = model(b["pixel_values"].cuda(), b["input_ids"].cuda(),
                                          b["attention_mask"].cuda())
            logits = head(ft.float()).squeeze(-1)
            tl, tv = b["token_label"].cuda(), b["token_valid"].cuda()
            loss = (bce(logits, tl) * tv).sum() / tv.sum()        # mask out special tokens
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"  ep{ep} {step}/{len(tr_loader)} loss {loss.item():.3f}", end="\r")

        vm = eval_tokens(model, head, va_loader)
        if vm["F1"] > best:
            best = vm["F1"]
            torch.save(head.state_dict(), args.token_out)
        print(f"\nep{ep} | loss {run/len(tr_loader):.3f} | val token {vm}")

    head.load_state_dict(torch.load(args.token_out))
    print("token grounding (test):", eval_tokens(model, head, te_loader))
    print("saved", args.token_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-path", required=True)
    ap.add_argument("--checkpoint", required=True, help="trained detector .pt")
    ap.add_argument("--task", choices=["box", "token", "both"], default="both")
    # box head
    ap.add_argument("--box-epochs", type=int, default=15)
    ap.add_argument("--box-lr", type=float, default=1e-3)
    ap.add_argument("--box-batch-size", type=int, default=512)
    ap.add_argument("--box-max-imgs", type=int, default=15000)
    ap.add_argument("--box-out", default="checkpoints/box_head.pt")
    # token head
    ap.add_argument("--token-epochs", type=int, default=4)
    ap.add_argument("--token-lr", type=float, default=1e-3)
    ap.add_argument("--token-batch-size", type=int, default=128)
    ap.add_argument("--token-out", default="checkpoints/token_head.pt")
    args = ap.parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    proc = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    model = build_detector(args.clip_path).cuda()
    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    if args.task in ("box", "both"):
        train_box_head(model, proc, args)
    if args.task in ("token", "both"):
        train_token_head(model, proc, args)


if __name__ == "__main__":
    main()