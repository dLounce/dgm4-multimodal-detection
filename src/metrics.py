"""Evaluation metrics for DGM4.

Follows the metric protocol from the DGM4 paper (Shao et al., CVPR 2023):

    detection (binary real/fake) ...... ACC, AUC, EER
    fine-grained type (multi-label) ... mAP, CF1 (macro-F1), OF1 (micro-F1)
    image grounding (bounding box) .... mean IoU, IoU@0.5, IoU@0.75
    text grounding (tampered tokens) .. precision, recall, F1 (micro)

Each function takes plain arrays/lists and returns a dict, so they're easy to
log or aggregate across runs.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ---------- binary real/fake: ACC, AUC, EER ----------
def binary_metrics(y_true, y_score):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    acc = accuracy_score(y_true, y_score >= 0.5)
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))          # equal error rate operating point
    return {"ACC": acc, "AUC": auc, "EER": float((fpr[i] + fnr[i]) / 2)}


# ---------- multi-label type [fs, fa, ts, ta]: mAP, CF1, OF1 ----------
def multilabel_metrics(y_true, y_score, thr=0.5):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_pred = (y_score >= thr).astype(int)
    ap = np.atleast_1d(average_precision_score(y_true, y_score, average=None))
    return {
        "mAP": float(ap.mean()),
        "CF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "OF1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "AP_per_class": [round(float(x), 3) for x in ap],
    }


# ---------- bbox grounding: IoU_mean, IoU50, IoU75 ----------
def box_iou(a, b):                                # (N,4),(N,4) as [x1,y1,x2,y2]
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda z: (z[:, 2] - z[:, 0]).clip(0) * (z[:, 3] - z[:, 1]).clip(0)
    return inter / np.clip(area(a) + area(b) - inter, 1e-6, None)


def bbox_metrics(pred, gt):
    iou = box_iou(pred, gt)
    return {
        "IoU_mean": float(iou.mean()),
        "IoU50": float((iou >= 0.5).mean()),
        "IoU75": float((iou >= 0.75).mean()),
    }


# ---------- token grounding (tampered tokens): P, R, F1 (micro) ----------
def token_metrics(preds, gts):
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        p, g = set(p), set(g)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000

    yb = rng.integers(0, 2, n)
    print("binary perfect:", binary_metrics(yb, yb.astype(float)))
    print("binary random :", binary_metrics(yb, rng.random(n)))

    gt = rng.integers(0, 100, (n, 2))
    gt = np.hstack([gt, gt + rng.integers(20, 60, (n, 2))])
    print("bbox   perfect:", bbox_metrics(gt, gt))
    print("bbox   shifted:", bbox_metrics(gt + 15, gt))

    g = [list(rng.choice(20, 4, replace=False)) for _ in range(n)]
    print("token  perfect:", token_metrics(g, g))