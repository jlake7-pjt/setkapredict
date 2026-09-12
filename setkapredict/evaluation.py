from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(np.mean(ids == i) * abs(y[ids == i].mean() - p[ids == i].mean())
                     for i in range(bins) if np.any(ids == i)))

def metrics(y, p):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, p >= .5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }
    out["selective"] = {}
    conf = np.maximum(p, 1-p)
    for threshold in (.60, .65, .70, .75, .80):
        keep = conf >= threshold
        out["selective"][str(threshold)] = {
            "coverage": float(keep.mean()),
            "accuracy": float(accuracy_score(y[keep], p[keep] >= .5)) if keep.any() else None,
            "n": int(keep.sum()),
        }
    return out

