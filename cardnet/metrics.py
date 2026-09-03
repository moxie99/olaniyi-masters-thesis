"""Evaluation metrics of Chapter 3, Table 3."""
import numpy as np


def _roc(pos, neg):
    """pos = scores for the positive (OOD) class, neg = in-distribution."""
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    return fpr, tpr, tp, fp


def auroc(pos, neg):
    fpr, tpr, _, _ = _roc(pos, neg)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def fpr_at_tpr(pos, neg, level=0.95):
    """FPR when TPR is held at `level` (FPR@95)."""
    fpr, tpr, _, _ = _roc(pos, neg)
    i = np.searchsorted(tpr, level, side="left")
    i = min(i, len(fpr) - 1)
    return float(fpr[i])


def aupr(pos, neg):
    fpr, tpr, tp, fp = _roc(pos, neg)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tpr
    return float(-np.trapezoid(prec, -rec)) if hasattr(np, "trapezoid") else float(-np.trapz(prec, -rec))


def ece(conf, correct, n_bins=15):
    """Expected Calibration Error."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def accuracy(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())
