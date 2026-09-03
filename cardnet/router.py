"""Stage 5 - Adaptive Decision Router (Chapter 3, Eq. 12, Alg. 5)."""
import numpy as np
import torch

ACCEPT, DEFER, FLAG = 0, 1, 2


def route(u, tau_acc, tau_def):
    """Eq. 12 -> {accept, defer, flag}."""
    d = torch.full_like(u, FLAG, dtype=torch.long)
    d[u < tau_def] = DEFER
    d[u < tau_acc] = ACCEPT
    return d


def calibrate_thresholds(u_id_clean, u_shifted, target_accept=0.90, target_flag=0.80):
    """tau_acc: the quantile of clean in-distribution uncertainty that accepts
    `target_accept` of clean traffic. tau_def: the quantile of shifted/attacked
    uncertainty above which `target_flag` of shifted traffic is flagged."""
    tau_acc = float(np.quantile(u_id_clean, target_accept))
    tau_def = float(np.quantile(u_shifted, 1.0 - target_flag))
    tau_def = max(tau_def, tau_acc + 1e-4)
    return tau_acc, tau_def
