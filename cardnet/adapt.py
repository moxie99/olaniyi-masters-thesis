"""Continual adaptation via LoRA + EWC (Chapter 3, Eq. 13, Alg. 5 lines 13-18)."""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .models import inject_lora


def fisher_diagonal(model, x, y, bs=128, max_batches=8):
    """Diagonal empirical Fisher F_i over the trainable (LoRA) parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    fisher = [torch.zeros_like(p) for p in params]
    nb = 0
    for i in range(0, len(x), bs):
        if nb >= max_batches:
            break
        model.zero_grad()
        logits = model(x[i:i + bs])
        loss = F.cross_entropy(logits, y[i:i + bs])
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for f, g in zip(fisher, grads):
            if g is not None:
                f += g.detach() ** 2
        nb += 1
    return [f / max(nb, 1) for f in fisher]


def adapt_cycle(model, x_buf, y_buf, x_prev, y_prev, lam_ewc=100.0,
                epochs=3, lr=1e-3, bs=64, rank=4, log=print):
    """One adaptation cycle: inject a LoRA adapter, fit it on the flagged
    buffer under the EWC penalty of Eq. 13, and return the adapted model."""
    model = copy.deepcopy(model)
    params = inject_lora(model, r=rank)
    model.train()

    fisher = fisher_diagonal(model, x_prev, y_prev)
    theta_star = [p.detach().clone() for p in params]

    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(x_buf))
        tot = 0.0
        for i in range(0, len(x_buf), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            task = F.cross_entropy(model(x_buf[idx]), y_buf[idx])
            pen = 0.0
            for f, p, ps in zip(fisher, params, theta_star):
                pen = pen + (f * (p - ps) ** 2).sum()
            loss = task + (lam_ewc / 2.0) * pen
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        hist.append(tot / len(x_buf))
        log(f"    adapt epoch {ep+1}/{epochs}  loss={hist[-1]:.4f}")
    model.eval()
    return model, hist
