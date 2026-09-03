"""Stage 3 - Certified-Robust Classification (Chapter 3, Eqs. 7-9, Alg. 3).

Randomized smoothing with Clopper-Pearson bounds and the Neyman-Pearson
certified radius, over a student trained by multi-teacher distillation.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, beta as beta_dist


def clopper_pearson(k, n, alpha):
    """Two-sided Clopper-Pearson interval for a Binomial proportion."""
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.isf(alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


class SmoothedClassifier:
    """g(x) of Eq. 7 with the certified l2 radius of Eq. 8."""

    def __init__(self, base, sigma=0.25, n=100, n0=25, alpha=0.001, bs=256):
        self.base, self.sigma = base, sigma
        self.n, self.n0, self.alpha, self.bs = n, n0, alpha, bs

    @torch.no_grad()
    def _counts(self, x, n):
        """Monte-Carlo class counts for a single image x (1,3,32,32)."""
        counts = torch.zeros(10)
        left = n
        while left > 0:
            b = min(self.bs, left)
            noise = torch.randn(b, *x.shape[1:]) * self.sigma
            logits = self.base((x + noise).clamp(0, 1))
            counts += torch.bincount(logits.argmax(1), minlength=10).float()
            left -= b
        return counts

    @torch.no_grad()
    def soft_probs(self, x, n=None):
        """Batched *soft* smoothed posterior p(c|x_hat) - the quantity Stage 4
        fuses. Averages softmax over n Gaussian draws for a whole batch."""
        n = n or self.n
        acc = torch.zeros(x.shape[0], 10)
        for _ in range(n):
            noise = torch.randn_like(x) * self.sigma
            acc += F.softmax(self.base((x + noise).clamp(0, 1)), dim=1)
        return acc / n

    @torch.no_grad()
    def certify(self, x):
        """Algorithm 3 for one image; returns (class or -1 for abstain, radius)."""
        c0 = self._counts(x, self.n0).argmax().item()
        counts = self._counts(x, self.n)
        k = int(counts[c0].item())
        pa, _ = clopper_pearson(k, self.n, 2 * self.alpha)
        if pa <= 0.5:
            return -1, 0.0
        # binary (one-vs-rest) form of Eq. 8: p_B <= 1 - p_A
        r = self.sigma * norm.ppf(pa)
        return c0, float(max(r, 0.0))


def distillation_loss(student_logits, teacher_logits_list, y, lam=0.6, T=4.0):
    """Eq. 9: (1-lambda) * CE  +  lambda * (1/M) * sum_m KL(student || teacher_m)."""
    ce = F.cross_entropy(student_logits, y)
    kl = 0.0
    ls = F.log_softmax(student_logits / T, dim=1)
    for tl in teacher_logits_list:
        pt = F.softmax(tl / T, dim=1)
        kl = kl + F.kl_div(ls, pt, reduction="batchmean") * (T * T)
    kl = kl / len(teacher_logits_list)
    return (1 - lam) * ce + lam * kl, ce.detach(), (kl.detach() if torch.is_tensor(kl) else kl)
