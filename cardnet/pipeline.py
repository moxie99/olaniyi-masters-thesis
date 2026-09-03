"""End-to-end CARD-Net inference (Chapter 3, Alg. 6) and its ablations."""
import time
import numpy as np
import torch

from . import fusion, router
from .certify import SmoothedClassifier


class CARDNet:
    """Composes Stages 1-5. `ablate` disables one stage at a time:
       'none' | 'no_purify' | 'no_detect' | 'no_smooth' | 'no_fusion'
    """

    def __init__(self, purifier, detector, student, sigma=0.25, n_smooth=32,
                 tau_acc=0.2, tau_def=0.5, ablate="none"):
        self.p, self.d, self.student = purifier, detector, student
        self.smooth = SmoothedClassifier(student, sigma=sigma, n=n_smooth)
        self.tau_acc, self.tau_def = tau_acc, tau_def
        self.ablate = ablate

    @torch.no_grad()
    def __call__(self, x, bs=128, return_all=True):
        out = {"pred": [], "conf": [], "u": [], "H": [], "s": [],
               "s_adv": [], "s_shift": [], "route": [], "T": []}
        t0 = time.time()
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]

            # ---- Stage 1 -------------------------------------------------
            if self.ablate == "no_purify":
                xh = xb
                Tb = torch.zeros(len(xb))
            else:
                s_pre = self.d.pre_severity(xb)
                Tb = self.p.depth(s_pre).float()
                xh = self.p.purify(xb, s_pre=s_pre)

            # ---- Stage 2 -------------------------------------------------
            if self.ablate == "no_detect":
                s = torch.zeros(len(xb))
                s_adv = torch.zeros(len(xb)); s_shift = torch.zeros(len(xb))
            else:
                s, s_adv, s_shift = self.d.severity(xb, xh)[:3]

            # ---- Stage 3 -------------------------------------------------
            if self.ablate == "no_smooth":
                p = torch.softmax(self.student(xh), dim=1)
            else:
                p = self.smooth.soft_probs(xh)

            # ---- Stage 4 -------------------------------------------------
            pt = p if self.ablate in ("no_fusion", "no_detect") else fusion.fuse(p, s)
            u, H = fusion.uncertainty(pt)

            # ---- Stage 5 -------------------------------------------------
            r = router.route(u, self.tau_acc, self.tau_def)

            out["pred"].append(pt.argmax(1)); out["conf"].append(pt.max(1).values)
            out["u"].append(u); out["H"].append(H); out["s"].append(s)
            out["s_adv"].append(s_adv); out["s_shift"].append(s_shift)
            out["route"].append(r); out["T"].append(Tb)

        res = {k: torch.cat(v).numpy() for k, v in out.items()}
        res["latency_s_per_img"] = (time.time() - t0) / len(x)
        return res


class BaselineWrapper:
    """Uniform result dict for an undefended / adversarially-trained baseline,
    so that every row of the results tables is produced by the same code path."""

    def __init__(self, model, detector=None):
        self.m, self.d = model, detector

    @torch.no_grad()
    def __call__(self, x, bs=256, **kw):
        t0 = time.time()
        ps, ss = [], []
        for i in range(0, len(x), bs):
            ps.append(torch.softmax(self.m(x[i:i + bs]), dim=1))
        p = torch.cat(ps)
        u, H = fusion.uncertainty(p)
        # baseline OOD score: maximum-softmax-probability (Hendrycks & Gimpel)
        s = 1.0 - p.max(1).values
        res = {"pred": p.argmax(1).numpy(), "conf": p.max(1).values.numpy(),
               "u": u.numpy(), "H": H.numpy(), "s": s.numpy(),
               "s_adv": np.zeros(len(x)), "s_shift": s.numpy(),
               "route": np.zeros(len(x), dtype=np.int64), "T": np.zeros(len(x))}
        res["latency_s_per_img"] = (time.time() - t0) / len(x)
        return res
