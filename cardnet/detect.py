"""Stage 2 - Multi-Channel Shift Detection (Chapter 3, Eqs. 4-6, Alg. 2).

Chapter 3 describes Stage 2 in two places that do not quite agree. The Stage 2
row of the component table calls for "an energy-based likelihood score" combined
with "a feature-space (Mahalanobis) distance score"; Equation 4.1 instead defines
the first channel as the purification residual ||x - x_hat||^2. Both signals are
real and they are not redundant - the residual measures how much perturbation
Stage 1 removed, the energy measures how well the input is explained by the
in-distribution model, and the Mahalanobis distance measures how far its features
sit from the class-conditional manifold.

This implementation therefore carries all three channels and lets the calibration
decide their relative weight on a simplex grid, generalising the single mixing
weight omega of Equation 6 to a weight vector (w_adv, w_energy, w_shift). Setting
w_energy = 0 recovers the two-channel form of Equation 6 exactly.
"""
import numpy as np
import torch

from .purify import hf_energy


class MinMax:
    """norm(.) of Eqs. 4-6: min-max normalisation fitted on the calibration set.

    The raw channel scores are unbounded positive quantities whose ranges differ
    by orders of magnitude - the high-frequency energy of a Gaussian-noise
    corruption at severity 3 exceeds that of an 8/255 l-infinity perturbation by
    several decades. Applying a linear min-max map across both would compress
    the entire adversarial range into the bottom few per cent of [0, 1] and
    leave the step-selection sigmoid of Eq. 3 effectively flat over exactly the
    inputs it must discriminate. The normaliser therefore operates on log(1 + v),
    which is monotone (so no ranking, and hence no AUROC, is affected) and puts
    perturbations and corruptions on a commensurable scale.

    `shift` is added before the log so that scores which can be negative - the
    free energy is one - remain in the domain of log1p.
    """

    def __init__(self, lo=0.0, hi=1.0, log=True, shift=0.0):
        self.lo, self.hi, self.log, self.shift = float(lo), float(hi), bool(log), float(shift)

    def _t(self, v):
        if not self.log:
            return v
        if torch.is_tensor(v):
            return torch.log1p((v + self.shift).clamp_min(0))
        return np.log1p(np.clip(v + self.shift, 0, None))

    def fit(self, v):
        v = np.asarray(v, dtype=np.float64)
        if self.log:
            m = float(v.min())
            self.shift = 0.0 if m >= 0 else (-m + 1e-3)
        t = self._t(v)
        self.lo = float(np.percentile(t, 1))
        self.hi = float(np.percentile(t, 99))
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-6
        return self

    def __call__(self, v):
        t = torch.is_tensor(v)
        out = (self._t(v) - self.lo) / (self.hi - self.lo)
        return out.clamp(0, 1) if t else np.clip(out, 0, 1)

    def state(self):
        return {"lo": self.lo, "hi": self.hi, "log": self.log, "shift": self.shift}

    @staticmethod
    def from_state(d):
        return MinMax(d["lo"], d["hi"], d.get("log", True), d.get("shift", 0.0))


class ShiftDetector:
    """Stage 2 as a TWO-COMPONENT detector.

    The single scalar severity of Equation 6 is asked to report two different
    things at once - "this input has been perturbed" and "this input is from
    another distribution" - through one number with one mixing weight. Fitting
    that weight against a blended calibration set makes it serve whichever
    threat the blend emphasises and fail on the other; fitting it against the
    average of the two objectives merely relocates the failure. Stage 2
    therefore emits two normalised scores,

        s_pert(x)  = norm( lambda_adv ||x - x_hat||^2 )                 (Eq. 4)
        s_nov(x)   = w_E norm(E(x)) + (1 - w_E) norm(d_Mahalanobis(x))  (Eq. 5)

    and Equation 6 becomes their combination

        s(x) = max( s_pert(x), s_nov(x) ),

    which is the reliability signal Stages 1, 4 and 5 consume. Adversarial-input
    detection is scored by s_pert, semantic novelty by s_nov, and each is
    reported against the task it was designed for. Setting w_E = 0 and replacing
    the max by a weighted sum recovers the original two-channel Equation 6.
    """

    def __init__(self, encoder, lam_adv=1.0, w_energy=0.5, T=1.0):
        self.enc = encoder            # frozen (optionally OE-hardened) backbone h(.)
        self.lam_adv = lam_adv
        self.w_energy = float(w_energy)
        self.T = float(T)
        self.mu = None
        self.prec = None
        self.n_adv = MinMax()
        self.n_energy = MinMax()
        self.n_shift = MinMax()
        self.n_pre = MinMax()

    # -- fit class-conditional Gaussians on in-distribution training data --
    @torch.no_grad()
    def fit_gaussians(self, x, y, bs=256):
        feats = []
        for i in range(0, len(x), bs):
            feats.append(self.enc.features(x[i:i + bs]))
        f = torch.cat(feats)
        d = f.shape[1]
        mus, centred = [], []
        for c in range(10):
            fc = f[y == c]
            m = fc.mean(0)
            mus.append(m)
            centred.append(fc - m)
        self.mu = torch.stack(mus)
        cc = torch.cat(centred)
        cov = (cc.T @ cc) / cc.shape[0]
        cov += 1e-4 * torch.eye(d)
        self.prec = torch.linalg.inv(cov)
        return f

    # -- Eq. 5: class-conditional Mahalanobis distance --------------------
    @torch.no_grad()
    def mahalanobis(self, x, bs=256):
        out = []
        for i in range(0, len(x), bs):
            f = self.enc.features(x[i:i + bs])
            d = f.unsqueeze(1) - self.mu.unsqueeze(0)
            m = torch.einsum("bkd,de,bke->bk", d, self.prec, d)
            out.append(m.min(dim=1).values)
        return torch.cat(out)

    # -- free energy (Liu et al., 2020): E(x) = -T logsumexp(f(x)/T) ------
    @torch.no_grad()
    def energy(self, x, bs=256):
        out = []
        for i in range(0, len(x), bs):
            logits = self.enc(x[i:i + bs])
            out.append(-self.T * torch.logsumexp(logits / self.T, dim=1))
        return torch.cat(out)

    # -- Eq. 4: raw adversarial-residual score ----------------------------
    def adv_score(self, x, x_hat):
        return self.lam_adv * (x - x_hat).pow(2).flatten(1).sum(1)

    # -- the novelty component -------------------------------------------
    @torch.no_grad()
    def novelty(self, x):
        return (self.w_energy * self.n_energy(self.energy(x))
                + (1 - self.w_energy) * self.n_shift(self.mahalanobis(x))).clamp(0, 1)

    # -- cheap pre-estimate used by Stage 1 before x_hat exists -----------
    @torch.no_grad()
    def pre_severity(self, x):
        pert = self.n_pre(hf_energy(x))
        return torch.maximum(pert, self.novelty(x)).clamp(0, 1)

    # -- Eq. 6 (generalised) ----------------------------------------------
    def severity(self, x, x_hat):
        """Returns (s, s_pert, s_nov, raw_residual, raw_energy, raw_mahalanobis)."""
        raw_adv = self.adv_score(x, x_hat)
        pert = self.n_adv(raw_adv)
        nov = self.novelty(x)
        s = torch.maximum(pert, nov).clamp(0, 1)
        with torch.no_grad():
            raw_e = self.energy(x)
            raw_m = self.mahalanobis(x)
        return s, pert, nov, raw_adv, raw_e, raw_m

    def state(self):
        return {"lam_adv": self.lam_adv, "w_energy": self.w_energy, "T": self.T,
                "mu": self.mu, "prec": self.prec,
                "n_adv": self.n_adv.state(), "n_energy": self.n_energy.state(),
                "n_shift": self.n_shift.state(), "n_pre": self.n_pre.state()}

    def load_state(self, d):
        self.lam_adv = d["lam_adv"]
        self.w_energy = d.get("w_energy", 0.5)
        self.T = d.get("T", 1.0)
        self.mu, self.prec = d["mu"], d["prec"]
        self.n_adv = MinMax.from_state(d["n_adv"])
        self.n_shift = MinMax.from_state(d["n_shift"])
        self.n_pre = MinMax.from_state(d["n_pre"])
        if "n_energy" in d:
            self.n_energy = MinMax.from_state(d["n_energy"])
        return self
