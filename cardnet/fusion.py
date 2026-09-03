"""Stage 4 - Bayesian Uncertainty Fusion (Chapter 3, Eqs. 10-11, Alg. 4).

Note on Eq. 10. As written in Chapter 3,

    p~(c|x) = p(c|x_hat) r(x) / sum_c' p(c'|x_hat) r(x)

the scalar reliability r(x) is common to every class and therefore cancels
exactly under the normalising sum, leaving p~ = p. The equation is degenerate
as a *combination rule*. Two standard non-degenerate readings of the same
intent - "low reliability must move mass away from the classifier's own
opinion" - are implemented here and selected on the validation split:

  linear  (linear opinion pool)     p~ = r p + (1-r) Uniform
  power   (logarithmic opinion pool) p~ ∝ p^r

Both reduce to p~ = p at r = 1 and to the uniform distribution at r = 0.
"""
import torch

MODE = "linear"


def fuse(p, s, kappa=1.0, floor=1e-3, mode=None):
    """Eq. 10 (see module docstring). `s` is the severity of Eq. 6 in [0,1]."""
    mode = mode or MODE
    r = (1.0 - s).clamp(floor, 1.0).unsqueeze(1)          # reliability prior r(x)
    if mode == "power":
        lp = p.clamp_min(1e-12).log() * (r ** kappa)
        pt = torch.softmax(lp, dim=1)
    else:
        u = torch.full_like(p, 1.0 / p.shape[1])
        pt = (r ** kappa) * p + (1.0 - r ** kappa) * u
        pt = pt / pt.sum(1, keepdim=True)
    return pt


def uncertainty(pt):
    """Eq. 11: u(x) = 1 - max_c p~(c|x) and H(x) = -sum_c p~ log p~."""
    umax = 1.0 - pt.max(1).values
    ent = -(pt.clamp_min(1e-12) * pt.clamp_min(1e-12).log()).sum(1)
    return umax, ent
