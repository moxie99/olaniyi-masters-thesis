"""Stage 1 - Diffusion-Based Adaptive Purification (Chapter 3, Eqs. 1-3, Alg. 1).

The forward process follows Eq. 1; the reverse process implements Eq. 2 on a
strided (DDIM) subsequence so that the number of network evaluations stays
inside the CPU budget documented in Chapter 4. Purification depth T(x) is
selected per input by Eq. 3.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.abar = torch.cumprod(alphas, dim=0).float()   # \bar{alpha}_t


def hf_energy(x):
    """Cheap high-frequency-residual proxy used for the *pre*-estimate of
    perturbation severity, so that T(x) can be chosen before x_hat exists."""
    k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    k = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=3)
    return (x - blur).pow(2).flatten(1).sum(1)


class AdaptivePurifier:
    """Stage 1.

    Parameters
    ----------
    unet    : trained noise predictor eps_theta
    t_max   : maximum permitted diffusion depth T_max (Eq. 3)
    stride  : DDIM step size; network evaluations = ceil(T(x)/stride)
    gamma, tau : learned scale / threshold of the step-selection sigmoid (Eq. 3)
    eta     : 0 -> deterministic DDIM reverse, 1 -> ancestral (DDPM) reverse
    """

    def __init__(self, unet, diff_cfg, t_max=150, stride=15,
                 gamma=6.0, tau=3.0, eta=0.0):
        self.unet = unet
        self.sched = DiffusionSchedule(diff_cfg["timesteps"],
                                       diff_cfg["beta_start"], diff_cfg["beta_end"])
        self.t_max, self.stride = t_max, stride
        self.gamma, self.tau, self.eta = gamma, tau, eta
        self.nfe = 0            # cumulative network function evaluations

    # -- Eq. 3 ----------------------------------------------------------
    def depth(self, s_pre):
        """T(x) = ceil(T_max * sigmoid(gamma*s - tau)), s in [0,1]."""
        t = torch.ceil(self.t_max * torch.sigmoid(self.gamma * s_pre - self.tau))
        return t.clamp(0, self.t_max).long()

    # -- Eq. 1 ----------------------------------------------------------
    def forward_diffuse(self, x, t):
        """q(x_t | x_0); x in [0,1] is mapped to the model's [-1,1] domain."""
        x = x * 2 - 1
        ab = self.sched.abar[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return ab.sqrt() * x + (1 - ab).sqrt() * noise

    # -- Eq. 2 on a strided subsequence ---------------------------------
    @torch.no_grad()
    def _reverse(self, x_t, t_start, chunk=48):
        """Reverse from a common t_start for the whole (bucketed) batch.

        The UNet's 16x16 self-attention allocates a B x H x W x H x W tensor, so
        the batch is processed in fixed-size chunks to keep peak memory inside
        the 7.8 GB budget of the execution environment.
        """
        if x_t.shape[0] > chunk:
            return torch.cat([self._reverse(x_t[i:i + chunk], t_start, chunk)
                              for i in range(0, x_t.shape[0], chunk)])
        seq = list(range(t_start, -1, -self.stride))
        if seq[-1] != 0:
            seq.append(0)
        for i in range(len(seq) - 1):
            t_cur, t_nxt = seq[i], seq[i + 1]
            tt = torch.full((x_t.shape[0],), t_cur, dtype=torch.long)
            eps = self.unet(x_t, tt)
            self.nfe += x_t.shape[0]
            ab_c = self.sched.abar[t_cur]
            ab_n = self.sched.abar[t_nxt]
            x0 = (x_t - (1 - ab_c).sqrt() * eps) / ab_c.sqrt()
            x0 = x0.clamp(-1, 1)
            if self.eta > 0:
                sigma = self.eta * ((1 - ab_n) / (1 - ab_c)).sqrt() * (1 - ab_c / ab_n).sqrt()
                c = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt()
                x_t = ab_n.sqrt() * x0 + c * eps + sigma * torch.randn_like(x_t)
            else:
                x_t = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
        return ((x_t + 1) / 2).clamp(0, 1)

    def purify(self, x, s_pre=None, fixed_t=None, n_buckets=4):
        """Algorithm 1. Inputs/outputs are images in [0,1].

        Inputs are bucketed by their selected depth so a whole bucket shares
        one reverse trajectory - this is what makes per-input adaptive depth
        affordable on CPU.
        """
        if fixed_t is not None:
            t = torch.full((x.shape[0],), int(fixed_t), dtype=torch.long)
        else:
            t = self.depth(s_pre)
        out = x.clone()
        # quantise depths onto a small grid so batching stays effective
        grid = torch.linspace(0, self.t_max, n_buckets + 1).long()[1:]
        for g in grid.tolist():
            lo = 0 if g == grid[0].item() else grid[(grid == g).nonzero()[0, 0] - 1].item()
            m = (t > lo) & (t <= g)
            if m.sum() == 0:
                continue
            xm = x[m]
            xt = self.forward_diffuse(xm, torch.full((xm.shape[0],), g, dtype=torch.long))
            out[m] = self._reverse(xt, g)
        return out
