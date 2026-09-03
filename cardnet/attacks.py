"""Threat-model simulation (Chapter 3, Table 2)."""
import torch
import torch.nn.functional as F


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, steps=10, alpha=None, random_start=True):
    alpha = alpha or eps / 4
    x0 = x.clone().detach()
    xa = x0 + (torch.empty_like(x0).uniform_(-eps, eps) if random_start else 0)
    xa = xa.clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = (x0 + (xa - x0).clamp(-eps, eps)).clamp(0, 1).detach()
    return xa


class BPDAPurified(torch.nn.Module):
    """Adaptive-attack surrogate: purification is run in the forward pass and
    approximated by the identity in the backward pass (BPDA, Athalye et al.
    2018). Optionally averages gradients over `eot` stochastic draws (EOT)."""

    def __init__(self, purifier, classifier, fixed_t=100, eot=1):
        super().__init__()
        self.p, self.c, self.fixed_t, self.eot = purifier, classifier, fixed_t, eot

    def forward(self, x):
        outs = 0
        for _ in range(self.eot):
            with torch.no_grad():
                xh = self.p.purify(x.detach(), fixed_t=self.fixed_t)
            # straight-through: value of x_hat, gradient of x
            xh = x + (xh - x).detach()
            outs = outs + self.c(xh)
        return outs / self.eot
