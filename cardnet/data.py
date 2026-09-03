import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPY = os.path.join(ROOT, "data", "npy")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load(name):
    x = np.load(os.path.join(NPY, f"{name}_x.npy"))
    yp = os.path.join(NPY, f"{name}_y.npy")
    y = np.load(yp) if os.path.exists(yp) else None
    return x, y


def to_tensor(x_uint8):
    """uint8 HWC array -> float tensor NCHW in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(x_uint8)).float().div_(255.)
    return t.permute(0, 3, 1, 2).contiguous()


def stratified_subset(y, n, seed=0):
    """n indices, balanced across the 10 classes, deterministic in `seed`."""
    rng = np.random.RandomState(seed)
    per = n // 10
    idx = []
    for c in range(10):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, per, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


class Normalize(torch.nn.Module):
    """Input normalisation kept *inside* the model so that all attacks and
    all purification operate in the natural [0,1] image space."""

    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))
