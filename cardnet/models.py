import json, os, sys
import torch
import torch.nn as nn

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

from pytorch_cifar_models import (cifar10_resnet20, cifar10_resnet32,
                                  cifar10_resnet56, cifar10_vgg16_bn)
from ddpm_torch.models.unet import UNet
from .data import Normalize

CKPT = os.path.join(ROOT, "checkpoints")
_ARCH = {"resnet20": cifar10_resnet20, "resnet32": cifar10_resnet32,
         "resnet56": cifar10_resnet56, "vgg16_bn": cifar10_vgg16_bn}


class Classifier(nn.Module):
    """Normalisation + backbone. Accepts images in [0,1]."""

    def __init__(self, arch="resnet20"):
        super().__init__()
        self.norm = Normalize()
        self.net = _ARCH[arch](pretrained=False)
        self.arch = arch

    def forward(self, x):
        return self.net(self.norm(x))

    def features(self, x):
        """Penultimate (post-global-pool) embedding, used by Stage 2."""
        n = self.net
        if self.arch.startswith("vgg"):
            h = n.features(self.norm(x))
            h = torch.flatten(h, 1)
            for layer in list(n.classifier)[:-1]:
                h = layer(h)
            return h
        h = n.conv1(self.norm(x))
        h = n.bn1(h); h = n.relu(h)
        h = n.layer1(h); h = n.layer2(h); h = n.layer3(h)
        h = n.avgpool(h)
        return torch.flatten(h, 1)


def load_pretrained(arch="resnet20"):
    m = Classifier(arch)
    sd = torch.load(os.path.join(CKPT, f"cifar10_{arch}.pt"), map_location="cpu")
    m.net.load_state_dict(sd)
    m.eval()
    return m


def load_local(path, arch="resnet20"):
    m = Classifier(arch)
    sd = torch.load(path, map_location="cpu")
    m.net.load_state_dict(sd["net"] if "net" in sd else sd)
    m.eval()
    return m


def load_unet():
    cfg = json.load(open(os.path.join(CODE, "configs", "cifar10.json")))
    u = UNet(out_channels=3, **cfg["model"])
    u.load_state_dict(torch.load(os.path.join(CKPT, "ddpm_cifar10_2040.pt"),
                                 map_location="cpu"))
    u.eval()
    for p in u.parameters():
        p.requires_grad_(False)
    return u, cfg["diffusion"]


# --------------------------------------------------------------------------
# LoRA adapters (used by the continual-adaptation module, Eq. 13)
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update dW = BA."""

    def __init__(self, base: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B)


def inject_lora(model: Classifier, r=4, alpha=8.0):
    """Inject a LoRA adapter into the classifier head and freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.net.fc = LoRALinear(model.net.fc, r=r, alpha=alpha)
    return [model.net.fc.A, model.net.fc.B]
