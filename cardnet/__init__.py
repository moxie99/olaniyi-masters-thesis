"""CARD-Net: Cascaded Adversarial-Robustness and Distribution-shift Defense Network.

Reference implementation of the five-stage architecture specified in
Chapter 3 of the thesis, scoped to the CPU compute budget documented in
Chapter 4.
"""
__all__ = ["data", "models", "purify", "detect", "certify", "fusion",
           "router", "adapt", "attacks", "corrupt", "metrics", "pipeline"]
