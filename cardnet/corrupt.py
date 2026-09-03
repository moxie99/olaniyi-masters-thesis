"""CIFAR-10-C style covariate shift.

The public CIFAR-10-C archive was not reachable from the execution
environment, so the 15 corruptions x 5 severities are regenerated locally with
the reference `imagecorruptions` implementation of Hendrycks & Dietterich
(2019) - the same code used to build the released archive.

`imagecorruptions` was written against an older scikit-image API in which
`gaussian` took a `multichannel` flag; current scikit-image replaced it with
`channel_axis`. A thin compatibility shim is installed before the package is
imported so the reference corruption functions run unmodified.
"""
import numpy as np
import skimage.filters as _skf

_orig_gaussian = _skf.gaussian


def _gaussian_comp(image, *args, **kwargs):
    mc = kwargs.pop("multichannel", None)
    if mc is not None and "channel_axis" not in kwargs:
        kwargs["channel_axis"] = -1 if mc else None
    return _orig_gaussian(image, *args, **kwargs)


_skf.gaussian = _gaussian_comp

# NumPy 2 removed the np.float_ alias that the reference `fog` corruption uses
if not hasattr(np, "float_"):
    np.float_ = np.float64
try:                                    # the package imports it by name
    import skimage
    skimage.filters.gaussian = _gaussian_comp
except Exception:
    pass

from imagecorruptions import corrupt, get_corruption_names  # noqa: E402
import imagecorruptions.corruptions as _ic                  # noqa: E402

_ic.gaussian = _gaussian_comp

CORRUPTIONS = get_corruption_names()   # the canonical 15


def apply_corruption(x_uint8, name, severity):
    """x_uint8: (N,32,32,3). imagecorruptions needs >=32px, which CIFAR meets."""
    out = np.empty_like(x_uint8)
    for i in range(len(x_uint8)):
        out[i] = corrupt(x_uint8[i], corruption_name=name, severity=severity)
    return out
