"""mojo-scipy-interpolate -- `scipy.interpolate`'s spline and RBF interpolators,
implemented in Mojo.

Names, argument order and defaults follow upstream, so swapping the import is
usually the whole migration for the covered subset. Arrays cross into Mojo by
address: a C-contiguous float64 NumPy array is never copied.
"""

from . import _rbfinterp, bsplines, interpolate, ppoly, rbf, rgi
from ._cubic import CubicHermiteSpline, CubicSpline, PchipInterpolator
from ._fitpack2 import RectBivariateSpline
from ._lib import build
from ._rbfinterp import (
    RBFInterpolator,
    cubic,
    gaussian,
    inverse_multiquadric,
    inverse_quadratic,
    linear,
    multiquadric,
    quintic,
    rbf_factory,
    thin_plate_spline,
)
from .bsplines import (
    BSpline,
    make_interp_spline,
    make_lsq_spline,
    splder,
    splantider,
)
from .interpolate import interp1d
from .ppoly import PPoly
from .rbf import Rbf
from .rgi import RegularGridInterpolator, interpn

__all__ = [
    "BSpline", "PPoly", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator",
    "RegularGridInterpolator", "interpn", "RectBivariateSpline",
    "interp1d",
    "Rbf", "RBFInterpolator", "rbf_factory",
    "linear", "thin_plate_spline", "cubic", "quintic", "multiquadric",
    "inverse_multiquadric", "inverse_quadratic", "gaussian",
    "build",
    "bsplines", "interpolate", "ppoly", "rbf", "rgi", "_rbfinterp",
]
