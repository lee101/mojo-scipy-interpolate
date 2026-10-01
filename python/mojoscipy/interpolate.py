"""`interp1d`, the legacy 1-D interpolator.

The linear, nearest and previous/next evaluation paths run in Mojo, mirroring
`_call_linear`, `_call_nearest` and `_call_previousnext`; the spline kinds are
built with `make_interp_spline` and evaluated by `BSpline.__call__`.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib, zeros
from .bsplines import make_interp_spline


def _do_extrapolate(fill_value):
    """Whether `fill_value` asks for extrapolation rather than a fill."""
    if isinstance(fill_value, str) and fill_value == "extrapolate":
        return True
    if isinstance(fill_value, tuple) and len(fill_value) == 2:
        return _do_extrapolate(fill_value[0]) and _do_extrapolate(fill_value[1])
    return False



def _shifted(x: np.ndarray, direction: float) -> np.ndarray:
    """`np.nextafter(x, direction)`, without a denormal.

    `nextafter(0.0, inf)` is 5e-324, and the vectorised build of the search
    flushes denormals to zero: the shifted knot would then compare equal to
    0.0 instead of strictly above it, and `searchsorted(..., 'right')` would
    return the knot instead of the one after it.  Substituting the smallest
    normal keeps the answer for every query that is not itself a denormal.
    """
    tiny = np.finfo(np.float64).tiny
    shifted = np.nextafter(x, direction)
    denormal = (shifted != 0.0) & (np.abs(shifted) < tiny)
    return f64(np.where(denormal, np.copysign(tiny, shifted), shifted))


class interp1d:
    """Interpolate a 1-D function (legacy interface).

    `kind` is one of ``'linear'``, ``'nearest'``, ``'nearest-up'``,
    ``'previous'``, ``'next'``, ``'zero'``, ``'slinear'``, ``'quadratic'``,
    ``'cubic'`` or an integer spline order.
    """

    def __init__(self, x, y, kind="linear", axis=-1, copy=True,
                 bounds_error=None, fill_value=np.nan, assume_sorted=False):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if not assume_sorted:
            ind = np.argsort(x, kind="mergesort")
            x = x[ind]
            y = np.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float64)

        # Backward compatibility
        self.axis = axis = int(axis) % y.ndim
        self._y_axis = self.axis
        self._y_extra_shape = y.shape[:self.axis] + y.shape[self.axis + 1:]

        # Interpolation goes internally along the first axis
        self.x = np.array(x, copy=copy) if copy else x
        self.y = y
        self._y = self._reshape_yi(y)
        self.bounds_error = bounds_error
        self.copy = copy
        self.dtype = y.dtype.type
        self.kind = kind

        if kind in ("zero", "slinear", "quadratic", "cubic"):
            order = {"zero": 0, "slinear": 1, "quadratic": 2, "cubic": 3}[kind]
            kind = "spline"
        elif isinstance(kind, (int, np.integer)):
            order = int(kind)
            kind = "spline"
        elif kind not in ("linear", "nearest", "nearest-up", "previous", "next"):
            raise NotImplementedError(
                f"{kind} is unsupported: Use fitpack routines for other types."
            )
        self._order = order if kind == "spline" else None

        if kind in ("linear", "nearest", "nearest-up", "previous", "next"):
            minval = 1
            if kind == "nearest":
                self._side = 0  # searchsorted(..., 'left')
            elif kind in ("previous",):
                self._side = 0
            else:
                self._side = 1  # 'right'
            self._call = kind
        else:
            minval = order + 1
            self._spline = make_interp_spline(self.x, self._y, k=order,
                                              check_finite=False)
            self._call = "spline"

        if len(self.x) < minval:
            raise ValueError(
                f"x and y arrays must have at least {minval} entries"
            )
        self.fill_value = fill_value

    def _reshape_yi(self, yi):
        """Move the interpolation axis to the front and flatten the rest."""
        return np.ascontiguousarray(
            np.moveaxis(yi, self._y_axis, 0).reshape(len(self.x), -1),
            dtype=np.float64,
        )

    def _finish_y(self, y, x_shape):
        """Reshape the interpolated values back to an N-D array like `y`."""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx + self._y_axis, nx + ny)))
            y = y.transpose(s)
        return y

    @property
    def fill_value(self):
        """The fill value."""
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        if _do_extrapolate(fill_value):
            if self.bounds_error:
                raise ValueError("Cannot extrapolate and raise at the same time.")
            self.bounds_error = False
            self._extrapolate = True
        else:
            fv = np.asarray(fill_value, dtype=float)
            if fv.ndim:
                fv = fv.reshape(-1)
                if fv.size == 1:
                    below = above = fv[0]
                elif fv.size == 2:
                    below, above = fv
                else:
                    raise ValueError(
                        "fill_value must be a scalar or a length-2 sequence"
                    )
            else:
                below = above = float(fv)
            self._fill_value_below = below
            self._fill_value_above = above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        self._fill_value_orig = fill_value

    def _y2d(self):
        """The data as the kernels want it: interpolation axis first, 2-D."""
        return self._y

    def _check_bounds(self, x_new):
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]
        if self.bounds_error and below_bounds.any():
            raise ValueError(
                f"A value ({x_new[np.argmax(below_bounds)]}) in x_new is below "
                f"the interpolation range's minimum value ({self.x[0]})."
            )
        if self.bounds_error and above_bounds.any():
            raise ValueError(
                f"A value ({x_new[np.argmax(above_bounds)]}) in x_new is above "
                f"the interpolation range's maximum value ({self.x[-1]})."
            )
        return below_bounds, above_bounds

    def _evaluate(self, x_new):
        x_new = np.ascontiguousarray(x_new.ravel(), dtype=np.float64)
        if self._call == "spline":
            return np.asarray(self._spline(x_new), dtype=np.float64)
        y2d = self._y2d()
        ycols = y2d.shape[1]
        n = len(self.x)
        nnew = x_new.size
        out = zeros(nnew * ycols)
        x2 = f64(self.x)
        if self._call == "linear":
            lib().msi_linear_interp(
                addr(x2), addr(y2d), n, addr(x_new), nnew, addr(out), ycols
            )
        elif self._call in ("nearest", "nearest-up"):
            x_bds = f64(0.5 * (self.x[:-1] + self.x[1:]))
            lib().msi_nearest_interp(
                addr(x_bds), n, addr(y2d), nnew, addr(x_new), self._side,
                addr(out), ycols,
            )
        else:  # previous / next
            ind = 0 if self._call == "previous" else 1
            direction = -np.inf if self._call == "previous" else np.inf
            x_shift = _shifted(self.x, direction)
            lib().msi_previous_next(
                addr(x_shift), len(x_shift), addr(y2d), n, nnew, addr(x_new),
                ind, self._side, addr(out), ycols,
            )
        y_new = out.reshape(nnew, ycols)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if nnew > 0:
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def __call__(self, x_new):
        """Evaluate the interpolant at `x_new`."""
        x_new = np.asarray(x_new, dtype=np.float64)
        return self._finish_y(self._evaluate(x_new), x_new.shape)
