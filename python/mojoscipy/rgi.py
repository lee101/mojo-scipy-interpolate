"""`RegularGridInterpolator` and `interpn`.

`find_indices` and the 2-D linear path run in Mojo, exactly as upstream's
`_rgi_cython` does; the general N-D linear evaluation runs through
`msi_evaluate_linear_nd`. The spline methods fold a 1-D `BSpline` along each
axis, which is what upstream's `_evaluate_spline` does.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib, zeros
from ._fitpack2 import _basis
from .bsplines import make_interp_spline

_ALL_METHODS = ["linear", "nearest", "slinear", "quintic", "cubic", "pchip"]
_SPLINE_METHODS = ["slinear", "quintic", "cubic", "pchip"]
_SPLINE_DEGREE_MAP = {"linear": 1, "slinear": 1, "cubic": 3, "quintic": 5, "pchip": 3}


def _ndim_coords_from_arrays(points, ndim=None):
    """Convert `points` into a tuple of `ndim` broadcast coordinate arrays."""
    if isinstance(points, tuple) and len(points) == 1:
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
    else:
        p = np.asarray(points)
        if p.dtype == object:
            raise ValueError("points must be a tuple of arrays or an array")
        ndim = p.ndim if ndim is None else ndim
        p = np.broadcast_arrays(*(p[..., i] for i in range(ndim)))
    if ndim is not None and len(p) != ndim:
        raise ValueError("points must have the same number of dimensions as grid")
    return p


class RegularGridInterpolator:
    """Interpolator of specified order on a rectilinear grid in N >= 1 dimensions.

    `values` holds one scalar per grid point. Methods are ``'linear'``,
    ``'nearest'``, ``'slinear'``, ``'quintic'``, ``'cubic'`` and ``'pchip'``, as
    upstream.
    """

    def __init__(self, points, values, method="linear", fill_value=np.nan,
                 bounds_error=True):
        if method not in _ALL_METHODS:
            raise NotImplementedError(f"Method '{method}' is not defined")
        self.grid = tuple(np.ascontiguousarray(p, dtype=np.float64) for p in points)
        if np.prod(self.grid[0].shape) == 0:
            raise ValueError("There are no evaluation points.")
        if len(self.grid) > np.ndim(values):
            raise ValueError(
                f"There are {len(self.grid)} point arrays, but values has "
                f"{np.ndim(values)} dimensions"
            )
        for p in self.grid:
            if p.ndim != 1:
                raise ValueError("Points should be 1-D arrays")
            if p.size < 2:
                raise ValueError("There should be at least 2 values per axis")
            if not np.all(np.diff(p) > 0.0):
                raise ValueError("Points should be strictly ascending")
        values = np.asarray(values, dtype=np.float64)
        self.ndim = len(self.grid)
        if values.ndim < self.ndim or values.shape[: self.ndim] != tuple(
            p.size for p in self.grid
        ):
            raise ValueError(
                f"There are {self.grid[0].size} points and {values.shape[0]} values"
            )
        self.values = np.ascontiguousarray(values)
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self._tensor = None
        if method in _SPLINE_METHODS:
            self._validate_grid_dimensions(self.grid, method)

    def _validate_grid_dimensions(self, points, method):
        k = _SPLINE_DEGREE_MAP[method]
        for i, point in enumerate(points):
            ndim = len(np.atleast_1d(point))
            if ndim <= k:
                raise ValueError(
                    f"There are {ndim} points in dimension {i}, but method "
                    f"{method} requires at least {k + 1} points per dimension."
                )

    def _find_indices(self, xi):
        """Locate each point between the grid edges (Mojo kernel)."""
        ndim = self.ndim
        npoints = xi.shape[1]
        grid_addrs = i64(np.array([g.ctypes.data for g in self.grid], dtype=np.int64))
        grid_lens = i64(np.array([g.size for g in self.grid], dtype=np.int64))
        indices = i64(np.zeros((ndim, npoints), dtype=np.int64))
        norm_distances = zeros(ndim * npoints)
        xit = f64(xi)
        lib().msi_find_indices(
            addr(grid_addrs), addr(grid_lens), ndim, addr(xit), npoints,
            addr(indices), addr(norm_distances),
        )
        return indices, norm_distances.reshape(ndim, npoints)

    def _find_out_of_bounds(self, xi):
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds

    def _prepare_xi(self, xi):
        ndim = self.ndim
        p = _ndim_coords_from_arrays(xi, ndim)
        xi = np.empty((np.prod(p[0].shape), len(p)), dtype=np.float64)
        for i, q in enumerate(p):
            xi[:, i] = np.asarray(q, dtype=np.float64).ravel()
        if xi.shape[-1] != ndim:
            raise ValueError(
                f"The requested sample points xi have dimension {xi.shape[-1]} "
                f"but this RegularGridInterpolator has dimension {ndim}"
            )
        nans = np.any(np.isnan(xi), axis=-1)
        if self.bounds_error:
            for i, col in enumerate(xi.T):
                if not np.logical_and(
                    np.all(self.grid[i][0] <= col), np.all(col <= self.grid[i][-1])
                ):
                    raise ValueError(
                        f"One of the requested xi is out of bounds in dimension {i}"
                    )
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)
        return xi, p[0].shape + (ndim,), ndim, nans, out_of_bounds

    def __call__(self, xi, method=None, *, nu=None):
        """Interpolate at `xi`."""
        method = self.method if method is None else method
        if method not in _ALL_METHODS:
            raise ValueError(f"Method '{method}' is not defined")
        if nu is not None and method not in _SPLINE_METHODS:
            raise ValueError(
                f"Can only compute derivatives for methods {_SPLINE_METHODS}, "
                f"got {method}."
            )
        xi, xi_shape, ndim, nans, out_of_bounds = self._prepare_xi(xi)

        if method == "linear":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_nearest(indices, norm_distances)
        else:
            result = self._evaluate_spline(xi, method, nu)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
        if np.any(nans):
            result[nans] = np.nan
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances):
        npoints = indices.shape[1]
        ntrail = max(int(np.prod(self.values.shape[self.ndim :])), 1)
        out = zeros(npoints * ntrail)
        if self.ndim == 2 and ntrail == 1 and self.values.ndim == 2:
            n0, n1 = self.grid[0].size, self.grid[1].size
            vals = self.values
            lib().msi_evaluate_linear_2d(
                addr(vals), n0, n1, addr(indices),
                addr(norm_distances), npoints, addr(out),
            )
        else:
            dims = i64(np.array([p.size for p in self.grid], dtype=np.int64))
            strides = i64(np.zeros(self.ndim, dtype=np.int64))
            vals = self.values.reshape(-1, ntrail)
            lib().msi_evaluate_linear_nd(
                addr(vals), addr(dims), self.ndim, ntrail,
                addr(indices), addr(norm_distances), npoints, addr(out),
                addr(strides),
            )
        return out.reshape(npoints, ntrail) if ntrail > 1 else out

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [
            np.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances)
        ]
        return self.values[tuple(idx_res)]

    def _tensor_spline(self, k):
        """The tensor-product B-spline, as `make_ndbspl` builds it.

        Folding the 1-D interpolation operator `A_d^-1` along each axis turns
        the grid values into the coefficients of the tensor product; evaluating
        it is then a contraction over the `k + 1` non-zero basis functions of
        each axis, which the Mojo kernel does directly.  The per-axis knot
        vectors, degrees, coefficient counts and coefficient strides are
        cached next to the coefficients, so a call does no NumPy work at all.
        """
        if self._tensor is None:
            n = self.ndim
            coeffs = self.values
            splines = []
            for d in range(n - 1, -1, -1):
                spl = make_interp_spline(self.grid[d], coeffs, k=k, axis=d)
                splines.append(spl)
                # A_d[grid, a] = B_a(grid point); the operator maps grid values
                # to spline coefficients
                a_d = _basis(spl, self.grid[d])
                coeffs = np.moveaxis(
                    np.tensordot(np.linalg.inv(a_d), coeffs, axes=([1], [d])), 0, d
                )
            splines = splines[::-1]
            coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
            nk = [int(s.c.shape[0]) for s in splines]
            cshape = nk + list(self.values.shape[n:])
            strides = []
            acc = int(np.prod(self.values.shape[n:])) if n else 1
            for d in range(n - 1, -1, -1):
                strides.insert(0, acc)
                acc *= nk[d]
            assert coeffs.shape == tuple(cshape), (coeffs.shape, cshape)
            # the knot vectors are kept in the cache: the kernel reads them
            # through the addresses taken here
            tvecs = [f64(s.t) for s in splines]
            self._tensor = (
                coeffs,
                tvecs,
                i64(np.array([t.ctypes.data for t in tvecs], dtype=np.int64)),
                i64(np.array([s.k for s in splines], dtype=np.int64)),
                i64(np.array(nk, dtype=np.int64)),
                i64(np.array(strides, dtype=np.int64)),
            )
        return self._tensor

    def _evaluate_tensor(self, xi, method, nu=None):
        coeffs, tvecs, t_addrs, degrees, ncoefs, cstrides = self._tensor_spline(
            _SPLINE_DEGREE_MAP[method]
        )
        n = self.ndim
        m = xi.shape[0]
        trail = self.values.shape[n:]
        ntrail = max(int(np.prod(trail)), 1)
        out = zeros(max(m * ntrail, 1))
        # one scratch slice per parallel chunk
        nchunks = max((m + 127) // 128, 1)
        wstride = lib().msi_tensor_spline_wstride(addr(degrees), n)
        work = zeros(nchunks * n * wstride)
        iscratch = i64(np.zeros(nchunks * 2 * n, dtype=np.int64))
        xf = f64(xi)
        lib().msi_evaluate_tensor_spline(
            addr(coeffs), addr(t_addrs), addr(degrees), addr(ncoefs),
            addr(cstrides), n, ntrail, 0 if nu is None else nu, addr(xf), m,
            addr(out), addr(work), addr(iscratch),
        )
        return out.reshape((m,) + trail)

    def _evaluate_spline(self, xi, method, nu=None):
        """Fold a 1-D interpolation along each axis, from the last one back."""
        if method != "pchip":
            return self._evaluate_tensor(xi, method, nu)
        m, n = xi.shape
        axes = tuple(range(self.values.ndim))
        axx = axes[:n][::-1] + axes[n:]
        values = self.values.transpose(axx)

        k = _SPLINE_DEGREE_MAP[method]
        last_dim = n - 1
        first_values = _spline_eval(
            self.grid[last_dim], values, xi[:, last_dim], k, method, nu
        )
        result = np.empty((m, *self.values.shape[n:]), dtype=np.float64)
        for j in range(m):
            folded = first_values[j, ...]
            for i in range(last_dim - 1, -1, -1):
                folded = _spline_eval(self.grid[i], folded, xi[j, i], k, method, nu)
            result[j, ...] = folded
        return result

    def __repr__(self):
        return f"<RegularGridInterpolator method={self.method}, ndim={self.ndim}>"


def _spline_eval(x, y, pt, k, method, nu=None):
    if method == "pchip":
        from ._cubic import PchipInterpolator

        return PchipInterpolator(x, y, axis=0)(pt, nu=nu)
    return make_interp_spline(x, y, k=k, axis=0)(pt, nu=0 if nu is None else nu)


def interpn(points, values, xi, method="linear", bounds_error=True, fill_value=np.nan):
    """Multidimensional linear interpolation on a rectilinear grid.

    Mirrors ``scipy.interpolate.interpn``.
    """
    values = np.asarray(values)
    if np.prod(points[0].shape) == 0:
        raise ValueError("Input array of coordinates is empty")
    if len(points) != values.ndim:
        raise ValueError(
            "The dimensions of the input points do not match the values array"
        )
    rgi = RegularGridInterpolator(
        points, values, method=method, bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return rgi(xi)
