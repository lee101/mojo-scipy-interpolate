"""`RBFInterpolator` and the RBF kernel family.

The kernel matrix, the polynomial matrix and the per-point evaluation vector are
built in Mojo; the dense solve stays in NumPy because upstream hands it to
LAPACK through `scipy.linalg.solve`. `neighbors=` uses a dense k-nearest-neighbour
scan rather than `scipy.spatial.KDTree`, which makes large `n` slower but
returns the same neighborhoods.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement
from math import comb

import numpy as np

from ._lib import addr, f64, i64, lib, zeros

# The RBF kernels, in the declaration order of `NAME_TO_FUNC` upstream.
NAME_TO_FUNC = {
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 2,
    "quintic": 3,
    "multiquadric": 4,
    "inverse_multiquadric": 5,
    "inverse_quadratic": 6,
    "gaussian": 7,
}

_AVAILABLE = set(NAME_TO_FUNC)
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}


def linear(r):
    return -r


def thin_plate_spline(r):
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    nz = r != 0
    out[nz] = r[nz] ** 2 * np.log(r[nz])
    return out


def cubic(r):
    return np.asarray(r, dtype=np.float64) ** 3


def quintic(r):
    return -np.asarray(r, dtype=np.float64) ** 5


def multiquadric(r):
    return -np.sqrt(np.asarray(r, dtype=np.float64) ** 2 + 1)


def inverse_multiquadric(r):
    return 1 / np.sqrt(np.asarray(r, dtype=np.float64) ** 2 + 1)


def inverse_quadratic(r):
    return 1 / (np.asarray(r, dtype=np.float64) ** 2 + 1)


def gaussian(r):
    return np.exp(-(np.asarray(r, dtype=np.float64) ** 2))


KERNELS = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}


def rbf_factory(function, use_dense=True):
    """Build a vectorised RBF of the named `function`."""
    if function in NAME_TO_FUNC:
        return KERNELS[function]
    if callable(function):
        return function
    raise ValueError("function must be callable or a name of a known RBF")


def _monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial."""
    nmonos = comb(degree + ndim, ndim)
    out = [[0] * ndim for _ in range(nmonos)]
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            for var in mono:
                out[count][var] += 1
            count += 1
    return out


def _kernel_matrix(x, kernel):
    """Return RBFs, with centers at `x`, evaluated at `x` (Mojo)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n, d = x.shape
    out = zeros(n * n)
    lib().msi_kernel_matrix(addr(x), n, d, NAME_TO_FUNC[kernel], addr(out))
    return out.reshape(n, n)


def _polynomial_matrix(x, powers):
    """Return monomials, with exponents from `powers`, evaluated at `x` (Mojo)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    powers = i64(np.ascontiguousarray(powers, dtype=np.int64))
    n, d = x.shape
    npow = powers.shape[0]
    out = np.zeros(n * npow, dtype=np.float64)
    lib().msi_polynomial_matrix(
        addr(x), n, d, addr(powers), d, npow, addr(out)
    )
    return out.reshape(n, npow)


# Largest system NumPy's LAPACK is used for; see
# `RBFInterpolator._build_and_solve_system`.  Measured on the pinned toolchain,
# where NumPy's OpenBLAS hands a solve to a 72-thread pool at n = 100.
_NUMPY_SOLVE_MAX = 96


def _build_system(y, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64)
    smoothing = np.ascontiguousarray(smoothing, dtype=np.float64)
    powers = i64(powers)
    p, ndim = y.shape
    npow = powers.shape[0]
    ncols = d.shape[1]
    n = p + npow
    lhs = np.empty(n * n, dtype=np.float64)
    rhs = np.empty(n * ncols, dtype=np.float64)
    shift = zeros(ndim)
    scale = zeros(ndim)
    work = zeros(2 * p * ndim)
    lib().msi_rbf_build_system(
        addr(y), p, ndim, float(epsilon), NAME_TO_FUNC[kernel], addr(powers),
        ndim, npow, addr(smoothing), addr(d), ncols, addr(lhs), n, addr(rhs),
        addr(shift), addr(scale), addr(work),
    )
    return lhs.reshape(n, n), rhs.reshape(n, ncols), shift, scale
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers, shift, scale):
    """Construct the coefficients needed to evaluate the RBF (Mojo)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    powers = i64(np.ascontiguousarray(powers, dtype=np.int64))
    q, d = x.shape
    p = y.shape[0]
    npow = powers.shape[0]
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    scale = np.ascontiguousarray(scale, dtype=np.float64)
    vec = zeros(q * (p + npow))
    scratch = zeros(p * d + 2 * d)
    lib().msi_build_evaluation_coefficients(
        addr(x), q, addr(y), p, d, NAME_TO_FUNC[kernel], float(epsilon),
        addr(powers), d, npow, addr(shift), addr(scale), addr(scratch), addr(vec),
    )
    return vec.reshape(q, p + npow)


class RBFInterpolator:
    """Radial basis function interpolator in N >= 1 dimensions.

    Parameters mirror upstream: `y`, `d`, `neighbors`, `smoothing`, `kernel`,
    `epsilon`, `degree`.
    """

    def __init__(self, y, d, neighbors=None, smoothing=0.0,
                 kernel="thin_plate_spline", epsilon=None, degree=None):
        y = np.ascontiguousarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")
        ny, ndim = y.shape

        d = np.asarray(d)
        if np.issubdtype(d.dtype, np.complexfloating):
            d_dtype = np.complex128
        else:
            d_dtype = np.float64
        d = np.ascontiguousarray(d, dtype=d_dtype)
        if d.shape[0] != ny:
            raise ValueError(
                f"Expected the first axis of `d` to have length {ny}."
            )
        d_shape = d.shape[1:]
        d = d.reshape(ny, -1)
        d = d.view(np.float64)

        if np.isscalar(smoothing) or np.ndim(smoothing) == 0:
            smoothing = np.full(ny, float(smoothing), dtype=np.float64)
        else:
            smoothing = np.ascontiguousarray(smoothing, dtype=np.float64)
            if smoothing.shape != (ny,):
                raise ValueError(
                    f"Expected `smoothing` to be a scalar or have shape ({ny},)."
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {sorted(_AVAILABLE)}.")

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{sorted(_SCALE_INVARIANT)}."
                )
        else:
            epsilon = float(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif -1 < degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} except -1 when "
                    f"`kernel` is '{kernel}'.",
                    UserWarning,
                    stacklevel=2,
                )

        if neighbors is None:
            nobs = ny
        else:
            neighbors = int(min(neighbors, ny))
            nobs = neighbors

        powers = np.array(_monomial_powers(ndim, degree), dtype=np.int64).reshape(-1, ndim)
        powers = powers.reshape(-1, ndim)
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}."
            )

        if neighbors is None:
            shift, scale, coeffs = self._build_and_solve_system(
                y, d, smoothing, kernel, epsilon, powers
            )
            self._shift, self._scale, self._coeffs = shift, scale, coeffs
        else:
            self._shift = self._scale = self._coeffs = None

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.powers = powers

    @staticmethod
    def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
        lhs, rhs, shift, scale = _build_system(
            y, d, smoothing, kernel, epsilon, powers
        )
        # `RBFInterpolator(neighbors=k)` builds and solves one system per
        # evaluation point, so the per-call cost of the solve is the whole
        # cost of the method.  Both routes reach the same LAPACK gesv; NumPy's
        # skips the finiteness scan, the structure probe and the norm estimate
        # `scipy.linalg.solve` does per call.  It only holds while LAPACK keeps
        # the solve on one thread: this toolchain's NumPy OpenBLAS asks for 72
        # threads and switches to them at n = 100.  Measured here at 68x68,
        # 0.31 ms through NumPy against 0.57 ms through SciPy; at 100x100,
        # 113 ms against 0.65 ms.  Above the measured threshold the two swap
        # places, so the route is chosen by size.  A singular system falls back
        # to SciPy, which returns its unchecked result rather than raising, so
        # the behaviour is the same either way.
        if lhs.shape[0] <= _NUMPY_SOLVE_MAX:
            try:
                return shift, scale, np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                pass
        from scipy.linalg import solve as _solve

        return shift, scale, _solve(lhs, rhs)

    def _chunk_evaluator(self, x, y, shift, scale, coeffs, memory_budget=1000000):
        """Evaluate the interpolation while controlling memory consumption."""
        nnei = y.shape[0] if self.neighbors is None else self.neighbors
        chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1
        nx = x.shape[0]
        if chunksize <= nx:
            out = np.empty((nx, self.d.shape[1]), dtype=np.float64)
            for i in range(0, nx, chunksize):
                chunk = _build_evaluation_coefficients(
                    x[i : i + chunksize], y, self.kernel, self.epsilon,
                    self.powers, shift, scale,
                ) @ coeffs
                out[i : i + chunksize] = chunk
        else:
            out = _build_evaluation_coefficients(
                x, y, self.kernel, self.epsilon, self.powers, shift, scale
            ) @ coeffs
        return out

    def __call__(self, x):
        """Evaluate the interpolant at `x`."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")
        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                f"Expected the second axis of `x` to have length {self.y.shape[1]}."
            )
        memory_budget = max(
            x.nbytes + self.y.nbytes + self.d.nbytes, 1_000_000
        )

        if self.neighbors is None:
            out = self._chunk_evaluator(
                x, self.y, self._shift, self._scale, self._coeffs,
                memory_budget=memory_budget,
            )
        else:
            out = np.empty((nx, self.d.shape[1]), dtype=float)
            # Multiple evaluation points may share a neighborhood; make the
            # neighborhoods unique so the coefficients are solved for once.
            yindices = np.sort(self._query(x), axis=1)
            uniq, inv = np.unique(yindices, return_inverse=True, axis=0)
            inv = np.reshape(inv, (-1,))
            # `order` groups the evaluation points by neighbourhood, so each
            # group's `xidx` is a contiguous int64 slice -- an order of
            # magnitude cheaper to index with than a Python list, and this
            # loop runs once per unique neighbourhood.
            order = np.argsort(inv, kind="stable")
            bounds = np.zeros(len(uniq) + 1, dtype=np.intp)
            np.cumsum(np.bincount(inv, minlength=len(uniq)), out=bounds[1:])
            xnbrs = x[order]
            for g, yidx in enumerate(uniq):
                xidx = order[bounds[g] : bounds[g + 1]]
                xnbr = xnbrs[bounds[g] : bounds[g + 1]]
                ynbr = self.y[yidx]
                dnbr = self.d[yidx]
                snbr = self.smoothing[yidx]
                shift, scale, coeffs = self._build_and_solve_system(
                    ynbr, dnbr, snbr, self.kernel, self.epsilon, self.powers
                )
                out[xidx] = self._chunk_evaluator(
                    xnbr, ynbr, shift, scale, coeffs, memory_budget=memory_budget
                )

        out = out.view(self.d_dtype)
        return out.reshape((nx,) + self.d_shape)

    def _query(self, x):
        """The `k` nearest observation indices for each evaluation point."""
        nx = x.shape[0]
        dist = zeros(nx * self.neighbors)
        idx = i64(np.zeros(nx * self.neighbors, dtype=np.int64))
        lib().msi_neighbors_knn(
            addr(self.y), self.y.shape[0], self.y.shape[1], addr(x), nx,
            self.neighbors, addr(dist), addr(idx),
        )
        return idx.reshape(nx, self.neighbors)
