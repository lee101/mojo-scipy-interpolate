"""`RectBivariateSpline` -- a bivariate spline on a rectangular grid.

Upstream's class is a thin FITPACK wrapper around ``rectbiv``. The port builds
the same tensor-product spline from two successive 1-D `make_interp_spline`
calls, which is exactly the construction the interpolated case ``s=0`` reduces
to, and evaluates it through the Mojo `BSpline.__call__` kernel: the two basis
matrices are evaluated first and contracted with the coefficient matrix.
The smoothing form (``s > 0``) is not ported.
"""

from __future__ import annotations

import numpy as np

from .bsplines import BSpline, make_interp_spline


def _basis(spl, pts, nu=0):
    """The values of every B-spline of `spl` at `pts`."""
    n = spl.c.shape[0]
    eye = np.eye(n, dtype=np.float64)
    return BSpline.construct_fast(spl.t, eye, spl.k, extrapolate=True)(pts, nu=nu)


class RectBivariateSpline:
    """A bivariate spline on a rectangular grid.

    `x`, `y` are the grid axes, `z[i, j]` the value at ``(x[i], y[j])``. Only the
    interpolating form is covered: `s` must be 0, and `kx`/`ky` are the degrees
    of the successive 1-D interpolations.
    """

    def __init__(self, x, y, z, kx=3, ky=3, s=0):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Expect x and y to be 1-D arrays.")
        if x.size < 4 or y.size < 4:
            raise ValueError("Expect x and y to have at least 4 entries.")
        if np.any(x[1:] < x[:-1]) or np.any(y[1:] < y[:-1]):
            raise ValueError("Expect x and y to be sorted in ascending order.")
        if not z.shape == (len(x), len(y)):
            raise ValueError(
                f"z must have shape ({len(x)}, {len(y)}), got {z.shape}"
            )
        if s:
            raise NotImplementedError(
                "RectBivariateSpline(s>0) is not ported; only the interpolating "
                "form s=0 is"
            )
        self.x, self.y, self.kx, self.ky, self.s = x, y, int(kx), int(ky), 0
        # the two successive 1-D interpolations; folding one into the other
        # leaves the tensor-product coefficient matrix `_c2`
        fx = make_interp_spline(x, z, k=self.kx, axis=0)
        fy = make_interp_spline(y, z.T, k=self.ky, axis=0)
        # ax[i, a] = B_a(x_i) is the x-spline collocation matrix on the grid
        ax = _basis(fx, x)
        self._c2 = np.linalg.solve(ax, fy.c.T)
        self._fx, self._fy = fx, fy

    def __call__(self, x, y, dx=0, dy=0):
        """Evaluate the spline at `(x, y)`, or its `dx`/`dy` derivative."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        # upstream evaluates on the full grid of the two inputs
        xg, yg = np.meshgrid(x, y, indexing="ij")
        shape = xg.shape
        xf = np.ascontiguousarray(xg.ravel(), dtype=np.float64)
        yf = np.ascontiguousarray(yg.ravel(), dtype=np.float64)
        wx = _basis(self._fx, xf, dx)
        wy = _basis(self._fy, yf, dy)
        out = np.einsum("ia,ab,ib->i", wx, self._c2, wy)
        return out.reshape(shape)

    def ev(self, x, y):
        """Evaluate at the paired points `(x[i], y[i])`."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        wx = _basis(self._fx, x)
        wy = _basis(self._fy, y)
        return np.einsum("ia,ab,ib->i", wx, self._c2, wy)


    def get_knots(self):
        """The interior knots of the two axes."""
        return self._fx.t, self._fy.t

    def get_coeffs(self):
        """The spline coefficients, flattened x-major as FITPACK returns them."""
        return np.ascontiguousarray(self._c2).ravel()

    def partial_derivative(self, dx, dy):
        """A new spline representing a partial derivative of this one.

        The derivative of a B-spline with coefficients `c` has coefficients
        `T @ c`, where `T` is the linear map `splder` applies.  Applying
        `splder` to the identity coefficients recovers `T` without needing
        FITPACK's `pardtc`, and the tensor-product coefficient matrix is then
        contracted along the corresponding axis.
        """
        from .bsplines import splder

        if dx == 0 and dy == 0:
            return self
        kx, ky = self.degrees
        if not (dx >= 0 and dy >= 0):
            raise ValueError("order of derivative must be positive or zero")
        if not (dx < kx and dy < ky):
            raise ValueError(
                "order of derivative must be less than degree of spline"
            )

        def _contract(spl, c2, order, along_x):
            nco = spl.c.shape[0]
            T = splder((spl.t, np.eye(nco), spl.k), order)[1]
            return T @ c2 if along_x else c2 @ T.T

        fx, fy, c2 = self._fx, self._fy, self._c2
        if dx:
            c2 = _contract(fx, c2, dx, True)
            fx = fx.derivative(dx)
        if dy:
            c2 = _contract(fy, c2, dy, False)
            fy = fy.derivative(dy)
        return _DerivedBivariateSpline(fx, fy, c2)

    @property
    def degrees(self):
        """The degrees of the two axes."""
        return self.kx, self.ky

    def integral(self, xa, xb, ya, yb):
        """Evaluate the integral of the spline over `[xa, xb] x [ya, yb]`."""
        ix = _basis_integrals(self._fx, xa, xb)
        iy = _basis_integrals(self._fy, ya, yb)
        return float(np.einsum("i,ij,j->", ix, self._c2, iy))


def _basis_integrals(spl, lo, hi):
    r"""``int_lo^hi B_a(x) dx`` for every basis function ``B_a`` of `spl`.

    `splantider` is linear in the coefficient vector, so applying it to the
    identity coefficients gives the antiderivative of every basis function at
    once: column ``a`` of the result is the antiderivative of ``B_a`` in the
    higher-order basis.  Contracting those columns with the integrals of the
    higher-order basis functions gives the per-basis definite integrals.
    """
    from .bsplines import splantider

    nco = spl.c.shape[0]
    tA, cA, kA = splantider((spl.t, np.eye(nco), spl.k), 1)
    nhi = cA.shape[0]
    gb = _basis(BSpline.construct_fast(tA, np.eye(nhi), kA, extrapolate=True),
                np.array([lo, hi]))
    dB = gb[0] - gb[1]
    return np.asarray(dB @ cA)



class _DerivedBivariateSpline(RectBivariateSpline):
    """A `RectBivariateSpline` whose axes carry already-taken derivatives."""

    def __init__(self, fx, fy, c2):
        self.x, self.y = fx.t, fy.t
        self.kx, self.ky, self.s = fx.k, fy.k, 0
        self._fx, self._fy, self._c2 = fx, fy, np.asarray(c2)

    def __repr__(self):
        return f"<RectBivariateSpline kx={self.kx}, ky={self.ky}>"
