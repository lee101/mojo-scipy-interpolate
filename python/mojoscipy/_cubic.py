"""`CubicSpline`, `PchipInterpolator` and `CubicHermiteSpline`.

The Hermite coefficients, the Fritsch-Butland derivative rule and the
tridiagonal collocation matrix are pure NumPy, exactly as upstream; the
factorisation is `dgbsv` in Mojo, which is the routine upstream reaches
through `scipy.linalg.solve_banded`, and evaluation goes through the Mojo
`PPoly` kernel.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib, zeros
from .ppoly import PPoly


def _solve_banded(ab, b):
    """`scipy.linalg.solve_banded((1, 1), ab, b)` through the Mojo `dgbsv`.

    The kernel takes LAPACK's band layout in a `(2 * kl + ku + 1)` x `n`
    buffer, element `A(i, j)` on band row `kl + ku + i - j`, so the three
    rows `solve_banded` hands over are copied in one row down and the
    fill-in row partial pivoting needs is added on top.
    """
    ab = np.asarray(ab, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    n = ab.shape[1]
    nrhs = b.shape[1] if b.ndim > 1 else 1
    b2 = b.reshape(n, nrhs)
    band = np.zeros((2 + 1 + 1, n), dtype=np.float64)
    band[1:] = ab

    ipiv = i64(np.zeros(n, dtype=np.int64))
    info = lib().msi_solve_banded(
        1, 1, addr(band), n, addr(b2), n, nrhs, addr(ipiv)
    )
    if info < 0:
        raise ValueError(
            f"illegal value in {-info}-th argument of internal dgbsv"
        )
    if info > 0:
        raise np.linalg.LinAlgError(
            f"U[k, {info - 1}] = 0 is an exact zero in the banded matrix, so "
            "the system is singular."
        )
    return b2 if b.ndim > 1 else b2.reshape(-1)


def _solve_dense(a, b):
    """`scipy.linalg.solve` through the port's dense LU with partial pivoting."""
    a = np.array(a, dtype=np.float64, order="C")
    b = np.array(b, dtype=np.float64, order="C")
    n = a.shape[0]
    nrhs = b.shape[1] if b.ndim > 1 else 1
    b2 = b.reshape(n, nrhs)
    ipiv = i64(np.zeros(n, dtype=np.int64))
    info = lib().msi_solve_dense(addr(a), addr(b2), n, nrhs, addr(ipiv))
    if info < 0:
        raise ValueError(
            f"illegal value in {-info}-th argument of internal gesv"
        )
    if info > 0:
        raise np.linalg.LinAlgError(
            f"U[{info - 1}, {info - 1}] = 0 is an exact zero in the matrix, so "
            "the system is singular."
        )
    return b2 if b.ndim > 1 else b2.reshape(-1)


def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.

    All data are converted to numpy arrays and checked for correctness.  Axes
    equal to `axis` of arrays `y` and `dydx` are moved to be the 0th axis.  The
    value of `axis` is converted to lie in [0, number of dimensions of `y`).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(np.float64, copy=False)

    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = np.complex128
    else:
        dtype = np.float64

    if dydx is not None:
        dydx = np.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError(
            f"The length of `y` along `axis`={axis} doesn't match the "
            "length of `x`"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not np.all(np.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = np.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = np.moveaxis(dydx, axis, 0)

    return x, dx, y, axis, dydx


class CubicHermiteSpline(PPoly):
    """Cubic Hermite interpolator; coefficients come from values and slopes."""

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = True
        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

        c = np.stack(
            (
                t / dxr,
                (slope - dydx[:-1]) / dxr - t,
                dydx[:-1],
                y[:-1],
            )
        )
        super().__init__(c, x, extrapolate=extrapolate, axis=0)


class PchipInterpolator(CubicHermiteSpline):
    """PCHIP shape-preserving interpolator (C1 smooth)."""

    def __init__(self, x, y, axis=0, extrapolate=None):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        xv = x.reshape((x.shape[0],) + (1,) * (y.ndim - 1))
        dk = self._find_derivatives(xv, y)
        super().__init__(x, y, dk, axis=0, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _edge_case(h0, h1, m0, m1):
        # one-sided three-point estimate for the derivative
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        # try to preserve shape
        mask = np.sign(d) != np.sign(m0)
        mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
        mmm = (~mask) & mask2
        d[mask] = 0.0
        d[mmm] = 3.0 * m0[mmm]
        return d

    @staticmethod
    def _find_derivatives(x, y):
        y_shape = y.shape
        if y.ndim == 1:
            x = x[:, None]
            y = y[:, None]

        hk = x[1:] - x[:-1]
        mk = (y[1:] - y[:-1]) / hk

        if y.shape[0] == 2:
            # edge case: only have two points, use linear interpolation
            dk = np.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return np.reshape(dk, y_shape)

        smk = np.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)

        w1 = 2 * hk[1:] + hk[:-1]
        w2 = hk[1:] + 2 * hk[:-1]

        # values where division by zero occurs are excluded by `condition`
        with np.errstate(divide="ignore", invalid="ignore"):
            whmean = (w1 / mk[:-1] + w2 / mk[1:]) / (w1 + w2)

        dk = np.zeros_like(y)
        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean[~condition]

        # special case endpoints, as suggested in
        # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
        dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return np.reshape(dk, y_shape)


def _validate_bc(bc_type, y, expected_deriv_shape, axis):
    """Validate and prepare boundary conditions.

    Returns the validated conditions for the curve start and end, and `y`
    recast to complex if one of the conditions is complex.
    """
    if isinstance(bc_type, str):
        if bc_type == "periodic":
            if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                raise ValueError(
                    f"The first and last `y` point along axis {axis} must "
                    "be identical (within machine precision) when "
                    "bc_type='periodic'."
                )

        bc_type = (bc_type, bc_type)
    else:
        if len(bc_type) != 2:
            raise ValueError(
                "`bc_type` must contain 2 elements to specify start and end "
                "conditions."
            )
        if "periodic" in bc_type:
            raise ValueError(
                "'periodic' `bc_type` is defined for both curve ends and "
                "cannot be used with other boundary conditions."
            )

    validated_bc = []
    for bc in bc_type:
        if isinstance(bc, str):
            if bc == "clamped":
                validated_bc.append((1, np.zeros(expected_deriv_shape)))
            elif bc == "natural":
                validated_bc.append((2, np.zeros(expected_deriv_shape)))
            elif bc in ["not-a-knot", "periodic"]:
                validated_bc.append(bc)
            else:
                raise ValueError(f"bc_type={bc} is not allowed.")
        else:
            try:
                deriv_order, deriv_value = bc
            except ValueError as exc:
                raise ValueError(
                    "A specified derivative value must be given in the form "
                    "(order, value)."
                ) from exc

            if deriv_order not in [1, 2]:
                raise ValueError(
                    "The specified derivative order must be 1 or 2."
                )

            deriv_value = np.asarray(deriv_value)
            if deriv_value.shape != expected_deriv_shape:
                raise ValueError(
                    f"`deriv_value` shape {deriv_value.shape} is not the "
                    f"expected one {expected_deriv_shape}."
                )

            if np.issubdtype(deriv_value.dtype, np.complexfloating):
                y = y.astype(complex, copy=False)

            validated_bc.append((deriv_order, deriv_value))

    return validated_bc, y


class CubicSpline(CubicHermiteSpline):
    """Piecewise cubic interpolator to fit values (C2 smooth).

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable.  The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    `bc_type` is one of ``'not-a-knot'`` (default), ``'periodic'``,
    ``'clamped'`` (zero first derivative at both ends) or ``'natural'`` (zero
    second derivative at both ends), or a 2-tuple giving the condition at the
    start and the end separately -- each either a string or a
    ``(order, deriv_values)`` pair with `order` 1 or 2.

    The tridiagonal system for the nodal derivatives is factorised in Mojo by
    the port's `dgbsv`, which is the same routine upstream reaches through
    `scipy.linalg.solve_banded`.
    """

    def __init__(self, x, y, axis=0, bc_type="not-a-knot", extrapolate=None):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        n = len(x)

        bc, y = _validate_bc(bc_type, y, y.shape[1:], axis)

        if extrapolate is None:
            if bc[0] == "periodic":
                extrapolate = "periodic"
            else:
                extrapolate = True

        if y.size == 0:
            # bail out early for zero-sized arrays
            s = np.zeros_like(y)
        else:
            dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
            slope = np.diff(y, axis=0) / dxr

            # If bc is 'not-a-knot' this change is just a convention.
            # If bc is 'periodic' then we already checked that y[0] == y[-1],
            # and the spline is just a constant, we handle this case in the
            # same way by setting the first derivatives to slope, which is 0.
            if n == 2:
                if bc[0] in ["not-a-knot", "periodic"]:
                    bc[0] = (1, slope[0])
                if bc[1] in ["not-a-knot", "periodic"]:
                    bc[1] = (1, slope[0])

            # This is a special case, when both conditions are 'not-a-knot'
            # and n == 3. In this case 'not-a-knot' can't be handled regularly
            # as the both conditions are identical. We handle this case by
            # constructing a parabola passing through given points.
            if n == 3 and bc[0] == "not-a-knot" and bc[1] == "not-a-knot":
                A = np.zeros((3, 3))  # This is a standard matrix.
                b = np.empty((3,) + y.shape[1:], dtype=y.dtype)

                A[0, 0] = 1
                A[0, 1] = 1
                A[1, 0] = dx[1]
                A[1, 1] = 2 * (dx[0] + dx[1])
                A[1, 2] = dx[0]
                A[2, 1] = 1
                A[2, 2] = 1

                b[0] = 2 * slope[0]
                b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
                b[2] = 2 * slope[1]

                m = b.shape[0]
                s = _solve_dense(A, b.reshape(m, -1)).reshape(b.shape)
            elif n == 3 and bc[0] == "periodic":
                # In case when number of points is 3 we compute the derivatives
                # manually
                t = (slope / dxr).sum(0) / (1.0 / dxr).sum(0)
                s = np.broadcast_to(t, (n,) + y.shape[1:])
            else:
                # Find derivative values at each x[i] by solving a tridiagonal
                # system.
                A = np.zeros((3, n))  # This is a banded matrix representation.
                b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

                # Filling the system for i=1..n-2
                #                         (x[i-1] - x[i]) * s[i-1] +\
                # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
                #                         (x[i] - x[i-1]) * s[i+1] =\
                #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
                #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i])

                A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
                A[0, 2:] = dx[:-1]                   # The upper diagonal
                A[-1, :-2] = dx[1:]                  # The lower diagonal

                b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

                bc_start, bc_end = bc

                if bc_start == "periodic":
                    # Due to the periodicity, and because y[-1] = y[0], the
                    # linear system has (n-1) unknowns/equations instead of n:
                    A = A[:, 0:-1]
                    A[1, 0] = 2 * (dx[-1] + dx[0])
                    A[0, 1] = dx[-1]

                    b = b[:-1]

                    # Also, due to the periodicity, the system is not
                    # tri-diagonal.  We need to compute a "condensed" matrix
                    # of shape (n-2, n-2).  The condensed matrix is obtained by
                    # removing the last column and last row of the (n-1, n-1)
                    # system matrix.  The removed values are saved in scalar
                    # variables with the (n-1, n-1) system matrix indices
                    # forming their names:
                    a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
                    a_m1_m2 = dx[-1]
                    a_m1_m1 = 2 * (dx[-1] + dx[-2])
                    a_m2_m1 = dx[-3]
                    a_0_m1 = dx[0]

                    b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
                    b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

                    Ac = A[:, :-1]
                    b1 = b[:-1]
                    b2 = np.zeros_like(b1)
                    b2[0] = -a_0_m1
                    b2[-1] = -a_m2_m1

                    # s1 and s2 are the solutions of (n-2, n-2) system
                    m = b1.shape[0]
                    s1 = _solve_banded(Ac, b1.reshape(m, -1)).reshape(b1.shape)

                    m = b2.shape[0]
                    s2 = _solve_banded(Ac, b2.reshape(m, -1)).reshape(b2.shape)

                    # computing the s[n-2] solution:
                    s_m1 = (
                        (b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1])
                        / (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1])
                    )

                    # s is the solution of the (n, n) system:
                    s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                    s[:-2] = s1 + s_m1 * s2
                    s[-2] = s_m1
                    s[-1] = s[0]
                else:
                    if bc_start == "not-a-knot":
                        A[1, 0] = dx[1]
                        A[0, 1] = x[2] - x[0]
                        d = x[2] - x[0]
                        b[0] = (
                            (dxr[0] + 2 * d) * dxr[1] * slope[0]
                            + dxr[0] ** 2 * slope[1]
                        ) / d
                    elif bc_start[0] == 1:
                        A[1, 0] = 1
                        A[0, 1] = 0
                        b[0] = bc_start[1]
                    elif bc_start[0] == 2:
                        A[1, 0] = 2 * dx[0]
                        A[0, 1] = dx[0]
                        b[0] = -0.5 * bc_start[1] * dx[0] ** 2 + 3 * (y[1] - y[0])

                    if bc_end == "not-a-knot":
                        A[1, -1] = dx[-2]
                        A[-1, -2] = x[-1] - x[-3]
                        d = x[-1] - x[-3]
                        b[-1] = (
                            dxr[-1] ** 2 * slope[-2]
                            + (2 * d + dxr[-1]) * dxr[-2] * slope[-1]
                        ) / d
                    elif bc_end[0] == 1:
                        A[1, -1] = 1
                        A[-1, -2] = 0
                        b[-1] = bc_end[1]
                    elif bc_end[0] == 2:
                        A[1, -1] = 2 * dx[-1]
                        A[-1, -2] = dx[-1]
                        b[-1] = 0.5 * bc_end[1] * dx[-1] ** 2 + 3 * (y[-1] - y[-2])

                    m = b.shape[0]
                    s = _solve_banded(A, b.reshape(m, -1)).reshape(b.shape)

        super().__init__(x, y, s, axis=0, extrapolate=extrapolate)
        self.axis = axis
