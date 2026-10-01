"""Piecewise polynomials in the local power basis (`scipy.interpolate.PPoly`).

Evaluation runs in Mojo via `_ppoly.evaluate`; the definite integral and the
coefficient transform behind `antiderivative` run via `_ppoly.integrate` and
`_ppoly.fix_continuity`.  Construction and the purely algebraic helpers stay in
NumPy, in upstream's order.
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np

from ._lib import addr, f64, lib, zeros


def _poch(m, nu):
    """`scipy.special.poch(m, nu)`: the rising factorial `m (m+1) ... (m+nu-1)`."""
    m = np.asarray(m, dtype=np.float64)
    return np.prod(m[..., None] + np.arange(nu), axis=-1)


class PPoly:
    r"""Piecewise polynomial in the local power basis.

    Parameters
    ----------
    c : (k, m, ...) ndarray
        Polynomial coefficients, highest order first.  The interval axis is
        `axis`, and it is moved to position 1 on construction, so `c` is
        stored as ``(k, m, ...)``.
    x : (m+1,) ndarray
        Breakpoints of the polynomials; must be sorted.
    extrapolate : bool, 'periodic' or None
        Whether to extrapolate beyond ``[x[0], x[-1]]`` or return NaN.  With
        ``'periodic'`` the query is wrapped into the base interval instead.
        ``None`` means ``True``.
    axis : int
        Axis of the input `c` holding the interval index.
    """

    extrapolate = True
    axis = 0

    def __init__(self, c, x, extrapolate=None, axis=0):
        c = np.asarray(c)
        x = np.ascontiguousarray(x, dtype=np.float64)

        if extrapolate is None:
            extrapolate = True
        elif extrapolate != "periodic":
            extrapolate = bool(extrapolate)

        if c.ndim < 2:
            raise ValueError(
                "Coefficients array must be at least 2-dimensional."
            )
        if not 0 <= axis < c.ndim - 1:
            raise ValueError(f"axis={axis} must be between 0 and {c.ndim - 1}")

        self.axis = axis
        if axis != 0:
            # move the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis != 0 means that we have c.shape (..., k, m, ...)
            #                                                ^
            #                                               axis
            # So we roll two of them.
            c = np.moveaxis(c, axis + 1, 0)
            c = np.moveaxis(c, axis + 1, 0)

        if x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if c.shape[1] != x.size - 1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = np.diff(x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        self.c = np.ascontiguousarray(c, dtype=np.float64)
        self.x = x
        self.extrapolate = extrapolate

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """Construct without the shape checks; same parameters as the constructor."""
        self = object.__new__(cls)
        self.c = np.asarray(c)
        self.x = np.asarray(x)
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _flat_c(self):
        """`c` as the contiguous `k` x `m` x `n` array the kernels want."""
        c = np.ascontiguousarray(self.c, dtype=np.float64)
        return c.reshape(c.shape[0], c.shape[1], -1)

    def _segment_integral(self, c, x, k, m, n, a, b, extrapolate=False):
        """`_ppoly.integrate` over `[a, b]`, assuming zero outside `x`."""
        dst = zeros(n)
        lib().msi_ppoly_integrate(
            addr(c), k, m, n, addr(x), int(x.size), float(a), float(b),
            int(bool(extrapolate)), addr(dst),
        )
        return dst

    def __call__(self, x, nu=0, extrapolate=None):
        """Evaluate the piecewise polynomial at `x`."""
        if extrapolate is None:
            extrapolate = self.extrapolate
        if nu is None:
            nu = 0
        if nu < 0:
            raise ValueError("Order of derivative cannot be negative")
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        xf = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == "periodic":
            xf = self.x[0] + (xf - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        c = self._flat_c()
        k, m, n = c.shape
        nxp = xf.shape[0]
        out = zeros(nxp * n)
        lib().msi_evaluate(
            addr(c), k, m, n, addr(f64(self.x)), addr(xf), nxp, nu,
            int(bool(extrapolate)), addr(out),
        )
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim + self.axis] + l[:x_ndim] + l[x_ndim + self.axis:]
            out = out.transpose(l)
        return out

    def integrate(self, a, b, extrapolate=None):
        """Definite integral between `a` and `b`."""
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        c = self._flat_c()
        k, m, n = c.shape
        x = f64(self.x)

        if extrapolate == "periodic":
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.
            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                res = n_periods * self._segment_integral(c, x, k, m, n, xs, xe)
            else:
                res = np.zeros(n)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            if b <= xe:
                res = res + self._segment_integral(c, x, k, m, n, a, b)
            else:
                res = res + self._segment_integral(c, x, k, m, n, a, xe)
                res = res + self._segment_integral(
                    c, x, k, m, n, xs, xs + left + a - xe
                )
        else:
            res = self._segment_integral(
                c, x, k, m, n, a, b, extrapolate
            )

        # Return
        res = res * sign
        return res.reshape(self.c.shape[2:])

    def derivative(self, nu=1):
        """A `PPoly` representing the `nu`-th derivative."""
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = _poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 = c2 * factor[(slice(None),) + (None,) * (c2.ndim - 1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """A `PPoly` representing the `nu`-th antiderivative."""
        if nu <= 0:
            return self.derivative(-nu)

        c = np.zeros(
            (self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
            dtype=self.c.dtype,
        )
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = _poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] = c[:-nu] / factor[(slice(None),) + (None,) * (c.ndim - 1)]

        # fix continuity of added degrees of freedom
        flat = np.ascontiguousarray(c, dtype=np.float64)
        flat = flat.reshape(flat.shape[0], flat.shape[1], -1)
        k, m, n = flat.shape
        lib().msi_fix_continuity(
            addr(flat), k, m, n, addr(f64(self.x)), int(self.x.size), nu - 1
        )
        c = flat.reshape(c.shape)

        if self.extrapolate == "periodic":
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        """Real solutions of the equation ``pp(x) == y``.

        Upstream finds the roots of each interval's polynomial as the
        eigenvalues of its companion matrix, which is a LAPACK call per
        interval, so that step is handed to the same routine rather than
        rewritten in Mojo.  See the coverage section of the README.
        """
        import scipy.interpolate as si

        if extrapolate is None:
            extrapolate = self.extrapolate
        if extrapolate == "periodic":
            # `real_roots` is compiled against a bool
            extrapolate = False
        # `c` is already stored as (k, m, ...), the layout `real_roots` wants,
        # so only the recorded interval axis has to be handed back.
        return si.PPoly(
            self.c, self.x, extrapolate=extrapolate, axis=self.axis
        ).solve(y, discontinuity)

    def roots(self, discontinuity=True, extrapolate=None):
        """Zeros of the piecewise polynomial."""
        return self.solve(0, discontinuity, extrapolate)

    def extend(self, c, x):
        """Extend the piecewise polynomial by adding more intervals."""
        c = np.asarray(c)
        x = np.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError(
                f"Shapes of x {x.shape} and c {c.shape} are incompatible"
            )
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError(
                f"Shapes of c {c.shape} and self.c {self.c.shape} are "
                "incompatible"
            )

        action = "append"
        if self.x[-1] >= self.x[0]:
            if x[0] < self.x[-1]:
                if x[0] == self.x[-1]:
                    action = "append"
                else:
                    raise ValueError(
                        f"x[0] {x[0]} is not >= x[-1] {self.x[-1]}"
                    )
            else:
                if x[0] == self.x[0]:
                    action = "prepend"
                else:
                    raise ValueError(
                        f"x[0] {x[0]} is not <= x[0] {self.x[0]}"
                    )
        else:
            if not x[0] <= self.x[0]:
                raise ValueError(
                    f"x[0] {x[0]} is not <= x[-1] {self.x[-1]}"
                )
            if x[0] == self.x[0]:
                action = "prepend"
            else:
                raise ValueError(
                    f"x[0] {x[0]} is not >= x[-1] {self.x[-1]}"
                )

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros(
            (k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
            dtype=np.result_type(self.c, c),
        )

        if action == "append":
            c2[k2 - self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2 - c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]
        else:
            c2[k2 - c.shape[0]:, :c.shape[1]] = c
            c2[k2 - self.c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """Build a `PPoly` from a FITPACK ``(t, c, k)`` triple."""
        import scipy.interpolate._fitpack as fitpack

        t, c, k = tck

        cvals = np.empty((k + 1, len(t) - 1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = fitpack.splev(t[:-1], (t, c, k), der=m)
            cvals[k - m, :] = y / factorial(m)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """Build a power-basis `PPoly` from a Bernstein-basis `BPoly`."""
        import scipy.interpolate as si

        if not isinstance(bp, si.BPoly):
            raise ValueError("bp must be a BPoly")
        dx = np.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,) * (bp.c.ndim - 2)

        c = np.zeros_like(bp.c)
        for a in range(k + 1):
            factor = (-1) ** a * comb(k, a) * bp.c[a]
            for s in range(a, k + 1):
                val = comb(k - a, s - a) * (-1) ** s
                c[k - s] += factor * val / dx[(slice(None),) + rest] ** s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)

    def __repr__(self):
        nintervals = self.c.shape[1]
        npolys = int(np.prod(self.c.shape[2:])) if self.c.ndim > 2 else 1
        return f"<PPoly degree={self.c.shape[0] - 1}, {nintervals} intervals, {npolys} polynomials>"
