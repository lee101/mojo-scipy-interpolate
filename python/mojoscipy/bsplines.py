"""B-splines: `BSpline`, `make_interp_spline`, `make_lsq_spline`.

Names, argument order and defaults follow `scipy.interpolate` so the import is
the only thing that changes. The evaluation, collocation and normal-equation
kernels run in Mojo; the O(n) coefficient recursions of `splder` /
`splantider`, knot insertion and the sparse assembly of `design_matrix` stay in
NumPy because upstream runs them as vectorised library calls, not as kernels.
"""

from __future__ import annotations

import operator

import numpy as np

from ._lib import addr, f64, i64, lib, zeros


def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if check_finite and not np.all(np.isfinite(x)):
        raise ValueError("array must not contain infs or nans")
    return x


def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, str):
        if deriv == "clamped":
            deriv = [(1, np.zeros(target_shape))]
        elif deriv == "not-a-knot":
            deriv = []
        else:
            raise ValueError(f"Unknown boundary condition : {deriv}")
    return deriv


def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except Exception as exc:
            raise ValueError("deriv must be a list of (nu, value) pairs") from exc
    else:
        ords, vals = [], []
    return np.array(ords, dtype=np.int64), np.array(vals, dtype=np.float64)


def find_interval(t, k, xval, prev_l, extrapolate):
    """`t[l] <= xval < t[l+1]`; the NumPy twin of the Mojo kernel.

    Used by `_insert`; a single scalar call, so it never reaches a kernel.
    """
    n = len(t) - k - 1
    tb, te = t[k], t[n]
    if xval != xval:
        return -1
    if (xval < tb or xval > te) and not extrapolate:
        return -1
    l = prev_l if (k < prev_l < n) else k
    while xval < t[l] and l != k:
        l -= 1
    l += 1
    while xval >= t[l] and l != n:
        l += 1
    return l - 1


class BSpline:
    """Univariate spline in the B-spline basis.

    Attributes are `t`, `c`, `k`, `extrapolate` and `axis`, as upstream.
    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        t = np.ascontiguousarray(t, dtype=np.float64)
        if extrapolate != "periodic":
            extrapolate = bool(extrapolate)
        c = np.array(c, dtype=np.float64, copy=True, order="C")
        k = operator.index(k)
        axis = operator.index(axis)
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError("Expect t to be a 1-D sorted array_like, but got t.")
        nt = len(t)
        if nt < k + 1:
            raise ValueError(f"Expect t to have at least {k + 1} elements.")
        if c.ndim == 0:
            raise ValueError("Expect c to have at least one dimension.")
        n = nt - k - 1
        if c.shape[0] != n:
            raise ValueError(f"Expect c to have {n} elements, but got {c.shape[0]}.")
        if c.ndim > 1 and axis != 0:
            c = np.moveaxis(c, axis, 0)
            axis = 0
        self.t, self.c, self.k = t, c, k
        self.extrapolate, self.axis = extrapolate, axis

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without doing the checks."""
        self = object.__new__(cls)
        self.t = np.ascontiguousarray(t, dtype=np.float64)
        self.c = np.ascontiguousarray(c, dtype=np.float64)
        self.k = operator.index(k)
        self.extrapolate = extrapolate
        self.axis = operator.index(axis)
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only)."""
        return self.t, self.c, self.k

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        `t` holds the internal knots; the degree is `len(t) - 2`.
        """
        t = np.asarray(t)
        k = t.shape[0] - 2
        if k < 0:
            raise ValueError("BSpline.basis_element requires at least 2 knots")
        t = f64(t)
        t = np.r_[(t[0] - 1.0,) * k, t, (t[-1] + 1.0,) * k]
        c = np.zeros_like(t)
        c[k] = 1.0
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        """Return the B-spline design matrix as a CSR sparse array."""
        from scipy.sparse import csr_array

        x = _as_float_array(x, True)
        t = _as_float_array(t, True)

        if extrapolate != "periodic":
            extrapolate = bool(extrapolate)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError("Expect t to be a 1-D sorted array_like.")
        if len(t) < 2 * k + 2:
            raise ValueError(f"Length t is not enough for k={k}.")

        if extrapolate == "periodic":
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
        ):
            raise ValueError(f"Out of bounds w/ x = {x}.")

        n = x.shape[0]
        data = zeros(n * (k + 1))
        indices = i64(np.zeros(n * (k + 1), dtype=np.int64))
        work = zeros(2 * k + 2)
        lib().msi_make_design_matrix(
            addr(x), n, addr(t), len(t), k, int(bool(extrapolate)),
            addr(indices), addr(data), addr(work),
        )
        indptr = np.arange(0, (n + 1) * (k + 1), k + 1, dtype=np.int32)
        return csr_array((data, indices, indptr), shape=(n, t.shape[0] - k - 1))

    def __call__(self, x, nu=0, extrapolate=None):
        """Evaluate the spline, or its `nu`-th derivative, at `x`."""
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        if extrapolate == "periodic":
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] - self.t[self.k])
            extrapolate = False

        cc = np.ascontiguousarray(self.c, dtype=np.float64)
        cc = cc.reshape(cc.shape[0], -1)
        nxp = x.shape[0]
        out = zeros(nxp * cc.shape[1])
        work = zeros(2 * self.k + 2)
        lib().msi_evaluate_spline(
            addr(self.t), len(self.t), addr(cc), cc.shape[1], self.k,
            addr(x), nxp, nu, int(bool(extrapolate)), addr(out), addr(work),
        )
        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            axes = list(range(out.ndim))
            axes = (
                axes[x_ndim : x_ndim + self.axis]
                + axes[:x_ndim]
                + axes[x_ndim + self.axis :]
            )
            out = out.transpose(axes)
        return out

    def derivative(self, nu=1):
        """Return a B-spline representing the `nu`-th derivative."""
        c = np.array(self.c, copy=True)
        t = self.t
        ct = t.shape[0] - c.shape[0]
        if ct > 0:
            c = np.concatenate((c, np.zeros((ct,) + c.shape[1:])))
        t, c, k = splder((t, c, self.k), nu)
        return self.construct_fast(
            t, c, k, extrapolate=self.extrapolate, axis=self.axis
        )

    def antiderivative(self, nu=1):
        """Return a B-spline representing the `nu`-th antiderivative."""
        c = np.array(self.c, copy=True)
        t = self.t
        ct = t.shape[0] - c.shape[0]
        if ct > 0:
            c = np.concatenate((c, np.zeros((ct,) + c.shape[1:])))
        t, c, k = splantider((t, c, self.k), nu)
        extrapolate = False if self.extrapolate == "periodic" else self.extrapolate
        return self.construct_fast(t, c, k, extrapolate=extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """Definite integral of the spline over the interval ``[a, b]``."""
        if extrapolate is None:
            extrapolate = self.extrapolate
        c = np.array(self.c, copy=True)
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.concatenate((c, np.zeros((ct,) + c.shape[1:])))

        # Swap integration bounds if needed.
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        n = self.t.size - self.k - 1

        if extrapolate != "periodic" and not extrapolate:
            # Shrink the integration interval, if needed.
            a = max(a, self.t[self.k])
            b = min(b, self.t[n])

        # Compute the antiderivative.
        ta, ca, ka = splantider((self.t, c, self.k), 1)
        spl = BSpline.construct_fast(ta, ca, ka, extrapolate=False)
        if extrapolate == "periodic":
            # Split the integral into the part over a full period and the rest.
            ts, te = self.t[self.k], self.t[n]
            n_periods, left = divmod(b - a, te - ts)
            integral = float(spl(te) - spl(ts)) * n_periods + (spl(a + left) - spl(a))
        else:
            integral = spl(b) - spl(a)
        return integral * sign

    def insert_knot(self, x, m=1):
        """Insert a new knot at `x` of multiplicity `m`."""
        x = float(x)
        t, c, k = self.t, self.c, self.k
        if x < t[k] or x > t[-k - 1]:
            raise ValueError(f"Can't insert a knot at {x}.")
        if m < 0:
            raise ValueError("m must be non-negative.")
        for _ in range(m):
            t, c = _insert(x, t, c, k, self.extrapolate == "periodic")
        return self.construct_fast(
            t, c, k, extrapolate=self.extrapolate, axis=self.axis
        )

    def __repr__(self):
        return (
            f"<BSpline degree={self.k}, t={self.t.size} knots, "
            f"c={self.c.shape}>"
        )


def _insert(xval, t, c, k, periodic=False):
    """Insert a single knot at `xval` (a port of FITPACK's `insert`)."""
    interval = find_interval(t, k, float(xval), k, False)
    if interval < 0:
        raise ValueError(f"Cannot insert the knot at {xval}.")

    # super edge case: a knot with multiplicity > k+1
    if t[interval] == t[interval + k + 1]:
        interval -= 1

    if periodic and (interval + 1 <= 2 * k) and (
        interval + 1 >= t.shape[0] - 2 * k
    ):
        raise ValueError("Not enough internal knots.")

    # knots
    tt = np.r_[t[: interval + 1], xval, t[interval + 1 :]]
    cc = np.zeros((c.shape[0] + 1,) + c.shape[1:], dtype=np.float64)
    # coefficients
    cc[interval + 1 :, ...] = c[interval:, ...]
    for i in range(interval, interval - k, -1):
        fac = (xval - tt[i]) / (tt[i + k + 1] - tt[i])
        cc[i, ...] = fac * c[i, ...] + (1.0 - fac) * c[i - 1, ...]
    cc[: interval - k + 1, ...] = c[: interval - k + 1, ...]

    if periodic:
        n = tt.shape[0]
        nk = n - k - 1
        n2k = n - 2 * k - 1
        period = tt[nk] - tt[k]
        if interval >= nk - k:
            tt[:k] = tt[nk - k : nk] - period
            cc[:k, ...] = cc[n2k : n2k + k, ...]
        if interval <= 2 * k - 1:
            tt[n - k :] = tt[k + 1 : k + 1 + k] + period
            cc[n2k : n2k + k, ...] = cc[:k, ...]
    return tt, cc


def splder(tck, n=1):
    """The `n`-th derivative of a B-spline ``(t, c, k)`` (FITPACK `splder`)."""
    t, c, k = tck
    if n < 0:
        return splantider(tck, -n)
    if n > k:
        raise ValueError(
            f"Order of derivative (n = {n!r}) must be <= order of spline "
            f"(k = {k!r})"
        )
    # upstream works with a coefficient vector padded to the knot count
    ct = t.shape[0] - c.shape[0]
    if ct > 0:
        c = np.concatenate((c, np.zeros((ct,) + c.shape[1:])))
    sh = (slice(None),) + ((None,) * len(c.shape[1:]))
    with np.errstate(invalid="raise", divide="raise"):
        try:
            for _ in range(n):
                dt = (t[k + 1 : -1] - t[1 : -k - 1])[sh]
                c = (c[1 : -1 - k, ...] - c[: -2 - k, ...]) * k / dt
                c = np.concatenate((c, np.zeros((k,) + c.shape[1:])))
                t = t[1:-1]
                k -= 1
        except FloatingPointError as exc:
            raise ValueError(
                "The spline has internal repeated knots and is not "
                f"differentiable {n} times"
            ) from exc
    return t, c, k


def splantider(tck, n=1):
    """The `n`-th antiderivative of a B-spline (FITPACK `splantider`)."""
    t, c, k = tck
    if n < 0:
        return splder(tck, -n)
    # upstream works with a coefficient vector padded to the knot count
    ct = t.shape[0] - c.shape[0]
    if ct > 0:
        c = np.concatenate((c, np.zeros((ct,) + c.shape[1:])))
    sh = (slice(None),) + (None,) * len(c.shape[1:])
    for _ in range(n):
        dt = (t[k + 1 :] - t[: -k - 1])[sh]
        c = np.cumulative_sum(c[: -k - 1, ...] * dt, axis=0) / (k + 1)
        c = np.concatenate(
            (np.zeros((1,) + c.shape[1:]), c, np.stack([c[-1, ...]] * (k + 2)))
        )
        t = np.concatenate((t[0:1], t, t[-1:]))
        k += 1
    return t, c, k


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True):
    """Create an interpolating B-spline with the given boundary conditions.

    Mirrors ``scipy.interpolate.make_interp_spline(x, y, k=3, t=None,
    bc_type=None, axis=0, check_finite=True)``.
    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == "not-a-knot" or bc_type == "periodic":
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as exc:
            raise ValueError(f"Unknown boundary condition: {bc_type}") from exc

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    axis = operator.index(axis)
    if axis >= y.ndim or axis < -y.ndim:
        raise np.exceptions.AxisError(axis, y.ndim)
    y = np.moveaxis(y, axis, 0)

    # sanity check the input
    if bc_type == "periodic" and not np.allclose(y[0], y[-1], atol=1e-15):
        raise ValueError(
            "First and last points does not match while periodic case expected"
        )
    if x.size != y.shape[0]:
        raise ValueError(f"Shapes of x {x.shape} and y {y.shape} are incompatible")
    if np.any(x[1:] == x[:-1]):
        raise ValueError("Expect x to not have duplicates")
    if x.ndim != 1 or np.any(x[1:] < x[:-1]):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    # special-case k=0 right away
    if k == 0:
        if any(v is not None for v in (t, deriv_l, deriv_r)):
            raise ValueError(
                "Too much info for k=0: t and bc_type can only be None."
            )

    # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        t = np.r_[x[0], x, x[-1]]
        c = np.ascontiguousarray(y, dtype=np.float64)
        return BSpline.construct_fast(t, c, 1, axis=axis)
    if bc_type == "periodic" and t is not None:
        raise NotImplementedError(
            "For periodic case t is constructed automatically and can not be "
            "passed manually"
        )

    # The glue mirrors the knot-vector length rule so it can size the buffers;
    # the knots themselves are built inside the Mojo kernel.
    nx = int(x.size)
    t_in = f64(t) if t is not None else zeros(1)
    nt_in = int(t_in.size) if t is not None else 0
    if nt_in == 0:
        if deriv_l is None and deriv_r is None:
            nt = nx + 2 * k if bc_type == "periodic" else nx + k + 1
        else:
            nt = nx + 2 * k
    else:
        nt = nt_in

    if k < 0:
        raise ValueError("Expect non-negative k.")
    if nt_in and (t_in.ndim != 1 or np.any(t_in[1:] < t_in[:-1])):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if nt < nx + k + 1:
        raise ValueError(f"Got {nt} knots, need at least {nx + k + 1}.")
    if nt_in and ((x[0] < t_in[k]) or (x[-1] > t_in[-k])):
        raise ValueError(f"Out of bounds w/ x = {x}.")

    ntco = nt - k - 1
    ycols = int(np.prod(y.shape[1:])) if y.ndim > 1 else 1
    y2 = np.ascontiguousarray(y.reshape(nx, ycols), dtype=np.float64)

    if k == 0:
        # upstream returns before the banded machinery exists
        t = np.r_[x, x[-1]]
        c = np.ascontiguousarray(y, dtype=np.float64)
        return BSpline.construct_fast(t, c, 0, axis=axis)

    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    dl_ords, dl_vals = _process_deriv_spec(deriv_l)
    dr_ords, dr_vals = _process_deriv_spec(deriv_r)
    nleft, nright = int(dl_ords.size), int(dr_ords.size)
    dl_ords = i64(dl_ords) if nleft else i64(np.zeros(1, dtype=np.int64))
    dr_ords = i64(dr_ords) if nright else i64(np.zeros(1, dtype=np.int64))
    dl_vals = f64(dl_vals.reshape(-1, ycols)) if nleft else zeros(1)
    dr_vals = f64(dr_vals.reshape(-1, ycols)) if nright else zeros(1)

    if not all(0 <= i <= k for i in dl_ords):
        raise ValueError(f"Bad boundary conditions at {x[0]}.")
    if not all(0 <= i <= k for i in dr_ords):
        raise ValueError(f"Bad boundary conditions at {x[-1]}.")

    # have `n` conditions for `nt` coefficients; need nt-n derivatives.
    # The periodic case is dispatched before this check, as upstream does.
    if bc_type != "periodic" and ntco - nx != nleft + nright:
        raise ValueError(
            "The number of derivatives at boundaries does not match: expected "
            f"{ntco - nx}, got {nleft}+{nright}"
        )

    # bail out if the `y` array is zero-sized
    if y.size == 0:
        c = np.zeros((ntco,) + y.shape[1:], dtype=np.float64)
        return BSpline.construct_fast(zeros(nt), c, k, axis=axis)

    bc_sel = 3 if bc_type == "periodic" else (0 if nleft == 0 and nright == 0 else 2)

    t_out = zeros(nt)
    c_out = zeros(ntco * ycols)
    ab = zeros(max((3 * k + 1) * ntco, (nx + k - 1) ** 2 if bc_sel == 3 else 1))
    rhs = zeros(ntco * ycols)
    work = zeros(2 * k + 2)
    naux = lib().msi_periodic_scratch(nx, k) if bc_sel == 3 else ntco * (k + 1) + 1
    aux = zeros(naux)
    ipiv = i64(np.zeros(ntco, dtype=np.int64))

    if bc_sel == 3:
        # the periodic kernel solves one column per call; the matrix is
        # rebuilt and re-solved per column anyway, so this costs nothing
        cols = []
        for c in range(ycols):
            ycol = np.ascontiguousarray(y2[:, c]).reshape(nx, 1)
            c_col = zeros(ntco)
            info = lib().msi_make_interp_spline(
                addr(x), nx, addr(ycol), 1, k, bc_sel,
                addr(dl_ords), addr(dl_vals), nleft, addr(dr_ords),
                addr(dr_vals), nright, addr(t_in), nt_in, addr(t_out),
                addr(c_col), addr(ab), addr(rhs), addr(work), addr(aux),
                addr(ipiv),
            )
            if info > 0:
                raise np.linalg.LinAlgError("Colocation matrix is singular.")
            cols.append(c_col)
        c = np.ascontiguousarray(
            np.stack(cols, axis=1).reshape((ntco,) + y.shape[1:])
        )
        return BSpline.construct_fast(
            t_out, c, k, extrapolate="periodic", axis=axis
        )

    info = lib().msi_make_interp_spline(
        addr(x), nx, addr(y2), ycols, k, bc_sel,
        addr(dl_ords), addr(dl_vals), nleft, addr(dr_ords), addr(dr_vals), nright,
        addr(t_in), nt_in, addr(t_out), addr(c_out), addr(ab), addr(rhs),
        addr(work), addr(aux), addr(ipiv),
    )
    if info > 0:
        raise np.linalg.LinAlgError("Colocation matrix is singular.")
    if info < 0:
        raise ValueError(f"illegal value in {-info}-th argument of internal gbsv")
    c = np.ascontiguousarray(c_out.reshape((ntco,) + y.shape[1:]))
    return BSpline.construct_fast(t_out, c, k, axis=axis)


def make_lsq_spline(
    x, y, t, k=3, w=None, axis=0, check_finite=True, *, method="norm-eq"
):
    """Create a smoothing B-spline satisfying the LSQ criterion.

    Mirrors ``scipy.interpolate.make_lsq_spline``; `method` must be
    ``'norm-eq'`` -- the port implements the normal-equations path, not the
    QR path.
    """
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    w = _as_float_array(w, check_finite) if w is not None else np.ones_like(x)
    k = operator.index(k)
    axis = operator.index(axis)
    if axis >= y.ndim or axis < -y.ndim:
        raise np.exceptions.AxisError(axis, y.ndim)
    y = np.moveaxis(y, axis, 0)

    if x.ndim != 1:
        raise ValueError("Expect x to be a 1-D sequence.")
    if x.shape[0] < k + 1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1D strictly increasing sequence.")
    if x.size != y.shape[0]:
        raise ValueError(f"Shapes of x {x.shape} and y {y.shape} are incompatible")
    if k > 0 and np.any((x < t[k]) | (x > t[-k])):
        raise ValueError(f"Out of bounds w/ x = {x}.")
    if x.size != w.size:
        raise ValueError(f"Shapes of x {x.shape} and w {w.shape} are incompatible")
    if method == "qr":
        raise NotImplementedError(
            "make_lsq_spline(method='qr') is not ported; use method='norm-eq'"
        )
    if method != "norm-eq":
        raise ValueError(f"Unknown {method =}.")
    if np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    nx = int(x.size)
    n = int(t.size - k - 1)
    ycols = int(np.prod(y.shape[1:])) if y.ndim > 1 else 1
    y2 = np.ascontiguousarray(y.reshape(nx, ycols), dtype=np.float64)

    ab = zeros((k + 1) * n)
    rhs = zeros(n * ycols)
    c_out = zeros(n * ycols)
    work = zeros(2 * k + 2)
    ipiv = i64(np.zeros(n, dtype=np.int64))
    w2 = f64(w)
    info = lib().msi_make_lsq_spline(
        addr(x), nx, addr(y2), ycols, addr(t), len(t), k, addr(w2),
        addr(c_out), addr(ab), addr(rhs), addr(work), addr(ipiv),
    )
    if info > 0:
        raise np.linalg.LinAlgError(
            "Expected a positive definite Gram matrix, got leading minor of "
            f"order {info} which is not positive definite."
        )
    c = np.ascontiguousarray(c_out.reshape((n,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, axis=axis)
