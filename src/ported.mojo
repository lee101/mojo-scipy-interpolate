"""Spline and RBF interpolation kernels, ported from ``scipy.interpolate``.

Sections follow upstream's module order so this file reads top to bottom the
way the package does:

    _bspl.pyx       _deBoor_D, find_interval, evaluate_spline, _colloc,
                    _handle_lhs_derivatives, _norm_eq_lsq, _make_design_matrix
    _bsplines.py    _augknt, _not_a_knot, _periodic_knots, _woodbury_algorithm,
                    make_interp_spline, make_lsq_spline
    _ppoly.pyx      evaluate_poly1, evaluate
    _rgi_cython.pyx find_interval_ascending, find_indices, evaluate_linear_2d
    _interpolate.py interp1d's _call_linear / _call_nearest / _call_previousnext
    _rbf.py         Rbf kernels and the collocation matrix
    _rbfinterp.py   the RBF kernels and the linear system assembly

Where upstream indexes a kernel by name (``NAME_TO_FUNC[kernel]``) or a
boundary condition by string (``bc_type``), the numeric kernels take an
``Int`` selector that the ctypes glue resolves from the same table. The branch
itself stays where upstream has it.

Banded matrices use the LAPACK band layout: element ``A(i, j)`` of a matrix
with ``kl`` lower and ``ku`` upper diagonals lives at
``ab[(kl + ku + i - j) * stride + j]`` in a flat buffer.
"""

from max.algorithm import parallelize
from std.math import exp, log, sqrt

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

# Worker count handed to every parallel region; explicit because the shorter
# `parallelize` forms misbehave when launched from a ctypes host.
comptime NWORKERS = 8
# Below this many outer iterations the launch cost dominates and the loops run
# serially, so they read exactly like the upstream ones.
comptime PARALLEL_MIN = 4096


def nan_value() -> Float64:
    var z = 0.0
    return z / z


# ===================================================================
# _bspl.pyx  /  src/__fitpack.h
# ===================================================================


def find_interval(
    t: Ptr, nt: Int, k: Int, xval: Float64, prev_l: Int, extrapolate: Bool
) -> Int:
    """`t[l] <= xval < t[l+1]`, by linear search with locality.

    Returns -1 when `xval` is nan, or when it is out of bounds and
    `extrapolate` is off. `prev_l` seeds the search and may be smaller than k.
    """
    var n = nt - k - 1
    var tb = t.unsafe_offset(k)[]
    var te = t.unsafe_offset(n)[]

    if xval != xval:
        # nan
        return -1

    if ((xval < tb) or (xval > te)) and not extrapolate:
        return -1

    var l = prev_l if (k < prev_l < n) else k

    # xval is in support, search for interval s.t. t.unsafe_offset(l)[] <= xval < t.unsafe_offset(l+1)[]
    while xval < t.unsafe_offset(l)[] and l != k:
        l -= 1

    l += 1
    # `l != n` alone is upstream's bound; the `nt - 1` term additionally stops a
    # degenerate spline (fewer than k coefficients) from reading past the knots
    while l < nt - 1 and xval >= t.unsafe_offset(l)[] and l != n:
        l += 1

    return l - 1


def deBoor_D(t: Ptr, x: Float64, k: Int, ell: Int, m: Int, result: Ptr):
    """The `k+1` non-zero B-splines on the interval `t[ell] <= x < t[ell+1]`.

    `result[0 .. k]` receives ``B_{ell-k}(x), ..., B_{ell}(x)``, or their
    `m`-th derivatives. `result[k+1 .. 2k+1]` is scratch of the same size, so
    the caller must hand in a buffer of `2k+2` doubles.
    """
    var hh = result.unsafe_offset(k + 1)
    var h = result
    result.unsafe_offset(0)[] = 1.0

    # k-m "standard" deBoor iterations; h ends up holding
    # B_{ell, k-m}(x), ..., B_{ell-k+m, k-m}(x).
    for j in range(1, k - m + 1):
        for q in range(j):
            hh.unsafe_offset(q)[] = h.unsafe_offset(q)[]
        h.unsafe_offset(0)[] = 0.0
        for n in range(1, j + 1):
            var ind = ell + n
            var xb = t.unsafe_offset(ind)[]
            var xa = t.unsafe_offset(ind - j)[]
            if xb == xa:
                h.unsafe_offset(n)[] = 0.0
                continue
            var w = hh.unsafe_offset(n - 1)[] / (xb - xa)
            h.unsafe_offset(n - 1)[] += w * (xb - x)
            h.unsafe_offset(n)[] = w * (x - xa)

    # m "derivative" recursions convert the values of beta into the m-th one.
    for j in range(k - m + 1, k + 1):
        for q in range(j):
            hh.unsafe_offset(q)[] = h.unsafe_offset(q)[]
        h.unsafe_offset(0)[] = 0.0
        for n in range(1, j + 1):
            var ind = ell + n
            var xb = t.unsafe_offset(ind)[]
            var xa = t.unsafe_offset(ind - j)[]
            if xb == xa:
                h.unsafe_offset(m)[] = 0.0
                continue
            var w = Float64(j) * hh.unsafe_offset(n - 1)[] / (xb - xa)
            h.unsafe_offset(n - 1)[] -= w
            h.unsafe_offset(n)[] = w


def evaluate_spline(
    t: Ptr,
    nt: Int,
    c: Ptr,
    ccols: Int,
    k: Int,
    xp: Ptr,
    nxp: Int,
    nu: Int,
    extrapolate: Bool,
    dst: Ptr,
    work: Ptr,
):
    """Evaluate a spline in the B-spline basis.

    `c` is `n` x `ccols` and `out` is `nxp` x `ccols`, both row-major; `work`
    is scratch of `2k+2` doubles.
    """
    var interval = k
    for ip in range(nxp):
        var xval = xp.unsafe_offset(ip)[]

        # Find correct interval
        interval = find_interval(t, nt, k, xval, interval, extrapolate)

        if interval < 0:
            # xval was nan etc
            for jp in range(ccols):
                dst.unsafe_offset(ip * ccols + jp)[] = nan_value()
            continue

        # Evaluate (k+1) b-splines which are non-zero on the interval.
        # on return, first k+1 elements of work are B_{m-k},..., B_{m}
        deBoor_D(t, xval, k, interval, nu, work)

        # Form linear combinations
        for jp in range(ccols):
            var acc = 0.0
            for a in range(k + 1):
                acc += c.unsafe_offset((interval + a - k) * ccols + jp)[] * work.unsafe_offset(a)[]
            dst.unsafe_offset(ip * ccols + jp)[] = acc


def colloc(
    x: Ptr, nx: Int, t: Ptr, nt: Int, k: Int, ab: Ptr, ncols: Int, offset: Int,
    work: Ptr,
):
    """Build the B-spline collocation matrix in the banded storage.

    `ab[kl + ku + row - clmn, clmn]` is `B_{clmn,k}(x[row])`, with `kl = ku = k`.
    """
    var kl = k
    var ku = k
    var left = k
    for j in range(nx):
        var xval = x.unsafe_offset(j)[]
        # find interval
        left = find_interval(t, nt, k, xval, left, False)

        # fill a row
        deBoor_D(t, xval, k, left, 0, work)
        # for a full matrix it would be ``A[j + offset, left-k:left+1] = bb``
        # in the banded storage, need to spread the row over. Entries that fall
        # outside the band are structurally zero, so they are simply skipped.
        for a in range(k + 1):
            var clmn = left - k + a
            var row = kl + ku + j + offset - clmn
            if 0 <= row <= 2 * kl + ku:
                ab.unsafe_offset(row * ncols + clmn)[] = work.unsafe_offset(a)[]


def handle_lhs_derivatives(
    t: Ptr,
    nt: Int,
    k: Int,
    xval: Float64,
    ab: Ptr,
    stride: Int,
    kl: Int,
    ku: Int,
    deriv_ords: IPtr,
    nords: Int,
    offset: Int,
    work: Ptr,
):
    """Fill in the collocation rows for known derivatives at `xval`."""
    # find where `xval` is in the knot vector, `t`
    var left = find_interval(t, nt, k, xval, k, False)

    # compute and fill in the derivatives @ xval
    for row in range(nords):
        var nu = Int(deriv_ords.unsafe_offset(row)[])
        deBoor_D(t, xval, k, left, nu, work)

        # if A were a full matrix, it would be just
        # ``A[row + offset, left-k:left+1] = bb``.
        for a in range(k + 1):
            var clmn = left - k + a
            ab.unsafe_offset((kl + ku + offset + row - clmn) * stride + clmn)[] = work.unsafe_offset(a)[]


def norm_eq_lsq(
    x: Ptr, nx: Int, t: Ptr, nt: Int, k: Int, y: Ptr, ycols: Int, w: Ptr,
    ab: Ptr, ncols: Int, rhs: Ptr, work: Ptr,
):
    """Normal equations for the B-spline LSQ problem.

    Fills `A.T @ A` (lower-banded, `k+1` rows) and `A.T @ y` for
    ``A.T @ A @ c = A.T @ y``.
    """
    var left = k
    for j in range(nx):
        var xval = x.unsafe_offset(j)[]
        var wval = w.unsafe_offset(j)[] * w.unsafe_offset(j)[]
        # find interval
        left = find_interval(t, nt, k, xval, left, False)

        # non-zero B-splines at xval
        deBoor_D(t, xval, k, left, 0, work)

        # non-zero values of A.T @ A: banded storage w/ lower=True
        # The colloq matrix in full storage would be
        #   A[j, left-k:left+1] = wrk,
        # Here we work out A.T @ A *in the banded storage* w/lower=True
        # see the docstring of `scipy.linalg.cholesky_banded`.
        for r in range(k + 1):
            var row = left - k + r
            for s in range(r + 1):
                var clmn = left - k + s
                ab.unsafe_offset((r - s) * ncols + clmn)[] += work.unsafe_offset(r)[] * work.unsafe_offset(s)[] * wval

            # ... and A.T @ y
            for ci in range(ycols):
                rhs.unsafe_offset(row * ycols + ci)[] += work.unsafe_offset(r)[] * y.unsafe_offset(j * ycols + ci)[] * wval


def make_design_matrix(
    x: Ptr, nx: Int, t: Ptr, nt: Int, k: Int, extrapolate: Bool, indices: IPtr,
    data: Ptr, work: Ptr,
):
    """Data and column indices of the CSR B-spline design matrix."""
    var ind = k
    for i in range(nx):
        var xval = x.unsafe_offset(i)[]

        # Find correct interval.
        ind = find_interval(t, nt, k, xval, ind, extrapolate)
        deBoor_D(t, xval, k, ind, 0, work)

        # data.unsafe_offset((k + 1) * i : (k + 1) * (i + 1))[] = work.unsafe_offset(:k + 1)[]
        # indices.unsafe_offset((k + 1) * i : (k + 1) * (i + 1))[] = np.arange(ind - k, ind + 1)
        for j in range(k + 1):
            var m = (k + 1) * i + j
            data.unsafe_offset(m)[] = work.unsafe_offset(j)[]
            indices.unsafe_offset(m)[] = Int64(ind - k + j)


# ===================================================================
# scipy.linalg banded factorisations (LAPACK dgbsv / dpbtrf, dgbtrf, dpbttrs)
# ===================================================================


def gbtrf(kl: Int, ku: Int, ab: Ptr, stride: Int, n: Int, ipiv: IPtr) -> Int:
    """In-place LU factorisation of a banded matrix (LAPACK's dgbtf2).

    `ab` arrives in LAPACK's band layout: `A(i, j)` sits on band row
    `kl + ku + i - j` of a `(2 * kl + ku + 1)` x `n` buffer. Rows below
    `kl + ku` are the fill-in rows partial pivoting needs, exactly as in
    LAPACK. On exit the multipliers sit on band rows `kl + ku + 1 ..
    kl + ku + kl` and the upper factor above the diagonal.
    `ipiv[j]` receives the row `j` was interchanged with.
    """
    var kv = kl + ku
    # JU is the index of the last column affected by the current stage
    var ju = 0
    for j in range(n):
        # set the fill-in elements of the column kv steps ahead to zero
        if j + kv < n:
            for i in range(kl):
                ab.unsafe_offset(i * stride + j + kv)[] = 0.0

        # find the pivot; KM is the number of subdiagonal entries here
        var km = min(kl, n - 1 - j)
        var jp = 0
        var vmax = abs(ab.unsafe_offset(kv * stride + j)[])
        for i in range(1, km + 1):
            var av = abs(ab.unsafe_offset((kv + i) * stride + j)[])
            if av > vmax:
                vmax = av
                jp = i
        ipiv.unsafe_offset(j)[] = Int64(j + jp)

        if ab.unsafe_offset((kv + jp) * stride + j)[] == 0.0:
            # the pivot is zero; report the first such column
            return j + 1

        if ju < min(j + ku + jp, n - 1):
            ju = min(j + ku + jp, n - 1)

        # apply the interchange to columns j .. ju; the two rows run along
        # parallel band diagonals, one step up and one column right per step
        if jp != 0:
            for m in range(ju - j + 1):
                var t = ab.unsafe_offset((kv + jp - m) * stride + j + m)[]
                ab.unsafe_offset((kv + jp - m) * stride + j + m)[] = (
                    ab.unsafe_offset((kv - m) * stride + j + m)[]
                )
                ab.unsafe_offset((kv - m) * stride + j + m)[] = t

        if km > 0:
            var inv = 1.0 / ab.unsafe_offset(kv * stride + j)[]
            # compute the multipliers
            for i in range(1, km + 1):
                ab.unsafe_offset((kv + i) * stride + j)[] *= inv
            # update the trailing submatrix within the band: a rank-1 update of
            # rows j+1 .. j+km by the pivot row's upper band
            for i in range(1, km + 1):
                var mult = ab.unsafe_offset((kv + i) * stride + j)[]
                if mult != 0.0:
                    for m in range(1, ju - j + 1):
                        ab.unsafe_offset((kv + i - m) * stride + j + m)[] -= (
                            mult * ab.unsafe_offset((kv - m) * stride + j + m)[]
                        )
    return 0


def gbtrs(
    kl: Int, ku: Int, ab: Ptr, stride: Int, n: Int, ipiv: IPtr, b: Ptr, nrhs: Int
):
    """Solve the factored banded system (LAPACK's dgbtrs).

    `b` is `n` x `nrhs`, row-major.
    """
    var kv = kl + ku
    # L = P(1) L(1) ... P(n-1) L(n-1): interchange, then eliminate, per column
    for j in range(n - 1):
        var lm = min(kl, n - 1 - j)
        var l = Int(ipiv.unsafe_offset(j)[])
        if l != j:
            for c in range(nrhs):
                var t = b.unsafe_offset(l * nrhs + c)[]
                b.unsafe_offset(l * nrhs + c)[] = b.unsafe_offset(j * nrhs + c)[]
                b.unsafe_offset(j * nrhs + c)[] = t
        for i in range(1, lm + 1):
            var mult = ab.unsafe_offset((kv + i) * stride + j)[]
            if mult != 0.0:
                for c in range(nrhs):
                    b.unsafe_offset((j + i) * nrhs + c)[] -= (
                        b.unsafe_offset(j * nrhs + c)[] * mult
                    )

    # U is upper triangular with bandwidth kl + ku (dtbsv)
    for j in range(n - 1, -1, -1):
        for c in range(nrhs):
            b.unsafe_offset(j * nrhs + c)[] /= ab.unsafe_offset(kv * stride + j)[]
        var ju = min(kv, j)
        for i in range(1, ju + 1):
            var mult = ab.unsafe_offset((kv - i) * stride + j)[]
            if mult != 0.0:
                for c in range(nrhs):
                    b.unsafe_offset((j - i) * nrhs + c)[] -= (
                        b.unsafe_offset(j * nrhs + c)[] * mult
                    )


def solve_banded(
    l: Int, u: Int, ab: Ptr, stride: Int, b: Ptr, n: Int, nrhs: Int, ipiv: IPtr
) -> Int:
    """`scipy.linalg.solve_banded` against LAPACK's dgbsv. Returns `info`.

    `stride` is the row stride, i.e. the number of columns `n`.
    """
    var kl = l
    var ku = u
    var kv = kl + ku

    var info = gbtrf(kl, ku, ab, stride, n, ipiv)
    if info != 0:
        return info
    gbtrs(kl, ku, ab, stride, n, ipiv, b, nrhs)
    return 0


def pbtrf(k: Int, ab: Ptr, stride: Int, n: Int) -> Int:
    """`scipy.linalg.cholesky_banded` (lower) against LAPACK's dpbtrf.

    `ab[(i - j) * stride + j]` is `A(i, j)` for `i >= j`, and `stride` is the
    number of columns `n`. Returns a positive `j` on the first non-positive
    pivot, like LAPACK's `info`.
    """
    for j in range(n):
        var ajj = ab.unsafe_offset(j)[]
        if ajj <= 0.0:
            return j + 1
        ajj = sqrt(ajj)
        ab.unsafe_offset(j)[] = ajj
        var kn = min(k, n - 1 - j)
        if kn > 0:
            for u in range(1, kn + 1):
                ab.unsafe_offset(u * stride + j)[] /= ajj
            # the diagonal of the next columns is left squared, reduced here
            for u in range(1, kn + 1):
                ab.unsafe_offset(j + u)[] -= (
                    ab.unsafe_offset(u * stride + j)[]
                    * ab.unsafe_offset(u * stride + j)[]
                )
                for v in range(1, kn - u + 1):
                    ab.unsafe_offset(v * stride + j + u)[] -= (
                        ab.unsafe_offset(u * stride + j)[]
                        * ab.unsafe_offset((u + v) * stride + j)[]
                    )
    return 0


def pbtrs(k: Int, ab: Ptr, stride: Int, b: Ptr, n: Int, nrhs: Int):
    """`cho_solve_banded` against LAPACK's dpbttrs; `b` is `n` x `nrhs`.

    Each column is finished before it is used, so the diagonal division
    precedes the update in both sweeps.
    """
    # forward sweep with the lower factor
    for j in range(n):
        for c in range(nrhs):
            b.unsafe_offset(j * nrhs + c)[] /= ab.unsafe_offset(j)[]
        var kn = min(k, n - 1 - j)
        for i in range(1, kn + 1):
            var mult = ab.unsafe_offset(i * stride + j)[]
            if mult != 0.0:
                for c in range(nrhs):
                    b.unsafe_offset((j + i) * nrhs + c)[] -= (
                        b.unsafe_offset(j * nrhs + c)[] * mult
                    )
    # backward sweep with its transpose
    for j in range(n - 1, -1, -1):
        for c in range(nrhs):
            b.unsafe_offset(j * nrhs + c)[] /= ab.unsafe_offset(j)[]
        var lo = j - k
        if lo < 0:
            lo = 0
        for i in range(lo, j):
            var mult = ab.unsafe_offset((j - i) * stride + i)[]
            if mult != 0.0:
                for c in range(nrhs):
                    b.unsafe_offset(i * nrhs + c)[] -= (
                        b.unsafe_offset(j * nrhs + c)[] * mult
                    )


def solve_dense(a: Ptr, b: Ptr, n: Int, nrhs: Int, ipiv: IPtr) -> Int:
    """LU with partial pivoting on a row-major `n` x `n` matrix.

    `b` is `n` x `nrhs`, row-major. The strict lower triangle of `a` ends up
    holding the unit-lower factor, so the back substitution only reads the
    upper triangle.
    """
    for j in range(n):
        var jp = j
        var vmax = abs(a.unsafe_offset(j * n + j)[])
        for i in range(j + 1, n):
            var av = abs(a.unsafe_offset(i * n + j)[])
            if av > vmax:
                vmax = av
                jp = i
        ipiv.unsafe_offset(j)[] = Int64(jp)
        if jp != j:
            for c in range(n):
                var t = a.unsafe_offset(j * n + c)[]
                a.unsafe_offset(j * n + c)[] = a.unsafe_offset(jp * n + c)[]
                a.unsafe_offset(jp * n + c)[] = t
            for c in range(nrhs):
                var t = b.unsafe_offset(j * nrhs + c)[]
                b.unsafe_offset(j * nrhs + c)[] = b.unsafe_offset(jp * nrhs + c)[]
                b.unsafe_offset(jp * nrhs + c)[] = t
        if a.unsafe_offset(j * n + j)[] == 0.0:
            return j + 1
        for i in range(j + 1, n):
            a.unsafe_offset(i * n + j)[] /= a.unsafe_offset(j * n + j)[]
        for i in range(j + 1, n):
            var mult = a.unsafe_offset(i * n + j)[]
            if mult != 0.0:
                for c in range(j + 1, n):
                    a.unsafe_offset(i * n + c)[] -= (
                        mult * a.unsafe_offset(j * n + c)[]
                    )
                for c in range(nrhs):
                    b.unsafe_offset(i * nrhs + c)[] -= (
                        mult * b.unsafe_offset(j * nrhs + c)[]
                    )
    # back substitution over the upper triangle only
    for j in range(n - 1, -1, -1):
        for c in range(nrhs):
            b.unsafe_offset(j * nrhs + c)[] /= a.unsafe_offset(j * n + j)[]
        for i in range(j):
            var mult = a.unsafe_offset(i * n + j)[]
            if mult != 0.0:
                for c in range(nrhs):
                    b.unsafe_offset(i * nrhs + c)[] -= (
                        b.unsafe_offset(j * nrhs + c)[] * mult
                    )
    return 0


def _periodic_knots(x: Ptr, nx: Int, k: Int, t: Ptr, dx: Ptr):
    """`_periodic_knots`: a vector of nodes on a circle."""
    var n = nx
    for i in range(n):
        t.unsafe_offset(k + i)[] = x.unsafe_offset(i)[]
    if k % 2 == 0:
        for i in range(n - 1):
            dx.unsafe_offset(i)[] = t.unsafe_offset(k + i + 1)[] - t.unsafe_offset(k + i)[]
        for i in range(1, n - 1):
            t.unsafe_offset(k + i)[] -= dx.unsafe_offset(i - 1)[] / 2
    for i in range(n - 1):
        dx.unsafe_offset(i)[] = t.unsafe_offset(k + i + 1)[] - t.unsafe_offset(k + i)[]
    for i in range(k):
        # filling first `k` elements in descending order
        t.unsafe_offset(k - i - 1)[] = (
            t.unsafe_offset(k - i)[] - dx.unsafe_offset(n - 2 - (i % (n - 1)))[]
        )
        # filling last `k` elements in ascending order
        t.unsafe_offset(n + k + i)[] = (
            t.unsafe_offset(n + k + i - 1)[] + dx.unsafe_offset(i % (n - 1))[]
        )


def _augknt(x: Ptr, nx: Int, k: Int, t: Ptr):
    """`_augknt`: a knot vector appropriate for the order-k interpolation."""
    for i in range(k):
        t.unsafe_offset(i)[] = x.unsafe_offset(0)[]
    for i in range(nx):
        t.unsafe_offset(k + i)[] = x.unsafe_offset(i)[]
    for i in range(k):
        t.unsafe_offset(k + nx + i)[] = x.unsafe_offset(nx - 1)[]


def _not_a_knot(x: Ptr, nx: Int, k: Int, t: Ptr, scratch: Ptr):
    """`_not_a_knot`: knot vector for the not-a-knot boundary condition.

    cf de Boor, XIII(12). For even k it is a bit ad hoc: Greville sites with
    the 2nd and 2nd-to-last data points omitted, a la not-a-knot.
    """
    var k2 = (k + 1) // 2 if k % 2 == 1 else k // 2
    var inner = nx
    if k % 2 == 1:
        for i in range(nx):
            t.unsafe_offset(i)[] = x.unsafe_offset(i)[]
    else:
        inner = nx - 1
        for i in range(nx - 1):
            t.unsafe_offset(i)[] = (x.unsafe_offset(i + 1)[] + x.unsafe_offset(i)[]) / 2
    # keep the interior sites, then pad with k+1 copies of each endpoint; the
    # source and destination windows overlap, so move through a scratch row
    var m = inner - 2 * k2
    if m < 0:
        m = 0
    for i in range(m):
        scratch.unsafe_offset(i)[] = t.unsafe_offset(k2 + i)[]
    for i in range(k + 1):
        t.unsafe_offset(i)[] = x.unsafe_offset(0)[]
    for i in range(m):
        t.unsafe_offset(k + 1 + i)[] = scratch.unsafe_offset(i)[]
    for i in range(k + 1):
        t.unsafe_offset(k + 1 + m + i)[] = x.unsafe_offset(nx - 1)[]


def _make_interp_per_full_matr(
    x: Ptr, nx: Int, y: Ptr, yoff: Int, t: Ptr, nt: Int, k: Int, b: Ptr,
    matr: Ptr, ipiv: IPtr, work: Ptr,
) -> Int:
    """`_make_interp_per_full_matr`: full system for the periodic case.

    First `k-1` rows impose periodicity, the last `n` rows interpolate. `b`
    receives the coefficients.
    """
    var n = nx
    var m = n + k - 1
    for i in range(m * m):
        matr.unsafe_offset(i)[] = 0.0

    # derivatives at x.unsafe_offset(0)[] and x[-1]:
    for i in range(k - 1):
        deBoor_D(t, x.unsafe_offset(0)[], k, k, i + 1, work)
        for q in range(k + 1):
            matr.unsafe_offset(i * m + q)[] += work.unsafe_offset(q)[]
        deBoor_D(t, x.unsafe_offset(n - 1)[], k, n + k - 1, i + 1, work)
        for q in range(k):
            matr.unsafe_offset(i * m + (m - k) + q)[] -= work.unsafe_offset(q)[]

    # colocation matrix
    for i in range(n):
        var xval = x.unsafe_offset(i)[]
        # find interval
        var left = k
        if xval != t.unsafe_offset(k)[]:
            # np.searchsorted(t, xval) - 1
            var lo = 0
            var hi = nt - 1
            while lo < hi:
                var mid = (lo + hi) // 2
                if t.unsafe_offset(mid)[] < xval:
                    lo = mid + 1
                else:
                    hi = mid
            left = lo - 1

        # fill a row
        deBoor_D(t, xval, k, left, 0, work)
        for q in range(k + 1):
            matr.unsafe_offset((i + k - 1) * m + (left - k) + q)[] = work.unsafe_offset(q)[]

    # RHS is ``k - 1`` zeros and ``n`` ordinates in this case.
    for i in range(k - 1):
        b.unsafe_offset(i)[] = 0.0
    for i in range(n):
        b.unsafe_offset(k - 1 + i)[] = y.unsafe_offset(yoff + i)[]
    return solve_dense(matr, b, m, 1, ipiv)


def periodic_scratch(nx: Int, k: Int) -> Int:
    """Doubles `make_interp_spline` needs for the periodic scratch space."""
    var m = nx + k - 1
    return m * m + nx


def make_interp_spline(
    x: Ptr,
    nx: Int,
    y: Ptr,
    ycols: Int,
    k: Int,
    bc_type: Int,
    dl: IPtr,
    dlv: Ptr,
    nleft: Int,
    dr: IPtr,
    drv: Ptr,
    nright: Int,
    t_in: Ptr,
    nt_in: Int,
    t_out: Ptr,
    c_out: Ptr,
    ab: Ptr,
    rhs: Ptr,
    work: Ptr,
    aux: Ptr,
    ipiv: IPtr,
) -> Int:
    """`make_interp_spline` with the `bc_type` string already resolved.

    `bc_type`: 0 = None/not-a-knot, 1 = 'clamped', 2 = an explicit
    ``(deriv_l, deriv_r)`` pair, 3 = 'periodic'. `t_in` is ignored when
    `nt_in` is 0. Returns 0 on success or a positive LAPACK `info`.
    """
    if k == 0:
        # special-case k=0 right away
        for i in range(nx):
            t_out.unsafe_offset(i)[] = x.unsafe_offset(i)[]
        t_out.unsafe_offset(nx)[] = x.unsafe_offset(nx - 1)[]
        for i in range(nx * ycols):
            c_out.unsafe_offset(i)[] = y.unsafe_offset(i)[]
        return 0

    if k == 1 and nt_in == 0:
        # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
        t_out.unsafe_offset(0)[] = x.unsafe_offset(0)[]
        for i in range(nx):
            t_out.unsafe_offset(1 + i)[] = x.unsafe_offset(i)[]
        t_out.unsafe_offset(nx + 1)[] = x.unsafe_offset(nx - 1)[]
        for i in range(nx * ycols):
            c_out.unsafe_offset(i)[] = y.unsafe_offset(i)[]
        return 0

    # come up with a sensible knot vector, if needed
    var nt = nt_in
    if nt == 0:
        if nleft == 0 and nright == 0:
            if bc_type == 3:
                _periodic_knots(x, nx, k, t_out, aux)
                nt = nx + 2 * k
            else:
                _not_a_knot(x, nx, k, t_out, aux)
                nt = nx + k + 1
        else:
            _augknt(x, nx, k, t_out)
            nt = nx + 2 * k
    else:
        for i in range(nt):
            t_out.unsafe_offset(i)[] = t_in.unsafe_offset(i)[]

    var n = nx
    var ntco = nt - k - 1

    if bc_type == 3:
        # `_make_periodic_spline`: the first k-1 rows impose periodicity, the
        # last n rows interpolate. Upstream solves this with Woodbury over the
        # cyclic band structure for n > k; the port solves the same full
        # system for every n, which agrees to round-off at a higher cost.
        var m = n + k - 1
        # One column per call: the caller (`make_interp_spline`) loops over the
        # columns of `y`.  The matrix is rebuilt and re-solved per column
        # either way, so nothing is given up, and a nested loop over the
        # columns here reads `y` wrongly on the pinned toolchain.
        var info = _make_interp_per_full_matr(
            x, nx, y, 0, t_out, nt, k, ab, aux, ipiv, work
        )
        if info != 0:
            return info
        for i in range(m):
            c_out.unsafe_offset(i)[] = ab.unsafe_offset(i)[]
        return 0

    # set up the LHS: the colocation matrix + derivatives at boundaries
    var kl = k
    var ku = k
    for i in range((2 * kl + ku + 1) * ntco):
        ab.unsafe_offset(i)[] = 0.0
    colloc(x, nx, t_out, nt, k, ab, ntco, nleft, work)
    if nleft > 0:
        handle_lhs_derivatives(
            t_out, nt, k, x.unsafe_offset(0)[], ab, ntco, kl, ku, dl, nleft, 0, work
        )
    if nright > 0:
        handle_lhs_derivatives(
            t_out, nt, k, x.unsafe_offset(nx - 1)[], ab, ntco, kl, ku, dr, nright,
            ntco - nright, work,
        )

    # set up the RHS: values to interpolate (+ derivative values, if any)
    for i in range(ntco * ycols):
        rhs.unsafe_offset(i)[] = 0.0
    for i in range(nleft):
        for c in range(ycols):
            rhs.unsafe_offset(i * ycols + c)[] = dlv.unsafe_offset(i * ycols + c)[]
    for i in range(ntco - nleft - nright):
        for c in range(ycols):
            rhs.unsafe_offset((nleft + i) * ycols + c)[] = y.unsafe_offset(i * ycols + c)[]
    for i in range(nright):
        for c in range(ycols):
            rhs.unsafe_offset((ntco - nright + i) * ycols + c)[] = drv.unsafe_offset(i * ycols + c)[]

    # solve Ab @ x = rhs
    var info = solve_banded(kl, ku, ab, ntco, rhs, ntco, ycols, ipiv)
    for i in range(ntco * ycols):
        c_out.unsafe_offset(i)[] = rhs.unsafe_offset(i)[]
    return info


def make_lsq_spline(
    x: Ptr, nx: Int, y: Ptr, ycols: Int, t: Ptr, nt: Int, k: Int, w: Ptr,
    c_out: Ptr, ab: Ptr, rhs: Ptr, work: Ptr, ipiv: IPtr,
) -> Int:
    """`make_lsq_spline` with ``method='norm-eq'``.

    Builds ``A.T @ A`` and ``A.T @ y`` for the collocation matrix `A` and solves
    the normal equations with a banded Cholesky. `c_out` receives the result.
    """
    var n = nt - k - 1
    var kb = k if k + 1 <= n else n - 1
    for i in range((kb + 1) * n):
        ab.unsafe_offset(i)[] = 0.0
    for i in range(n * ycols):
        rhs.unsafe_offset(i)[] = 0.0
    norm_eq_lsq(x, nx, t, nt, k, y, ycols, w, ab, n, rhs, work)

    var info = pbtrf(kb, ab, n, n)
    if info != 0:
        return info
    pbtrs(kb, ab, n, rhs, n, ycols)
    for i in range(n * ycols):
        c_out.unsafe_offset(i)[] = rhs.unsafe_offset(i)[]
    return 0


def evaluate_poly1(
    s: Float64, c: Ptr, ci: Int, m: Int, n: Int, cj: Int, dx: Int, k: Int
) -> Float64:
    """`_ppoly.evaluate_poly1`: one local polynomial, derivative or antiderivative.

    `c` is the flat `k` x `m` x `n` coefficient array, leading coefficient
    first: the polynomial for interval `ci` and polynomial `cj` is
    `c[0, ci, cj] * s^(k-1) + ... + c[k-1, ci, cj]`, in the caller's own frame
    (the caller passes `x - x[ci]`).  `dx` is the order of the derivative to
    evaluate, or the negated order of the antiderivative; antiderivatives
    assume zero integration constants.  `k` is `c.shape[0]`, passed in
    explicitly rather than read back out of the pointer.
    """
    var res = 0.0
    var z = 1.0

    if dx < 0:
        for _k in range(-dx):
            z *= s

    for kp in range(k):
        # prefactor of term after differentiation
        var prefactor: Float64
        if dx == 0:
            prefactor = 1.0
        elif dx > 0:
            # derivative
            if kp < dx:
                continue
            else:
                prefactor = 1.0
                for kk in range(kp, kp - dx, -1):
                    prefactor *= Float64(kk)
        else:
            # antiderivative
            prefactor = 1.0
            for kk in range(kp, kp - dx):
                prefactor /= Float64(kk + 1)

        res = (
            res
            + c.unsafe_offset(((k - kp - 1) * m + ci) * n + cj)[] * z * prefactor
        )

        # compute x**max(k-dx,0)
        if kp < k - 1 and kp >= dx:
            z *= s

    return res

# ===================================================================
# _poly_common.pxi
# ===================================================================


def find_interval_ascending(
    x: Ptr, nx: Int, xval: Float64, prev_interval: Int, extrapolate: Bool
) -> Int:
    """`x.unsafe_offset(i)[] <= xval < x.unsafe_offset(i+1)[]` on a sorted ascending array, by binary search."""
    var interval = prev_interval
    var a = x.unsafe_offset(0)[]
    var b = x.unsafe_offset(nx - 1)[]
    if interval < 0 or interval >= nx:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        var low = 0
        var high = nx - 2
        if xval >= x.unsafe_offset(interval)[]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x.unsafe_offset(low + 1)[]:
            high = low

        while low < high:
            var mid = (high + low) // 2
            if xval < x.unsafe_offset(mid)[]:
                # mid < high
                high = mid
            elif xval >= x.unsafe_offset(mid + 1)[]:
                low = mid + 1
            else:
                # x.unsafe_offset(mid)[] <= xval < x.unsafe_offset(mid+1)[]
                low = mid
                break

        interval = low

    return interval


def evaluate(
    c: Ptr,
    k: Int,
    m: Int,
    n: Int,
    x: Ptr,
    xp: Ptr,
    nxp: Int,
    dx: Int,
    extrapolate: Bool,
    dst: Ptr,
):
    """`_ppoly.evaluate`: evaluate a piecewise polynomial.

    `c` is `k` x `m` x `n` row-major, `x` has `m+1` breakpoints and `out` is
    `nxp` x `n`.
    """
    var interval = 0
    for ip in range(nxp):
        var xval = xp.unsafe_offset(ip)[]

        # Find correct interval
        var i = find_interval_ascending(x, m + 1, xval, interval, extrapolate)
        if i < 0:
            # xval was nan etc
            for jp in range(n):
                dst.unsafe_offset(ip * n + jp)[] = nan_value()
            continue
        interval = i

        # Evaluate the local polynomial(s)
        for jp in range(n):
            dst.unsafe_offset(ip * n + jp)[] = evaluate_poly1(
                xval - x.unsafe_offset(interval)[], c, i, m, n, jp, dx, k
            )


def fix_continuity(c: Ptr, k: Int, m: Int, n: Int, x: Ptr, nx: Int, order: Int):
    """`_ppoly.fix_continuity`: make a piecewise polynomial differentiable.

    `c` is `k` x `m` x `n` row-major, leading coefficient first, and `x` has
    `nx` breakpoints.  The last `order + 1` coefficients of each interval are
    overwritten so that the derivatives of order `order` down to `0` agree
    across every interior breakpoint.
    """
    if order < 0:
        return
    for ip in range(1, nx - 1):
        var xval = x.unsafe_offset(ip)[]
        var interval = ip - 1

        for jp in range(n):
            # start at the highest derivative: the lower ones depend on the
            # higher ones, but not the other way round
            for dx in range(order, -1, -1):
                # dx-th derivative of the previous interval's polynomial
                var res = evaluate_poly1(
                    xval - x.unsafe_offset(interval)[], c, interval, m, n, jp,
                    dx, k,
                )

                # set the dx-th coefficient of this interval so that the dx-th
                # derivative is continuous
                for kp in range(dx):
                    res /= Float64(kp + 1)

                c.unsafe_offset(((k - dx - 1) * m + ip) * n + jp)[] = res


def ppoly_integrate(
    c: Ptr,
    k: Int,
    m: Int,
    n: Int,
    x: Ptr,
    nx: Int,
    a: Float64,
    b: Float64,
    extrapolate: Bool,
    dst: Ptr,  # upstream names this `out`, which is a Mojo keyword
):
    """`_ppoly.integrate`: the integral over `[a, b]`, zero outside `x`.

    The polynomial is assumed to be zero outside `[x[0], x[-1]]`, and `a` must
    not exceed `b`.  `dx = -1` selects the antiderivative in `evaluate_poly1`.
    """
    var start_interval = find_interval_ascending(x, nx, a, 0, extrapolate)
    var end_interval = find_interval_ascending(x, nx, b, 0, extrapolate)
    if start_interval < 0 or end_interval < 0:
        for jp in range(n):
            dst.unsafe_offset(jp)[] = nan_value()
        return

    for jp in range(n):
        var vtot = 0.0
        for interval in range(start_interval, end_interval + 1):
            # local antiderivative, end point
            var vb: Float64
            if interval == end_interval:
                vb = evaluate_poly1(
                    b - x.unsafe_offset(interval)[],
                    c, interval, m, n, jp, -1, k,
                )
            else:
                vb = evaluate_poly1(
                    x.unsafe_offset(interval + 1)[]
                    - x.unsafe_offset(interval)[],
                    c, interval, m, n, jp, -1, k,
                )

            # local antiderivative, start point
            var va: Float64
            if interval == start_interval:
                va = evaluate_poly1(
                    a - x.unsafe_offset(interval)[],
                    c, interval, m, n, jp, -1, k,
                )
            else:
                va = evaluate_poly1(0.0, c, interval, m, n, jp, -1, k)

            # integral
            vtot += vb - va

        dst.unsafe_offset(jp)[] = vtot


# ===================================================================
# _rgi_cython.pyx
# ===================================================================


def upper_bound(x: Ptr, n: Int, v: Float64) -> Int:
    """First index in `[0, n]` with `x[i] > v`, or `n`.

    The branchless binary search: `x` ascending, the answer is held in the
    window `[base, base + len]` and the window halves every step, so each `v`
    costs the same `ceil(log2(n))` loads and no mispredicted branch.  An
    unsorted query stream is what makes the branchy form slow -- a sorted one
    is answered by the locality hint in the caller instead.
    """
    var base = 0
    var len = n
    while len > 1:
        var half = len // 2
        if x.unsafe_offset(base + half)[] <= v:
            base += half
            len -= half
        else:
            len = half
    if base < n and x.unsafe_offset(base)[] <= v:
        base += 1
    return base


def _find_indices_axis(
    lo: Int,
    hi: Int,
    grid_i: Ptr,
    grid_i_size: Int,
    col: Ptr,
    indices: IPtr,
    norm_distances: Ptr,
):
    """`find_indices` for points `lo .. hi` of one axis' column of `xi`."""
    var nx = grid_i_size
    var a = grid_i.unsafe_offset(0)[]
    var b = grid_i.unsafe_offset(nx - 1)[]
    var hint = 0
    var j = lo
    while j < hi:
        var value = col.unsafe_offset(j)[]
        var index = 0
        if not (a <= value <= b):
            # out-of-bounds or nan
            if value < a:
                index = 0
            elif value > b:
                index = nx - 2
            else:
                index = -1
        elif value == b:
            # make the interval closed from the right
            index = nx - 2
        elif grid_i.unsafe_offset(hint)[] <= value and (
            value < grid_i.unsafe_offset(hint + 1)[]
        ):
            # the previous point's interval, for a monotone query stream
            index = hint
        else:
            index = max(upper_bound(grid_i, nx, value) - 1, 0)

        indices.unsafe_offset(j)[] = Int64(index)
        if index >= 0:
            hint = index
            var g0 = grid_i.unsafe_offset(index)[]
            var denom = grid_i.unsafe_offset(index + 1)[] - g0
            norm_distances.unsafe_offset(j)[] = (value - g0) / denom
        else:
            # out of bounds, or xi is nan
            norm_distances.unsafe_offset(j)[] = nan_value()
        j += 1


def find_indices(
    grid_addrs: IPtr,
    grid_lens: IPtr,
    ndim: Int,
    xi: Ptr,
    npoints: Int,
    indices: IPtr,
    norm_distances: Ptr,
):
    """`find_indices`: locate each point between the grid edges.

    `xi` is `ndim` x `npoints`; `grid_addrs` points at each axis' sorted grid
    array. `indices` / `norm_distances` are `ndim` x `npoints`.  Points are
    independent, so above `PARALLEL_MIN` each axis' column is spread over the
    workers; a chunk restarts the locality hint with a full search.
    """
    # iterate through dimensions
    for i in range(ndim):
        var grid_i = Ptr(unsafe_from_address=Int(grid_addrs.unsafe_offset(i)[]))
        var grid_i_size = grid_lens.unsafe_offset(i)[]
        var col = xi.unsafe_offset(i * npoints)
        var idx = indices.unsafe_offset(i * npoints)
        var nrm = norm_distances.unsafe_offset(i * npoints)

        if grid_i_size == 1:
            # special case length-one dimensions
            for j in range(npoints):
                # Setting it to -1 is a hack: evaluate_linear looks at indices
                # [i, i+1] which both end up =0 with wraparound.
                idx.unsafe_offset(j)[] = Int64(-1)
                # nrm.unsafe_offset(j)[] is already zero
        elif npoints < PARALLEL_MIN:
            _find_indices_axis(
                0, npoints, grid_i, Int(grid_i_size), col, idx, nrm
            )
        else:
            def work(w: Int) {imm}:
                var lo = w * 64
                _find_indices_axis(
                    lo,
                    min(lo + 64, npoints),
                    grid_i,
                    Int(grid_i_size),
                    col,
                    idx,
                    nrm,
                )

            parallelize(work, (npoints + 63) // 64, NWORKERS)


def evaluate_linear_2d(
    values: Ptr, n0: Int, n1: Int, indices: IPtr, norm_distances: Ptr, npoints: Int,
    dst: Ptr,
):
    """`evaluate_linear_2d`: the 2-D linear path of `RegularGridInterpolator`.

    `values` is `n0` x `n1` row-major.  Points are independent, so above
    `PARALLEL_MIN` they are spread over the workers.
    """
    if npoints >= PARALLEL_MIN and n0 > 1 and n1 > 1:
        def work(w: Int) {imm}:
            var lo = w * 64
            var hi = min(lo + 64, npoints)
            var point = lo
            while point < hi:
                var i0 = Int(indices.unsafe_offset(point)[])
                var i1 = Int(indices.unsafe_offset(npoints + point)[])
                if i0 >= 0 and i1 >= 0:
                    var y0 = norm_distances.unsafe_offset(point)[]
                    var y1 = norm_distances.unsafe_offset(npoints + point)[]
                    var r0 = values.unsafe_offset(i0 * n1 + i1)[]
                    var r1 = values.unsafe_offset(i0 * n1 + i1 + 1)[]
                    var r2 = values.unsafe_offset((i0 + 1) * n1 + i1)[]
                    var r3 = values.unsafe_offset((i0 + 1) * n1 + i1 + 1)[]
                    dst.unsafe_offset(point)[] = (
                        r0 * (1 - y0) * (1 - y1)
                        + r1 * (1 - y0) * y1
                        + r2 * y0 * (1 - y1)
                        + r3 * y0 * y1
                    )
                else:
                    # xi was nan
                    dst.unsafe_offset(point)[] = nan_value()
                point += 1

        parallelize(work, (npoints + 63) // 64, NWORKERS)
        return

    if n1 == 1:
        # linear interpolation along axis=0
        for point in range(npoints):
            var i0 = Int(indices.unsafe_offset(point)[])
            if i0 >= 0:
                var y0 = norm_distances.unsafe_offset(point)[]
                dst.unsafe_offset(point)[] = values.unsafe_offset(i0 * n1)[] * (1 - y0) + values.unsafe_offset((i0 + 1) * n1)[] * y0
            else:
                # xi was nan: find_interval returns -1
                dst.unsafe_offset(point)[] = nan_value()
    elif n0 == 1:
        # linear interpolation along axis=1
        for point in range(npoints):
            var i1 = Int(indices.unsafe_offset(npoints + point)[])
            if i1 >= 0:
                var y1 = norm_distances.unsafe_offset(npoints + point)[]
                dst.unsafe_offset(point)[] = values.unsafe_offset(i1)[] * (1 - y1) + values.unsafe_offset(i1 + 1)[] * y1
            else:
                dst.unsafe_offset(point)[] = nan_value()
    else:
        for point in range(npoints):
            var i0 = Int(indices.unsafe_offset(point)[])
            var i1 = Int(indices.unsafe_offset(npoints + point)[])
            if i0 >= 0 and i1 >= 0:
                var y0 = norm_distances.unsafe_offset(point)[]
                var y1 = norm_distances.unsafe_offset(npoints + point)[]

                var result = 0.0
                result = result + values.unsafe_offset(i0 * n1 + i1)[] * (1 - y0) * (1 - y1)
                result = result + values.unsafe_offset(i0 * n1 + i1 + 1)[] * (1 - y0) * y1
                result = result + values.unsafe_offset((i0 + 1) * n1 + i1)[] * y0 * (1 - y1)
                result = result + values.unsafe_offset((i0 + 1) * n1 + i1 + 1)[] * y0 * y1
                dst.unsafe_offset(point)[] = result
            else:
                # xi was nan
                dst.unsafe_offset(point)[] = nan_value()


def evaluate_linear_nd(
    values: Ptr,
    dims: IPtr,
    ndim: Int,
    ntrail: Int,
    indices: IPtr,
    norm_distances: Ptr,
    npoints: Int,
    dst: Ptr,
    strides: IPtr,
):
    """`RegularGridInterpolator._evaluate_linear` for arbitrary dimension.

    `dims` is the grid shape, `values` is ``prod(dims) x ntrail`` row-major and
    `out` is ``npoints x ntrail``. `strides` is scratch of `ndim` entries.
    """
    var stride = Int64(1)
    for d in range(ndim - 1, -1, -1):
        strides.unsafe_offset(d)[] = stride
        stride *= dims.unsafe_offset(d)[]

    for point in range(npoints):
        for c in range(ntrail):
            # iterate over the vertices of the hypercube, as upstream's
            # `itertools.product` over the (index, 1 - y) / (index + 1, y) pairs
            var value = 0.0
            for h in range(1 << ndim):
                var idx = 0
                var weight = 1.0
                for d in range(ndim):
                    var i = Int(indices.unsafe_offset(d * npoints + point)[])
                    var yi = norm_distances.unsafe_offset(d * npoints + point)[]
                    if ((h >> d) & 1) == 1:
                        i += 1
                        weight *= yi
                    else:
                        weight *= 1 - yi
                    if i < 0:
                        # a length-one axis: upstream relies on numpy's negative
                        # indexing here, and the weight of the +1 corner is 0
                        i += Int(dims.unsafe_offset(d)[])
                    idx += i * Int(strides.unsafe_offset(d)[])
                value += values.unsafe_offset(idx * ntrail + c)[] * weight
            dst.unsafe_offset(point * ntrail + c)[] = value



def tensor_spline_wstride(degrees: IPtr, ndim: Int) -> Int:
    """Per-axis de Boor scratch stride shared by all axes of a tensor product."""
    var w = 0
    for d in range(ndim):
        w = max(w, 2 * Int(degrees.unsafe_offset(d)[]) + 2)
    return w


def _tensor_spline_rows(
    lo: Int,
    hi: Int,
    coeffs: Ptr,
    t_addrs: IPtr,
    degrees: IPtr,
    ncoefs: IPtr,
    cstrides: IPtr,
    ndim: Int,
    ntrail: Int,
    nu: Int,
    x: Ptr,
    dst: Ptr,
    work: Ptr,
    iscratch: IPtr,
    wstride: Int,
):
    """Tensor-product B-spline evaluation for points `lo .. hi`.

    A tensor-product B-spline of degree `k_d` along axis `d` has only
    `k_d + 1` non-zero basis functions per axis at any point, so the
    contraction is over `prod(k_d + 1)` corners -- 16 for a bicubic, against
    the `nk[0] * ... * nk[ndim-1]` terms a dense basis matrix would touch.
    """
    var total = 1
    for d in range(ndim):
        total *= Int(degrees.unsafe_offset(d)[]) + 1

    var p = lo
    while p < hi:
        var out = dst.unsafe_offset(p * ntrail)
        var bad = False
        for d in range(ndim):
            var k = Int(degrees.unsafe_offset(d)[])
            var nk = Int(ncoefs.unsafe_offset(d)[])
            var t = Ptr(unsafe_from_address=Int(t_addrs.unsafe_offset(d)[]))
            var xval = x.unsafe_offset(p * ndim + d)[]
            var l = find_interval(t, nk + k + 1, k, xval, k, True)
            if l < 0:
                bad = True
                break
            var w = work.unsafe_offset(d * wstride)
            deBoor_D(t, xval, k, l, nu, w)
            iscratch.unsafe_offset(d)[] = Int64(l - k)
            iscratch.unsafe_offset(ndim + d)[] = Int64(0)

        if bad:
            for tr in range(ntrail):
                out.unsafe_offset(tr)[] = nan_value()
            p += 1
            continue

        for tr in range(ntrail):
            out.unsafe_offset(tr)[] = 0.0

        var c = 0
        while c < total:
            var base = 0
            var w = 1.0
            for d in range(ndim):
                var a = Int(iscratch.unsafe_offset(ndim + d)[])
                base += (Int(iscratch.unsafe_offset(d)[]) + a) * Int(
                    cstrides.unsafe_offset(d)[]
                )
                w *= work.unsafe_offset(d * wstride + a)[]
            for tr in range(ntrail):
                out.unsafe_offset(tr)[] += coeffs.unsafe_offset(base + tr)[] * w
            # odometer over the per-axis corner index, so no division
            var d = ndim - 1
            while d >= 0:
                var a = Int(iscratch.unsafe_offset(ndim + d)[]) + 1
                if a <= Int(degrees.unsafe_offset(d)[]):
                    iscratch.unsafe_offset(ndim + d)[] = Int64(a)
                    break
                iscratch.unsafe_offset(ndim + d)[] = Int64(0)
                d -= 1
            c += 1
        p += 1


def evaluate_tensor_spline(
    coeffs: Ptr,
    t_addrs: IPtr,
    degrees: IPtr,
    ncoefs: IPtr,
    cstrides: IPtr,
    ndim: Int,
    ntrail: Int,
    nu: Int,
    x: Ptr,
    m: Int,
    dst: Ptr,
    work: Ptr,
    iscratch: IPtr,
):
    """`RegularGridInterpolator`'s spline methods, over the whole grid.

    `coeffs` is the ``nk[0] x ... x nk[ndim-1] x ntrail`` coefficient tensor,
    row-major, and `cstrides[d]` is axis `d`'s element stride in it -- the
    trailing value axes are contiguous after the grid axes, so they collapse
    to `ntrail`.  `t_addrs[d]`, `degrees[d]` and `ncoefs[d]` describe axis
    `d`'s 1-D spline (`t` has `ncoefs[d] + degrees[d] + 1` knots).  `x` is
    `m` x `ndim` and `dst` is `m` x `ntrail`.  `work` is
    `nchunks * ndim * wstride` doubles and `iscratch` `nchunks * 2 * ndim`
    Int64, one slice per chunk.
    """
    var wstride = tensor_spline_wstride(degrees, ndim)
    if m < PARALLEL_MIN:
        _tensor_spline_rows(
            0, m, coeffs, t_addrs, degrees, ncoefs, cstrides, ndim, ntrail, nu,
            x, dst, work, iscratch, wstride,
        )
    else:
        def work_chunk(w: Int) {imm}:
            var lo = w * 128
            _tensor_spline_rows(
                lo,
                min(lo + 128, m),
                coeffs,
                t_addrs,
                degrees,
                ncoefs,
                cstrides,
                ndim,
                ntrail,
                nu,
                x,
                dst,
                work.unsafe_offset(w * ndim * wstride),
                iscratch.unsafe_offset(w * 2 * ndim),
                wstride,
            )

        parallelize(work_chunk, (m + 127) // 128, NWORKERS)


# ===================================================================
# _interpolate.py -- interp1d
# ===================================================================


def searchsorted_left(x: Ptr, lo: Int, hi: Int, v: Float64) -> Int:
    """`numpy.searchsorted(..., side='left')` over `[lo, hi)`: the first index with `x[i] >= v`, or `hi`."""
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if v <= x.unsafe_offset(mid)[]:
            b = mid
        else:
            a = mid + 1
    return a


def searchsorted_right(x: Ptr, lo: Int, hi: Int, v: Float64) -> Int:
    """`numpy.searchsorted(..., side='right')` over `[lo, hi)`: the first index with `x[i] > v`, or `hi`."""
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if x.unsafe_offset(mid)[] <= v:
            a = mid + 1
        else:
            b = mid
    return a


def searchsorted_left_hinted(x: Ptr, n: Int, v: Float64, hint: Int) -> Int:
    """`searchsorted_left(x, 0, n, v)` narrowed by the previous answer `hint`.

    `x` ascending and `hint` the previous query's answer: `x[hint] >= v` puts
    the answer at or below `hint`, and `x[hint] < v` puts it strictly above.
    Either way the range searched shrinks, and a query that lands in the
    interval the previous one did -- a sorted or near-sorted `x_new` -- is
    answered by the two comparisons alone. A wrong guess costs one extra
    comparison, never more steps than the plain binary search.

    `hint` is the previous answer, so it can be `n`; it is clamped to the
    last index, which keeps the narrowing valid (a `hint` of `n` only ever
    produces the `hint + 1` branch, and that still returns `n`).
    """
    var h = min(hint, n - 1)
    if x.unsafe_offset(h)[] >= v:
        if h == 0 or x.unsafe_offset(h - 1)[] < v:
            return h
        return searchsorted_left(x, 0, h, v)
    return searchsorted_left(x, h + 1, n, v)


def searchsorted_right_hinted(x: Ptr, n: Int, v: Float64, hint: Int) -> Int:
    """`searchsorted_right(x, 0, n, v)` narrowed by the previous answer `hint`."""
    var h = min(hint, n - 1)
    if x.unsafe_offset(h)[] > v:
        if h == 0 or x.unsafe_offset(h - 1)[] <= v:
            return h
        return searchsorted_right(x, 0, h, v)
    return searchsorted_right(x, h + 1, n, v)


def _linear_interp_rows(
    lo: Int, hi: Int, x: Ptr, y: Ptr, n: Int, x_new: Ptr, y_new: Ptr, ycols: Int
):
    var hint = 0
    var i = lo
    while i < hi:
        var v = x_new.unsafe_offset(i)[]
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.  The search is narrowed by the previous
        #    answer, so a sorted `x_new` costs a comparison per query.
        var idx = searchsorted_left_hinted(x, n, v, hint)
        # 3. Clip so that they are within the range of self.x indices and at
        #    least 1.  Removes mis-interpolation of x_new[i] == x[0].
        if idx < 1:
            idx = 1
        if idx > n - 1:
            idx = n - 1
        hint = idx
        # 4. de Boor's algorithm for linear interpolation.
        var a = idx - 1
        var x_lo = x.unsafe_offset(a)[]
        var t = (v - x_lo) / (x.unsafe_offset(idx)[] - x_lo)
        var y_lo = y.unsafe_offset(a * ycols)
        var y_hi = y.unsafe_offset(idx * ycols)
        var dst = y_new.unsafe_offset(i * ycols)
        for c in range(ycols):
            dst.unsafe_offset(c)[] = t * y_hi.unsafe_offset(c)[] + (
                1.0 - t
            ) * y_lo.unsafe_offset(c)[]
        i += 1


def linear_interp(
    x: Ptr, y: Ptr, n: Int, x_new: Ptr, nnew: Int, y_new: Ptr, ycols: Int
):
    """`interp1d._call_linear`: de Boor linear interpolation.

    `x` must be sorted ascending. `y` is `n` x `ycols` row-major and
    `y_new` is `nnew` x `ycols`.  Rows are independent, so above
    `PARALLEL_MIN` they are spread over the workers; each chunk restarts the
    search hint with a full binary search.
    """
    if nnew < PARALLEL_MIN:
        _linear_interp_rows(0, nnew, x, y, n, x_new, y_new, ycols)
    else:
        def work(w: Int) {imm}:
            var lo = w * 256
            _linear_interp_rows(
                lo, min(lo + 256, nnew), x, y, n, x_new, y_new, ycols
            )

        parallelize(work, (nnew + 255) // 256, NWORKERS)


def _nearest_interp_rows(
    lo: Int, hi: Int, x_bds: Ptr, n: Int, y: Ptr, x_new: Ptr, side: Int,
    y_new: Ptr, ycols: Int,
):
    var hint = 0
    var i = lo
    while i < hi:
        var v = x_new.unsafe_offset(i)[]
        var idx = 0
        if side == 0:
            idx = searchsorted_left_hinted(x_bds, n - 1, v, hint)
        else:
            idx = searchsorted_right_hinted(x_bds, n - 1, v, hint)
        hint = idx
        if idx < 0:
            idx = 0
        if idx > n - 1:
            idx = n - 1
        var row = y.unsafe_offset(idx * ycols)
        var dst = y_new.unsafe_offset(i * ycols)
        for c in range(ycols):
            dst.unsafe_offset(c)[] = row.unsafe_offset(c)[]
        i += 1


def nearest_interp(
    x_bds: Ptr, n: Int, y: Ptr, nnew: Int, x_new: Ptr, side: Int, y_new: Ptr,
    ycols: Int,
):
    """`interp1d._call_nearest`: nearest neighbour about the half-way points.

    `side` is 0 for `searchsorted(..., 'left')` (kind='nearest') and 1 for
    'right' (kind='nearest-up'). `x_bds` holds the `n-1` half-way points.
    """
    if nnew < PARALLEL_MIN:
        _nearest_interp_rows(0, nnew, x_bds, n, y, x_new, side, y_new, ycols)
    else:
        def work(w: Int) {imm}:
            var lo = w * 256
            _nearest_interp_rows(
                lo, min(lo + 256, nnew), x_bds, n, y, x_new, side, y_new, ycols
            )

        parallelize(work, (nnew + 255) // 256, NWORKERS)


def _previous_next_rows(
    lo: Int, hi: Int, x_shift: Ptr, n_shift: Int, y: Ptr, n: Int, x_new: Ptr,
    ind: Int, side: Int, y_new: Ptr, ycols: Int,
):
    var hint = 0
    var i = lo
    while i < hi:
        var v = x_new.unsafe_offset(i)[]
        var idx = 0
        if side == 0:
            idx = searchsorted_left_hinted(x_shift, n_shift, v, hint)
        else:
            idx = searchsorted_right_hinted(x_shift, n_shift, v, hint)
        hint = idx
        var clo = 1 - ind
        var chi = n - ind
        if idx < clo:
            idx = clo
        if idx > chi:
            idx = chi
        var j = idx + ind - 1
        var row = y.unsafe_offset(j * ycols)
        var dst = y_new.unsafe_offset(i * ycols)
        for c in range(ycols):
            dst.unsafe_offset(c)[] = row.unsafe_offset(c)[]
        i += 1


def previous_next(
    x_shift: Ptr, n_shift: Int, y: Ptr, n: Int, nnew: Int, x_new: Ptr, ind: Int,
    side: Int, y_new: Ptr, ycols: Int,
):
    """`interp1d._call_previousnext`: the previous/next neighbour of `x_new`.

    `ind` is 1 for 'previous' and 0 for 'next'; `side` is 0 for 'left' and 1
    for 'right'.
    """
    if nnew < PARALLEL_MIN:
        _previous_next_rows(
            0, nnew, x_shift, n_shift, y, n, x_new, ind, side, y_new, ycols
        )
    else:
        def work(w: Int) {imm}:
            var lo = w * 256
            _previous_next_rows(
                lo, min(lo + 256, nnew), x_shift, n_shift, y, n, x_new, ind,
                side, y_new, ycols,
            )

        parallelize(work, (nnew + 255) // 256, NWORKERS)


# ===================================================================
# _rbf.py -- the legacy `Rbf`
# ===================================================================


def _h_multiquadric(eps: Float64, r: Float64) -> Float64:
    return sqrt((1.0 / eps * r) ** 2 + 1)


def _h_inverse_multiquadric(eps: Float64, r: Float64) -> Float64:
    return 1.0 / sqrt((1.0 / eps * r) ** 2 + 1)


def _h_gaussian(eps: Float64, r: Float64) -> Float64:
    return exp(-((1.0 / eps * r) ** 2))


def _h_linear(eps: Float64, r: Float64) -> Float64:
    return r


def _h_cubic(eps: Float64, r: Float64) -> Float64:
    return r * r * r


def _h_quintic(eps: Float64, r: Float64) -> Float64:
    return r * r * r * r * r


def _h_thin_plate(eps: Float64, r: Float64) -> Float64:
    # xlogy(r**2, r)
    if r == 0.0:
        return 0.0
    return r * r * log(r)


def rbf_kernel(kernel: Int, eps: Float64, r: Float64) -> Float64:
    """The `Rbf._h_*` family, selected by index.

    0 multiquadric, 1 inverse_multiquadric, 2 gaussian, 3 linear, 4 cubic,
    5 quintic, 6 thin_plate -- the order of `RBF_FUNCTIONS` in the glue.
    """
    if kernel == 0:
        return _h_multiquadric(eps, r)
    elif kernel == 1:
        return _h_inverse_multiquadric(eps, r)
    elif kernel == 2:
        return _h_gaussian(eps, r)
    elif kernel == 3:
        return _h_linear(eps, r)
    elif kernel == 4:
        return _h_cubic(eps, r)
    elif kernel == 5:
        return _h_quintic(eps, r)
    return _h_thin_plate(eps, r)


def rbf_function_matrix(kernel: Int, eps: Float64, r: Ptr, n: Int, dst: Ptr):
    """`Rbf._init_function`: apply the kernel to a whole array of radii."""
    for i in range(n):
        dst.unsafe_offset(i)[] = rbf_kernel(kernel, eps, r.unsafe_offset(i)[])


def rbf_function_dot(
    kernel: Int,
    eps: Float64,
    r: Ptr,
    npoints: Int,
    ncentres: Int,
    nodes: Ptr,
    ncols: Int,
    dst: Ptr,
):
    """`Rbf.__call__`: ``np.dot(self._function(r), self.nodes)``.

    `r` is the `npoints` x `ncentres` matrix of radii.
    """
    for i in range(npoints):
        var row = r.unsafe_offset(i * ncentres)
        for c in range(ncols):
            var acc = 0.0
            for j in range(ncentres):
                acc += rbf_kernel(kernel, eps, row.unsafe_offset(j)[]) * nodes.unsafe_offset(
                    j * ncols + c
                )[]
            dst.unsafe_offset(i * ncols + c)[] = acc


def euclidean2_row(a: Ptr, d: Int, y: Ptr, m: Int, dst: Ptr):
    """One row of `cdist(x, y, 'euclidean')`, squared."""
    for j in range(m):
        dst.unsafe_offset(j)[] = _euclidean2(a, d, y.unsafe_offset(j * d))


def _euclidean2(a: Ptr, n: Int, b: Ptr) -> Float64:
    var acc = 0.0
    for i in range(n):
        var e = a.unsafe_offset(i)[] - b.unsafe_offset(i)[]
        acc += e * e
    return acc


def cdist_euclidean(x: Ptr, n: Int, d: Int, y: Ptr, m: Int, dst: Ptr):
    """`scipy.spatial.distance.cdist(x, y, 'euclidean')` for small `d`.

    `dst` holds the *squared* distances; the Python glue takes the square root
    where upstream needs the norm itself.
    """
    if n < PARALLEL_MIN:
        for i in range(n):
            euclidean2_row(x.unsafe_offset(i * d), d, y, m, dst.unsafe_offset(i * m))
    else:
        def work(w: Int) {imm}:
            var lo = w * 64
            var hi = min(lo + 64, n)
            var i = lo
            while i < hi:
                euclidean2_row(x.unsafe_offset(i * d), d, y, m, dst.unsafe_offset(i * m))
                i += 1

        parallelize(work, (n + 63) // 64, NWORKERS)


# ===================================================================
# _rbfinterp.py
# ===================================================================


def linear(r: Float64) -> Float64:
    return -r


def thin_plate_spline(r: Float64) -> Float64:
    if r == 0:
        return 0.0
    return r * r * log(r)


def cubic(r: Float64) -> Float64:
    return r * r * r


def quintic(r: Float64) -> Float64:
    return -(r * r * r * r * r)


def multiquadric(r: Float64) -> Float64:
    return -sqrt(r * r + 1)


def inverse_multiquadric(r: Float64) -> Float64:
    return 1.0 / sqrt(r * r + 1)


def inverse_quadratic(r: Float64) -> Float64:
    return 1.0 / (r * r + 1)


def gaussian(r: Float64) -> Float64:
    return exp(-(r * r))


def rbf(name: Int, r: Float64) -> Float64:
    """`NAME_TO_FUNC[name]`, selected by the index the Python glue resolved.

    0 linear, 1 thin_plate_spline, 2 cubic, 3 quintic, 4 multiquadric,
    5 inverse_multiquadric, 6 inverse_quadratic, 7 gaussian -- the declaration
    order of `NAME_TO_FUNC` in ``_rbfinterp_pythran.py``.
    """
    if name == 0:
        return linear(r)
    elif name == 1:
        return thin_plate_spline(r)
    elif name == 2:
        return cubic(r)
    elif name == 3:
        return quintic(r)
    elif name == 4:
        return multiquadric(r)
    elif name == 5:
        return inverse_multiquadric(r)
    elif name == 6:
        return inverse_quadratic(r)
    return gaussian(r)


def norm(x: Ptr, n: Int, y: Ptr) -> Float64:
    """`np.linalg.norm(x - y)`."""
    var acc = 0.0
    for i in range(n):
        var e = x.unsafe_offset(i)[] - y.unsafe_offset(i)[]
        acc += e * e
    return sqrt(acc)


def kernel_vector(
    x: Ptr, d: Int, y: Ptr, p: Int, kernel: Int, dst: Ptr
):
    """`kernel_vector`: RBFs, with centers at `y`, at the point `x`."""
    for i in range(p):
        dst.unsafe_offset(i)[] = rbf(kernel, norm(x, d, y.unsafe_offset(i * d)))


def polynomial_vector(
    x: Ptr, d: Int, powers: IPtr, ndim: Int, npow: Int, dst: Ptr
):
    """`polynomial_vector`: monomials, with exponents from `powers`, at `x`."""
    for i in range(npow):
        var v = 1.0
        for dd in range(ndim):
            var xv = x.unsafe_offset(dd)[]
            var q = 1.0
            for _ in range(powers.unsafe_offset(i * ndim + dd)[]):
                q *= xv
            v *= q
        dst.unsafe_offset(i)[] = v


def _kernel_matrix_rows(lo: Int, hi: Int, x: Ptr, d: Int, kernel: Int, dst: Ptr, n: Int):
    for i in range(lo, hi):
        for j in range(i + 1):
            var v = rbf(kernel, norm(x.unsafe_offset(i * d), d, x.unsafe_offset(j * d)))
            dst.unsafe_offset(i * n + j)[] = v
            dst.unsafe_offset(j * n + i)[] = v


def kernel_matrix(x: Ptr, n: Int, d: Int, kernel: Int, dst: Ptr):
    """`kernel_matrix`: RBFs, with centers at `x`, evaluated at `x`."""
    if n < PARALLEL_MIN:
        _kernel_matrix_rows(0, n, x, d, kernel, dst, n)
    else:
        def work(w: Int) {imm}:
            _kernel_matrix_rows(w * 64, min(w * 64 + 64, n), x, d, kernel, dst, n)

        parallelize(work, (n + 63) // 64, NWORKERS)


def polynomial_matrix(
    x: Ptr, n: Int, d: Int, powers: IPtr, ndim: Int, npow: Int, dst: Ptr
):
    """`polynomial_matrix`: monomials, with exponents from `powers`, at `x`."""
    for i in range(n):
        for j in range(npow):
            var v = 1.0
            for dd in range(ndim):
                var xv = x.unsafe_offset(i * d + dd)[]
                var q = 1.0
                for _ in range(powers.unsafe_offset(j * ndim + dd)[]):
                    q *= xv
                v *= q
            dst.unsafe_offset(i * npow + j)[] = v


def rbf_build_system(
    y: Ptr,
    npts: Int,
    d: Int,
    eps: Float64,
    kernel: Int,
    powers: IPtr,
    ndim: Int,
    npow: Int,
    smoothing: Ptr,
    dv: Ptr,
    ncols: Int,
    lhs: Ptr,
    n: Int,
    rhs: Ptr,
    shift: Ptr,
    scale: Ptr,
    work: Ptr,
):
    """`_build_system`: the `(p + r)` square system the RBF solve runs on.

    `y` is `p` x `d`, `dv` is `p` x `ncols`, `lhs` is `n` x `n` and `rhs` is
    `n` x `ncols`, with `n = p + npow`.  `shift` and `scale` are `d` doubles
    and `work` is `2 * p * d` doubles of scratch for the scaled coordinates.

    `RBFInterpolator(neighbors=k)` runs this once per evaluation point, so it
    is the whole cost of the method: upstream compiles it with pythran, and
    here it is one pass instead of a dozen NumPy calls per system.
    """
    var yeps = work
    var yhat = work.unsafe_offset(npts * d)

    # shift and scale: the polynomial domain mapped onto [-1, 1]
    for a in range(d):
        var lo = y.unsafe_offset(a)[]
        var hi = lo
        var i = 1
        while i < npts:
            var v = y.unsafe_offset(i * d + a)[]
            lo = min(lo, v)
            hi = max(hi, v)
            i += 1
        shift.unsafe_offset(a)[] = (hi + lo) / 2
        var s = (hi - lo) / 2
        # the scale is zero for a single point, or an axis all points share
        scale.unsafe_offset(a)[] = s if s != 0.0 else 1.0

    for a in range(d):
        var sh = shift.unsafe_offset(a)[]
        var sc = scale.unsafe_offset(a)[]
        var i = 0
        while i < npts:
            var v = y.unsafe_offset(i * d + a)[]
            yeps.unsafe_offset(i * d + a)[] = v * eps
            yhat.unsafe_offset(i * d + a)[] = (v - sh) / sc
            i += 1

    # the kernel and polynomial blocks of the left-hand side, plus the
    # smoothing on the diagonal
    for i in range(npts):
        var row = lhs.unsafe_offset(i * n)
        var yi = yeps.unsafe_offset(i * d)
        for j in range(i + 1):
            var v = rbf(kernel, norm(yi, d, yeps.unsafe_offset(j * d)))
            row.unsafe_offset(j)[] = v
            lhs.unsafe_offset(j * n + i)[] = v
        var yhi = yhat.unsafe_offset(i * d)
        for j in range(npow):
            var v = 1.0
            for a in range(ndim):
                var q = 1.0
                for _ in range(powers.unsafe_offset(j * ndim + a)[]):
                    q *= yhi.unsafe_offset(a)[]
                v *= q
            row.unsafe_offset(npts + j)[] = v
            lhs.unsafe_offset((npts + j) * n + i)[] = v
        row.unsafe_offset(i)[] += smoothing.unsafe_offset(i)[]

    # the zero polynomial block, and the right-hand side
    for j in range(npts, n):
        var row = lhs.unsafe_offset(j * n)
        for i in range(npts, n):
            row.unsafe_offset(i)[] = 0.0
        var rrow = rhs.unsafe_offset(j * ncols)
        for c in range(ncols):
            rrow.unsafe_offset(c)[] = 0.0
    for i in range(npts):
        var rrow = rhs.unsafe_offset(i * ncols)
        var drow = dv.unsafe_offset(i * ncols)
        for c in range(ncols):
            rrow.unsafe_offset(c)[] = drow.unsafe_offset(c)[]


def build_evaluation_coefficients(
    x: Ptr,
    q: Int,
    y: Ptr,
    p: Int,
    d: Int,
    kernel: Int,
    epsilon: Float64,
    powers: IPtr,
    ndim: Int,
    npow: Int,
    shift: Ptr,
    scale: Ptr,
    scratch: Ptr,
    vec: Ptr,
):
    """`_build_evaluation_coefficients`.

    `vec` is the `q` x `(p + npow)` result; `scratch` is `p * d + 2 * d` doubles.
    """
    var cap = p * d
    var yeps = scratch.unsafe_offset(d)
    var xeps = scratch.unsafe_offset(d + cap)
    # yeps = y * epsilon
    for a in range(cap):
        yeps.unsafe_offset(a)[] = y.unsafe_offset(a)[] * epsilon
    for i in range(q):
        var row = scratch
        # xhat = (x[i] - shift) / scale, and xeps = x[i] * epsilon
        for a in range(d):
            var v = x.unsafe_offset(i * d + a)[]
            row.unsafe_offset(a)[] = (
                (v - shift.unsafe_offset(a)[]) / scale.unsafe_offset(a)[]
            )
            xeps.unsafe_offset(a)[] = v * epsilon
        kernel_vector(
            xeps, d, yeps, p, kernel, vec.unsafe_offset(i * (p + npow))
        )
        polynomial_vector(
            row, d, powers, ndim, npow,
            vec.unsafe_offset(i * (p + npow) + p),
        )



def _knn_row(
    q: Ptr, d: Int, train: Ptr, n: Int, k: Int, dist: Ptr, idx: IPtr
):
    # distances start at +inf so the first `k` centres always land
    for c in range(k):
        dist.unsafe_offset(c)[] = 1.0e308
        idx.unsafe_offset(c)[] = 0
    for c in range(n):
        var dd = 0.0
        for t in range(d):
            var e = q.unsafe_offset(t)[] - train.unsafe_offset(c * d + t)[]
            dd += e * e
        if dd >= dist.unsafe_offset(k - 1)[]:
            continue
        var j = k - 1
        while j > 0 and dist.unsafe_offset(j - 1)[] > dd:
            dist.unsafe_offset(j)[] = dist.unsafe_offset(j - 1)[]
            idx.unsafe_offset(j)[] = idx.unsafe_offset(j - 1)[]
            j -= 1
        dist.unsafe_offset(j)[] = dd
        idx.unsafe_offset(j)[] = Int64(c)


def neighbors_knn(
    train: Ptr, n: Int, d: Int, query: Ptr, m: Int, k: Int, dist: Ptr, idx: IPtr
):
    """`scipy.spatial.KDTree.query(x, k)` under the squared-Euclidean metric.

    Dense: the tree is not ported, so every query point is scored against all
    `n` centres and the `k` smallest are kept. Equal distances break towards
    the lower index, matching the k-d tree's left-first descent. `dist` and
    `idx` are `m` x `k`.
    """
    if m < PARALLEL_MIN:
        for q in range(m):
            _knn_row(
                query.unsafe_offset(q * d), d, train, n, k,
                dist.unsafe_offset(q * k), idx.unsafe_offset(q * k),
            )
    else:
        def work_chunk(w: Int) {imm}:
            var lo = w * 64
            var hi = min(lo + 64, m)
            var q = lo
            while q < hi:
                _knn_row(
                    query.unsafe_offset(q * d), d, train, n, k,
                    dist.unsafe_offset(q * k), idx.unsafe_offset(q * k),
                )
                q += 1

        parallelize(work_chunk, (m + 63) // 64, NWORKERS)
