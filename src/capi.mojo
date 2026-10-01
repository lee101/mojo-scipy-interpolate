"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. Callers own all
memory, including scratch, so nothing here allocates and nothing can leak.
"""

from ported import (
    IPtr,
    Ptr,
    PARALLEL_MIN,
    build_evaluation_coefficients,
    cdist_euclidean,
    colloc,
    evaluate,
    evaluate_linear_2d,
    evaluate_linear_nd,
    evaluate_spline,
    find_indices,
    kernel_matrix,
    fix_continuity,
    linear_interp,
    make_design_matrix,
    make_interp_spline,
    make_lsq_spline,
    nearest_interp,
    neighbors_knn,
    periodic_scratch,
    polynomial_matrix,
    previous_next,
    rbf_function_dot,
    rbf_function_matrix,
    solve_banded,
    ppoly_integrate,
    solve_dense,
    tensor_spline_wstride,
    evaluate_tensor_spline,
    rbf_build_system,
)


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)



def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("msi_rbf_build_system")
def msi_rbf_build_system(
    y: Int, npts: Int, d: Int, eps: Float64, kernel: Int, powers: Int,
    ndim: Int,
    npow: Int, smoothing: Int, dv: Int, ncols: Int, lhs: Int, n: Int, rhs: Int,
    shift: Int, scale: Int, work: Int,
) abi("C"):
    rbf_build_system(
        p(y), npts, d, eps, kernel, ip(powers), ndim, npow, p(smoothing), p(dv),
        ncols, p(lhs), n, p(rhs), p(shift), p(scale), p(work),
    )


# ------------------------------------------------------------------ _bspl
@export("msi_evaluate_spline")
def msi_evaluate_spline(
    t: Int,
    nt: Int,
    c: Int,
    ccols: Int,
    k: Int,
    x: Int,
    nxp: Int,
    nu: Int,
    extrapolate: Int,
    dst: Int,
    work: Int,
) abi("C"):
    evaluate_spline(
        p(t), nt, p(c), ccols, k, p(x), nxp, nu, extrapolate != 0, p(dst), p(work)
    )


@export("msi_make_design_matrix")
def msi_make_design_matrix(
    x: Int, nx: Int, t: Int, nt: Int, k: Int, extrapolate: Int, indices: Int,
    data: Int, work: Int,
) abi("C"):
    make_design_matrix(
        p(x), nx, p(t), nt, k, extrapolate != 0, ip(indices), p(data), p(work)
    )


# ------------------------------------------------------- banded solvers
@export("msi_solve_banded")
def msi_solve_banded(
    l: Int, u: Int, ab: Int, ldab: Int, b: Int, n: Int, nrhs: Int, ipiv: Int
) abi("C") -> Int:
    return solve_banded(l, u, p(ab), ldab, p(b), n, nrhs, ip(ipiv))


@export("msi_solve_dense")
def msi_solve_dense(
    a: Int, b: Int, n: Int, nrhs: Int, ipiv: Int
) abi("C") -> Int:
    return solve_dense(p(a), p(b), n, nrhs, ip(ipiv))


# ------------------------------------------------------------- _bsplines
@export("msi_make_interp_spline")
def msi_make_interp_spline(
    x: Int,
    nx: Int,
    y: Int,
    ycols: Int,
    k: Int,
    bc_type: Int,
    dl: Int,
    dlv: Int,
    nleft: Int,
    dr: Int,
    drv: Int,
    nright: Int,
    t_in: Int,
    nt_in: Int,
    t_out: Int,
    c_out: Int,
    ab: Int,
    rhs: Int,
    work: Int,
    aux: Int,
    ipiv: Int,
) abi("C") -> Int:
    return make_interp_spline(
        p(x), nx, p(y), ycols, k, bc_type, ip(dl), p(dlv), nleft, ip(dr), p(drv),
        nright, p(t_in), nt_in, p(t_out), p(c_out), p(ab), p(rhs), p(work), p(aux),
        ip(ipiv),
    )


@export("msi_periodic_scratch")
def msi_periodic_scratch(nx: Int, k: Int) abi("C") -> Int:
    return periodic_scratch(nx, k)


@export("msi_make_lsq_spline")
def msi_make_lsq_spline(
    x: Int, nx: Int, y: Int, ycols: Int, t: Int, nt: Int, k: Int, w: Int,
    c_out: Int, ab: Int, rhs: Int, work: Int, ipiv: Int,
) abi("C") -> Int:
    return make_lsq_spline(
        p(x), nx, p(y), ycols, p(t), nt, k, p(w), p(c_out), p(ab), p(rhs),
        p(work), ip(ipiv),
    )


# ------------------------------------------------------------------ _ppoly
@export("msi_evaluate")
def msi_evaluate(
    c: Int, k: Int, m: Int, n: Int, x: Int, xp: Int, nxp: Int, dx: Int,
    extrapolate: Int, dst: Int,
) abi("C"):
    evaluate(p(c), k, m, n, p(x), p(xp), nxp, dx, extrapolate != 0, p(dst))


@export("msi_fix_continuity")
def msi_fix_continuity(
    c: Int, k: Int, m: Int, n: Int, x: Int, nx: Int, order: Int,
) abi("C"):
    fix_continuity(p(c), k, m, n, p(x), nx, order)


@export("msi_ppoly_integrate")
def msi_ppoly_integrate(
    c: Int, k: Int, m: Int, n: Int, x: Int, nx: Int, a: Float64, b: Float64,
    extrapolate: Int, dst: Int,
) abi("C"):
    ppoly_integrate(
        p(c), k, m, n, p(x), nx, a, b, extrapolate != 0, p(dst)
    )


# --------------------------------------------------------- _rgi_cython
@export("msi_find_indices")
def msi_find_indices(
    grid_addrs: Int, grid_lens: Int, ndim: Int, xi: Int, npoints: Int,
    indices: Int, norm_distances: Int,
) abi("C"):
    find_indices(
        ip(grid_addrs), ip(grid_lens), ndim, p(xi), npoints, ip(indices),
        p(norm_distances),
    )


@export("msi_evaluate_linear_2d")
def msi_evaluate_linear_2d(
    values: Int, n0: Int, n1: Int, indices: Int, norm_distances: Int,
    npoints: Int, dst: Int,
) abi("C"):
    evaluate_linear_2d(
        p(values), n0, n1, ip(indices), p(norm_distances), npoints, p(dst)
    )


@export("msi_evaluate_linear_nd")
def msi_evaluate_linear_nd(
    values: Int, dims: Int, ndim: Int, ntrail: Int, indices: Int,
    norm_distances: Int, npoints: Int, dst: Int, strides: Int,
) abi("C"):
    evaluate_linear_nd(
        p(values), ip(dims), ndim, ntrail, ip(indices), p(norm_distances), npoints,
        p(dst), ip(strides),
    )


# ------------------------------------------------- tensor-product spline
@export("msi_evaluate_tensor_spline")
def msi_evaluate_tensor_spline(
    coeffs: Int, t_addrs: Int, degrees: Int, ncoefs: Int, cstrides: Int,
    ndim: Int, ntrail: Int, nu: Int, x: Int, m: Int, dst: Int, work: Int,
    iscratch: Int,
) abi("C"):
    evaluate_tensor_spline(
        p(coeffs), ip(t_addrs), ip(degrees), ip(ncoefs), ip(cstrides), ndim,
        ntrail, nu, p(x), m, p(dst), p(work), ip(iscratch),
    )


@export("msi_tensor_spline_wstride")
def msi_tensor_spline_wstride(degrees: Int, ndim: Int) abi("C") -> Int:
    return tensor_spline_wstride(ip(degrees), ndim)


# ---------------------------------------------------------- _interpolate
@export("msi_linear_interp")
def msi_linear_interp(
    x: Int, y: Int, n: Int, x_new: Int, nnew: Int, y_new: Int, ycols: Int
) abi("C"):
    linear_interp(p(x), p(y), n, p(x_new), nnew, p(y_new), ycols)


@export("msi_nearest_interp")
def msi_nearest_interp(
    x_bds: Int, n: Int, y: Int, nnew: Int, x_new: Int, side: Int, y_new: Int,
    ycols: Int,
) abi("C"):
    nearest_interp(p(x_bds), n, p(y), nnew, p(x_new), side, p(y_new), ycols)


@export("msi_previous_next")
def msi_previous_next(
    x_shift: Int, n_shift: Int, y: Int, n: Int, nnew: Int, x_new: Int, ind: Int,
    side: Int, y_new: Int, ycols: Int,
) abi("C"):
    previous_next(
        p(x_shift), n_shift, p(y), n, nnew, p(x_new), ind, side, p(y_new), ycols
    )


# ------------------------------------------------------------------ _rbf
@export("msi_rbf_function_matrix")
def msi_rbf_function_matrix(
    kernel: Int, eps: Float64, r: Int, n: Int, dst: Int
) abi("C"):
    rbf_function_matrix(kernel, eps, p(r), n, p(dst))


@export("msi_rbf_function_dot")
def msi_rbf_function_dot(
    kernel: Int, eps: Float64, r: Int, npoints: Int, ncentres: Int, nodes: Int,
    ncols: Int, dst: Int,
) abi("C"):
    rbf_function_dot(
        kernel, eps, p(r), npoints, ncentres, p(nodes), ncols, p(dst)
    )


@export("msi_cdist_euclidean")
def msi_cdist_euclidean(
    x: Int, n: Int, d: Int, y: Int, m: Int, dst: Int
) abi("C"):
    cdist_euclidean(p(x), n, d, p(y), m, p(dst))


# ------------------------------------------------------------ _rbfinterp
@export("msi_kernel_matrix")
def msi_kernel_matrix(x: Int, n: Int, d: Int, kernel: Int, dst: Int) abi("C"):
    kernel_matrix(p(x), n, d, kernel, p(dst))


@export("msi_polynomial_matrix")
def msi_polynomial_matrix(
    x: Int, n: Int, d: Int, powers: Int, ndim: Int, npow: Int, dst: Int
) abi("C"):
    polynomial_matrix(p(x), n, d, ip(powers), ndim, npow, p(dst))


@export("msi_build_evaluation_coefficients")
def msi_build_evaluation_coefficients(
    x: Int,
    q: Int,
    y: Int,
    pcount: Int,
    d: Int,
    kernel: Int,
    epsilon: Float64,
    powers: Int,
    ndim: Int,
    npow: Int,
    shift: Int,
    scale: Int,
    scratch: Int,
    vec: Int,
) abi("C"):
    build_evaluation_coefficients(
        p(x), q, p(y), pcount, d, kernel, epsilon, ip(powers), ndim, npow, p(shift),
        p(scale), p(scratch), p(vec),
    )


@export("msi_neighbors_knn")
def msi_neighbors_knn(
    train: Int, n: Int, d: Int, query: Int, m: Int, k: Int, dist: Int, idx: Int
) abi("C"):
    neighbors_knn(p(train), n, d, p(query), m, k, p(dist), ip(idx))
