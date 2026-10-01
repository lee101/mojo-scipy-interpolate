"""Parity of the regular-grid, legacy-interp1d and RBF layers with upstream."""

from __future__ import annotations

import numpy as np
import pytest

import mojoscipy as msi
import scipy.interpolate as si

GRID1 = np.linspace(0, 1, 7)
GRID2 = np.linspace(0, 2, 6)
VALUES = np.add.outer(GRID1**2, np.sin(GRID2))
QUERY = np.array([[0.3, 0.5], [0.5, 1.5], [0.0, 0.0], [1.0, 2.0]])
KINDS = ("linear", "nearest", "cubic", "quintic", "slinear", "pchip")


@pytest.mark.parametrize("method", KINDS)
def test_regular_grid_interpolator_2d(method):
    ours = msi.RegularGridInterpolator((GRID1, GRID2), VALUES, method=method)
    theirs = si.RegularGridInterpolator((GRID1, GRID2), VALUES, method=method)
    # upstream solves the tensor collocation system with an iterative solver
    # (atol=1e-6), so the spline methods agree only to that tolerance
    assert np.allclose(ours(QUERY), theirs(QUERY), equal_nan=True, atol=1e-4)


@pytest.mark.parametrize("method", ("linear", "nearest", "cubic"))
def test_regular_grid_interpolator_3d(method):
    g = (np.linspace(0, 1, 4), np.linspace(0, 1, 5), np.linspace(0, 1, 6))
    z = np.einsum("i,j,k->ijk", g[0], g[1] ** 2, g[2] - 1)
    q = np.array([[0.3, 0.5, 0.7], [0.1, 0.2, 0.9]])
    ours = msi.RegularGridInterpolator(g, z, method=method)
    theirs = si.RegularGridInterpolator(g, z, method=method)
    if method == "cubic":
        # upstream switches to `make_ndbspl`, whose iterative solve is only
        # accurate to its 1e-6 tolerance; this port solves the tensor
        # collocation system exactly
        assert np.allclose(ours(q), theirs(q), atol=1e-4)
    else:
        assert np.allclose(ours(q), theirs(q))


def test_regular_grid_interpolator_trailing_dims():
    z = np.stack([VALUES, 2.0 * VALUES], axis=-1)
    ours = msi.RegularGridInterpolator((GRID1, GRID2), z)
    theirs = si.RegularGridInterpolator((GRID1, GRID2), z)
    assert ours(QUERY).shape == theirs(QUERY).shape
    assert np.allclose(ours(QUERY), theirs(QUERY))


def test_regular_grid_bounds_and_nan():
    z = VALUES
    ours = msi.RegularGridInterpolator((GRID1, GRID2), z, bounds_error=True)
    theirs = si.RegularGridInterpolator((GRID1, GRID2), z, bounds_error=True)
    for bad in (np.array([[-1.0, 0.5]]), np.array([[0.2, np.nan]])):
        with pytest.raises(ValueError):
            ours(bad, method="linear")
        with pytest.raises(ValueError):
            theirs(bad, method="linear")


def test_regular_grid_nan_propagates():
    z = VALUES
    ours = msi.RegularGridInterpolator((GRID1, GRID2), z, bounds_error=False)
    theirs = si.RegularGridInterpolator((GRID1, GRID2), z, bounds_error=False)
    q = np.array([[0.2, np.nan], [0.5, 0.5]])
    a = ours(q, method="linear")
    b = theirs(q, method="linear")
    assert np.array_equal(np.isnan(a), np.isnan(b))



def test_regular_grid_fill_value():
    z = VALUES
    ours = msi.RegularGridInterpolator(
        (GRID1, GRID2), z, bounds_error=False, fill_value=-1.0
    )
    theirs = si.RegularGridInterpolator(
        (GRID1, GRID2), z, bounds_error=False, fill_value=-1.0
    )
    q = np.array([[-1.0, 0.5], [0.3, 0.5]])
    assert np.allclose(ours(q), theirs(q))


def test_interpn():
    ours = msi.interpn((GRID1, GRID2), VALUES, QUERY)
    theirs = si.interpn((GRID1, GRID2), VALUES, QUERY)
    assert np.allclose(ours, theirs)


def test_rect_bivariate_spline():
    gx = np.linspace(0, 1, 9)
    gy = np.linspace(0, 2, 8)
    z = np.sin(gx[:, None] * 3) + gy[None, :] ** 2
    ours = msi.RectBivariateSpline(gx, gy, z)
    theirs = si.RectBivariateSpline(gx, gy, z)
    qx = np.array([0.2, 0.9])
    qy = np.array([0.5, 1.7])
    assert np.allclose(ours(qx, qy), theirs(qx, qy))
    assert np.allclose(ours.ev(qx, qy), theirs.ev(qx, qy))
    assert np.allclose(ours(0.3, 0.8), theirs(0.3, 0.8), atol=1e-9)


def test_rect_bivariate_spline_rejects_smoothing():
    gx = np.linspace(0, 1, 9)
    gy = np.linspace(0, 2, 8)
    z = np.add.outer(gx, gy)
    with pytest.raises(NotImplementedError):
        msi.RectBivariateSpline(gx, gy, z, s=1.0)


def test_pchip_matches_upstream():
    x = np.linspace(0, 1, 9)
    y = np.sin(4 * x)
    xx = np.linspace(0, 1, 23)
    ours = msi.PchipInterpolator(x, y)
    theirs = si.PchipInterpolator(x, y)
    assert np.allclose(ours(xx), theirs(xx))
    Y = np.stack([y, 2.0 * y], axis=1)
    assert np.allclose(msi.PchipInterpolator(x, Y)(xx), si.PchipInterpolator(x, Y)(xx))


@pytest.mark.parametrize(
    "kind", ["linear", "nearest", "nearest-up", "previous", "next",
             "zero", "slinear", "quadratic", "cubic", 0, 2]
)
def test_interp1d_matches_upstream(kind):
    x = np.linspace(0, 10, 21)
    y = np.sin(x)
    xn = np.linspace(0.3, 9.7, 29)
    ours = msi.interp1d(x, y, kind=kind)
    theirs = si.interp1d(x, y, kind=kind)
    assert np.allclose(ours(xn), theirs(xn), equal_nan=True, atol=1e-12)


def test_interp1d_fill_value():
    x = np.linspace(0, 1, 11)
    y = x**2
    ours = msi.interp1d(x, y, fill_value=-1.0, bounds_error=False)
    theirs = si.interp1d(x, y, fill_value=-1.0, bounds_error=False)
    xn = np.array([-1.0, 0.5, 2.0])
    assert np.allclose(ours(xn), theirs(xn), equal_nan=True)


def test_interp1d_bounds_error():
    x = np.linspace(0, 1, 11)
    ours = msi.interp1d(x, x, bounds_error=True)
    with pytest.raises(ValueError):
        ours(np.array([-1.0]))



@pytest.mark.parametrize("kind", ["linear", "nearest", "nearest-up", "previous", "next"])
def test_interp1d_large_query_count(kind):
    """The search-hint and the parallel path, on a query array over the
    `PARALLEL_MIN` threshold the two switch on."""
    # axis=-1 is the interpolation axis, so `y` is (targets, points) and the
    # result is (targets, queries), as upstream returns it
    x = np.linspace(0, 10, 501)
    y = np.stack([np.sin(x), np.cos(x)], axis=0)
    ours = msi.interp1d(x, y, kind=kind)
    theirs = si.interp1d(x, y, kind=kind)
    # sorted, unsorted, and exactly on the knots: the three ways the search
    # hint can be right, unhelpful, or off by an interval
    assert np.allclose(ours(np.linspace(0, 10, 9000)),
                       theirs(np.linspace(0, 10, 9000)), atol=1e-12)
    shuffled = np.random.default_rng(0).random(9000) * 10
    assert np.allclose(ours(shuffled), theirs(shuffled), atol=1e-12)
    # out of range is left to `test_interp1d_large_out_of_range_fill`:
    # upstream raises there unless `bounds_error=False` is passed


def test_interp1d_large_out_of_range_fill():
    x = np.linspace(0, 1, 101)
    y = np.sin(4 * x)
    q = np.linspace(-0.5, 1.5, 9000)
    ours = msi.interp1d(x, y, kind="linear", bounds_error=False, fill_value=-7.0)
    theirs = si.interp1d(x, y, kind="linear", bounds_error=False, fill_value=-7.0)
    assert np.allclose(ours(q), theirs(q), equal_nan=True, atol=1e-12)




@pytest.mark.parametrize("kind", ["linear", "nearest", "nearest-up", "previous", "next"])
def test_interp1d_unsorted_queries_spanning_the_range(kind):
    """Unsorted queries drive the search hint backwards as well as forwards,
    and past the ends of the array it searches."""
    x = np.linspace(0, 10, 501)
    y = np.stack([np.sin(x), np.cos(x)], axis=0)
    ours = msi.interp1d(x, y, kind=kind)
    theirs = si.interp1d(x, y, kind=kind)
    for n in (5000, 9000):
        q = np.random.default_rng(1).random(n) * 10
        assert np.allclose(ours(q), theirs(q), atol=1e-12), (kind, n)


@pytest.mark.parametrize("kind", ["previous", "next"])
def test_interp1d_knot_at_zero(kind):
    """`nextafter(0, inf)` is 5e-324.  The port substitutes the smallest
    normal so the shifted knot still compares strictly above zero, which the
    vectorised search would otherwise see as a tie."""
    x = np.linspace(0, 10, 501)
    y = np.stack([np.sin(x), np.cos(x)], axis=0)
    ours = msi.interp1d(x, y, kind=kind)
    theirs = si.interp1d(x, y, kind=kind)
    for n in (300, 4096, 9000):
        q = np.linspace(0, 10, n)
        assert np.allclose(ours(q), theirs(q), atol=1e-12), (kind, n)


def test_regular_grid_linear_large_query_count():
    """`find_indices` and `evaluate_linear_2d` above `PARALLEL_MIN`."""
    g = np.linspace(0, 1, 61)
    z = np.add.outer(g**2, np.sin(g))
    rng = np.random.default_rng(4)
    q = rng.random((9000, 2))
    ours = msi.RegularGridInterpolator((g, g), z, method="linear")
    theirs = si.RegularGridInterpolator((g, g), z, method="linear")
    assert np.allclose(ours(q), theirs(q), atol=1e-12)


def test_regular_grid_find_indices_on_grid_points():
    """Every query exactly on a knot, which is where the branchless
    `upper_bound` and the locality hint could disagree."""
    g1 = np.linspace(0, 1, 7)
    g2 = np.linspace(0, 2, 6)
    q = np.array([(a, b) for a in g1 for b in g2])
    ours = msi.RegularGridInterpolator((g1, g2), VALUES, method="linear")
    theirs = si.RegularGridInterpolator((g1, g2), VALUES, method="linear")
    assert np.allclose(ours(q), theirs(q), atol=1e-12)
    # and the same points, repeated, so the hint carries over between them
    assert np.allclose(ours(np.tile(q, (3, 1))), theirs(np.tile(q, (3, 1))),
                       atol=1e-12)


@pytest.mark.parametrize("method", ["cubic", "slinear", "quintic"])
def test_regular_grid_spline_large_query_count(method):
    """The tensor-product kernel on both sides of `PARALLEL_MIN`."""
    g = np.linspace(0, 1, 12)
    z = np.add.outer(g**2, np.sin(g))
    rng = np.random.default_rng(5)
    ours = msi.RegularGridInterpolator((g, g), z, method=method)
    theirs = si.RegularGridInterpolator((g, g), z, method=method)
    for q in (rng.random((20, 2)), rng.random((9000, 2)), rng.random((4095, 2)),
              rng.random((4097, 2))):
        assert np.allclose(ours(q), theirs(q), atol=1e-4), q.shape


def test_regular_grid_spline_trailing_dims():
    z = np.stack([VALUES, 2.0 * VALUES, 3.0 * VALUES], axis=-1)
    q = np.random.default_rng(6).random((5000, 2))
    for method in ("slinear", "cubic", "quintic"):
        ours = msi.RegularGridInterpolator((GRID1, GRID2), z, method=method)
        theirs = si.RegularGridInterpolator((GRID1, GRID2), z, method=method)
        assert ours(q).shape == theirs(q).shape
        assert np.allclose(ours(q), theirs(q), atol=1e-4), method


def test_regular_grid_spline_nan_and_derivatives():
    g = np.linspace(0, 1, 9)
    z = np.outer(g**2, np.sin(g))
    ours = msi.RegularGridInterpolator((g, g), z, method="cubic",
                                       bounds_error=False)
    theirs = si.RegularGridInterpolator((g, g), z, method="cubic",
                                        bounds_error=False)
    q = np.array([[0.2, 0.3], [0.2, np.nan], [np.nan, 0.4], [0.5, 0.5]])
    a = ours(q, method="cubic")
    b = theirs(q, method="cubic")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a, b, atol=1e-4, equal_nan=True)
    # `nu` derivatives go through the same exact fold, so the port is checked
    # against it directly: upstream's derivative path inherits its iterative
    # solve's 1e-6 coefficient error, which the second derivative amplifies
    # to O(1), so it is not a reference at nu=2.
    from mojoscipy.bsplines import make_interp_spline as _mis
    from mojoscipy._fitpack2 import _basis
    coeffs = ours._tensor_spline(3)[0]
    axes, c = [], ours.values
    for axis in (1, 0):
        axes.append(_mis(g, c, k=3, axis=axis))
    axes.reverse()
    q2 = np.random.default_rng(7).random((5000, 2))
    for nu in (0, 1, 2):
        exact = np.einsum(
            "ia,ab,ib->i",
            _basis(axes[0], q2[:, 0], nu), coeffs, _basis(axes[1], q2[:, 1], nu),
        )
        assert np.allclose(ours(q2, method="cubic", nu=nu), exact, atol=1e-9)
        assert np.allclose(exact, theirs(q2, method="cubic", nu=[nu, nu]),
                           atol=1e-4 if nu == 0 else 5.0)


def test_rbf_interpolator_neighbors_shared_neighbourhoods():
    """`neighbors=k` groups queries that share a neighbourhood; the grouping
    must give each query the answer of its own neighbourhood."""
    rng = np.random.default_rng(11)
    y = rng.random((60, 2))
    d = np.sin(y[:, 0])
    # half the queries are the data points themselves, so they all collapse
    # onto the same neighbourhood; the rest are scattered
    x = np.vstack([y, rng.random((40, 2))])
    ours = msi.RBFInterpolator(y, d, neighbors=8)
    theirs = si.RBFInterpolator(y, d, neighbors=8)
    assert np.allclose(ours(x), theirs(x), atol=1e-7)
    single = msi.RBFInterpolator(y, d, neighbors=8)
    for i in range(0, len(x), 7):
        assert np.allclose(single(x[i : i + 1]), theirs(x[i : i + 1]), atol=1e-7)


def test_rbf_interpolator_solve_route_both_sides():
    """`neighbors=k` picks the LAPACK route by system size; both sides of the
    threshold must give the same answer as upstream."""
    rng = np.random.default_rng(12)
    y = rng.random((400, 2))
    d = np.sin(y[:, 0])
    x = rng.random((12, 2))
    for neighbors in (32, 200):
        ours = msi.RBFInterpolator(y, d, neighbors=neighbors)
        theirs = si.RBFInterpolator(y, d, neighbors=neighbors)
        assert np.allclose(ours(x), theirs(x), atol=1e-7), neighbors


@pytest.mark.parametrize("smoothing", [0.0, 0.25])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_rbf_build_system_degenerate_axis(smoothing, degree):
    """`rbf_build_system` owns the shift/scale, including the zero scale of an
    axis every point shares -- where upstream's monomial block is rank
    deficient, so both must fail the same way."""
    rng = np.random.default_rng(13)
    y = np.column_stack([rng.random(50), np.zeros(50), rng.random(50)])
    d = rng.random((50, 2))
    x = rng.random((10, 3))
    kwargs = {"smoothing": smoothing, "degree": degree}
    if degree == 0:
        ours = msi.RBFInterpolator(y, d, **kwargs)
        theirs = si.RBFInterpolator(y, d, **kwargs)
        assert np.allclose(ours(x), theirs(x), atol=1e-7)
        return
    with pytest.raises(np.linalg.LinAlgError):
        msi.RBFInterpolator(y, d, **kwargs)
    with pytest.raises(np.linalg.LinAlgError):
        si.RBFInterpolator(y, d, **kwargs)


@pytest.mark.parametrize("neighbors", [8, 64])
def test_rbf_interpolator_tiny_scale_axis(neighbors):
    """An axis whose range is tiny but non-zero: `rbf_build_system` scales it
    up, and the two must still agree."""
    rng = np.random.default_rng(14)
    y = np.column_stack([rng.random(80), 1e-9 * rng.random(80)])
    d = rng.random(80)
    x = np.column_stack([rng.random(30), 1e-9 * rng.random(30)])
    ours = msi.RBFInterpolator(y, d, neighbors=neighbors)
    theirs = si.RBFInterpolator(y, d, neighbors=neighbors)
    # the tiny scale makes the system badly conditioned by construction
    assert np.allclose(ours(x), theirs(x), atol=1e-3), neighbors




# ---------------------------------------------------------------- RBF


def _rbf_data(n=40, d=3, seed=1):
    y = np.random.default_rng(seed).random((n, d))
    values = np.sin(y[:, 0]) + y[:, 1] ** 2
    x = np.random.default_rng(seed + 1).random((12, d))
    return y, values, x


@pytest.mark.parametrize(
    "kernel,epsilon",
    [
        ("thin_plate_spline", None), ("linear", None), ("cubic", None),
        ("quintic", None), ("multiquadric", 1.5),
        ("inverse_multiquadric", 1.5), ("inverse_quadratic", 1.5),
        ("gaussian", 1.5),
    ],
)
def test_rbf_interpolator_kernels(kernel, epsilon):
    y, d, x = _rbf_data()
    kwargs = {} if epsilon is None else {"epsilon": epsilon}
    ours = msi.RBFInterpolator(y, d, kernel=kernel, **kwargs)
    theirs = si.RBFInterpolator(y, d, kernel=kernel, **kwargs)
    assert np.allclose(ours(x), theirs(x), atol=1e-8)


@pytest.mark.parametrize(
    "kwargs", [{"smoothing": 0.1}, {"degree": 2}, {"degree": -1}, {"degree": 0}]
)
def test_rbf_interpolator_options(kwargs):
    y, d, x = _rbf_data()
    ours = msi.RBFInterpolator(y, d, **kwargs)
    theirs = si.RBFInterpolator(y, d, **kwargs)
    assert np.allclose(ours(x), theirs(x), atol=1e-8)


def test_rbf_interpolator_smoothing_vector():
    y, d, x = _rbf_data()
    s = np.linspace(0, 0.5, len(y))
    ours = msi.RBFInterpolator(y, d, smoothing=s)
    theirs = si.RBFInterpolator(y, d, smoothing=s)
    assert np.allclose(ours(x), theirs(x), atol=1e-8)


def test_rbf_interpolator_multi_target():
    y, d, x = _rbf_data()
    d2 = np.stack([d, 2.0 * d], axis=1)
    ours = msi.RBFInterpolator(y, d2)
    theirs = si.RBFInterpolator(y, d2)
    assert ours(x).shape == theirs(x).shape
    assert np.allclose(ours(x), theirs(x))


@pytest.mark.parametrize("neighbors", [5, 8, 16])
def test_rbf_interpolator_neighbors(neighbors):
    y, d, x = _rbf_data()
    ours = msi.RBFInterpolator(y, d, neighbors=neighbors)
    theirs = si.RBFInterpolator(y, d, neighbors=neighbors)
    assert np.allclose(ours(x), theirs(x), atol=1e-7)


def test_rbf_interpolator_interpolates():
    y, d, _ = _rbf_data()
    ours = msi.RBFInterpolator(y, d)
    assert np.allclose(ours(y), d, atol=1e-8)


def test_rbf_interpolator_validation():
    y, d, _ = _rbf_data()
    with pytest.raises(ValueError):
        msi.RBFInterpolator(y, d, kernel="not-a-kernel")
    with pytest.raises(ValueError):
        msi.RBFInterpolator(y, d, kernel="gaussian")  # epsilon missing


@pytest.mark.parametrize(
    "function,epsilon",
    [
        ("multiquadric", 0.4), ("inverse", 0.4), ("gaussian", 0.4),
        ("linear", None), ("cubic", None), ("quintic", None),
        ("thin_plate", None), ("inverse multiquadric", 0.4),
    ],
)
def test_rbf_legacy_kernels(function, epsilon):
    rng = np.random.default_rng(3)
    coords = rng.random((3, 25))
    d = rng.random(25)
    q = rng.random((3, 25))
    kwargs = {} if epsilon is None else {"epsilon": epsilon}
    ours = msi.Rbf(*coords, d, function=function, **kwargs)
    theirs = si.Rbf(*coords, d, function=function, **kwargs)
    assert np.allclose(ours(*q), theirs(*q), atol=1e-7)


def test_rbf_legacy_smooth_and_matrix():
    rng = np.random.default_rng(3)
    coords = rng.random((3, 20))
    d = rng.random(20)
    assert np.allclose(
        msi.Rbf(*coords, d).A, si.Rbf(*coords, d).A, atol=1e-12
    )
    assert np.allclose(
        msi.Rbf(*coords, d, smooth=0.1).nodes,
        si.Rbf(*coords, d, smooth=0.1).nodes,
        atol=1e-12,
    )


def test_rbf_legacy_multi_target():
    rng = np.random.default_rng(3)
    coords = rng.random((3, 20))
    d = rng.random((20, 2))
    q = rng.random((3, 20))
    ours = msi.Rbf(*coords, d, mode="N-D")
    theirs = si.Rbf(*coords, d, mode="N-D")
    assert np.allclose(ours(*q), theirs(*q), atol=1e-12)


def test_rbf_legacy_callable():
    rng = np.random.default_rng(3)
    coords = rng.random((3, 20))
    d = rng.random(20)
    q = rng.random((3, 20))

    def kernel(r):
        return np.exp(-(r / 0.7) ** 2)

    ours = msi.Rbf(*coords, d, function=kernel)
    theirs = si.Rbf(*coords, d, function=kernel)
    assert np.allclose(ours(*q), theirs(*q))



KERNEL_FORMULAS = {
    "linear": lambda r: -r,
    "thin_plate_spline": lambda r: np.where(r == 0, 0.0, r**2 * np.log(np.where(r == 0, 1.0, r))),
    "cubic": lambda r: r**3,
    "quintic": lambda r: -(r**5),
    "multiquadric": lambda r: -np.sqrt(r**2 + 1),
    "inverse_multiquadric": lambda r: 1 / np.sqrt(r**2 + 1),
    "inverse_quadratic": lambda r: 1 / (r**2 + 1),
    "gaussian": lambda r: np.exp(-(r**2)),
}


@pytest.mark.parametrize("name", sorted(KERNEL_FORMULAS))
def test_rbf_kernels_match_their_formula(name):
    r = np.array([0.0, 0.3, 1.0, 2.5])
    want = KERNEL_FORMULAS[name](r)
    assert np.allclose(getattr(msi, name)(r), want, atol=1e-15), name
    assert np.allclose(msi.rbf_factory(name)(r), want, atol=1e-15), name


def test_cubic_hermite_spline_matches_upstream():
    x = np.linspace(0, 1, 9)
    y = np.sin(3 * x)
    dydx = np.cos(3 * x)
    ours = msi.CubicHermiteSpline(x, y, dydx)
    theirs = si.CubicHermiteSpline(x, y, dydx)
    xx = np.linspace(0, 1, 23)
    assert np.allclose(ours(xx), theirs(xx))
    assert np.allclose(ours(xx, nu=1), theirs(xx, nu=1), atol=1e-10)
    assert np.allclose(ours(xx, nu=2), theirs(xx, nu=2), atol=1e-10)


def test_ppoly_matches_upstream():
    x = np.linspace(0, 2, 7)
    c = np.stack([np.full(6, 0.5), np.linspace(0, 1, 6), np.full(6, 2.0)])
    ours = msi.PPoly(c, x)
    theirs = si.PPoly(c, x)
    xx = np.linspace(-0.5, 2.5, 31)
    assert np.allclose(ours(xx), theirs(xx))
    for nu in (0, 1, 2, 3):
        assert np.allclose(ours(xx, nu=nu), theirs(xx, nu=nu), atol=1e-10), nu
    assert np.allclose(
        ours(xx, extrapolate=False), theirs(xx, extrapolate=False), equal_nan=True
    )
    # `antiderivative` goes through `_ppoly.fix_continuity`, so the
    # integration constant is fixed by upstream's own continuity pass and both
    # sides must agree on the coefficients themselves, not only on values.
    assert np.allclose(ours.antiderivative().c, theirs.antiderivative().c, atol=1e-12)
    assert np.allclose(ours.antiderivative(2).c, theirs.antiderivative(2).c, atol=1e-12)
    assert np.allclose(ours.derivative().c, theirs.derivative().c, atol=1e-12)
    assert np.allclose(ours.derivative(2).c, theirs.derivative(2).c, atol=1e-12)
    assert np.allclose(ours(xx, nu=3), theirs(xx, nu=3), atol=1e-10)
    for lo, hi in ((0.2, 1.4), (1.4, 0.2), (0.0, 2.0), (0.5, 0.5)):
        assert abs(ours.integrate(lo, hi) - theirs.integrate(lo, hi)) < 1e-12, (lo, hi)
    # the definite integral must agree with the interval-by-interval one: `c` is
    # leading coefficient first, so interval `i` holds
    # `sum_j c[j, i] * u^(k-j)`.
    k = c.shape[0]
    want = 0.0
    for i in range(c.shape[1]):
        lo, hi = max(x[i], 0.2), min(x[i + 1], 1.4)
        if hi <= lo:
            continue
        u0, u1 = lo - x[i], hi - x[i]
        want += sum(c[j, i] * (u1 ** (k - j) - u0 ** (k - j)) / (k - j)
                    for j in range(k))
    assert abs(ours.integrate(0.2, 1.4) - want) < 1e-12
    assert abs(ours.integrate(1.4, 0.2) + want) < 1e-12


def test_ppoly_derivative_and_antiderivative_are_inverse():
    x = np.linspace(0, 1, 3)
    c = np.array([[1.0, -0.5], [0.25, 2.0], [-3.0, 0.5], [0.75, 1.0]])
    p = msi.PPoly(c, x)
    # the knot at 0.5 is a slope discontinuity, so the finite differences stay
    # inside the two intervals
    xx = np.r_[np.linspace(0.0, 0.49, 9), np.linspace(0.51, 1.0, 9)]
    h = 1e-6
    fd = (p(xx + h) - p(xx - h)) / (2 * h)
    assert np.allclose(p.derivative()(xx), fd, atol=1e-6)
    # a second derivative against the difference of the first
    fd2 = (p.derivative()(xx + h) - p.derivative()(xx - h)) / (2 * h)
    assert np.allclose(p.derivative(2)(xx), fd2, atol=1e-5)
    # integrating and differentiating are inverses
    back = p.antiderivative().derivative()
    assert np.allclose(back.c, p.c, atol=1e-12)
    assert np.allclose(back(xx), p(xx), atol=1e-12)
    assert np.allclose(p.antiderivative(2).derivative(2).c, p.c, atol=1e-12)
