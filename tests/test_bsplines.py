"""Parity of the B-spline machinery against `scipy.interpolate`."""

from __future__ import annotations

import numpy as np
import pytest

import mojoscipy as msi
import scipy.interpolate as si


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = np.sort(rng.random(30))
    x[0], x[-1] = 0.0, 1.0
    x = np.unique(x)
    return x, np.sin(3 * x) + 0.1 * x


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_make_interp_spline_matches(data, k):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=k)
    theirs = si.make_interp_spline(x, y, k=k)
    assert np.allclose(ours.t, theirs.t)
    assert np.allclose(ours.c, theirs.c)


def test_make_interp_spline_interpolates(data):
    x, y = data
    assert np.allclose(msi.make_interp_spline(x, y)(x), y, atol=1e-12)


def test_make_interp_spline_k0(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=0)
    theirs = si.make_interp_spline(x, y, k=0)
    assert np.allclose(ours.t, theirs.t)
    assert np.allclose(ours.c, theirs.c)


def test_make_interp_spline_axis(data):
    x, y = data
    Y = np.stack([y, 2.0 * y], axis=1)
    ours = msi.make_interp_spline(x, Y, k=3, axis=0)
    theirs = si.make_interp_spline(x, Y, k=3, axis=0)
    assert np.allclose(ours.c, theirs.c)
    xx = np.linspace(0, 1, 17)
    assert np.allclose(ours(xx), theirs(xx))


def test_make_interp_spline_clamped(data):
    x, y = data
    for bc in ("clamped", ([(1, 0.0)], [(1, 0.0)])):
        ours = msi.make_interp_spline(x, y, k=3, bc_type=bc)
        theirs = si.make_interp_spline(x, y, k=3, bc_type=bc)
        assert np.allclose(ours.c, theirs.c)


@pytest.mark.parametrize("n", [5, 6, 8, 12, 20])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_make_interp_spline_periodic(n, k):
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)
    ours = msi.make_interp_spline(x, y, k=k, bc_type="periodic")
    theirs = si.make_interp_spline(x, y, k=k, bc_type="periodic")
    assert np.allclose(ours.t, theirs.t)
    assert np.allclose(ours.c, theirs.c)
    assert np.allclose(ours(x), theirs(x))


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("ncols", [2, 3])
def test_make_interp_spline_periodic_multi_column(k, ncols):
    x = np.linspace(0, 2 * np.pi, 11)
    y = np.stack([np.sin((c + 1) * x) for c in range(ncols)], axis=1)
    y[0] = y[-1]
    ours = msi.make_interp_spline(x, y, k=k, bc_type="periodic")
    theirs = si.make_interp_spline(x, y, k=k, bc_type="periodic")
    assert ours.c.shape == theirs.c.shape
    assert np.allclose(ours.t, theirs.t)
    assert np.allclose(ours.c, theirs.c)
    xx = np.linspace(0, 2 * np.pi, 23)
    assert np.allclose(ours(xx), theirs(xx))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_make_lsq_spline_matches(k):
    x = np.linspace(0, 1, 40)
    y = np.sin(3 * x)
    t = np.r_[[0.0] * k, np.linspace(0, 1, 30), [1.0] * k]
    ours = msi.make_lsq_spline(x, y, t, k=k)
    theirs = si.make_lsq_spline(x, y, t, k=k, method="norm-eq")
    assert np.allclose(ours.c, theirs.c)
    xx = np.linspace(0, 1, 25)
    assert np.allclose(ours(xx), theirs(xx))


def test_make_lsq_spline_weights():
    x = np.linspace(0, 1, 40)
    y = np.sin(3 * x)
    t = np.r_[[0.0] * 3, np.linspace(0, 1, 30), [1.0] * 3]
    w = np.linspace(0.5, 1.5, 40)
    ours = msi.make_lsq_spline(x, y, t, k=3, w=w)
    theirs = si.make_lsq_spline(x, y, t, k=3, w=w, method="norm-eq")
    assert np.allclose(ours.c, theirs.c)


@pytest.mark.parametrize("nu", [0, 1, 2, 3])
def test_bspline_evaluate_and_derivative(data, nu):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    xx = np.linspace(0, 1, 37)
    assert np.allclose(ours(xx, nu=nu), theirs(xx, nu=nu), atol=1e-9)


def test_bspline_out_of_bounds_nan(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    ours.extrapolate = False
    theirs.extrapolate = False
    xx = np.array([-0.5, 0.5, 1.5])
    assert np.array_equal(np.isnan(ours(xx)), np.isnan(theirs(xx)))


def test_bspline_extrapolate(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    ours.extrapolate = True
    theirs.extrapolate = True
    xx = np.array([-0.5, 0.5, 1.5])
    assert ours.extrapolate and theirs.extrapolate
    assert np.allclose(ours(xx), theirs(xx))


@pytest.mark.parametrize("t", [[0, 1, 2, 3, 4], [0, 1, 1, 2], [0, 0.5, 1.0]])
def test_bspline_basis_element(t):
    ours = msi.BSpline.basis_element(np.asarray(t, dtype=float))
    theirs = si.BSpline.basis_element(np.asarray(t, dtype=float))
    assert ours.k == theirs.k
    assert np.array_equal(ours.t, theirs.t)
    assert np.array_equal(ours.c, theirs.c)
    lo, hi = float(t[0]), float(t[-1])
    xx = np.linspace(lo, hi, 31)
    assert np.allclose(ours(xx), theirs(xx), atol=1e-12)


def test_bspline_design_matrix():
    t = np.array([0.0, 0.0, 0.0, 0.3, 0.6, 0.9, 1.0, 1.0, 1.0])
    xx = np.linspace(0.05, 0.55, 21)
    ours = msi.BSpline.design_matrix(xx, t, 3, extrapolate=True)
    theirs = si.BSpline.design_matrix(xx, t, 3, extrapolate=True)
    assert np.allclose(ours.toarray(), theirs.toarray())


def test_bspline_derivative_antiderivative(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    assert np.allclose(ours.derivative(2).c, theirs.derivative(2).c, atol=1e-9)
    assert np.allclose(ours.antiderivative().c, theirs.antiderivative().c, atol=1e-9)
    assert np.allclose(
        ours.derivative(1)(np.array([0.25, 0.5])),
        theirs.derivative(1)(np.array([0.25, 0.5])),
        atol=1e-9,
    )


def test_bspline_integrate(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    assert np.allclose(ours.integrate(0.1, 0.8), theirs.integrate(0.1, 0.8), atol=1e-10)
    assert np.allclose(ours.integrate(0.8, 0.1), theirs.integrate(0.8, 0.1), atol=1e-10)


def test_bspline_insert_knot(data):
    x, y = data
    ours = msi.make_interp_spline(x, y, k=3)
    theirs = si.make_interp_spline(x, y, k=3)
    a = ours.insert_knot(0.5)
    b = theirs.insert_knot(0.5)
    assert np.allclose(a.t, b.t)
    assert np.allclose(a.c, b.c)
    xx = np.linspace(0, 1, 19)
    assert np.allclose(a(xx), b(xx), atol=1e-9)


def test_bspline_tck_roundtrip(data):
    x, y = data
    spl = msi.make_interp_spline(x, y, k=3)
    t, c, k = spl.tck
    assert t is spl.t and c is spl.c and k == 3


def _padded(t, c):
    """FITPACK's representation pads the coefficient vector to the knot count."""
    ct = t.shape[0] - c.shape[0]
    if ct > 0:
        c = np.concatenate((c, np.zeros(ct, dtype=c.dtype)))
    return c


def test_splder_matches_upstream(data):
    x, y = data
    t, c, k = msi.make_interp_spline(x, y, k=3).tck
    c = _padded(t, c)
    for n in (1, 2, 3):
        to, co, ko = msi.splder((t, c, k), n)
        tr, cr, kr = si.splder((t, c, k), n)
        assert np.allclose(to, tr)
        assert np.allclose(co, cr)
        assert ko == kr


def test_splantider_matches_upstream(data):
    x, y = data
    t, c, k = msi.make_interp_spline(x, y, k=3).tck
    c = _padded(t, c)
    for n in (1, 2):
        to, co, ko = msi.splantider((t, c, k), n)
        tr, cr, kr = si.splantider((t, c, k), n)
        assert np.allclose(to, tr)
        assert np.allclose(co, cr)
        assert ko == kr


def test_make_interp_spline_rejects_duplicates(data):
    x, y = data
    with pytest.raises(ValueError):
        msi.make_interp_spline(np.r_[x, x[-1]], np.r_[y, y[-1]])


def test_make_lsq_spline_qr_not_supported():
    x = np.linspace(0, 1, 10)
    with pytest.raises(NotImplementedError):
        msi.make_lsq_spline(x, x, np.r_[[0.0] * 3, np.linspace(0, 1, 12), [1.0] * 3], method="qr")
