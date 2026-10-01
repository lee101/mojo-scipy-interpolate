"""Numerical parity of the banded factorisations against LAPACK."""

from __future__ import annotations

import numpy as np
import pytest

from mojoscipy._lib import addr, f64, i64, lib

LAPACK_CASES = [
    (1, 1, 8), (3, 3, 10), (2, 4, 9), (0, 3, 6), (4, 1, 7), (3, 3, 5),
    (1, 0, 4), (2, 0, 5), (0, 2, 5), (5, 1, 12), (1, 1, 3), (1, 1, 2),
    (6, 6, 20), (1, 7, 9), (0, 0, 6), (2, 3, 7), (3, 1, 9), (5, 2, 15),
]


def _band(A, l, u):
    """A[i, j] -> band row `l + u + i - j` of an `(2l+u+1) x n` array."""
    n = A.shape[0]
    f = np.zeros((2 * l + u + 1, n))
    for i in range(n):
        for j in range(max(i - l, 0), min(i + u + 1, n)):
            f[l + u + i - j, j] = A[i, j]
    return f


def _unband(F, l, u):
    n = F.shape[1]
    kv = l + u
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(i - l, 0), min(i + u + 1, n)):
            A[i, j] = F[kv + i - j, j]
    return A


@pytest.mark.parametrize("l,u,n", LAPACK_CASES)
@pytest.mark.parametrize("dominating", [True, False])
def test_solve_banded_matches_dense(l, u, n, dominating):
    rng = np.random.default_rng(l * 100 + u * 10 + n + dominating)
    A = _unband(_band(rng.standard_normal((n, n)), l, u), l, u)
    if dominating:
        A += n * np.eye(n)
    b = rng.standard_normal((n, 2))
    buf = _band(A, l, u).copy()
    out = np.ascontiguousarray(b.copy())
    ipiv = i64(np.zeros(n, dtype=np.int64))
    info = lib().msi_solve_banded(l, u, addr(buf), n, addr(out), n, 2, addr(ipiv))
    assert info == 0
    assert np.allclose(out, np.linalg.solve(A, b))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20])
@pytest.mark.parametrize("dominating", [True, False])
def test_solve_dense_matches_numpy(n, dominating):
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n))
    if dominating:
        A += n * np.eye(n)
    b = rng.standard_normal((n, 2))
    a_buf = np.ascontiguousarray(A.copy())
    out = np.ascontiguousarray(b.copy())
    ipiv = i64(np.zeros(n, dtype=np.int64))
    info = lib().msi_solve_dense(addr(a_buf), addr(out), n, 2, addr(ipiv))
    assert info == 0
    assert np.allclose(out, np.linalg.solve(A, b))


def test_solve_dense_reports_singularity():
    A = np.zeros((3, 3))
    A[1, 1] = 1.0
    a_buf = A.copy()
    out = np.ones(3)
    ipiv = i64(np.zeros(3, dtype=np.int64))
    info = lib().msi_solve_dense(addr(a_buf), addr(out), 3, 1, addr(ipiv))
    assert info == 1  # LAPACK reports the index of the zero pivot


def test_every_exported_symbol_has_a_ctypes_signature():
    """A missing entry means ctypes truncates the pointers to 32 bits."""
    from mojoscipy import _lib

    handle = lib()
    for name in _lib._SIGNATURES:
        assert hasattr(handle, name)
    for name in dir(handle):
        if name.startswith("msi_"):
            assert name in _lib._SIGNATURES, name
