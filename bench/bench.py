"""mojo-scipy-interpolate against scipy.interpolate on the same data.

Run with `pixi run bench`: the task holds a machine-wide flock so a concurrent
factory job cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojoscipy as msi  # noqa: E402
import scipy.interpolate as si  # noqa: E402


def timeit(fn, repeat: int = 3) -> float:
    """Best-of-`repeat` wall time in milliseconds."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def grid(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


CASES: list = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------ splines

@case("BSpline eval (2M points, k=3)")
def _():
    x = np.linspace(0, 1, 20_000)
    spl_m = msi.make_interp_spline(x, np.sin(3 * x), k=3)
    spl_s = si.make_interp_spline(x, np.sin(3 * x), k=3)
    q = np.linspace(0, 1, 2_000_000)
    return (lambda: spl_m(q), lambda: spl_s(q))


@case("BSpline eval + 1st derivative (2M, k=3)")
def _():
    x = np.linspace(0, 1, 20_000)
    spl_m = msi.make_interp_spline(x, np.sin(3 * x), k=3)
    spl_s = si.make_interp_spline(x, np.sin(3 * x), k=3)
    q = np.linspace(0, 1, 2_000_000)
    return (lambda: spl_m(q, nu=1), lambda: spl_s(q, nu=1))


@case("BSpline design_matrix (200k x 4, k=3)")
def _():
    t = np.r_[[0.0] * 3, np.linspace(0, 1, 500), [1.0] * 3]
    x = np.linspace(0, 1, 200_000)
    return (
        lambda: msi.BSpline.design_matrix(x, t, 3, extrapolate=True).toarray(),
        lambda: si.BSpline.design_matrix(x, t, 3, extrapolate=True).toarray(),
    )


@case("make_interp_spline k=3 (20k points)")
def _():
    x = np.linspace(0, 1, 20_000)
    y = np.sin(3 * x)
    return (lambda: msi.make_interp_spline(x, y, k=3),
            lambda: si.make_interp_spline(x, y, k=3))


@case("make_lsq_spline k=3 (20k points, 4k knots)")
def _():
    x = np.linspace(0, 1, 20_000)
    y = np.sin(3 * x)
    t = np.r_[[0.0] * 3, np.linspace(0, 1, 4_000), [1.0] * 3]
    return (lambda: msi.make_lsq_spline(x, y, t, k=3),
            lambda: si.make_lsq_spline(x, y, t, k=3, method="norm-eq"))


# --------------------------------------------------------------- grid / 1-D

@case("RegularGridInterpolator linear (1M queries, 200x200)")
def _():
    g = np.linspace(0, 1, 200)
    z = np.add.outer(g**2, np.sin(g))
    q = np.random.default_rng(0).random((1_000_000, 2))
    a = msi.RegularGridInterpolator((g, g), z, method="linear")
    b = si.RegularGridInterpolator((g, g), z, method="linear")
    return (lambda: a(q), lambda: b(q))


@case("RegularGridInterpolator cubic (100k queries, 40x40)")
def _():
    g = np.linspace(0, 1, 40)
    z = np.add.outer(g**2, np.sin(g))
    q = np.random.default_rng(0).random((100_000, 2))
    a = msi.RegularGridInterpolator((g, g), z, method="cubic")
    b = si.RegularGridInterpolator((g, g), z, method="cubic")
    return (lambda: a(q), lambda: b(q))


@case("interp1d linear (5M points)")
def _():
    x = np.linspace(0, 10, 5_000)
    y = np.sin(x)
    q = np.linspace(0.1, 9.9, 5_000_000)
    a = msi.interp1d(x, y, kind="linear")
    b = si.interp1d(x, y, kind="linear")
    return (lambda: a(q), lambda: b(q))


# ------------------------------------------------------------------- RBF

@case("RBF kernel matrix (2000 centres, 3-D)")
def _():
    from scipy.interpolate import _rbfinterp_pythran as pythran

    y = np.ascontiguousarray(grid(2000, 3, 1) * 1.5)
    return (
        lambda: msi._rbfinterp._kernel_matrix(y, "thin_plate_spline"),
        lambda: pythran._kernel_matrix(y, "thin_plate_spline"),
    )


@case("RBFInterpolator build (1200 centres, 3-D, TPS)")
def _():
    y = grid(1200, 3, 1)
    d = np.sin(y[:, 0])
    return (lambda: msi.RBFInterpolator(y, d),
            lambda: si.RBFInterpolator(y, d))


@case("RBFInterpolator eval (20k queries, 2000 centres, TPS)")
def _():
    y = grid(2000, 3, 1)
    d = np.sin(y[:, 0])
    q = grid(20_000, 3, 2)
    a = msi.RBFInterpolator(y, d)
    b = si.RBFInterpolator(y, d)
    return (lambda: a(q), lambda: b(q))


@case("RBFInterpolator neighbours=64 (5k queries, 2000 centres)")
def _():
    y = grid(2000, 3, 1)
    d = np.sin(y[:, 0])
    q = grid(5_000, 3, 2)
    a = msi.RBFInterpolator(y, d, neighbors=64)
    b = si.RBFInterpolator(y, d, neighbors=64)
    return (lambda: a(q), lambda: b(q))


@case("Rbf legacy eval (20k queries, 2000 nodes, multiquadric)")
def _():
    coords = grid(3, 2000, 3)
    d = np.sin(coords[0])
    q = grid(3, 20_000, 4)
    a = msi.Rbf(*coords, d, function="multiquadric", epsilon=0.5)
    b = si.Rbf(*coords, d, function="multiquadric", epsilon=0.5)
    return (lambda: a(*q), lambda: b(*q))


def machine() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
            else:
                name = "unknown CPU"
    except OSError:
        name = "unknown CPU"
    return (
        f"{name}, {os.cpu_count()} logical CPUs, "
        f"Linux {platform.release()}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}"
    )


def main() -> None:
    print(f"machine: {machine()}\n")
    header = f"| {'case':<52} | {'mojo-scipy-interpolate':>22} | {'scipy.interpolate':>19} | {'ratio':>7} |"
    print(header)
    print("|" + "-" * 54 + "|" + "-" * 24 + "|" + "-" * 21 + "|" + "-" * 9 + "|")
    for name, build in CASES:
        ours, theirs = build()
        a = timeit(ours)
        b = timeit(theirs)
        ratio = b / a
        note = ""
        if a > b:
            note = " slower"
        print(
            f"| {name:<52} | {a:>19.1f} ms | {b:>16.1f} ms | {ratio:>6.2f}x |{note}"
        )


if __name__ == "__main__":
    main()
