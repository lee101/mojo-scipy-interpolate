"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOSCIPYINTERPOLATE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-scipy-interpolate.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "msi_evaluate_spline": ([I] * 11, None),
    "msi_make_design_matrix": ([I, I, I, I, I, I, I, I, I], None),
    "msi_solve_banded": ([I] * 8, I),
    "msi_make_interp_spline": ([I] * 21, I),
    "msi_periodic_scratch": ([I, I], I),
    "msi_solve_dense": ([I] * 5, I),
    "msi_make_lsq_spline": ([I] * 13, I),
    "msi_evaluate": ([I, I, I, I, I, I, I, I, I, I], None),
    "msi_fix_continuity": ([I] * 7, None),
    "msi_ppoly_integrate": ([I, I, I, I, I, I, F, F, I, I], None),
    "msi_find_indices": ([I] * 7, None),
    "msi_evaluate_linear_2d": ([I] * 7, None),
    "msi_evaluate_tensor_spline": ([I] * 13, None),
    "msi_tensor_spline_wstride": ([I, I], I),
    "msi_evaluate_linear_nd": ([I] * 9, None),
    "msi_linear_interp": ([I] * 7, None),
    "msi_nearest_interp": ([I] * 8, None),
    "msi_previous_next": ([I] * 10, None),
    "msi_rbf_function_matrix": ([I, F, I, I, I], None),
    "msi_rbf_function_dot": ([I, F, I, I, I, I, I, I], None),
    "msi_cdist_euclidean": ([I] * 6, None),
    "msi_rbf_build_system": ([I, I, I, F, I] + [I] * 12, None),
    "msi_kernel_matrix": ([I] * 5, None),
    "msi_polynomial_matrix": ([I] * 7, None),
    "msi_neighbors_knn": ([I] * 8, None),
    "msi_build_evaluation_coefficients": ([I] * 6 + [F] + [I] * 7, None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOSCIPYINTERPOLATE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOSCIPYINTERPOLATE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-scipy-interpolate.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    env = dict(os.environ)
    if "MODULAR_HOME" not in env:
        mojo = shutil.which("mojo")
        if mojo:
            env["MODULAR_HOME"] = os.path.join(
                os.path.dirname(os.path.dirname(mojo)), "share", "max"
            )
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = (
        np.array(a, dtype=np.float64, order="C", copy=True)
        if copy
        else np.ascontiguousarray(a, dtype=np.float64)
    )
    return arr


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    """The data address of `a`.

    The caller must keep `a` alive across the FFI call: a temporary such as
    `addr(f64(x))` is released as soon as this returns, so bind the array to
    a local first.
    """
    return a.ctypes.data


def zeros(n: int) -> np.ndarray:
    return np.zeros(max(int(n), 1), dtype=np.float64)


def main() -> int:
    """`python -m mojoscipy._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
