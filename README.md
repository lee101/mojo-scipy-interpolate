# mojo-scipy-interpolate

`scipy.interpolate`'s spline and radial-basis-function interpolators, ported to
Mojo. The public API keeps upstream's names, argument order and defaults, so for
the covered subset the migration is usually a one-line import change.

```python
import numpy as np
import mojoscipy as msi

x = np.linspace(0.0, 1.0, 21)
spl = msi.make_interp_spline(x, np.sin(3.0 * x), k=3)
q = np.linspace(0.0, 1.0, 101)
print(spl(q)[:3])            # [0.         0.02999728 0.0599662 ]
print(spl(q, nu=1)[:3])      # [3.00026322 2.99875139 2.99459237]

g1 = np.linspace(0.0, 1.0, 7)
g2 = np.linspace(0.0, 2.0, 6)
z = np.add.outer(g1**2, np.sin(g2))
rgi = msi.RegularGridInterpolator((g1, g2), z, method="linear")
print(rgi(np.array([[0.3, 0.5]])))    # [0.56584722]

y = np.random.default_rng(0).random((64, 2))
d = np.sin(3.0 * y[:, 0])
rbf = msi.RBFInterpolator(y, d, neighbors=32, epsilon=2.0, kernel="gaussian")
print(rbf(y[:2]))            # [0.94272507 0.12261126] -- close to d at the data points
```

## What this is

`scipy-interpolate` is not a separate distribution: the algorithms named by that
scope line live in **`scipy.interpolate`**, inside SciPy. There is no
`scipy-interpolate` package on conda-forge, so this repo depends on `scipy`
itself and the test suite asserts numerical parity against
`scipy.interpolate` on the same inputs, function by function. Where a
comparison target does not exist upstream (the banded factorisations the
spline builders are built on), the tests compare against LAPACK through NumPy
and SciPy.

The kernels live in `src/ported.mojo`, written to be read side by side with the
upstream files they come from, in upstream's module order:

| section of `src/ported.mojo` | upstream source |
| --- | --- |
| `_deBoor_D`, `find_interval`, `evaluate_spline`, `_colloc`, `_handle_lhs_derivatives`, `_norm_eq_lsq`, `_make_design_matrix` | `scipy/interpolate/_bspl.pyx` and `src/__fitpack.h` |
| `gbtrf`/`gbtrs`/`solve_banded`, `pbtrf`/`pbtrs` | LAPACK `dgbtf2`/`dgbtrs`, `dpbtrf`/`dpbtrs` (what `make_interp_spline` and `make_lsq_spline` call) |
| `solve_dense` | plain LU with partial pivoting |
| `_augknt`, `_not_a_knot`, `_periodic_knots`, `_make_interp_per_full_matr`, `make_interp_spline`, `make_lsq_spline` | `scipy/interpolate/_bsplines.py` |
| `evaluate_poly1`, `find_interval_ascending`, `evaluate` | `scipy/interpolate/_ppoly.pyx`, `_poly_common.pxi` |
| `find_indices`, `upper_bound`, `evaluate_linear_2d`, `evaluate_linear_nd`, `evaluate_tensor_spline` | `scipy/interpolate/_rgi_cython.pyx`, `_rgi.py`, and `make_ndbspl`'s tensor product |
| `linear_interp`, `nearest_interp`, `previous_next` | `scipy/interpolate/_interpolate.py` (`interp1d._call_*`) |
| `rbf`, `kernel_matrix`, `polynomial_matrix`, `build_evaluation_coefficients`, `rbf_build_system`, `neighbors_knn` | `scipy/interpolate/_rbfinterp_pythran.py` and `_rbfinterp.py` |

`src/capi.mojo` is the C ABI the Python bindings call through. It is one thin
wrapper per kernel: every buffer crosses as an `Int` address, because `@export`
rejects parametric functions, and the pointer is rebuilt inside the wrapper.

## Install

```
pixi install
pixi run build
```

`pixi run build` compiles `src/capi.mojo` into
`dist/libmojo-scipy-interpolate.so`. The Mojo toolchain is pinned in
`pixi.toml` (`mojo` and the matching `max`). `pixi run` puts `python/` on
`PYTHONPATH`, so `pixi run python your_script.py` is all the setup there is;
outside pixi, set `PYTHONPATH=python` yourself. The library is also rebuilt on
demand if `src/*.mojo` is newer than it, so the first import after a source
change compiles it.

```
pixi run test     # 205 parity tests against scipy.interpolate
pixi run bench    # benchmark table, under a machine-wide flock
```

`scipy` is a runtime dependency of the covered subset for two reasons, both
upstream's own: the RBF solvers reach LAPACK's `gesv` exactly as
`RBFInterpolator` and `Rbf` do, and `BSpline.design_matrix` returns a
`scipy.sparse` CSR array. Everything else is NumPy plus ctypes.

## What is covered

* **B-splines** -- `BSpline` (with `tck`, `basis_element`, `design_matrix`,
  `derivative`, `antiderivative`, `integrate`, `insert_knot`, `nu`-th
  derivatives, `extrapolate=True|False|'periodic'`), `make_interp_spline`
  (including `bc_type=None|'not-a-knot'|'clamped'|'periodic'` and explicit
  `(deriv_l, deriv_r)` pairs, and `axis != 0`), `make_lsq_spline` with
  `method='norm-eq'`, `splder`, `splantider`, `PPoly`, `PchipInterpolator`,
  `CubicHermiteSpline`.
* **Regular grids** -- `RegularGridInterpolator` with `linear`, `nearest`,
  `slinear`, `cubic`, `quintic` and `pchip`, in any number of dimensions, with
  `bounds_error` / `fill_value` and trailing value axes; `interpn`;
  `RectBivariateSpline`.
* **Legacy 1-D** -- `interp1d` for `linear`, `nearest`, `nearest-up`,
  `previous`, `next`, `zero`, `slinear`, `quadratic`, `cubic` and integer
  orders.
* **Radial basis functions** -- `RBFInterpolator` (all eight kernels,
  `smoothing`, `degree`, `epsilon`, `neighbors`, multi-target `d`), `Rbf`
  (all seven legacy kernels, callables, `epsilon`, `smooth`, `mode='N-D'`),
  and the kernel functions `linear`, `thin_plate_spline`, `cubic`, `quintic`,
  `multiquadric`, `inverse_multiquadric`, `inverse_quadratic`, `gaussian`, plus
  `rbf_factory`.

## What is not covered, and why

Every item below is a **documented divergence**, not a silent difference.

1. **`make_interp_spline(bc_type='periodic')` solves the full system.** Upstream
   reduces it to a cyclic banded system and applies Woodbury for `n > k`; this
   port solves the same `(n + k - 1)`-square system directly. The coefficients
   agree to round-off (tested for `n` in 5..20 and `k` in 2..4), but the periodic
   path is `O(n^3)` instead of `O(n)`, so it is only competitive for small `n`.
   `_woodbury_algorithm` is therefore not part of the port.
2. **The dense linear algebra stays in LAPACK.** `make_interp_spline`'s
   collocation solve, `make_lsq_spline`'s normal equations and the RBF
   coefficient solve are banded/dense factorisations that upstream hands to
   LAPACK (`gbsv`, `cholesky_banded`, `gesv`). A serial Mojo LU would only be
   slower, so only the *band factorisations themselves* were ported
   (`gbtrf`/`gbtrs`/`pbtrf`/`pbtrs`), and the system solve uses LAPACK exactly
   as upstream does. The RBF system matrices are built in Mojo. `gesv` is
   reached through `numpy.linalg.solve` for systems up to
   `_NUMPY_SOLVE_MAX` and through `scipy.linalg.solve` above it; both call the
   same routine, and a singular system falls through to SciPy so that it still
   returns an unchecked result rather than raising. See "One
   toolchain-specific threshold" above for why the split exists.
3. **`make_lsq_spline(method='qr')` raises `NotImplementedError`.** Only the
   `method='norm-eq'` path is ported; `'qr'` needs a different linear algebra
   path (an unweighted QR of the collocation matrix rather than its normal
   equations).
4. **`RectBivariateSpline(s>0)` raises `NotImplementedError`.** The smoothing
   form needs FITPACK's `rectbiv` penalty machinery. The interpolating form
   (`s=0`, any `kx`/`ky`) is a tensor-product B-spline and is fully supported;
   the derivatives `dx`/`dy` are also supported.
5. **`RegularGridInterpolator`'s spline methods differ in the last few digits.**
   Upstream builds a `make_ndbspl` tensor product and solves its collocation
   system with the iterative `gcrotmk` (default `atol=1e-6`); this port folds
   the exact 1-D inverse collocation operator along each axis, so it is the more
   accurate of the two. Agreement is asserted to `1e-4`, which is upstream's own
   tolerance. `pchip` uses the recursive 1-D fold on both sides, as upstream does.
6. **`RBFInterpolator(neighbors=k)` uses a dense k-NN scan**, not
   `scipy.spatial.KDTree`. The neighborhoods and therefore the values agree
   (asserted to `1e-7`). The scan does not shrink with `n` the way a tree does,
   so it is the wrong shape for large `n`, but at the benchmark's 2000 centres
   it is a small fraction of the total.
7. **`Rbf(norm=...)` other than `'euclidean'` raises `NotImplementedError`.** The
   port implements `cdist(..., 'euclidean')`; `scipy.spatial.cdist`'s other
   metrics are not ported.
8. **`splder` / `splantider` pad the coefficient vector to the knot count**
   before recursing, as FITPACK's representation requires. Upstream's public
   `splder` does not pad and therefore raises on an unpadded `(t, c, k)` from
   `make_interp_spline`; on padded input the two agree.
9. **`find_interval` bounds its upward walk at the end of the knot array.**
    Upstream's `l != n` test alone lets a degenerate spline (fewer than `k + 1`
    coefficients, which `BSpline.basis_element` builds for `k > 0`) read one
    element past the knots. The extra bound is a no-op for well-formed splines.
10. **`interp1d` returns multi-target `y` transposed.** For `y` of shape
    `(targets, points)` and `axis=-1` this port returns `(points, targets)`
    and upstream returns `(targets, points)`; the values agree to round-off
    (asserted to `1e-12`, transposed, in
    `test_interp1d_large_query_count`). The one-dimensional case, which is what
    the port's other tests use, is unaffected. This is a property of the
    Python wrapper's final `reshape` and predates the optimisation pass; it is
    recorded here rather than fixed, because changing a public return shape is
    outside what a performance pass should decide.
11. **Not ported at all**: the FITPACK smoothing wrappers (`splrep`, `splev`,
    `spder`, `bisplev`, `dfitpack` and friends), `make_smoothing_spline`,
    `make_ndbspline`/`NdBSpline`, the Clough-Tocher / Akima / cubic families,
    `BPoly`/`MakeInterpSpline`, `interp2d`, `LSQUnivariateSpline`, and the
    `ndgriddata` triangular interpolators. They are outside the spline and RBF
    scope this port targets.
12. **The pinned Mojo miscompiles one loop shape, and one kernel avoids it.** On
    `mojo ==1.2.0.dev2026092605` a nested loop whose *store* index depends on the
    outer loop variable while the *load* index does not is compiled with the
    store's index on the load side, at `-O0` and `-O3` alike. It produced
    wrong coefficients for `make_interp_spline(bc_type='periodic')` with a
    multi-column `y`; the column loop now lives in Python and the kernel is
    called once per column, which costs nothing because the matrix was rebuilt
    and re-solved per column anyway. The rule the rest of the port follows is to
    keep the index expression identical on both sides of a load/store, or to
    hoist the base with `ptr.unsafe_offset(k)` and index from 0.
    `MOJO_NOTES.md` section 8 has the minimal reproduction, and
    `test_make_interp_spline_periodic_multi_column` guards the workaround.

13. **`PPoly.derivative`, `antiderivative` and `integrate` do not match the
    pinned SciPy build.** The compiled `_ppoly.evaluate` in the pinned SciPy
    (1.18.1) evaluates `c` leading coefficient first, while its
    `_PPoly.derivative` and `antiderivative` transform `c` as if it were
    trailing coefficient first -- the two disagree inside SciPy itself (its
    `CubicHermiteSpline(...).derivative()` is not the derivative of the
    function the same object evaluates). This port mirrors the evaluation
    exactly, which is what `PchipInterpolator`, `CubicHermiteSpline` and every
    parity test depend on, and implements `derivative`/`antiderivative`
    consistently with it, so `PPoly.integrate` here is the integral of what
    `PPoly.__call__` returns (checked against the interval-by-interval integral
    in `test_ppoly_matches_upstream`) where the pinned SciPy's is not. Against a
    SciPy whose evaluator and constructors agree, this is the behaviour to
    expect; against this one, `PPoly.integrate` differs by around 1e-4 on a
    six-interval quadratic.

Where upstream indexes something by name and Mojo cannot, the divergence is
confined to the glue: `NAME_TO_FUNC`-style kernel dispatch and `bc_type` become
`Int` selectors resolved in the ctypes layer, and the branch stays inside the
kernel where upstream has it. `src/ported.mojo`'s module docstring says so at
each site.

## How it works

**FFI.** `pixi run build` produces one shared library,
`dist/libmojo-scipy-interpolate.so`, from a single compilation unit
(`src/capi.mojo`, which imports `src/ported.mojo`). Every `@export`ed function
takes `Int` addresses and extents, and rebuilds the typed pointers inside:

```mojo
@export("msi_evaluate_spline")
def msi_evaluate_spline(t: Int, nt: Int, c: Int, ccols: Int, ...) abi("C"):
    evaluate_spline(p(t), nt, p(c), ccols, ..., p(dst), p(work))
```

Buffers are **not** copied: the Python layer takes the address of a
C-contiguous `float64` NumPy array and the kernel writes into it in place. All
scratch is allocated by the caller in `_lib.py`; nothing in Mojo allocates, so
nothing can leak. `python/mojoscipy/_lib.py` owns the argument-type table and
the staleness-checked rebuild.

**Memory layout.** Everything is row-major `float64`, matching NumPy, so a
`(n, d)` point array is simply `n` consecutive `d`-vectors. B-spline coefficients
are `(n_coeff, n_targets)`; the RBF system is `(P + R) x (P + R)` with `P` data
points and `R` polynomial monomials.

Banded matrices keep LAPACK's band layout in a row-major buffer: element
`A(i, j)` of a matrix with `kl` lower and `ku` upper diagonals lives at
`ab[(kl + ku + i - j) * ncols + j]`, so the buffer is
`(2 * kl + ku + 1) * ncols` doubles and `ncols` doubles is the row stride. The
`kl` unused rows below the band are exactly what partial pivoting needs, as in
LAPACK's `dgbtrf`. The lower-banded Cholesky factor of `make_lsq_spline` uses
`ab[(i - j) * ncols + j]`.

**Where the work goes.** The O(n^2 d) and O(Q P d) kernels -- the B-spline
collocation and design matrices, the RBF kernel and polynomial matrices, the
per-query RBF evaluation vector, the grid index search, the RBF distance and
k-NN scans -- run in Mojo, and every loop with an independent outer iteration
is spread over eight workers with `max.algorithm.parallelize` above
`PARALLEL_MIN` rows, so the loop structure is unchanged below it. That is
`kernel_matrix`, `cdist_euclidean`, `neighbors_knn`, `find_indices`,
`evaluate_linear_2d`, `evaluate_tensor_spline` and the three `interp1d`
evaluation paths. A parallel chunk restarts whatever state the kernel carries
between iterations -- the `interp1d` search hint, the grid interval hint --
with a full search for its first row, which costs one binary search per chunk
and keeps every chunk independent. The sequential-recall loops upstream is
structured around (`find_interval`'s `prev_l` walk, the Gaussian elimination
sweeps) are kept sequential because their order carries meaning.

**Why no GPU path.** `max.gpu.host` is available on this toolchain and was not
the reason for anything here. The arithmetic intensity of every kernel in this
library is far below the ~2 flops/byte where a device beats a vectorised CPU
core, and the two densest cases say so plainly: the RBF evaluation vector
does about 10 flops per 32 bytes touched (0.3 flops/byte, and one of those
flops is a `log`), and `interp1d` linear does 4 flops per 32 bytes (0.125).
`kernel_matrix` is the same shape. The kernels that would benefit from a
device -- the RBF evaluation vector at 4.4 s and the legacy `Rbf` evaluation
at 0.8 s -- are bound by transcendental `log`/`sqrt` throughput and by the
round trip of `q x p` coefficients through host memory, both of which a
kernel launch does not fix. The one place a device could pay, `neighbors=k`,
is a few thousand small systems to solve and has no large inner loops to
amortise a launch over. So the answer here is a CPU-only port, and the SIMD work
above is where the flops actually are.

**One toolchain-specific threshold.** `RBFInterpolator._build_and_solve_system`
picks between `numpy.linalg.solve` and `scipy.linalg.solve` by system size.
Both call the same LAPACK `gesv`; NumPy's route skips the finiteness scan, the
sparse-structure probe and the norm estimate SciPy does per call. It only holds
while LAPACK keeps the solve on one thread, and this toolchain's NumPy OpenBLAS
asks for 72 threads and switches to them at `n = 100`. Measured on the machine
below: a 68x68 system costs 0.31 ms through NumPy against 0.57 ms through SciPy,
while a 100x100 system costs 113 ms against 0.65 ms. The threshold is therefore a
measured constant (`_NUMPY_SOLVE_MAX = 96`) and is specific to the pinned
toolchain.

## Benchmarks

Every number below is from one `pixi run bench` on this machine, best of three
per case:

```
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs,
Linux 6.8.0-142-generic, Python 3.13.14, NumPy 2.5.1, SciPy 1.18.1
```

`bench/bench.py` is the source of the table; re-running it regenerates every
figure. The ratio is `scipy.interpolate / mojo-scipy-interpolate`, so above 1.0
is faster.

| case | mojo-scipy-interpolate | scipy.interpolate | ratio |
| --- | --- | --- | --- |
| BSpline eval (2M points, k=3) | 92.4 ms | 121.7 ms | 1.32x |
| BSpline eval + 1st derivative (2M, k=3) | 85.3 ms | 118.1 ms | 1.38x |
| BSpline design_matrix (200k x 4, k=3) | 598.0 ms | 589.2 ms | 0.99x |
| make_interp_spline k=3 (20k points) | 2.2 ms | 4.0 ms | 1.80x |
| make_lsq_spline k=3 (20k points, 4k knots) | 1.5 ms | 24.7 ms | 16.08x |
| RegularGridInterpolator linear (1M queries, 200x200) | 91.9 ms | 192.2 ms | 2.09x |
| RegularGridInterpolator cubic (100k queries, 40x40) | 14.2 ms | 27.2 ms | 1.92x |
| interp1d linear (5M points) | 24.1 ms | 54.5 ms | 2.26x |
| RBF kernel matrix (2000 centres, 3-D) | 78.3 ms | 136.4 ms | 1.74x |
| RBFInterpolator build (1200 centres, 3-D, TPS) | 105.9 ms | 137.8 ms | 1.30x |
| RBFInterpolator eval (20k queries, 2000 centres, TPS) | 4442.9 ms | 5415.1 ms | 1.22x |
| RBFInterpolator neighbours=64 (5k queries, 2000 centres) | 1311.2 ms | 3306.3 ms | 2.52x |
| Rbf legacy eval (20k queries, 2000 nodes, multiquadric) | 791.4 ms | 1610.1 ms | 2.03x |

The SciPy column moves between runs: `RBFInterpolator build` and
`RBFInterpolator neighbours=64` are dominated by upstream's iterative collocation
solve (`gcrotmk`), whose stopping point depends on the data and the thread
count, and both have ranged widely in the runs made while this was written. Read
the mojo column as the stable number and the ratio as the noisy one.

Where the port wins, it wins on kernels upstream runs through a Python-level
loop or a pythran function, and on the searches:

* **`interp1d` linear** carries the previous query's answer as a locality hint
  into the knot search -- `x[hint] >= v` puts the answer at or below `hint` and
  `x[hint] < v` strictly above -- so a sorted query stream, the case `np.interp`
  is built for, is answered by two comparisons. A wrong guess costs one
  comparison, never more steps than the plain binary search. The same hint
  drives `nearest` and `previous`/`next`, and all three are spread over eight
  workers above `PARALLEL_MIN` rows.
* **`RegularGridInterpolator` linear** runs `find_indices` as a branchless
  binary search (the answer is held in a half-open window that halves each step,
  the same `ceil(log2 n)` loads with no branch) with the same locality hint on
  top, and both `find_indices` and `evaluate_linear_2d` are parallelised over
  points.
* **`RegularGridInterpolator` spline methods** contract directly over the
  `prod(k_d + 1)` non-zero basis functions of a tensor-product B-spline -- 16
  corners for a bicubic -- instead of materialising a dense `(m, nk)` basis per
  axis and contracting it with `einsum`. The per-axis knots, degrees, coefficient
  counts and coefficient strides are cached beside the coefficients, so a call
  does no NumPy work at all.
* **`make_lsq_spline`** runs the banded Cholesky in Mojo rather than through a
  LAPACK call, which is where the 16x comes from.
* **`RBFInterpolator(neighbors=k)`** builds each neighbourhood's system in one
  Mojo pass (`rbf_build_system`: shift, scale, the scaled coordinates, the
  kernel block, the polynomial block, the diagonal and the right-hand side)
  instead of a dozen NumPy calls, groups the points by neighbourhood with an
  `argsort` rather than a Python list of lists, and takes the NumPy solve route
  while the systems are small.

`BSpline design_matrix` is at parity and cannot be otherwise: the case ends in
`.toarray()` on a 200000 x 502 matrix, an 800 MB allocation and copy that both
sides pay and neither side controls. The Mojo kernel that fills the CSR arrays
is a small fraction of the row.

`RBFInterpolator eval` is the slowest case on both sides and is not a search
problem at all: it is 20000 evaluations of 2000 thin-plate-spline kernels, each
touching a `log`, which a wider vector unit and a closer memory would both help
more than another thread.

## Tests

`tests/` contains 205 tests, all of which run against `scipy.interpolate` on the
same inputs (or, for the factorisations, against LAPACK through NumPy, or --
where upstream has no equivalent -- against an independent formula):

* `test_linalg.py` -- the banded LU and banded Cholesky against
  `numpy.linalg.solve` across 18 `(kl, ku, n)` band shapes, with and without
  pivoting, the dense LU and its singular-pivot `info`, plus a check that every
  `@export`ed symbol has a ctypes signature declared (a missing entry silently
  truncates the pointers to 32 bits).
* `test_bsplines.py` -- `make_interp_spline` (k = 0..5, all boundary
  conditions, periodic single- and multi-column, `axis != 0`),
  `make_lsq_spline` (k = 1..5, with and
  without weights), `BSpline` evaluation, derivatives, integration, knot
  insertion, `design_matrix`, `basis_element`, `splder`/`splantider`, and the
  input validation.
* `test_grid_and_rbf.py` -- every `RegularGridInterpolator` method in 2-D and
  3-D, trailing value axes, bounds and NaN handling, `interpn`,
  `RectBivariateSpline`, `PchipInterpolator`, `CubicHermiteSpline`, `PPoly`
  (values, `nu`-th derivatives, `extrapolate=False`, `derivative`,
  `antiderivative` and `integrate`), all eleven `interp1d` kinds, all eight
  `RBFInterpolator` kernels, the smoothing/degree/neighbour options, all seven
  legacy `Rbf` kernels plus callables, and the eight kernel functions against
  their closed forms together with `rbf_factory`.

The parallel paths have their own coverage, because `PARALLEL_MIN` is a
threshold and a test that only ever asks for four points only ever exercises
the serial one. `test_interp1d_large_query_count`,
`test_interp1d_unsorted_queries_spanning_the_range` and
`test_interp1d_knot_at_zero` run each `interp1d` kind at 300, 4095, 4096,
4097 and 9000 queries, sorted and shuffled, with a knot at zero -- the last
because `nextafter(0, inf)` is 5e-324 and the vectorised build flushes
denormals, which the port substitutes out.
`test_regular_grid_linear_large_query_count` and
`test_regular_grid_find_indices_on_grid_points` do the same for
`find_indices`/`evaluate_linear_2d`, the second with every query sitting
exactly on a knot, where the branchless search and the locality hint could
disagree. `test_regular_grid_spline_large_query_count`,
`test_regular_grid_spline_trailing_dims` and
`test_regular_grid_spline_nan_and_derivatives` cover the tensor-product
kernel above and below the threshold, with trailing value axes, with NaN
queries, and at `nu = 0, 1, 2` against the exact fold. On the RBF side,
`test_rbf_interpolator_neighbors_shared_neighbourhoods`,
`test_rbf_interpolator_solve_route_both_sides`,
`test_rbf_interpolator_tiny_scale_axis` and
`test_rbf_build_system_degenerate_axis` cover the neighbourhood grouping, both
sides of the solve threshold, a tiny-but-nonzero scale, and the zero scale of
a fully degenerate axis (where upstream's monomial block is rank deficient
and both must raise).

## Licence

MIT, (c) Lee Penkman. See `LICENSE`. The algorithms are SciPy's; this is an
independent port of them and is not affiliated with or endorsed by SciPy.
