"""The legacy `Rbf` class from `scipy.interpolate._rbf`.

The `_h_*` kernels, the pairwise distance matrix and the evaluation dot
product run in Mojo. The dense solve stays in NumPy because upstream hands it
to LAPACK through `scipy.linalg.solve` / `lu_factor`, and a serial Mojo LU
would only be slower.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib, zeros

# `_h_<name>` kernels, in the order `rbf_kernel` expects.
RBF_FUNCTIONS = {
    "multiquadric": 0,
    "inverse": 1,
    "inverse_multiquadric": 1,
    "inverse multiquadric": 1,
    "gaussian": 2,
    "linear": 3,
    "cubic": 4,
    "quintic": 5,
    "thin_plate": 6,
    "thin-plate": 6,
    "thin plate": 6,
}


class Rbf:
    """Class for radial basis function interpolation of functions from N-D
    scattered data to an M-D domain (legacy)."""

    def __init__(self, *args, **kwargs):
        self.xi = np.asarray(
            [np.asarray(a, dtype=np.float64).flatten() for a in args[:-1]]
        )
        self.N = self.xi.shape[-1]

        self.mode = kwargs.pop("mode", "1-D")

        if self.mode == "1-D":
            self.di = np.asarray(args[-1]).flatten()
            self._target_dim = 1
        elif self.mode == "N-D":
            self.di = np.asarray(args[-1])
            self._target_dim = self.di.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        if not all(x.size == self.di.shape[0] for x in self.xi):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop("norm", "euclidean")
        self.epsilon = kwargs.pop("epsilon", None)
        if self.epsilon is None:
            # default epsilon is the "average distance between nodes" based on
            # a bounding hypercube
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges) / self.N, 1.0 / edges.size)

        self.smooth = kwargs.pop("smooth", 0.0)
        self.function = kwargs.pop("function", "multiquadric")

        for item, value in kwargs.items():
            setattr(self, item, value)

        # Compute weights
        A = self.A
        if self._target_dim > 1:
            self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
            import scipy.linalg

            lu, piv = scipy.linalg.lu_factor(A)
            for i in range(self._target_dim):
                self.nodes[:, i] = scipy.linalg.lu_solve((lu, piv), self.di[:, i])
        else:
            self.nodes = np.linalg.solve(A, np.asarray(self.di, dtype=np.float64))

    @property
    def A(self):
        """The collocation matrix, ``kernel(r) - smooth * I``."""
        r = self._pairwise_norm(self.xi.T, self.xi.T)
        return self._init_function(r) - np.eye(self.N) * self.smooth

    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            key = RBF_FUNCTIONS.get(self.function)
            if key is None:
                raise ValueError(
                    "function must be a callable or one of "
                    + ", ".join(sorted(set(RBF_FUNCTIONS)))
                )
            self._kernel_id = key
        elif callable(self.function):
            a0 = self.function(np.asarray(r, dtype=np.float64))
            if np.shape(a0) != np.shape(r):
                raise ValueError(
                    "Callable must take array and return array of the same shape"
                )
            self._callable = self.function
            return np.asarray(a0, dtype=np.float64)
        r = np.ascontiguousarray(r, dtype=np.float64)
        out = zeros(r.size)
        lib().msi_rbf_function_matrix(
            self._kernel_id, float(self.epsilon), addr(r), r.size, addr(out)
        )
        return out.reshape(np.shape(r))

    def _call_norm(self, x1, x2):
        """`cdist(x1, x2, self.norm)` with both arrays as `(npoints, ndim)`."""
        if self.norm != "euclidean":
            raise NotImplementedError(
                "only norm='euclidean' is ported; Rbf(norm=...) with other "
                "metrics is not"
            )
        return self._pairwise_norm(
            np.ascontiguousarray(x1, dtype=np.float64),
            np.ascontiguousarray(x2, dtype=np.float64),
        )

    def _pairwise_norm(self, a, b):
        """`squareform(pdist(a))` for one argument, `cdist(a, b)` for two."""
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        n, d = a.shape
        m = b.shape[0]
        dst = zeros(n * m)
        lib().msi_cdist_euclidean(addr(a), n, d, addr(b), m, addr(dst))
        return np.sqrt(dst.reshape(n, m))

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all(x.shape == y.shape for x in args for y in args):
            raise ValueError("Array lengths must be equal")
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float64)
        r = self._call_norm(xa.T, self.xi.T)
        npoints, ncentres = r.shape
        ncols = 1 if self.nodes.ndim == 1 else self.nodes.shape[1]
        nodes = np.ascontiguousarray(
            self.nodes.reshape(self.N, ncols), dtype=np.float64
        )
        if callable(getattr(self, "_callable", None)):
            # `np.dot(self._function(r), self.nodes)`, with a user kernel
            vals = np.asarray(self._callable(r), dtype=np.float64)
            vals = vals.reshape(npoints, ncentres) @ nodes
        else:
            r = np.ascontiguousarray(r, dtype=np.float64)
            vals = zeros(npoints * ncols)
            lib().msi_rbf_function_dot(
                self._kernel_id, float(self.epsilon), addr(r), npoints,
                ncentres, addr(nodes), ncols, addr(vals),
            )
        return np.asarray(vals, dtype=np.float64).reshape(shp)
