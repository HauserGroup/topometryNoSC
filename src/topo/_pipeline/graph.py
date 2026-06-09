"""Base graph / base kernel phase of :class:`topo.topograph.TopOGraph`.

Mixin (see :class:`topo.uom.UoMMixin`) holding the neighbourhood-graph and base
kernel construction steps. Methods operate on ``self`` and call the shared
``self._build_kernel`` helper defined on ``TopOGraph``.
"""

import logging
import time
from typing import cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from topo.base.ann import kNN
from topo.base.graph_matrix import as_csr_matrix
from topo.tpgraph.kernels import Kernel

logger = logging.getLogger(__name__)


class GraphBuildMixin:
    """Base neighbourhood graph and base kernel construction."""

    # Interface contract — attributes supplied by TopOGraph
    n: int | None
    m: int | None
    verbosity: int
    base_knn: int
    base_metric: str
    n_jobs: int
    _n_jobs_effective: int
    _backend_resolved: str
    bases_graph_verbose: bool
    runtimes: dict[str, float]
    base_kernel_version: str
    low_memory: bool
    BaseKernelDict: dict[str, Kernel]
    base_kernel: Kernel | None
    base_nbrs_class: BaseEstimator | None
    base_knn_graph: csr_matrix | None
    knn_X_: csr_matrix | None
    P_X_: csr_matrix | None

    def _build_kernel(self, *args, **kwargs) -> tuple[Kernel, dict[str, Kernel]]:
        raise NotImplementedError

    def _build_base_graph(self, X: np.ndarray | csr_matrix | None) -> None:
        """Build or reuse the base kNN graph in input space.

        If ``X`` is None, a fitted ``base_kernel`` must be available and its fitted
        kNN graph is reused. If ``base_metric='precomputed'``, ``X`` itself is
        treated as the square base graph/distance matrix. Otherwise, exact/HNSW
        kNN construction is delegated to ``topo.base.ann.kNN``.
        """
        if X is None:
            if self.base_kernel is None:
                raise ValueError("X was not passed and no base_kernel was provided.")
            if not isinstance(self.base_kernel, Kernel):
                raise ValueError("base_kernel must be a topo.tpgraph.Kernel instance.")
            if self.base_kernel.knn_ is None:
                raise ValueError("base_kernel has not been fitted or lacks `knn_`.")

            knn = cast(
                csr_matrix,
                as_csr_matrix(self.base_kernel.knn_, "base_kernel.knn_", copy=True),
            )
            if knn.shape[0] != knn.shape[1]:  # type: ignore[reportOptionalSubscript]
                raise ValueError("base_kernel.knn_ must be a square graph.")

            self.n = int(knn.shape[0])  # type: ignore[reportOptionalSubscript]
            kernel_X = getattr(self.base_kernel, "X", None)
            if (
                kernel_X is not None
                and hasattr(kernel_X, "shape")
                and len(kernel_X.shape) == 2
            ):
                self.m = int(kernel_X.shape[1])
            else:
                self.m = int(knn.shape[1])  # type: ignore[reportOptionalSubscript]

            self.base_knn_graph = knn
            self.knn_X_ = self.base_knn_graph
            return

        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("X must be a 2-D array-like or sparse matrix.")

        self.n, self.m = int(shape[0]), int(shape[1])

        if self.base_metric == "precomputed":
            if self.n != self.m:
                raise ValueError("When base_metric='precomputed', X must be square.")
            self.base_knn_graph = as_csr_matrix(X, "base precomputed graph", copy=True)
            self.knn_X_ = self.base_knn_graph
            return

        if self.verbosity >= 1:
            logger.info("Computing neighborhood graph (X space)...")

        t0 = time.time()
        self.base_nbrs_class, self.base_knn_graph = kNN(
            X,
            n_neighbors=self.base_knn,
            metric=self.base_metric,
            n_jobs=self._n_jobs_effective,
            backend=self._backend_resolved,
            return_instance=True,
            verbose=self.bases_graph_verbose,
        )
        self.runtimes["kNN_X"] = time.time() - t0
        self.knn_X_ = self.base_knn_graph

        if self.verbosity >= 1:
            logger.info("  Base kNN computed in %.3fs", self.runtimes["kNN_X"])

    def _build_base_kernel(self, X) -> None:
        """Build or reuse the base diffusion/kernel operator on input space."""
        if self.base_kernel is not None:
            if not isinstance(self.base_kernel, Kernel):
                raise ValueError("base_kernel must be a topo.tpgraph.Kernel instance.")
            if getattr(self.base_kernel, "P", None) is None:
                raise ValueError("base_kernel exists but does not expose fitted `P`.")
            self.P_X_ = csr_matrix(self.base_kernel.P)
            return

        if self.base_kernel_version in self.BaseKernelDict:
            self.base_kernel = self.BaseKernelDict[self.base_kernel_version]
            if getattr(self.base_kernel, "P", None) is None:
                raise RuntimeError(
                    f"Cached base kernel {self.base_kernel_version!r} is not fitted."
                )
            self.P_X_ = csr_matrix(self.base_kernel.P)
            return

        if self.base_knn_graph is None:
            self._build_base_graph(X)

        if self.base_knn_graph is None:
            raise RuntimeError("Cannot build base kernel before base kNN graph exists.")

        t0 = time.time()
        self.base_kernel, self.BaseKernelDict = self._build_kernel(
            self.base_knn_graph,
            self.base_knn,
            self.base_kernel_version,
            self.BaseKernelDict,
            low_memory=self.low_memory,
            data_for_expansion=X,
            base=True,
        )
        self.runtimes["Kernel_X"] = time.time() - t0
        self.P_X_ = csr_matrix(self.base_kernel.P)

        if self.verbosity >= 1:
            logger.info(
                "  Base kernel (%s) in %.3fs",
                self.base_kernel_version,
                self.runtimes["Kernel_X"],
            )
