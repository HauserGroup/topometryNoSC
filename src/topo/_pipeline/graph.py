"""Base graph / base kernel phase of :class:`topo.topograph.TopOGraph`.

Mixin (see :class:`topo.uom.UoMMixin`) holding the neighbourhood-graph and base
kernel construction steps. Methods operate on ``self`` and call the shared
``self._build_kernel`` helper defined on ``TopOGraph``.
"""

import logging
import time

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
    backend: str
    bases_graph_verbose: bool
    runtimes: dict[str, float]
    base_kernel_version: str
    low_memory: bool
    BaseKernelDict: dict[str, Kernel]
    base_kernel: Kernel | None
    base_nbrs_class: BaseEstimator | None
    base_knn_graph: csr_matrix | None

    def _build_kernel(self, *args, **kwargs) -> tuple[Kernel, dict[str, Kernel]]:
        raise NotImplementedError

    def _build_base_graph(self, X: np.ndarray | csr_matrix | None, **kwargs):
        if X is None:
            if self.base_kernel is None:
                raise ValueError("X was not passed and no base_kernel provided.")
            if not isinstance(self.base_kernel, Kernel):
                raise ValueError("base_kernel must be a topo.tpgraph.Kernel instance.")
            if self.base_kernel.knn_ is None:
                raise ValueError("base_kernel has not been fitted.")
            self.n = self.base_kernel.knn_.shape[0]
            self.m = self.base_kernel.knn_.shape[1]
            self.base_knn_graph = self.base_kernel.knn_
        else:
            shape = X.shape
            if shape is None:
                raise ValueError("X must have a 2-D shape.")
            self.n, self.m = int(shape[0]), int(shape[1])
            if self.base_metric == "precomputed":
                if self.n != self.m:
                    raise ValueError(
                        "When base_metric='precomputed', X must be square."
                    )
                self.base_knn_graph = as_csr_matrix(X, "base kNN graph", copy=True)

        if self.base_knn_graph is None:
            if X is None:
                raise ValueError("X was not passed and no base graph could be built.")
            if self.verbosity >= 1:
                logger.info("Computing neighborhood graph (X space)...")
            t0 = time.time()
            self.base_nbrs_class, self.base_knn_graph = kNN(
                X,
                n_neighbors=self.base_knn,
                metric=self.base_metric,
                n_jobs=getattr(self, "_n_jobs_effective", self.n_jobs),
                backend=getattr(self, "_backend_resolved", self.backend),
                return_instance=True,
                verbose=self.bases_graph_verbose,
                **kwargs,
            )
            self.runtimes["kNN_X"] = time.time() - t0
            if self.verbosity >= 1:
                logger.info(f"  Base kNN computed in {self.runtimes['kNN_X']:.3f}s")

    def _build_base_kernel(self, X, **kwargs):
        if self.base_kernel_version in self.BaseKernelDict:
            self.base_kernel = self.BaseKernelDict[self.base_kernel_version]
        else:
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
            if self.verbosity >= 1:
                logger.info(
                    f"  Base kernel ({self.base_kernel_version}) in {self.runtimes['Kernel_X']:.3f}s"
                )
