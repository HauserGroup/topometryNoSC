"""Eigenbasis / scaffold phase of :class:`topo.topograph.TopOGraph`.

Mixin holding intrinsic-dimension sizing, the dual (DM / msDM) eigenbasis
construction and refined scaffold-graph building. Methods operate on ``self``.
"""

import copy
import logging
import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from topo.base.ann import kNN
from topo.spectral.eigen import EigenDecomposition
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing
from topo.tpgraph.kernels import Kernel

logger = logging.getLogger(__name__)


class EigenBuildMixin:
    """Intrinsic-dimension sizing, eigenbasis and scaffold-graph construction."""

    # Interface contract — attributes supplied by TopOGraph
    id_max_components: int
    id_method: str
    id_ks: int | Sequence[int]
    backend: str
    id_metric: str
    n_jobs: int
    id_quantile: float
    id_min_components: int
    id_headroom: float
    random_state: int | np.random.RandomState | None
    _id_details: dict[str, Any]
    _scaffold_components_ms: int | None
    _scaffold_components_dm: int | None
    n_eigs: int
    n_eigs_: int | None
    selected_scaffold_components_: int | None
    global_dimensionality: int | float | None
    local_dimensionality: np.ndarray | None
    verbosity: int
    base_kernel_version: str
    EigenbasisDict: dict[str, EigenDecomposition]
    eigensolver: str
    eigen_tol: float
    diff_t: int
    bases_graph_verbose: bool
    runtimes: dict[str, float]
    current_eigenbasis: str | None
    eigenbasis: EigenDecomposition | None
    graph_knn: int
    graph_metric: str
    graph_kernel_version: str
    GraphKernelDict: dict[str, Kernel]
    low_memory: bool
    graph_kernel: Kernel | None
    current_graphkernel: str | None
    _knn_msZ: csr_matrix | None
    _knn_Z: csr_matrix | None
    _kernel_msZ: Kernel | None
    _kernel_Z: Kernel | None
    base_kernel: Kernel | None

    def _build_kernel(self, *args, **kwargs) -> tuple[Kernel, dict[str, Kernel]]:
        raise NotImplementedError

    def spectral_layout(self, *args, **kwargs) -> np.ndarray:
        return cast(Any, super()).spectral_layout(*args, **kwargs)

    def _run_projections(self, *args, **kwargs) -> None:
        return cast(Any, super())._run_projections(*args, **kwargs)

    def _automated_sizing(self, X: np.ndarray | csr_matrix):
        shape = X.shape
        if shape is None:
            raise ValueError("X must have a 2-D shape.")
        n = int(shape[0])
        max_cap = min(int(self.id_max_components), max(2, n - 2))

        res = cast(Any, automated_scaffold_sizing)(
            X,
            method=self.id_method,
            ks=cast(Any, self.id_ks),
            backend=getattr(self, "_backend_resolved", self.backend),
            metric=self.id_metric,
            n_jobs=getattr(self, "_n_jobs_effective", self.n_jobs),
            quantile=self.id_quantile,
            min_components=int(self.id_min_components),
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=getattr(self, "_random_state_resolved", self.random_state),
            return_details=True,
        )
        n_eigs_automated, id_details = cast(tuple[int, dict[str, Any]], res)
        self._id_details[self.id_method] = id_details
        k_sel = int(max(2, min(n_eigs_automated, max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel
        self.selected_scaffold_components_ = k_sel
        base_n_eigs = self.n_eigs if self.n_eigs_ is None else self.n_eigs_
        self.n_eigs_ = int(max(base_n_eigs, k_sel))
        if "global_id" in id_details:
            self.global_dimensionality = id_details["global_id"]
        elif "quantile_value" in id_details:
            self.global_dimensionality = id_details["quantile_value"]
        else:
            self.global_dimensionality = None
        self.local_dimensionality = id_details.get("local_id", None)

    # ------------------------------------------------------------------
    # fit() — decomposed into stages (Phase 4)
    # ------------------------------------------------------------------

    def _fit_global(self, X: Any, **kwargs: Any):
        """Global (non-UoM) scaffold construction."""
        if self.verbosity >= 1:
            logger.info("Computing eigenbasis → DM/msDM scaffolds...")

        dm_key = f"DM with {self.base_kernel_version}"
        ms_key = f"msDM with {self.base_kernel_version}"

        # Eigendecomposition (shared spectrum, different transforms)
        if dm_key not in self.EigenbasisDict:
            t0 = time.time()
            dm_eig = EigenDecomposition(
                n_components=self.n_eigs_ if self.n_eigs_ is not None else self.n_eigs,
                method="DM",
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                weight=True,
                t=self.diff_t,
                random_state=getattr(self, "_random_state_resolved", self.random_state),
                verbose=self.bases_graph_verbose,
            ).fit(self.base_kernel)
            self.EigenbasisDict[dm_key] = dm_eig
            self.runtimes[dm_key] = time.time() - t0
            if self.verbosity >= 1:
                logger.info(f"  DM/msDM eigenpairs in {self.runtimes[dm_key]:.3f}s")
        else:
            dm_eig = self.EigenbasisDict[dm_key]

        if ms_key not in self.EigenbasisDict:
            ms_eig = copy.deepcopy(dm_eig)
            ms_eig.method = "msDM"
            self.EigenbasisDict[ms_key] = ms_eig
        else:
            ms_eig = self.EigenbasisDict[ms_key]

        self.current_eigenbasis = ms_key
        self.eigenbasis = self.EigenbasisDict[ms_key]

        # Scaffold-space kNN + refined kernels
        self._build_scaffold_graphs(X, dm_eig, ms_eig, dm_key, ms_key, **kwargs)

        self.graph_kernel = self._kernel_msZ
        self.current_graphkernel = f"{self.graph_kernel_version} from {ms_key}"

        # Spectral layout + projections
        if self._kernel_msZ is None:
            raise RuntimeError("msDM scaffold kernel was not built.")
        _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)
        self._run_projections()
        return self

    def _build_scaffold_graphs(
        self, X: Any, dm_eig: Any, ms_eig: Any, dm_key: str, ms_key: str, **kwargs: Any
    ):
        """Build kNN and refined kernels in both scaffold spaces."""
        ms_components = self._scaffold_components_ms
        if ms_components is None:
            ms_components = int(
                self.n_eigs_ if self.n_eigs_ is not None else self.n_eigs
            )
        dm_components = self._scaffold_components_dm
        if dm_components is None:
            dm_components = int(
                self.n_eigs_ if self.n_eigs_ is not None else self.n_eigs
            )

        # msZ
        if self.verbosity >= 1:
            logger.info("Computing kNN (msZ space)...")
        t0 = time.time()
        ms_target = ms_eig.transform(X)[:, :ms_components]
        self._knn_msZ = kNN(
            ms_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=getattr(self, "_n_jobs_effective", self.n_jobs),
            backend=getattr(self, "_backend_resolved", self.backend),
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs,
        )
        self.runtimes["kNN_msZ"] = time.time() - t0

        # Z (DM)
        if self.verbosity >= 1:
            logger.info("Computing kNN (Z/DM space)...")
        t0 = time.time()
        dm_target = dm_eig.transform(X)[:, :dm_components]
        self._knn_Z = kNN(
            dm_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=getattr(self, "_n_jobs_effective", self.n_jobs),
            backend=getattr(self, "_backend_resolved", self.backend),
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs,
        )
        self.runtimes["kNN_Z"] = time.time() - t0

        # Refined kernels
        t0 = time.time()
        self._kernel_msZ, self.GraphKernelDict = self._build_kernel(
            self._knn_msZ,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=f" from {ms_key}",
            low_memory=self.low_memory,
            data_for_expansion=ms_target,
            base=False,
        )
        self.runtimes["Kernel_msZ"] = time.time() - t0

        t0 = time.time()
        self._kernel_Z, self.GraphKernelDict = self._build_kernel(
            self._knn_Z,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=f" from {dm_key}",
            low_memory=self.low_memory,
            data_for_expansion=dm_target,
            base=False,
        )
        self.runtimes["Kernel_Z"] = time.time() - t0

    # ------------------------------------------------------------------
    # Spectral scaffold accessor
    # ------------------------------------------------------------------
