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


def copy_eigendecomposition(eig: EigenDecomposition) -> EigenDecomposition:
    """Return an independent copy of a fitted EigenDecomposition object."""
    out: EigenDecomposition = copy.deepcopy(eig)
    return out


def _as_scaffold_array(value: Any, name: str) -> np.ndarray:
    """Return a fitted scaffold transform as a 2-D dense array."""
    if value is None:
        raise RuntimeError(f"{name} transform returned None.")
    if isinstance(value, tuple):
        raise RuntimeError(f"{name} transform returned a tuple, expected a matrix.")

    arr = np.asarray(value)
    if arr.ndim != 2:
        raise RuntimeError(f"{name} transform did not return a 2-D matrix.")
    if arr.shape[0] < 2:
        raise RuntimeError(f"{name} scaffold must contain at least 2 samples.")
    if arr.shape[1] < 1:
        raise RuntimeError(f"{name} scaffold has no usable components.")

    return arr


class EigenBuildMixin:
    """Intrinsic-dimension sizing, eigenbasis and scaffold-graph construction."""

    # Interface contract — attributes supplied by TopOGraph
    id_max_components: int
    id_method: str
    id_ks: int | Sequence[int]
    _backend_resolved: str
    id_metric: str
    _n_jobs_effective: int
    id_quantile: float
    id_min_components: int
    id_headroom: float
    _random_state_resolved: np.random.RandomState
    _id_details: dict[str, Any]
    _scaffold_components_ms: int | None
    _scaffold_components_dm: int | None
    n_eigs: int
    n_eigs_: int | None
    selected_scaffold_components_: int | None
    Z_: np.ndarray | csr_matrix | None
    msZ_: np.ndarray | csr_matrix | None
    evals_Z_: np.ndarray | None
    evals_msZ_: np.ndarray | None
    global_dimensionality: int | float | None
    local_dimensionality: np.ndarray | None
    verbosity: int
    base_kernel_version: str
    eigensolver: str
    eigen_tol: float
    diff_t: int
    bases_graph_verbose: bool
    runtimes: dict[str, float]
    eigenbasis: EigenDecomposition | None
    graph_knn: int
    graph_metric: str
    graph_kernel_version: str
    low_memory: bool
    knn_Z_: csr_matrix | None
    knn_msZ_: csr_matrix | None
    P_Z_: csr_matrix | None
    P_msZ_: csr_matrix | None
    K_Z_: csr_matrix | None
    K_msZ_: csr_matrix | None
    _knn_msZ: csr_matrix | None
    _knn_Z: csr_matrix | None
    base_kernel: Kernel | None

    def _build_kernel(self, *args, **kwargs) -> Kernel:
        raise NotImplementedError

    def spectral_layout(self, *args, **kwargs) -> np.ndarray:
        return cast(Any, super()).spectral_layout(*args, **kwargs)

    def _run_projections(self, *args, **kwargs) -> None:
        return cast(Any, super())._run_projections(*args, **kwargs)

    def _automated_sizing(self, X: np.ndarray | csr_matrix) -> None:
        """Estimate scaffold size from intrinsic-dimensionality diagnostics."""
        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("X must be a 2-D array-like or sparse matrix.")

        n = int(shape[0])
        if n < 3:
            raise ValueError("Automated scaffold sizing requires at least 3 samples.")

        max_cap = min(int(self.id_max_components), max(2, n - 2))
        min_components = min(int(self.id_min_components), max_cap)

        res = cast(Any, automated_scaffold_sizing)(
            X,
            method=self.id_method,
            ks=cast(Any, self.id_ks),
            backend=self._backend_resolved,
            metric=self.id_metric,
            n_jobs=self._n_jobs_effective,
            quantile=float(self.id_quantile),
            min_components=min_components,
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=self._random_state_resolved,
            return_details=True,
        )

        n_eigs_automated, id_details = cast(tuple[int, dict[str, Any]], res)

        self._id_details[self.id_method] = id_details

        k_sel = int(max(2, min(int(n_eigs_automated), max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel
        self.selected_scaffold_components_ = k_sel

        base_n_eigs = self.n_eigs if self.n_eigs_ is None else self.n_eigs_
        self.n_eigs_ = int(max(int(base_n_eigs), k_sel))

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

    def _fit_global(self, X: Any):
        """Build the global non-UoM spectral scaffold pipeline.

        This private pipeline phase assumes that input validation, environment
        setup, base graph construction, base kernel construction, and automated
        scaffold sizing have already run.
        """
        del X
        if self.base_kernel is None:
            raise RuntimeError("Cannot build eigenbasis before base_kernel is fitted.")

        if self.verbosity >= 1:
            logger.info("Computing eigenbasis -> DM/msDM scaffolds...")

        n_components = int(self.n_eigs_ if self.n_eigs_ is not None else self.n_eigs)
        if n_components < 1:
            raise ValueError("n_eigs_ must be >= 1 before eigendecomposition.")

        dm_key = f"DM with {self.base_kernel_version}"

        t0 = time.time()
        dm_eig = EigenDecomposition(
            n_components=n_components,
            method="DM",
            eigensolver=self.eigensolver,
            eigen_tol=self.eigen_tol,
            drop_first=True,
            t=self.diff_t,
            random_state=self._random_state_resolved,
            verbose=self.bases_graph_verbose,
        ).fit(self.base_kernel)
        self.runtimes[dm_key] = time.time() - t0

        if self.verbosity >= 1:
            logger.info("  DM eigenpairs in %.3fs", self.runtimes[dm_key])

        # The msDM object reuses the fitted decomposition but changes the transform
        # mode. If EigenDecomposition later gains a dedicated clone/copy method, use
        # that instead of relying on this internal object copy.
        ms_eig = copy_eigendecomposition(dm_eig)
        ms_eig.method = "msDM"

        self.evals_Z_ = np.asarray(dm_eig.eigenvalues, dtype=float)
        self.evals_msZ_ = np.asarray(ms_eig.eigenvalues, dtype=float)
        self.eigenbasis = ms_eig

        self._build_scaffold_graphs(dm_eig, ms_eig)

        if self.P_msZ_ is None or self.K_msZ_ is None:
            raise RuntimeError("msDM scaffold operator/affinity was not built.")
        if self.P_Z_ is None or self.K_Z_ is None:
            raise RuntimeError("DM scaffold operator/affinity was not built.")

        _ = self.spectral_layout(graph=self.K_msZ_, n_components=2)
        self._run_projections()

        return self

    def _build_scaffold_graphs(
        self,
        dm_eig: EigenDecomposition,
        ms_eig: EigenDecomposition,
    ) -> None:
        """Build kNN graphs and refined kernels in both scaffold spaces."""
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

        ms_coords = _as_scaffold_array(ms_eig.transform(X=None), "msDM")
        dm_coords = _as_scaffold_array(dm_eig.transform(X=None), "DM")

        if ms_coords.shape[0] != dm_coords.shape[0]:
            raise RuntimeError("DM and msDM scaffolds have different row counts.")

        n_samples = int(ms_coords.shape[0])
        if int(self.graph_knn) >= n_samples:
            raise ValueError(
                f"graph_knn={self.graph_knn} must be smaller than scaffold "
                f"sample count={n_samples}."
            )

        ms_components = min(int(ms_components), int(ms_coords.shape[1]))
        dm_components = min(int(dm_components), int(dm_coords.shape[1]))

        if ms_components < 1:
            raise RuntimeError("msDM scaffold has no usable components.")
        if dm_components < 1:
            raise RuntimeError("DM scaffold has no usable components.")

        self._scaffold_components_ms = ms_components
        self._scaffold_components_dm = dm_components
        self.selected_scaffold_components_ = max(ms_components, dm_components)

        ms_target = ms_coords[:, :ms_components]
        dm_target = dm_coords[:, :dm_components]

        self.msZ_ = ms_target
        self.Z_ = dm_target

        if self.verbosity >= 1:
            logger.info("Computing kNN (msZ space)...")

        t0 = time.time()
        self._knn_msZ = kNN(
            ms_target,
            n_neighbors=int(self.graph_knn),
            metric=self.graph_metric,
            n_jobs=self._n_jobs_effective,
            backend=self._backend_resolved,
            verbose=self.bases_graph_verbose,
        )
        self.runtimes["kNN_msZ"] = time.time() - t0
        self.knn_msZ_ = self._knn_msZ

        if self.verbosity >= 1:
            logger.info("Computing kNN (Z/DM space)...")

        t0 = time.time()
        self._knn_Z = kNN(
            dm_target,
            n_neighbors=int(self.graph_knn),
            metric=self.graph_metric,
            n_jobs=self._n_jobs_effective,
            backend=self._backend_resolved,
            verbose=self.bases_graph_verbose,
        )
        self.runtimes["kNN_Z"] = time.time() - t0
        self.knn_Z_ = self._knn_Z

        t0 = time.time()
        kernel_msZ = self._build_kernel(
            self._knn_msZ,
            int(self.graph_knn),
            self.graph_kernel_version,
            data_for_expansion=ms_target,
            base=False,
        )
        self.runtimes["Kernel_msZ"] = time.time() - t0
        self.P_msZ_ = csr_matrix(kernel_msZ.P)
        self.K_msZ_ = csr_matrix(kernel_msZ.K)

        t0 = time.time()
        kernel_Z = self._build_kernel(
            self._knn_Z,
            int(self.graph_knn),
            self.graph_kernel_version,
            data_for_expansion=dm_target,
            base=False,
        )
        self.runtimes["Kernel_Z"] = time.time() - t0
        self.P_Z_ = csr_matrix(kernel_Z.P)
        self.K_Z_ = csr_matrix(kernel_Z.K)
