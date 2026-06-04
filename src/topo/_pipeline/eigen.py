"""Eigenbasis / scaffold phase of :class:`topo.topograph.TopOGraph`.

Mixin holding intrinsic-dimension sizing, the dual (DM / msDM) eigenbasis
construction and refined scaffold-graph building. Methods operate on ``self``.
"""

import copy
import logging
import time
from typing import Any, cast

from topo.base.ann import kNN
from topo.spectral.eigen import EigenDecomposition
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing

logger = logging.getLogger(__name__)


class EigenBuildMixin:
    """Intrinsic-dimension sizing, eigenbasis and scaffold-graph construction."""

    def _automated_sizing(self, X):
        n = X.shape[0]
        max_cap = min(int(self.id_max_components), max(2, n - 2))

        res = cast(Any, automated_scaffold_sizing)(
            X,
            method=self.id_method,
            ks=cast(Any, self.id_ks),
            backend=self.backend,
            metric=self.id_metric,
            n_jobs=self.n_jobs,
            quantile=self.id_quantile,
            min_components=int(self.id_min_components),
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=self.random_state,
            return_details=True,
        )
        n_eigs_automated, id_details = cast(tuple[int, dict[str, Any]], res)
        self._id_details[self.id_method] = id_details
        k_sel = int(max(2, min(n_eigs_automated, max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel
        self.n_eigs = int(max(self.n_eigs, k_sel))
        self.global_dimensionality = k_sel
        self.local_dimensionality = id_details.get("local_id_mle", None)

    # ------------------------------------------------------------------
    # fit() — decomposed into stages (Phase 4)
    # ------------------------------------------------------------------

    def _fit_global(self, X, **kwargs):
        """Global (non-UoM) scaffold construction."""
        if self.verbosity >= 1:
            logger.info("Computing eigenbasis → DM/msDM scaffolds...")

        dm_key = f"DM with {self.base_kernel_version}"
        ms_key = f"msDM with {self.base_kernel_version}"

        # Eigendecomposition (shared spectrum, different transforms)
        if dm_key not in self.EigenbasisDict:
            t0 = time.time()
            dm_eig = EigenDecomposition(
                n_components=self.n_eigs,
                method="DM",
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                weight=True,
                t=self.diff_t,
                random_state=self.random_state,
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
        _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)
        self._run_projections()
        return self

    def _build_scaffold_graphs(self, X, dm_eig, ms_eig, dm_key, ms_key, **kwargs):
        """Build kNN and refined kernels in both scaffold spaces."""
        # msZ
        if self.verbosity >= 1:
            logger.info("Computing kNN (msZ space)...")
        t0 = time.time()
        ms_target = ms_eig.transform(X)[:, : self._scaffold_components_ms]
        self._knn_msZ = kNN(
            ms_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs,
        )
        self.runtimes["kNN_msZ"] = time.time() - t0

        # Z (DM)
        if self.verbosity >= 1:
            logger.info("Computing kNN (Z/DM space)...")
        t0 = time.time()
        dm_target = dm_eig.transform(X)[:, : self._scaffold_components_dm]
        self._knn_Z = kNN(
            dm_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
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
            data_for_expansion=ms_eig.transform(X),
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
            data_for_expansion=dm_eig.transform(X),
            base=False,
        )
        self.runtimes["Kernel_Z"] = time.time() - t0

    # ------------------------------------------------------------------
    # Spectral scaffold accessor
    # ------------------------------------------------------------------
