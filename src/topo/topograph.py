# TopoMetry high-level API — the TopOGraph class
#
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
#
"""High-level :class:`TopOGraph` orchestrator.

The user-facing entry point that ties the pipeline together: base graph and
kernel construction, the dual (DM / msDM) spectral scaffold, refined graphs and
2-D projections. Composes the mixins in :mod:`topo._pipeline` and
:mod:`topo.uom`, exposing scikit-learn-style ``fit``/``transform`` plus
``save``/``load`` helpers.
"""

import copy
import gc
import logging
import warnings
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from topo import analysis as _analysis
from topo._pipeline.eigen import EigenBuildMixin
from topo._pipeline.graph import GraphBuildMixin
from topo._pipeline.layout import LayoutBuildMixin
from topo.tpgraph.kernels import Kernel
from topo.uom import UoMMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid kernel versions & data-driven dispatch
# ---------------------------------------------------------------------------

VALID_KERNEL_VERSIONS = frozenset(
    [
        "cknn",
        "fuzzy",
        "bw_adaptive",
        "bw_adaptive_alpha_decaying",
        "bw_adaptive_nbr_expansion",
        "bw_adaptive_alpha_decaying_nbr_expansion",
        "gaussian",
    ]
)

# Each entry maps to the Kernel constructor kwargs that *differ* between versions.
# Common kwargs (metric, n_neighbors, backend, …) are merged at call time.
_KERNEL_CONFIGS: dict[str, dict[str, Any]] = {
    "cknn": dict(
        cknn=True,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "fuzzy": dict(
        cknn=False,
        fuzzy=True,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive_alpha_decaying": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=True,
        sigma=None,
    ),
    "bw_adaptive_nbr_expansion": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=True,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive_alpha_decaying_nbr_expansion": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=True,
        alpha_decaying=True,
        sigma=None,
    ),
    "gaussian": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=False,
        expand_nbr_search=False,
        alpha_decaying=False,
        # sigma is filled dynamically from self.sigma
    ),
}


# ============================================================================
# TopOGraph
# ============================================================================


class TopOGraph(  # pyright: ignore[reportIncompatibleVariableOverride]
    GraphBuildMixin,
    EigenBuildMixin,
    LayoutBuildMixin,
    UoMMixin,
    BaseEstimator,
    TransformerMixin,
):
    """Geometry-aware estimator for spectral scaffolds, operators and layouts.

    Learns spectral scaffolds, refined operators and 2-D layouts from data.

    Use this for standard end-to-end workflows. It orchestrates building
    neighborhood graphs, diffusion operators, spectral scaffolds, and 2-D layouts.

    **Notation Glossary:**
    * `X` = original input data
    * `Z` = fixed-time spectral scaffold
    * `msZ` = multiscale spectral scaffold
    * `P` = diffusion / Markov transition operator
    * `kNN` = sparse nearest-neighbor distance graph
    * `ID` = intrinsic dimensionality

    **Example**
    ```python
    from topo import TopOGraph
    model = TopOGraph(
        base_knn=30,
        graph_knn=30,
        projection_methods=["MAP", "PaCMAP"],
        random_state=42,
    )
    model.fit(X)
    Y = model.msTopoMAP
    Z = model.spectral_scaffold(multiscale=True)
    ```

    Parameters
    ----------
    base_knn : int, default=30
        k-nearest neighbors for the base graph on input space.
    graph_knn : int, default=30
        k-nearest neighbors for the refined graph built in spectral scaffold space.
    min_eigs : int, default=128
        Minimum number of eigenpairs to compute for the scaffold.
    base_kernel : topo.tpgraph.Kernel or None, default None
        Pre-fitted kernel to reuse; if provided, ``fit`` skips base graph construction.
    laplacian_type : {"normalized", "unnormalized", "random_walk"}, default="normalized"
        Laplacian normalization for spectral computations.
    base_kernel_version : {"cknn", "fuzzy", "bw_adaptive", "bw_adaptive_alpha_decaying", "bw_adaptive_nbr_expansion", "bw_adaptive_alpha_decaying_nbr_expansion", "gaussian"}, default="bw_adaptive"
        Kernel choice for the base graph.
    graph_kernel_version : {"cknn", "fuzzy", "bw_adaptive", "bw_adaptive_alpha_decaying", "bw_adaptive_nbr_expansion", "bw_adaptive_alpha_decaying_nbr_expansion", "gaussian"}, default="bw_adaptive"
        Kernel choice for scaffold graphs.
    backend : {"hnswlib", "nmslib", "faiss", "annoy", "sklearn"}, default="hnswlib"
        Approximate nearest-neighbor backend.
    base_metric : str, default="cosine"
        Distance metric for the base kNN graph (e.g., "cosine", "euclidean", or "precomputed").
    graph_metric : str, default="euclidean"
        Distance metric for kNN in scaffold space.
    diff_t : int, default=0
        Diffusion time for single-time scaffold.
    sigma : float, default=0.1
        Bandwidth for Gaussian kernels.
    delta : float, default=1.0
        Radius parameter for cKNN kernels.
    n_jobs : int, default=-1
        Threads for kNN searches; -1 uses all cores.
    low_memory : bool, default=False
        Avoid caching large kernel objects.
    eigen_tol : float, default=1e-8
        Tolerance for the eigensolver.
    eigensolver : {"arpack", "lobpcg", "amg", "dense"}, default="arpack"
        Solver for eigendecomposition.
    projection_methods : list of str, default=["MAP", "PaCMAP"]
        Layouts to compute during ``fit``. Supported strings include "MAP", "PaCMAP", "UMAP", "Isomap", "t-SNE", etc.
    cache : bool, default=True
        Cache kernel / eigen objects in dictionaries for reuse.
    verbosity : int, default=0
        Logging verbosity (0=silent, 1=major, 2+=layout, 3=debug).
    random_state : int or RandomState, default 42
        Random seed.
    id_method : {"fsa", "mle"}, default="fsa"
        Intrinsic-dimensionality estimator for scaffold sizing.
    id_ks : int or iterable, default 50
        Neighborhood sizes for I.D. estimation.
    id_metric : str, default="euclidean"
        Metric for I.D. estimation.
    id_quantile : float, default=0.99
        Quantile of local intrinsic-dimensionality estimates used to choose
        the scaffold dimensionality. Higher values allocate more eigenvectors.
    id_min_components : int, default=128
        Lower bound on the number of spectral components computed, regardless
        of the intrinsic-dimensionality estimate.
    id_max_components : int, default=1024
        Upper bound on the number of spectral components computed.
    id_headroom : float, default=0.5
        Fractional safety margin added to the intrinsic-dimensionality estimate
        when selecting the number of spectral components.
    uom : bool, default False
        Enable Union-of-Manifolds (block-diagonal scaffolds).

    Fitted Attributes
    -----------------
    BaseKernelDict : dict
        Cached base kernels.
    EigenbasisDict : dict
        Cached spectral scaffolds.
    GraphKernelDict : dict
        Cached refined graph kernels.
    ProjectionDict : dict
        Cached 2-D projections.
    global_dimensionality : float
        Global intrinsic dimensionality estimate.
    """

    def __init__(
        self,
        base_knn: int = 30,
        graph_knn: int = 30,
        min_eigs: int = 128,
        n_jobs: int = -1,
        projection_methods: list[str] | None = None,
        base_kernel=None,
        base_kernel_version: str = "bw_adaptive",
        graph_kernel_version: str = "bw_adaptive",
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        diff_t: int = 0,
        delta: float = 1.0,
        sigma: float = 0.1,
        low_memory: bool = False,
        eigen_tol: float = 1e-8,
        eigensolver: str = "arpack",
        backend: str = "hnswlib",
        cache: bool = True,
        verbosity: int = 0,
        random_state=42,
        laplacian_type: str = "normalized",
        # Intrinsic-dimensionality sizing
        id_method: str = "fsa",
        id_ks=50,
        id_metric: str = "euclidean",
        id_quantile: float = 0.99,
        id_min_components: int = 128,
        id_max_components: int = 1024,
        id_headroom: float = 0.5,
        # UoM
        uom: bool = False,
    ):
        if projection_methods is None:
            projection_methods = ["MAP", "PaCMAP"]

        # Core config
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.min_eigs = min_eigs
        self.n_eigs = min_eigs
        self.n_jobs = n_jobs
        self.projection_methods = cast(Any, projection_methods)
        self.base_kernel = base_kernel
        self.base_kernel_version = base_kernel_version
        self.graph_kernel_version = graph_kernel_version
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.diff_t = diff_t
        self.delta = delta
        self.sigma = sigma
        self.low_memory = low_memory
        self.eigen_tol = eigen_tol
        self.eigensolver = eigensolver
        self.backend = backend
        self.cache = cache
        self.verbosity = verbosity
        self.random_state = random_state
        self.laplacian_type = laplacian_type

        # ID config
        self.id_method = id_method
        self.id_ks = id_ks
        self.id_metric = id_metric
        self.id_quantile = id_quantile
        self.id_min_components = id_min_components
        self.id_max_components = id_max_components
        self.id_headroom = id_headroom

        # Fitted state
        self.n = cast(Any, None)
        self.m = cast(Any, None)
        self.n_eigs_ = None
        self.selected_scaffold_components_ = None
        self._backend_resolved = backend
        self._n_jobs_effective = n_jobs
        self._random_state_resolved = None
        self.base_nbrs_class = None
        self.base_knn_graph = None
        self.eigenbasis = None
        self.current_eigenbasis = cast(Any, None)
        self.current_graphkernel = cast(Any, None)
        self.graph_kernel = None
        self.SpecLayout = None
        self.global_dimensionality = None
        self.local_dimensionality = None
        self._id_details: dict[str, Any] = {"mle": None, "fsa": None}
        self._scaffold_components_dm = None
        self._scaffold_components_ms = None

        # Dual-scaffold products
        self._knn_msZ = None
        self._knn_Z = None
        self._kernel_msZ = None
        self._kernel_Z = None

        # MAP snapshots
        self.msTopoMAP_snapshots = []
        self.TopoMAP_snapshots = []

        # Verbosity toggles (derived)
        self.bases_graph_verbose = False
        self.layout_verbose = False

        # Legacy / benchmarking dictionaries
        self.BaseKernelDict = {}
        self.EigenbasisDict = {}
        self.GraphKernelDict = {}
        self.ProjectionDict = {}
        self.LocalScoresDict: dict[str, Any] = {}
        self.RiemannMetricDict: dict[str, Any] = {}
        self.runtimes = {}

        # UoM state (from mixin)
        self._init_uom_state()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self, N_CHAR_MAX=700):
        """Return a short summary of the fitted state and active scaffold."""
        if self.n is None:
            return "TopOGraph (not fitted)"
        parts = [f"TopOGraph with {self.n} samples"]
        if self.m is not None:
            parts[0] += f" × {self.m} features"
        for label, d in [
            ("Base Kernels", self.BaseKernelDict),
            ("Eigenbases", self.EigenbasisDict),
            ("Graph Kernels", self.GraphKernelDict),
            ("Projections", self.ProjectionDict),
        ]:
            if d:
                parts.append(f"  {label}: {', '.join(d.keys())}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Backend / random-state helpers
    # ------------------------------------------------------------------

    def _parse_backend(self):
        from topo._optional import best_ann_backend

        self._backend_resolved = best_ann_backend(self.backend)

    def _parse_random_state(self):
        if self.random_state is None:
            self._random_state_resolved = np.random.RandomState()
        elif isinstance(self.random_state, int):
            self._random_state_resolved = np.random.RandomState(self.random_state)
        else:
            self._random_state_resolved = self.random_state

    # ------------------------------------------------------------------
    # Data-driven kernel builder (Phase 3)
    # ------------------------------------------------------------------

    def _build_kernel(
        self,
        knn,
        n_neighbors,
        kernel_version,
        results_dict,
        prefix="",
        suffix="",
        low_memory=False,
        base=True,
        data_for_expansion=None,
    ) -> tuple[Any, dict]:
        """
        Build a :class:`Kernel` from a kNN graph and a named *kernel_version*.

        Returns ``(kernel, results_dict)``.
        """
        kernel_key = f"{prefix}{kernel_version}{suffix}"
        if kernel_key in results_dict:
            return results_dict[kernel_key], results_dict

        cfg = _KERNEL_CONFIGS[kernel_version].copy()

        # Expansion versions need the original data + correct metric
        if cfg.get("expand_nbr_search"):
            metric = self.base_metric if base else self.graph_metric
            if data_for_expansion is None:
                raise ValueError(
                    f"data_for_expansion is required for kernel version '{kernel_version}'."
                )
            if metric == "precomputed":
                raise ValueError(
                    f"kernel version '{kernel_version}' expands neighbor search and "
                    "therefore requires raw feature data; it cannot be used with "
                    "metric='precomputed'. Use a non-expansion kernel version or pass "
                    "raw data."
                )
            if data_for_expansion.shape[0] != knn.shape[0]:
                raise ValueError(
                    "data_for_expansion must have the same number of rows as the kNN graph."
                )
            fit_input = data_for_expansion
        else:
            metric = "precomputed"
            fit_input = knn

        # Gaussian uses self.sigma
        if kernel_version == "gaussian":
            cfg["sigma"] = self.sigma

        kernel = Kernel(
            metric=metric,
            n_neighbors=n_neighbors,
            pairwise=False,
            backend=self._backend_resolved,
            n_jobs=self._n_jobs_effective,
            laplacian_type=self.laplacian_type,
            semi_aniso=False,
            anisotropy=1.0,
            cache_input=False,
            verbose=self.bases_graph_verbose,
            random_state=self._random_state_resolved,
            **cfg,
        ).fit(fit_input)

        gc.collect()
        if not low_memory:
            results_dict[kernel_key] = kernel
        return kernel, results_dict

    # ------------------------------------------------------------------
    # Intrinsic-dimension sizing
    # ------------------------------------------------------------------

    def fit(self, X=None, **kwargs):
        """Run the full pipeline on ``X``.

        Builds base kNN → base kernel P(X) → dual eigenbases (DM + msDM) →
        refined scaffold graphs → 2-D projections. When ``uom=True``, detects
        disconnected components and builds per-component scaffolds and
        block-diagonal operators.
        """
        self._validate_inputs(X)
        self._setup_environment()
        self._build_base_graph(X, **kwargs)
        self._build_base_kernel(X, **kwargs)

        if self.base_metric != "precomputed":
            sizing_X = X
            if sizing_X is None:
                if self.base_kernel is None or self.base_kernel.X is None:
                    raise ValueError("Input data is required for automated sizing.")
                sizing_X = self.base_kernel.X
            self._automated_sizing(sizing_X)
            if self.verbosity >= 1:
                logger.info(
                    f"Automated sizing → target components: {self._scaffold_components_ms} "
                    f"(n_eigs_={self.n_eigs_})"
                )

        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        if self.uom_enabled:
            return self._fit_uom(X, **kwargs)

        return self._fit_global(X, **kwargs)

    # -- Stage helpers --

    def _validate_inputs(self, X):
        if self.base_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(f"Invalid base_kernel_version: {self.base_kernel_version}")
        if self.graph_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(
                f"Invalid graph_kernel_version: {self.graph_kernel_version}"
            )

        if X is None:
            if self.base_kernel is None:
                raise ValueError("X was not passed and no base_kernel was provided.")
            return

        if not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError("X must be a 2-D array-like or sparse matrix.")

        n_samples, n_features = X.shape
        if n_samples < 3:
            raise ValueError("X must contain at least 3 samples.")
        if n_features < 1:
            raise ValueError("X must contain at least 1 feature.")

        if self.base_metric == "precomputed":
            if n_samples != n_features:
                raise ValueError(
                    "When base_metric='precomputed', X must be a square "
                    "(n_samples, n_samples) sparse or dense graph/distance matrix."
                )
        else:
            if issparse(X):
                if not np.isfinite(X.data).all():
                    raise ValueError("Sparse X contains NaN or infinite values.")
            else:
                X_arr = np.asarray(X)
                if not np.isfinite(X_arr).all():
                    raise ValueError("X contains NaN or infinite values.")

        for name in ("base_knn", "graph_knn"):
            k = int(getattr(self, name))
            if k < 1:
                raise ValueError(f"{name} must be >= 1.")
            if k >= n_samples:
                raise ValueError(
                    f"{name}={k} must be smaller than n_samples={n_samples}."
                )

        max_eigs = max(1, n_samples - 2)
        if self.n_eigs > max_eigs:
            self.n_eigs_ = max_eigs
            warnings.warn(f"Clamped n_eigs to {max_eigs} (n_samples={n_samples})")
        else:
            self.n_eigs_ = int(self.n_eigs)

    def _setup_environment(self):
        from topo._logging import configure

        configure(self.verbosity)
        self._parse_backend()
        self._parse_random_state()
        self._n_jobs_effective = self.n_jobs
        if self.n_jobs == -1:
            try:
                from joblib import cpu_count

                self._n_jobs_effective = cpu_count()
            except Exception:
                pass
        if self.n_eigs_ is None:
            self.n_eigs_ = int(self.n_eigs)
        self.layout_verbose = self.verbosity >= 2
        self.bases_graph_verbose = self.verbosity >= 3

    def spectral_scaffold(self, multiscale: bool = True) -> np.ndarray:
        """Return spectral scaffold coordinates.

        Parameters
        ----------
        multiscale : bool, default=True
            If True, return the multiscale diffusion-map scaffold. If False,
            return the fixed-time diffusion-map scaffold.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Spectral coordinates used to build refined graph operators and layouts.

        Raises
        ------
        AttributeError
            If called before `fit`.
        """
        if self.uom_enabled:
            arr = self.msZ_uom if multiscale else self.Z_uom
            if arr is None:
                raise AttributeError(
                    "UoM scaffold not available. Call .fit(X, uom=True)."
                )
            return arr
        key = f"{'msDM' if multiscale else 'DM'} with {self.base_kernel_version}"
        if key not in self.EigenbasisDict:
            raise AttributeError("Scaffold not found. Call .fit() first.")
        return self.EigenbasisDict[key].transform(X=None)

    # ------------------------------------------------------------------
    # Properties (public API)
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> np.ndarray | dict[str, Any]:
        """Eigenvalues of the active spectral scaffold.

        For diffusion maps (DM/msDM), large values (near 1) indicate smooth,
        globally persistent geometric modes, while values near 0 indicate local noise.
        In UoM mode, returns a dictionary containing the eigenvalues per component.

        Returns
        -------
        evals : ndarray or dict
            The eigenvalues array, or a dict in UoM mode.
        """
        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list:
            mode = getattr(self, "_uom_active_mode", "msDM")
            per_comp = (
                self.uom_eigenvalues_ms_list
                if mode == "msDM"
                else self.uom_eigenvalues_dm_list
            )
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]
            return {"mode": mode, "per_component": per_comp, "component_sizes": sizes}
        if self.current_eigenbasis is None:
            raise AttributeError("Eigenvalues unavailable. Call .fit() first.")
        return self.EigenbasisDict[self.current_eigenbasis].eigenvalues

    @property
    def knn_msZ(self) -> csr_matrix:
        """The k-nearest-neighbors graph built in the msDM scaffold space.

        Represents the refined neighborhood structure after multiscale spectral
        denoising. Used to build the final projection layout.

        Returns
        -------
        knn : scipy.sparse.csr_matrix
            The sparse distance matrix.
        """
        if self.uom_enabled and self.knn_msZ_uom is not None:
            return self.knn_msZ_uom
        if self._knn_msZ is None:
            raise AttributeError("knn_msZ unavailable. Call .fit() first.")
        return self._knn_msZ

    @property
    def knn_Z(self) -> csr_matrix:
        """The k-nearest-neighbors graph built in the fixed-time DM scaffold space.

        Represents the refined neighborhood structure after fixed-time spectral
        denoising.

        Returns
        -------
        knn : scipy.sparse.csr_matrix
            The sparse distance matrix.
        """
        if self.uom_enabled and self.knn_Z_uom is not None:
            return self.knn_Z_uom
        if self._knn_Z is None:
            raise AttributeError("knn_Z unavailable. Call .fit() first.")
        return self._knn_Z

    @property
    def P_of_msZ(self) -> csr_matrix:
        """The diffusion operator (Markov matrix) on the msDM scaffold.

        Captures the random walk transition probabilities on the refined
        multiscale spectral manifold.

        Returns
        -------
        P : scipy.sparse.csr_matrix
            The sparse Markov transition matrix.
        """
        if self.uom_enabled and self.P_of_msZ_uom is not None:
            return (
                self.P_of_msZ_uom
                if isinstance(self.P_of_msZ_uom, csr_matrix)
                else csr_matrix(self.P_of_msZ_uom)
            )
        if self._kernel_msZ is None:
            raise AttributeError("P_of_msZ unavailable. Call .fit() first.")
        P = self._kernel_msZ.P
        return P if isinstance(P, csr_matrix) else csr_matrix(P)

    @P_of_msZ.setter
    def P_of_msZ(self, value) -> None:
        raise AttributeError("P_of_msZ is a read-only fitted property.")

    @property
    def P_of_Z(self) -> csr_matrix:
        """The diffusion operator (Markov matrix) on the fixed-time DM scaffold.

        Returns
        -------
        P : scipy.sparse.csr_matrix
            The sparse Markov transition matrix.
        """
        if self.uom_enabled and self.P_of_Z_uom is not None:
            return (
                self.P_of_Z_uom
                if isinstance(self.P_of_Z_uom, csr_matrix)
                else csr_matrix(self.P_of_Z_uom)
            )
        if self._kernel_Z is None:
            raise AttributeError("P_of_Z unavailable. Call .fit() first.")
        P = self._kernel_Z.P
        return P if isinstance(P, csr_matrix) else csr_matrix(P)

    @P_of_Z.setter
    def P_of_Z(self, value) -> None:
        raise AttributeError("P_of_Z is a read-only fitted property.")

    @property
    def knn_X(self) -> csr_matrix:
        """The base k-nearest-neighbors graph in the original input space.

        Contains the raw neighbor distances before any density correction
        or spectral decomposition is applied.

        Returns
        -------
        knn : scipy.sparse.csr_matrix
            The sparse distance matrix.
        """
        if self.uom_enabled and self.knn_X_uom is not None:
            return self.knn_X_uom
        if self.base_knn_graph is None:
            raise AttributeError("knn_X unavailable. Call .fit() first.")
        return self.base_knn_graph

    @property
    def P_of_X(self) -> csr_matrix:
        """The base diffusion operator on the original input space.

        The density-corrected Markov transition matrix from which the
        spectral scaffolds (eigenbases) are initially decomposed.

        Returns
        -------
        P : scipy.sparse.csr_matrix
            The sparse Markov transition matrix.
        """
        if self.uom_enabled and self.P_of_X_uom is not None:
            return self.P_of_X_uom
        if self.base_kernel is None:
            raise AttributeError("P_of_X unavailable. Call .fit() first.")
        return self.base_kernel.P

    @property
    def global_id(self) -> float:
        """The estimated global intrinsic dimensionality of the dataset.

        A scalar value representing the effective number of continuous dimensions
        required to represent the data manifold, computed using the specified
        `id_method` (e.g., MLE or FSA).

        Returns
        -------
        dim : float
            The global intrinsic dimensionality estimate.
        """
        if self.global_dimensionality is None:
            raise NotFittedError(
                "Global intrinsic dimensionality is unavailable. Call fit(X) first "
                "with an input-space metric that supports automated sizing."
            )
        return float(self.global_dimensionality)

    @property
    def intrinsic_dim(self) -> dict[str, Any]:
        """Structured intrinsic-dimensionality information.

        Returns a dictionary containing the method used, the global ID estimate,
        and per-sample local ID estimates. Useful for identifying regions of
        varying geometric complexity.

        Returns
        -------
        info : dict
            Dictionary with keys 'method', 'global', 'local', and 'details'.
        """
        if self.global_dimensionality is None and self.local_dimensionality is None:
            raise NotFittedError(
                "Intrinsic-dimensionality estimates are unavailable. Call fit(X) first."
            )
        det = (self._id_details or {}).get(self.id_method)
        return {
            "method": self.id_method,
            "global": self.global_dimensionality,
            "local": self.local_dimensionality,
            "selected_components": self.selected_scaffold_components_,
            "details": det,
        }

    # --- Embedding properties ---

    @property
    def TopoMAP(self) -> np.ndarray:
        """2-D MAP layout optimized on the fixed-time DM refined graph.

        A low-dimensional visualization preserving the fuzzy simplicial set
        cross-entropy of the DM spectral scaffold.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, 2)
            The fitted two-dimensional embedding.
        """
        return self._get_projection("MAP", multiscale=False)

    @property
    def msTopoMAP(self) -> np.ndarray:
        """2-D MAP layout optimized on the msDM refined graph.

        A low-dimensional visualization preserving the fuzzy simplicial set
        cross-entropy of the multiscale spectral scaffold. This is typically
        the default and most robust representation.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, 2)
            The fitted two-dimensional embedding.
        """
        return self._get_projection("MAP", multiscale=True)

    @property
    def TopoPaCMAP(self) -> np.ndarray:
        """2-D PaCMAP layout optimized on the fixed-time DM refined graph.

        A low-dimensional visualization preserving the pairwise relationships
        of the DM spectral scaffold.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, 2)
            The fitted two-dimensional embedding.
        """
        return self._get_projection("PaCMAP", multiscale=False)

    @property
    def msTopoPaCMAP(self) -> np.ndarray:
        """2-D PaCMAP layout optimized on the msDM refined graph.

        A low-dimensional visualization preserving the pairwise relationships
        of the multiscale spectral scaffold.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, 2)
            The fitted two-dimensional embedding.
        """
        return self._get_projection("PaCMAP", multiscale=True)

    # ------------------------------------------------------------------
    # Analysis convenience wrappers (delegate to topo.analysis)
    # ------------------------------------------------------------------

    def _select_P_operator(self, which: str = "msZ"):
        """Resolve a fitted diffusion operator by name."""
        which = str(which).lower()
        if which == "x":
            P = self.P_of_X
        elif which == "z":
            P = self.P_of_Z
        elif which == "msz":
            P = self.P_of_msZ
        else:
            raise ValueError("`which` must be one of {'X', 'Z', 'msZ'}.")

        if P is None:
            raise ValueError(
                f"Diffusion operator '{which}' is not available. "
                "Call fit() first, or choose an operator that was computed."
            )
        return P

    def spectral_selectivity(
        self,
        Z=None,
        evals=None,
        multiscale=True,
        use_scaffold_components=True,
        smooth_P=None,
        smooth_t=0,
        out_prefix="spectral",
        return_dict=True,
        **kwargs,
    ):
        """Per-sample spectral selectivity (delegates to ``topo.analysis``).

        Returns
        -------
        result : dict or None
            Dictionary containing selectivity scores.
        """
        if Z is None:
            Z = self.spectral_scaffold(multiscale=multiscale)
        Z = np.asarray(Z)
        if use_scaffold_components and self._scaffold_components_ms is not None:
            Z = Z[:, : int(self._scaffold_components_ms)]
        if evals is None:
            key = f"{'msDM' if multiscale else 'DM'} with {self.base_kernel_version}"
            eigenbasis = self.EigenbasisDict[key]
            ev = np.asarray(eigenbasis.eigenvalues)
            evals = (
                ev[1 : Z.shape[1] + 1]
                if ev.shape[0] >= Z.shape[1] + 1
                else ev[: Z.shape[1]]
            )

        P = self._select_P_operator(smooth_P) if smooth_P else None
        result = _analysis.spectral_selectivity(
            Z,
            evals,
            P=cast(Any, P),
            smooth_t=smooth_t,
            **kwargs,  # type: ignore
        )

        for k, v in result.items():
            self.LocalScoresDict[f"{out_prefix}_{k}"] = v
        return result if return_dict else None

    def filter_signal(self, signal, t: int = 8, which: str = "msZ"):
        """Diffusion-filter a 1-D signal.

        Returns
        -------
        filtered_signal : ndarray
            The smoothed signal.
        """
        return _analysis.filter_signal(signal, self._select_P_operator(which), t)

    def impute(self, X, t: int = 8, which: str = "msZ", **kwargs):
        """Diffusion-based imputation.

        Returns
        -------
        imputed_X : ndarray
            The imputed matrix.
        """
        return _analysis.impute(X, self._select_P_operator(which), t, **kwargs)

    def riemann_diagnostics(self, Y=None, L=None, diffusion_op=None, **kwargs):
        """Riemann metric + deformation scalars.

        Notes
        -----
        This method may require O(n_samples^2) memory if all-pairs distances are
        materialized internally. For large datasets, use landmark mode or avoid.

        Returns
        -------
        metrics : dict
            Dictionary of computed Riemannian metrics and deformation fields.
        """
        if Y is None:
            for prop in ("TopoMAP", "msTopoMAP", "TopoPaCMAP", "msTopoPaCMAP"):
                try:
                    Y = getattr(self, prop)
                    break
                except AttributeError:
                    continue
            if Y is None:
                Y = self.project(projection_method="MAP", multiscale=False)
        if L is None:
            if self.base_kernel is None:
                raise ValueError(
                    "No base kernel available. Call fit() before riemann_diagnostics()."
                )
            L = self.base_kernel.L
        P = self._select_P_operator(diffusion_op) if diffusion_op else None
        result = _analysis.riemann_diagnostics(Y, L, diffusion_op=P, **kwargs)
        self.RiemannMetricDict["last"] = result
        return result

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, filename: str = "topograph.pkl", remove_base_class: bool = True):
        """Save this TopOGraph to a pickle file."""
        save_topograph(self, filename, remove_base_class)

    def spectral_layout(self, *args: Any, **kwargs: Any) -> Any:
        """Disambiguate inherited ``spectral_layout`` implementations.

        Multiple base classes expose an attribute with this name. We delegate
        explicitly to :class:`LayoutBuildMixin` to preserve the current MRO
        behavior while removing ambiguity for static analysis and future
        maintenance.
        """
        return LayoutBuildMixin.spectral_layout(self, *args, **kwargs)


# =========================================================================
# Module-level I/O helpers
# =========================================================================


def save_topograph(
    tg: TopOGraph, filename: str = "topograph.pkl", remove_base_class: bool = True
):
    """Save a TopOGraph object to a pickle file without mutating the live object."""
    import pickle

    if not isinstance(tg, TopOGraph):
        raise TypeError("`tg` must be a TopOGraph instance.")

    obj = copy.copy(tg)
    if remove_base_class:
        obj.base_nbrs_class = None

    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    if getattr(tg, "verbosity", 0) >= 1:
        logger.info(f"TopOGraph saved at {filename}")


def load_topograph(filename: str) -> TopOGraph:
    """Load a TopOGraph from a trusted pickle file.

    Pickle can execute arbitrary code while loading. Only load files from a
    trusted source.
    """
    import pickle

    with open(filename, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TopOGraph):
        raise TypeError(f"Pickle did not contain a TopOGraph: {type(obj)!r}")
    return obj
