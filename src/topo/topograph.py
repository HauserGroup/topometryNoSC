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

    Parameters
    ----------
    base_knn : int, default 30
        k-nearest neighbors for the base graph on input space.
    graph_knn : int, default 30
        k-nearest neighbors for the refined graph built in spectral scaffold space.
    min_eigs : int, default 128
        Minimum number of eigenpairs to compute for the scaffold.
    base_kernel : topo.tpgraph.Kernel or None, default None
        Pre-fitted kernel to reuse; if provided, ``fit`` skips base graph construction.
    laplacian_type : str, default 'normalized'
        Laplacian normalization for spectral computations.
    base_kernel_version : str, default 'bw_adaptive'
        Kernel choice for the base graph.
    graph_kernel_version : str, default 'bw_adaptive'
        Kernel choice for scaffold graphs.
    backend : str, default 'hnswlib'
        Approximate nearest-neighbor backend.
    base_metric : str, default 'cosine'
        Distance metric for the base kNN graph.
    graph_metric : str, default 'euclidean'
        Distance metric for kNN in scaffold space.
    diff_t : int, default 0
        Diffusion time for single-time scaffold.
    sigma : float, default 0.1
        Bandwidth for Gaussian kernels.
    delta : float, default 1.0
        Radius parameter for cKNN kernels.
    n_jobs : int, default -1
        Threads for kNN searches; -1 uses all cores.
    low_memory : bool, default False
        Avoid caching large kernel objects.
    eigen_tol : float, default 1e-8
        Tolerance for the eigensolver.
    eigensolver : str, default 'arpack'
        Solver for eigendecomposition.
    projection_methods : list[str], default ['MAP', 'PaCMAP']
        Layouts to compute during ``fit``.
    cache : bool, default True
        Cache kernel / eigen objects in dictionaries for reuse.
    verbosity : int, default 0
        Logging verbosity (0=silent, 1=major, 2+=layout, 3=debug).
    random_state : int or RandomState, default 42
        Random seed.
    id_method : str, default 'fsa'
        Intrinsic-dimensionality estimator for scaffold sizing.
    id_ks : int or iterable, default 50
        Neighborhood sizes for I.D. estimation.
    id_metric : str, default 'euclidean'
        Metric for I.D. estimation.
    id_quantile : float, default 0.99
    id_min_components : int, default 128
    id_max_components : int, default 1024
    id_headroom : float, default 0.5
    uom : bool, default False
        Enable Union-of-Manifolds (block-diagonal scaffolds).
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

        self.backend = best_ann_backend(self.backend)

    def _parse_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)

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
        else:
            metric = "precomputed"

        # Gaussian uses self.sigma
        if kernel_version == "gaussian":
            cfg["sigma"] = self.sigma

        kernel = Kernel(
            metric=metric,
            n_neighbors=n_neighbors,
            pairwise=False,
            backend=self.backend,
            n_jobs=self.n_jobs,
            laplacian_type=self.laplacian_type,
            semi_aniso=False,
            anisotropy=1.0,
            cache_input=False,
            verbose=self.bases_graph_verbose,
            random_state=self.random_state,
            **cfg,
        ).fit(knn)

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
                    f"(n_eigs={self.n_eigs})"
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
            self.n_eigs = max_eigs
            warnings.warn(f"Clamped n_eigs to {max_eigs} (n_samples={n_samples})")

    def _setup_environment(self):
        from topo._logging import configure

        configure(self.verbosity)
        self._parse_backend()
        self._parse_random_state()
        if self.n_jobs == -1:
            try:
                from joblib import cpu_count

                self.n_jobs = cpu_count()
            except Exception:
                pass
        self.layout_verbose = self.verbosity >= 2
        self.bases_graph_verbose = self.verbosity >= 3

    def spectral_scaffold(self, multiscale: bool = True):
        """Return the fitted spectral scaffold coordinates.

        Parameters
        ----------
        multiscale : bool, default=True
            If ``True``, return the multiscale diffusion-map scaffold (``msZ``),
            which weights eigenvectors by ``lambda / (1 - lambda)`` and is the
            default representation for most downstream TopoMetry layouts. If
            ``False``, return the fixed-time diffusion-map scaffold (``Z``).

        Returns
        -------
        ndarray, shape (n_samples, n_components)
            Spectral coordinates for the fitted samples. The number of columns is
            determined by ``n_eigs`` and, when automatic sizing is enabled, the
            estimated intrinsic dimensionality.
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
    def eigenvalues(self):
        """Eigenvalues from the fitted spectral decomposition.

        For the standard pipeline this returns the eigenvalues associated with
        the currently active eigenbasis, usually the ``msDM`` scaffold built from
        the base kernel. Diffusion-map eigenvalues are ordered from largest to
        smallest; values near 1 represent smooth, persistent geometric modes,
        while values near 0 represent rapidly decaying modes.

        In Union-of-Manifolds mode, the return value is a dictionary with the
        active mode, one eigenvalue array per component, and the corresponding
        component sizes.

        Returns
        -------
        ndarray or dict
            Eigenvalues for the fitted global scaffold, or per-component
            eigenvalues when ``uom=True``.
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
    def knn_msZ(self):
        """Sparse kNN distance graph built in multiscale scaffold space.

        ``msZ`` is the multiscale diffusion-map representation returned by
        ``spectral_scaffold(multiscale=True)``. This graph is used to build the
        refined ``msDM`` kernel and the ``P_of_msZ`` operator that underlies
        ``msTopoMAP`` and ``msTopoPaCMAP``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse graph with one row per sample and nonzero entries storing
            neighbor distances in ``msZ`` space.
        """
        if self.uom_enabled and self.knn_msZ_uom is not None:
            return self.knn_msZ_uom
        if self._knn_msZ is None:
            raise AttributeError("knn_msZ unavailable. Call .fit() first.")
        return self._knn_msZ

    @property
    def knn_Z(self):
        """Sparse kNN distance graph built in fixed-time DM scaffold space.

        ``Z`` is the diffusion-map representation returned by
        ``spectral_scaffold(multiscale=False)``. This graph is used to build the
        refined ``DM`` kernel and the ``P_of_Z`` operator that underlies
        ``TopoMAP`` and ``TopoPaCMAP``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse graph with one row per sample and nonzero entries storing
            neighbor distances in ``Z`` space.
        """
        if self.uom_enabled and self.knn_Z_uom is not None:
            return self.knn_Z_uom
        if self._knn_Z is None:
            raise AttributeError("knn_Z unavailable. Call .fit() first.")
        return self._knn_Z

    @property
    def P_of_msZ(self) -> csr_matrix:
        """Diffusion operator built from the multiscale scaffold graph.

        This is the refined graph operator computed after building kNN and kernel
        weights in ``msZ`` space. It is the operator used by multiscale layouts
        such as ``msTopoMAP`` and by analysis helpers when ``which='msZ'``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Fitted diffusion/operator matrix with shape
            ``(n_samples, n_samples)``.
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

    @property
    def P_of_Z(self) -> csr_matrix:
        """Diffusion operator built from the fixed-time DM scaffold graph.

        This is the refined graph operator computed after building kNN and kernel
        weights in ``Z`` space. It is the operator used by fixed-time layouts
        such as ``TopoMAP`` and by analysis helpers when ``which='Z'``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Fitted diffusion/operator matrix with shape
            ``(n_samples, n_samples)``.
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

    @property
    def knn_X(self):
        """Sparse kNN distance graph built from the input representation.

        This is the first graph in the pipeline, constructed directly from the
        data passed to :meth:`fit` using ``base_metric`` and ``base_knn``. It is
        useful for debugging neighbor search, comparing the input graph with
        scaffold-space graphs, and building custom diagnostics.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse graph with one row per sample and nonzero entries storing
            neighbor distances in input space.
        """
        if self.uom_enabled and self.knn_X_uom is not None:
            return self.knn_X_uom
        if self.base_knn_graph is None:
            raise AttributeError("knn_X unavailable. Call .fit() first.")
        return self.base_knn_graph

    @property
    def P_of_X(self):
        """Diffusion operator built from the input-space base kernel.

        This operator is computed from the base affinity graph before the
        spectral scaffold is built. It is the reference operator for the first
        diffusion/eigendecomposition step and is useful when comparing how much
        structure is preserved by scaffold-space or layout-space operators.

        Returns
        -------
        scipy.sparse matrix
            Fitted diffusion/operator matrix with shape
            ``(n_samples, n_samples)``.
        """
        if self.uom_enabled and self.P_of_X_uom is not None:
            return self.P_of_X_uom
        if self.base_kernel is None:
            raise AttributeError("P_of_X unavailable. Call .fit() first.")
        return self.base_kernel.P

    @property
    def global_id(self):
        """Global intrinsic-dimensionality estimate used to size the scaffold.

        The value is estimated during :meth:`fit` from neighbor-distance
        statistics using the configured ``id_method`` and ``id_ks`` settings. It
        represents the package's estimate of the data's effective geometric
        dimensionality and is used, with headroom and bounds, to choose how many
        spectral components to keep.

        Returns
        -------
        int, float, or None
            Estimated global intrinsic dimension after fitting. ``None`` means
            the model has not been fitted or automatic sizing has not populated
            the estimate.
        """
        return self.global_dimensionality

    @property
    def intrinsic_dim(self):
        """Detailed intrinsic-dimensionality diagnostics from fitting.

        This property exposes both the global estimate and the per-sample/local
        estimates used to size the spectral scaffold. The exact contents of
        ``details`` depend on the configured estimator, for example ``fsa`` or
        ``mle``.

        Returns
        -------
        dict
            Dictionary with ``method``, ``global``, ``local``, and ``details``.
            ``local`` is typically an array of per-sample estimates; ``details``
            contains estimator-specific intermediate values.
        """
        det = (self._id_details or {}).get(self.id_method)
        return {
            "method": self.id_method,
            "global": self.global_dimensionality,
            "local": self.local_dimensionality,
            "details": det,
        }

    # --- Embedding properties ---

    @property
    def TopoMAP(self):
        """Two-dimensional MAP layout from the fixed-time DM branch.

        This is the default MAP-style visualization computed from the refined
        graph/operator built in ``Z`` space. It is a display embedding with shape
        ``(n_samples, 2)``; use it for visualization, not as a lossless
        replacement for the spectral scaffold.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Stored projection from ``ProjectionDict``. Call :meth:`fit` or
            :meth:`project` first if it is unavailable.
        """
        return self._get_projection("MAP", multiscale=False)

    @property
    def msTopoMAP(self):
        """Two-dimensional MAP layout from the multiscale msDM branch.

        This is the MAP-style visualization computed from the refined
        graph/operator built in ``msZ`` space. Because ``msZ`` aggregates
        diffusion behavior across scales, this layout is often a useful default
        view for data with hierarchical or multiscale structure.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Stored projection from ``ProjectionDict``. Call :meth:`fit` or
            :meth:`project` first if it is unavailable.
        """
        return self._get_projection("MAP", multiscale=True)

    @property
    def TopoPaCMAP(self):
        """Two-dimensional PaCMAP layout from the fixed-time DM branch.

        PaCMAP is an optional projection method that balances nearby,
        mid-near, and farther point pairs. This property returns the PaCMAP view
        computed from the fixed-time diffusion scaffold ``Z``.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Stored PaCMAP projection. Requires the PaCMAP optional dependency and
            a fitted or explicitly projected ``PaCMAP`` layout.
        """
        return self._get_projection("PaCMAP", multiscale=False)

    @property
    def msTopoPaCMAP(self):
        """Two-dimensional PaCMAP layout from the multiscale msDM branch.

        This property returns the PaCMAP view computed from the multiscale
        diffusion scaffold ``msZ``. It is useful as a comparison to ``msTopoMAP``
        because PaCMAP and MAP optimize different layout objectives.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Stored PaCMAP projection. Requires the PaCMAP optional dependency and
            a fitted or explicitly projected ``PaCMAP`` layout.
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
        """Compute per-sample diagnostics in the spectral scaffold.

        The returned scores summarize how strongly each sample is associated
        with particular scaffold axes and whether its local neighborhood is
        axis-like or radially separated. These diagnostics are exploratory:
        they help identify samples driven by a small number of spectral modes,
        but they are not supervised labels or clustering scores.

        Parameters
        ----------
        Z : array-like, optional
            Scaffold coordinates to analyze. If omitted, uses
            :meth:`spectral_scaffold`.
        evals : array-like, optional
            Eigenvalues matching the scaffold columns. If omitted, uses the
            fitted eigenvalues for the selected DM/msDM branch.
        multiscale : bool, default=True
            Whether the default scaffold should be ``msZ`` (``True``) or ``Z``
            (``False``).
        use_scaffold_components : bool, default=True
            If automatic sizing selected fewer useful components than were
            computed, restrict diagnostics to that selected width.
        smooth_P : {'X', 'Z', 'msZ'} or None, default=None
            Optional fitted operator used to diffusion-smooth the scalar
            diagnostic fields.
        smooth_t : int, default=0
            Number of smoothing steps when ``smooth_P`` is provided.
        out_prefix : str, default='spectral'
            Prefix used when storing results in ``LocalScoresDict``.
        return_dict : bool, default=True
            If ``True``, return the result dictionary. If ``False``, only store
            the result in ``LocalScoresDict``.
        **kwargs
            Forwarded to :func:`topo.analysis.spectral_selectivity`.

        Returns
        -------
        dict or None
            Dictionary with ``EAS``, ``RayScore``, ``LAC``, ``axis``,
            ``axis_sign``, and ``radius`` when ``return_dict=True``.
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
        """Smooth a per-sample scalar signal over a fitted diffusion operator.

        This applies ``P^t`` to a one-dimensional signal, where ``P`` is one of
        the fitted operators from the pipeline. It is useful for denoising or
        visualizing a continuous per-sample quantity, such as a score, intensity,
        or expression value, according to the learned graph geometry.

        Parameters
        ----------
        signal : array-like, shape (n_samples,)
            One value per fitted sample.
        t : int, default=8
            Number of diffusion steps. Larger values produce stronger smoothing.
        which : {'X', 'Z', 'msZ'}, default='msZ'
            Operator to use: input-space base operator (``'X'``), fixed-time DM
            scaffold operator (``'Z'``), or multiscale scaffold operator
            (``'msZ'``).

        Returns
        -------
        ndarray, shape (n_samples,)
            Diffusion-smoothed signal.
        """
        return _analysis.filter_signal(signal, self._select_P_operator(which), t)

    def impute(self, X, t: int = 8, which: str = "msZ", **kwargs):
        """Diffuse a data matrix over a fitted graph operator.

        Each column of ``X`` is treated as a graph signal and smoothed by
        applying ``P^t``. This can be used as a simple graph-based denoising or
        imputation step, but it should be interpreted as smoothing over the
        learned geometry rather than recovery of unobserved ground truth.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data values to diffuse. The number of rows must match the fitted
            samples.
        t : int, default=8
            Number of diffusion steps. Larger values smooth more aggressively.
        which : {'X', 'Z', 'msZ'}, default='msZ'
            Fitted operator to use for diffusion.
        **kwargs
            Forwarded to :func:`topo.analysis.impute`, such as ``output``.

        Returns
        -------
        ndarray or scipy.sparse matrix
            Diffused version of ``X``. The output format follows
            :func:`topo.analysis.impute`.
        """
        return _analysis.impute(X, self._select_P_operator(which), t, **kwargs)

    def riemann_diagnostics(self, Y=None, L=None, diffusion_op=None, **kwargs):
        """Measure local distortion in a two-dimensional embedding.

        This estimates a Riemannian metric field for a 2-D layout and derives
        scalar distortion summaries such as anisotropy, log-determinant, and
        deformation. Use it to identify where a visualization stretches,
        compresses, or anisotropically distorts the fitted graph geometry.

        Parameters
        ----------
        Y : array-like, shape (n_samples, 2), optional
            Embedding to diagnose. If omitted, the first available fitted layout
            is used, trying ``TopoMAP``, ``msTopoMAP``, ``TopoPaCMAP``, and
            ``msTopoPaCMAP`` before computing a MAP projection.
        L : array-like or sparse matrix, optional
            Graph Laplacian/operator used as the reference geometry. If omitted,
            uses ``base_kernel.L``.
        diffusion_op : {'X', 'Z', 'msZ'} or None, default=None
            Optional fitted operator used to smooth deformation maps when
            ``diffusion_t`` is passed through ``kwargs``.
        **kwargs
            Forwarded to :func:`topo.analysis.riemann_diagnostics`.

        Returns
        -------
        dict
            Dictionary stored in ``RiemannMetricDict['last']``. Common keys are
            ``G`` for the local metric tensor, ``anisotropy``, ``logdetG``,
            ``deformation``, and ``limits``.
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
    """Load a TopOGraph from a pickle file."""
    import pickle

    with open(filename, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TopOGraph):
        warnings.warn("Loaded object is not a TopOGraph.", RuntimeWarning)
    return obj
