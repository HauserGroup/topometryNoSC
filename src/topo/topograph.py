# TopoMetry high-level API — the TopOGraph class
#
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
#
"""High-level :class:`TopOGraph` orchestrator.

The user-facing entry point that ties the pipeline together: base graph and
kernel construction, the dual (DM / msDM) spectral scaffold, refined graphs and
2-D projections. Composes the mixins in :mod:`topo._pipeline` and
:mod:`topo.uom`, exposing ``fit`` plus analysis, projection, and persistence
helpers.
"""

import gc
import logging
import warnings
from collections.abc import Sequence
from os import PathLike
from typing import Any, cast

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

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

VALID_LAPLACIAN_TYPES = frozenset(["normalized", "unnormalized", "random_walk"])

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


def _is_finite_matrix(X) -> bool:
    """Return True if dense/sparse matrix contains only finite numeric values."""
    if issparse(X):
        return bool(np.isfinite(X.data).all())
    return bool(np.isfinite(np.asarray(X)).all())


# ============================================================================
# TopOGraph
# ============================================================================


class TopOGraph(
    GraphBuildMixin,
    EigenBuildMixin,
    LayoutBuildMixin,
    UoMMixin,
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
    base_kernel_version : str, default="bw_adaptive"
        Kernel choice for the base graph.
    graph_kernel_version : str, default="bw_adaptive"
        Kernel choice for scaffold graphs.
    backend : {"sklearn", "hnswlib"}, default="sklearn"
        Nearest-neighbor backend. Use "sklearn" for exact search and "hnswlib"
        for optional dense-vector HNSW search.
    base_metric : str, default="cosine"
        Distance metric for the base kNN graph.
    graph_metric : str, default="euclidean"
        Distance metric for kNN in scaffold space.
    diff_t : int, default=0
        Diffusion time for single-time scaffold.
    sigma : float, default=0.1
        Bandwidth for Gaussian kernels.
    delta : float, default=1.0
        Unitless edge threshold for CkNN kernels.
    cknn_candidate_neighbors : int or None, default=None
        Number of candidate neighbors tested in approximate CkNN mode.
    cknn_exact : bool, default=False
        If True, threshold all pairwise distances for paper-faithful CkNN
        construction. This is quadratic in samples.
    n_jobs : int, default=-1
        Threads for kNN searches; -1 uses all cores.
    low_memory : bool, default=False
        Avoid caching large kernel objects.
    eigen_tol : float, default=1e-8
        Tolerance for the eigensolver.
    eigensolver : {"arpack", "lobpcg", "amg", "dense"}, default="arpack"
        Solver for eigendecomposition.
    projection_methods : sequence of str or None, default=None
        Layouts to compute during ``fit``. If None, uses ["MAP", "PaCMAP"].
    verbosity : int, default=0
        Logging verbosity.
    random_state : int, RandomState, or None, default=42
        Random seed.
    id_method : {"fsa", "mle"}, default="fsa"
        Intrinsic-dimensionality estimator for scaffold sizing.
    id_ks : int or iterable, default=50
        Neighborhood sizes for I.D. estimation.
    id_metric : str, default="euclidean"
        Metric for I.D. estimation.
    id_quantile : float, default=0.99
        Quantile of local intrinsic-dimensionality estimates used to choose
        the scaffold dimensionality.
    id_min_components : int, default=128
        Lower bound on the number of spectral components computed.
    id_max_components : int, default=1024
        Upper bound on the number of spectral components computed.
    id_headroom : float, default=0.5
        Fractional safety margin added to the intrinsic-dimensionality estimate.
    uom : bool, default=False
        Enable Union-of-Manifolds block-diagonal scaffolds.
    """

    def __init__(
        self,
        base_knn: int = 30,
        graph_knn: int = 30,
        min_eigs: int = 128,
        n_jobs: int = -1,
        projection_methods: Sequence[str] | None = None,
        base_kernel=None,
        base_kernel_version: str = "bw_adaptive",
        graph_kernel_version: str = "bw_adaptive",
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        diff_t: int = 0,
        delta: float = 1.0,
        cknn_candidate_neighbors: int | None = None,
        cknn_exact: bool = False,
        sigma: float = 0.1,
        low_memory: bool = False,
        eigen_tol: float = 1e-8,
        eigensolver: str = "arpack",
        backend: str = "sklearn",
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
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.min_eigs = min_eigs
        self.n_jobs = n_jobs
        self.projection_methods: list[str] = (
            ["MAP", "PaCMAP"]
            if projection_methods is None
            else list(projection_methods)
        )
        self.base_kernel = base_kernel
        self.base_kernel_version = base_kernel_version
        self.graph_kernel_version = graph_kernel_version
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.diff_t = diff_t
        self.delta = delta
        self.cknn_candidate_neighbors = cknn_candidate_neighbors
        self.cknn_exact = cknn_exact
        self.sigma = sigma
        self.low_memory = low_memory
        self.eigen_tol = eigen_tol
        self.eigensolver = eigensolver
        self.backend = backend
        self.verbosity = verbosity
        self.random_state = random_state
        self.laplacian_type = laplacian_type

        # Validate backend and resolve fallback
        self._backend_resolved = backend
        self.id_method = id_method
        self.id_ks = id_ks
        self.id_metric = id_metric
        self.id_quantile = id_quantile
        self.id_min_components = id_min_components
        self.id_max_components = id_max_components
        self.id_headroom = id_headroom
        self.uom = uom

        # Runtime / derived config.
        self.n_eigs = min_eigs

        # If CkNN is used with a normalized Laplacian, use unnormalized by
        # default. This preserves the intended binary graph semantics while
        # still allowing explicit non-default laplacian_type choices.
        uses_cknn = base_kernel_version == "cknn" or graph_kernel_version == "cknn"

        self.laplacian_type = (
            "unnormalized"
            if uses_cknn and laplacian_type == "normalized"
            else laplacian_type
        )

        # Fitted state
        self.n: int | None = None
        self.m: int | None = None
        self.n_eigs_: int | None = None
        self.selected_scaffold_components_: int | None = None
        self._n_jobs_effective = n_jobs
        self._random_state_resolved: RandomState
        self.base_knn_graph: csr_matrix | None = None
        self.knn_X_: csr_matrix | None = None
        self.knn_Z_: csr_matrix | None = None
        self.knn_msZ_: csr_matrix | None = None
        self.P_X_: csr_matrix | None = None
        self.P_Z_: csr_matrix | None = None
        self.P_msZ_: csr_matrix | None = None
        self.K_Z_: csr_matrix | None = None
        self.K_msZ_: csr_matrix | None = None
        self.eigenbasis: Any = None
        self.graph_kernel: Kernel | None = None
        self.SpecLayout: np.ndarray | None = None
        self.global_dimensionality: int | float | None = None
        self.local_dimensionality: np.ndarray | None = None
        self._id_details: dict[str, Any] = {"mle": None, "fsa": None}
        self._scaffold_components_dm: int | None = None
        self._scaffold_components_ms: int | None = None

        # Dual-scaffold products
        self.Z_: np.ndarray | csr_matrix | None = None
        self.msZ_: np.ndarray | csr_matrix | None = None
        self.evals_Z_: np.ndarray | None = None
        self.evals_msZ_: np.ndarray | None = None
        self._knn_msZ: csr_matrix | None = None
        self._knn_Z: csr_matrix | None = None

        # MAP snapshots
        self.msTopoMAP_snapshots: list[Any] = []
        self.TopoMAP_snapshots: list[Any] = []

        # Verbosity toggles (derived)
        self.bases_graph_verbose = False
        self.layout_verbose = False

        # Legacy / benchmarking dictionaries
        self.ProjectionDict: dict[str, np.ndarray] = {}
        self.LocalScoresDict: dict[str, Any] = {}
        self.RiemannMetricDict: dict[str, Any] = {}
        self.runtimes: dict[str, Any] = {}

        # UoM state (from mixin)
        self._init_uom_state()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return a short summary of the fitted state and active scaffold."""
        if self.n is None:
            return "TopOGraph (not fitted)"

        parts = [f"TopOGraph with {self.n} samples"]
        if self.m is not None:
            parts[0] += f" × {self.m} features"

        for label, d in [
            ("Projections", self.ProjectionDict),
        ]:
            if d:
                parts.append(f"  {label}: {', '.join(map(str, d.keys()))}")

        out = "\n".join(parts)
        if len(out) > N_CHAR_MAX:
            return out[: max(0, N_CHAR_MAX - 3)] + "..."
        return out

    # ------------------------------------------------------------------
    # Data-driven kernel builder
    # ------------------------------------------------------------------

    def _build_kernel(
        self,
        knn,
        n_neighbors,
        kernel_version,
        *,
        base: bool = True,
        data_for_expansion=None,
    ) -> Kernel:
        """Build a :class:`Kernel` from a kNN graph and a named kernel version.

        This internal helper assumes ``_setup_environment()`` has already run.
        """
        if kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(f"Invalid kernel_version: {kernel_version}")

        if knn is None:
            raise ValueError("knn must not be None.")

        if not hasattr(knn, "shape") or len(knn.shape) != 2:
            raise ValueError("knn must be a 2-D sparse/dense matrix.")

        if int(knn.shape[0]) != int(knn.shape[1]):
            raise ValueError("knn must be a square graph.")

        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1.")

        cfg = _KERNEL_CONFIGS[str(kernel_version)].copy()

        uses_raw_data = bool(cfg.get("expand_nbr_search")) or kernel_version == "cknn"

        if uses_raw_data:
            metric = self.base_metric if base else self.graph_metric

            if metric == "precomputed":
                fit_input = knn
            else:
                if data_for_expansion is None:
                    raise ValueError(
                        "data_for_expansion is required for kernel version "
                        f"{kernel_version!r} with metric={metric!r}."
                    )

                if (
                    not hasattr(data_for_expansion, "shape")
                    or len(data_for_expansion.shape) != 2
                ):
                    raise ValueError(
                        "data_for_expansion must be a 2-D array-like or sparse matrix."
                    )

                if int(data_for_expansion.shape[0]) != int(knn.shape[0]):
                    raise ValueError(
                        "data_for_expansion must have the same number of rows "
                        "as the kNN graph."
                    )

                fit_input = data_for_expansion

            if cfg.get("expand_nbr_search") and metric == "precomputed":
                raise ValueError(
                    f"kernel version {kernel_version!r} expands neighbor search and "
                    "therefore requires raw feature data; it cannot be used with "
                    "metric='precomputed'. Use a non-expansion kernel version or pass "
                    "raw data."
                )
        else:
            metric = "precomputed"
            fit_input = knn

        if kernel_version == "gaussian":
            cfg["sigma"] = float(self.sigma)

        if kernel_version == "cknn":
            cfg["cknn_delta"] = float(self.delta)
            cfg["cknn_candidate_neighbors"] = self.cknn_candidate_neighbors
            cfg["cknn_exact"] = bool(self.cknn_exact)

        kernel = Kernel(
            metric=metric,
            n_neighbors=int(n_neighbors),
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
        return kernel

    # ------------------------------------------------------------------
    # Fit orchestration
    # ------------------------------------------------------------------

    def fit(self, X=None):
        """Run the full pipeline on ``X``.

        Builds base kNN → base kernel P(X) → dual eigenbases (DM + msDM) →
        refined scaffold graphs → 2-D projections. When ``uom=True``, detects
        disconnected components and builds per-component scaffolds and
        block-diagonal operators.
        """
        self._validate_inputs(X)
        self._setup_environment()
        self._build_base_graph(X)
        self._build_base_kernel(X)

        if self.base_metric != "precomputed":
            sizing_X = self._resolve_sizing_input(X)
            self._automated_sizing(sizing_X)

            if self.verbosity >= 1:
                logger.info(
                    "Automated sizing → target components: %s (n_eigs_=%s)",
                    self._scaffold_components_ms,
                    self.n_eigs_,
                )

        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        if self.uom_enabled:
            out = self._fit_uom(X)
        else:
            out = self._fit_global(X)

        self._check_fitted_pipeline_state()
        return out

    # ------------------------------------------------------------------
    # Input validation / setup
    # ------------------------------------------------------------------

    def _validate_inputs(self, X) -> None:
        if self.base_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(f"Invalid base_kernel_version: {self.base_kernel_version}")
        if self.graph_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(
                f"Invalid graph_kernel_version: {self.graph_kernel_version}"
            )
        if self.backend not in {"sklearn", "hnswlib"}:
            raise ValueError("backend must be one of {'sklearn', 'hnswlib'}.")
        if self.laplacian_type not in VALID_LAPLACIAN_TYPES:
            raise ValueError(
                f"laplacian_type must be one of {sorted(VALID_LAPLACIAN_TYPES)}."
            )
        if int(self.n_jobs) < -1 or int(self.n_jobs) == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")

        for name in (
            "base_knn",
            "graph_knn",
            "min_eigs",
            "id_min_components",
            "id_max_components",
        ):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1.")

        if int(self.id_min_components) > int(self.id_max_components):
            raise ValueError("id_min_components must be <= id_max_components.")

        if not (0.0 < float(self.id_quantile) <= 1.0):
            raise ValueError("id_quantile must be in the interval (0, 1].")

        if float(self.id_headroom) < 0.0:
            raise ValueError("id_headroom must be non-negative.")

        if int(self.diff_t) < 0:
            raise ValueError("diff_t must be non-negative.")

        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be positive.")

        if float(self.delta) <= 0.0:
            raise ValueError("delta must be positive for CkNN.")

        if float(self.eigen_tol) < 0.0:
            raise ValueError("eigen_tol must be non-negative.")

        if (
            self.cknn_candidate_neighbors is not None
            and int(self.cknn_candidate_neighbors) < 1
        ):
            raise ValueError("cknn_candidate_neighbors must be >= 1.")

        if X is None:
            if self.base_kernel is None:
                raise ValueError("X was not passed and no base_kernel was provided.")

            if not isinstance(self.base_kernel, Kernel):
                raise ValueError("base_kernel must be a topo.tpgraph.Kernel instance.")

            if getattr(self.base_kernel, "P", None) is None:
                raise ValueError(
                    "base_kernel must expose a fitted diffusion operator `P`."
                )

            if self.base_metric != "precomputed":
                kernel_X = getattr(self.base_kernel, "X", None)
                if kernel_X is None:
                    raise ValueError(
                        "base_kernel must expose original input data as `X` when automated "
                        "sizing is required."
                    )

            return

        if not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError("X must be a 2-D array-like or sparse matrix.")

        n_samples, n_features = X.shape
        if n_samples < 3:
            raise ValueError("X must contain at least 3 samples.")
        if n_features < 1:
            raise ValueError("X must contain at least 1 feature.")

        if not _is_finite_matrix(X):
            raise ValueError("X contains NaN or infinite values.")

        if self.base_metric == "precomputed" and n_samples != n_features:
            raise ValueError(
                "When base_metric='precomputed', X must be a square "
                "(n_samples, n_samples) sparse or dense graph/distance matrix."
            )

        for name in ("base_knn", "graph_knn"):
            k = int(getattr(self, name))
            if k >= n_samples:
                raise ValueError(
                    f"{name}={k} must be smaller than n_samples={n_samples}."
                )

        if self.cknn_candidate_neighbors is not None:
            candidate_k = int(self.cknn_candidate_neighbors)
            if candidate_k >= n_samples:
                raise ValueError(
                    "cknn_candidate_neighbors must be smaller than n_samples."
                )

        max_eigs = max(1, n_samples - 2)
        if int(self.n_eigs) > max_eigs:
            self.n_eigs_ = max_eigs
            warnings.warn(
                f"Clamped n_eigs to {max_eigs} (n_samples={n_samples})",
                stacklevel=2,
            )
        else:
            self.n_eigs_ = int(self.n_eigs)

    def _setup_environment(self) -> None:
        """Resolve runtime configuration used by internal pipeline phases."""
        from topo._logging import configure
        from topo.base.ann import _resolve_n_jobs

        configure(self.verbosity)

        if self.backend not in {"sklearn", "hnswlib"}:
            raise ValueError("backend must be one of {'sklearn', 'hnswlib'}.")

        self._n_jobs_effective = _resolve_n_jobs(self.n_jobs)
        self._backend_resolved = self.backend
        self._random_state_resolved = check_random_state(self.random_state)
        self.uom_enabled = bool(self.uom)

        if self.n_eigs_ is None:
            self.n_eigs_ = int(self.n_eigs)

        self.layout_verbose = self.verbosity >= 2
        self.bases_graph_verbose = self.verbosity >= 3

    def _check_fitted_pipeline_state(self) -> None:
        """Validate that canonical fitted pipeline state is available."""
        if self.P_X_ is None:
            raise RuntimeError("fit() completed without fitted input-space operator.")

        if self.Z_ is None:
            raise RuntimeError("fit() completed without fitted DM scaffold.")
        if self.msZ_ is None:
            raise RuntimeError("fit() completed without fitted msDM scaffold.")

        if self.P_Z_ is None:
            raise RuntimeError("fit() completed without fitted DM scaffold operator.")
        if self.P_msZ_ is None:
            raise RuntimeError("fit() completed without fitted msDM scaffold operator.")

        if self.K_Z_ is None:
            raise RuntimeError("fit() completed without fitted DM scaffold affinity.")
        if self.K_msZ_ is None:
            raise RuntimeError("fit() completed without fitted msDM scaffold affinity.")

        if self.knn_X_ is None:
            raise RuntimeError("fit() completed without fitted input-space kNN graph.")
        if self.knn_Z_ is None:
            raise RuntimeError("fit() completed without fitted DM scaffold kNN graph.")
        if self.knn_msZ_ is None:
            raise RuntimeError(
                "fit() completed without fitted msDM scaffold kNN graph."
            )

        if self.eigenbasis is None and not self.uom_enabled:
            raise RuntimeError("fit() completed without an active eigenbasis.")

        if self.graph_kernel is None and not self.uom_enabled:
            raise RuntimeError("fit() completed without an active graph_kernel.")

    # ------------------------------------------------------------------
    # Scaffold access
    # ------------------------------------------------------------------

    def spectral_scaffold(self, multiscale: bool = True) -> np.ndarray | csr_matrix:
        """Return fitted spectral scaffold coordinates."""
        arr = self.msZ_ if multiscale else self.Z_

        if arr is None:
            raise AttributeError("Scaffold unavailable. Call .fit() first.")

        return arr

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> np.ndarray | dict[str, Any]:
        """Eigenvalues of the active spectral scaffold."""
        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list:
            mode = getattr(self, "_uom_active_mode", "msDM")
            per_comp = (
                self.uom_eigenvalues_ms_list
                if mode == "msDM"
                else self.uom_eigenvalues_dm_list
            )
            comps = getattr(self, "uom_components_", None) or []
            sizes = [int(ix.size) for ix in comps]
            return {"mode": mode, "per_component": per_comp, "component_sizes": sizes}

        if self.eigenbasis is None:
            raise AttributeError("Eigenvalues unavailable. Call .fit() first.")

        return self.eigenbasis.eigenvalues

    @property
    def knn_msZ(self) -> csr_matrix:
        """The k-nearest-neighbors graph built in the msDM scaffold space."""
        if self.knn_msZ_ is None:
            raise AttributeError("knn_msZ unavailable. Call .fit() first.")
        return self.knn_msZ_

    @property
    def knn_Z(self) -> csr_matrix:
        """The k-nearest-neighbors graph built in the fixed-time DM scaffold space."""
        if self.knn_Z_ is None:
            raise AttributeError("knn_Z unavailable. Call .fit() first.")
        return self.knn_Z_

    @property
    def P_of_msZ(self) -> csr_matrix:
        """The diffusion operator on the msDM scaffold."""
        if self.P_msZ_ is None:
            raise AttributeError("P_of_msZ unavailable. Call .fit() first.")
        return self.P_msZ_

    @property
    def P_of_Z(self) -> csr_matrix:
        """The diffusion operator on the fixed-time DM scaffold."""
        if self.P_Z_ is None:
            raise AttributeError("P_of_Z unavailable. Call .fit() first.")
        return self.P_Z_

    @property
    def knn_X(self) -> csr_matrix:
        """The base k-nearest-neighbors graph in the original input space."""
        if self.knn_X_ is None:
            raise AttributeError("knn_X unavailable. Call .fit() first.")
        return self.knn_X_

    @property
    def P_of_X(self) -> csr_matrix:
        """The base diffusion operator on the original input space."""
        if self.P_X_ is None:
            raise AttributeError("P_of_X unavailable. Call .fit() first.")
        return self.P_X_

    @property
    def global_id(self) -> float:
        """The estimated global intrinsic dimensionality of the dataset."""
        if self.global_dimensionality is None:
            raise NotFittedError(
                "Global intrinsic dimensionality is unavailable. Call fit(X) first "
                "with an input-space metric that supports automated sizing."
            )
        return float(self.global_dimensionality)

    @property
    def intrinsic_dim(self) -> dict[str, Any]:
        """Structured intrinsic-dimensionality information."""
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
        """2-D MAP layout optimized on the fixed-time DM refined graph."""
        return self._get_projection("MAP", multiscale=False)

    @property
    def msTopoMAP(self) -> np.ndarray:
        """2-D MAP layout optimized on the msDM refined graph."""
        return self._get_projection("MAP", multiscale=True)

    @property
    def TopoPaCMAP(self) -> np.ndarray:
        """2-D PaCMAP layout optimized on the fixed-time DM refined graph."""
        return self._get_projection("PaCMAP", multiscale=False)

    @property
    def msTopoPaCMAP(self) -> np.ndarray:
        """2-D PaCMAP layout optimized on the msDM refined graph."""
        return self._get_projection("PaCMAP", multiscale=True)

    # ------------------------------------------------------------------
    # Analysis convenience wrappers
    # ------------------------------------------------------------------

    def _select_P_operator(self, which: str = "msZ") -> csr_matrix:
        """Resolve a fitted diffusion operator by name."""
        which_norm = str(which).lower()

        if which_norm == "x":
            return self.P_of_X
        if which_norm == "z":
            return self.P_of_Z
        if which_norm == "msz":
            return self.P_of_msZ

        raise ValueError("`which` must be one of {'X', 'Z', 'msZ'}.")

    def _resolve_sizing_input(self, X) -> NDArray[Any] | csr_matrix:
        """Resolve input data used for automated scaffold sizing."""
        if X is not None:
            return cast(NDArray[Any] | csr_matrix, X)

        if self.base_kernel is None:
            raise ValueError("Input data is required for automated sizing.")

        if self.base_kernel.X is None:
            raise ValueError("Input data is required for automated sizing.")

        return cast(NDArray[Any] | csr_matrix, self.base_kernel.X)

    def _resolve_optional_operator(self, op):
        """Resolve None/string/operator inputs used by analysis wrappers."""
        if op is None:
            return None
        if isinstance(op, str):
            return self._select_P_operator(op)
        return op

    def spectral_selectivity(
        self,
        Z=None,
        evals=None,
        multiscale: bool = True,
        use_scaffold_components: bool = True,
        smooth_P=None,
        smooth_t: int = 0,
        out_prefix: str = "spectral",
        return_dict: bool = True,
        **kwargs,
    ):
        """Per-sample spectral selectivity delegated to ``topo.analysis``."""
        if Z is None:
            Z = self.spectral_scaffold(multiscale=multiscale)

        Z_arr = np.asarray(Z)
        if Z_arr.ndim != 2:
            raise ValueError("Z must be a 2-D scaffold matrix.")

        if use_scaffold_components:
            n_keep = (
                self._scaffold_components_ms
                if multiscale
                else self._scaffold_components_dm
            )
            if n_keep is not None:
                Z_arr = Z_arr[:, : min(int(n_keep), Z_arr.shape[1])]

        if evals is None:
            evals = self.evals_msZ_ if multiscale else self.evals_Z_
            if evals is None:
                raise AttributeError(
                    "Eigenvalues unavailable for spectral_selectivity. "
                    "Pass `evals` explicitly or call .fit() in global mode."
                )

            ev = np.asarray(evals, dtype=float)

            # Eigenvalues often include the trivial first mode; if present, drop it.
            evals = (
                ev[1 : Z_arr.shape[1] + 1]
                if ev.shape[0] >= Z_arr.shape[1] + 1
                else ev[: Z_arr.shape[1]]
            )

        P = self._resolve_optional_operator(smooth_P)

        result = _analysis.spectral_selectivity(
            Z_arr,
            evals,
            P=P,
            smooth_t=smooth_t,
            **kwargs,
        )

        for k, v in result.items():
            self.LocalScoresDict[f"{out_prefix}_{k}"] = v

        return result if return_dict else None

    def filter_signal(self, signal, t: int = 8, which: str = "msZ"):
        """Diffusion-filter a one-dimensional signal."""
        return _analysis.filter_signal(signal, self._select_P_operator(which), t)

    def impute(self, X, t: int = 8, which: str = "msZ", **kwargs):
        """Diffusion-based imputation."""
        return _analysis.impute(X, self._select_P_operator(which), t, **kwargs)

    def riemann_diagnostics(self, Y=None, L=None, diffusion_op=None, **kwargs):
        """Riemann metric + deformation scalars."""
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
                raise ValueError("No base kernel available. Call fit() first.")
            L = self.base_kernel.L

        P = self._resolve_optional_operator(diffusion_op)

        result = _analysis.riemann_diagnostics(Y, L, diffusion_op=P, **kwargs)
        self.RiemannMetricDict["last"] = result
        return result

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, filename: str | PathLike[str] = "topograph.pkl") -> None:
        """Save this TopOGraph to a pickle file."""
        save_topograph(self, filename)

    def spectral_layout(self, *args: Any, **kwargs: Any) -> Any:
        """Disambiguate inherited ``spectral_layout`` implementations."""
        return LayoutBuildMixin.spectral_layout(self, *args, **kwargs)


# =========================================================================
# Module-level I/O helpers
# =========================================================================


def save_topograph(
    tg: TopOGraph,
    filename: str | PathLike[str] = "topograph.pkl",
) -> None:
    """Save a TopOGraph object to a pickle file."""
    import pickle

    if not isinstance(tg, TopOGraph):
        raise TypeError("`tg` must be a TopOGraph instance.")

    with open(filename, "wb") as f:
        pickle.dump(tg, f, pickle.HIGHEST_PROTOCOL)

    if tg.verbosity >= 1:
        logger.info("TopOGraph saved at %s", filename)


def load_topograph(filename: str | PathLike[str]) -> TopOGraph:
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
