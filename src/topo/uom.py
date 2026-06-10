# Union-of-Manifolds (UoM) logic extracted from TopOGraph.
"""Union-of-Manifolds (UoM) scaffold construction.

Standalone helpers plus the :class:`UoMMixin` that :class:`topo.topograph.TopOGraph`
inherits. UoM partitions the graph into manifold components, builds a per-component
eigenbasis and consolidates them into a single scaffold, keeping the orchestrator slim.
"""

import copy
import logging
import warnings
from typing import Any

import numpy as np
from scipy.sparse import block_diag, csr_matrix, diags

from topo.base.ann import kNN
from topo.base.graph_matrix import as_float32_csr
from topo.spectral.eigen import EigenDecomposition
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _as_1d_labels(labels, n: int | None = None) -> np.ndarray:
    """Return labels as a one-dimensional integer array."""
    out = np.asarray(labels, dtype=int).ravel()
    if n is not None and out.shape[0] != n:
        raise ValueError(f"labels must have length {n}, got {out.shape[0]}.")
    return out


def _sparse_identity(n: int) -> csr_matrix:
    """Return an n-by-n CSR identity matrix with float32 dtype."""
    diag = np.ones(int(n), dtype=np.float32)
    return csr_matrix(diags(diag, offsets=0, shape=(int(n), int(n)), format="csr"))


def _symmetrize_geometric(P: Any) -> csr_matrix:
    """Return geometric symmetrization on overlapping support."""
    P_csr = as_float32_csr(P, "P")

    shape = P_csr.shape
    if shape is None or len(shape) != 2:
        raise ValueError("P must be a 2-D sparse matrix.")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if n_rows != n_cols:
        raise ValueError("P must be square.")

    S = csr_matrix(P_csr.multiply(P_csr.T))
    if S.nnz == 0:
        return S

    S.data = np.sqrt(S.data.astype(np.float64)).astype(np.float32, copy=False)
    S.eliminate_zeros()
    return csr_matrix(S)


def _symmetrize_sum(A) -> csr_matrix:
    """Return additive undirected symmetrization with zero diagonal."""
    A = as_float32_csr(A, "A")
    S = csr_matrix(A + A.T)
    S.setdiag(0)
    S.eliminate_zeros()
    return S


def _normalized_laplacian(A: Any) -> csr_matrix:
    """Compute zero-degree-safe symmetric normalized graph Laplacian."""
    A_csr = as_float32_csr(A, "A")

    shape = A_csr.shape
    if shape is None or len(shape) != 2:
        raise ValueError("A must be a 2-D sparse matrix.")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if n_rows != n_cols:
        raise ValueError("A must be square.")

    d = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)
    inv_sqrt = np.zeros_like(d, dtype=np.float64)
    positive = d > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(d[positive])

    Dmh = diags(inv_sqrt.astype(np.float32), format="csr")
    I = _sparse_identity(n_rows)
    L = I - (Dmh @ A_csr @ Dmh)
    L = csr_matrix(L)
    L.eliminate_zeros()
    return L


def _eigengap_k(vals, k_max: int, k_min: int = 2) -> int:
    """Choose cluster count from eigengap of ascending Laplacian eigenvalues."""
    vals = np.asarray(vals, dtype=float).ravel()
    k_max = int(max(1, k_max))
    k_min = int(max(1, k_min))

    if vals.size <= 2:
        return int(max(k_min, min(k_max, 2)))

    gaps = np.diff(vals)
    if gaps.size <= 1:
        return int(max(k_min, min(k_max, 2)))

    # Skip the first gap because lambda_0 is the trivial zero eigenvalue.
    j = int(np.argmax(gaps[1:])) + 1
    return int(max(k_min, min(k_max, j + 1)))


def mbkm(X, n_clusters: int, random_state=0) -> np.ndarray:
    """MiniBatch KMeans clustering over rows of ``X``."""
    from sklearn.cluster import MiniBatchKMeans

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")

    n_samples = X.shape[0]
    if n_samples < 1:
        raise ValueError("X must contain at least one sample.")

    n_use = int(n_clusters)
    if n_use < 1:
        raise ValueError("n_clusters must be >= 1.")
    n_use = min(n_use, n_samples)

    if n_use == 1:
        return np.zeros(n_samples, dtype=int)

    batch = int(min(2048, max(256, 8 * n_use * n_use)))
    km = MiniBatchKMeans(
        n_clusters=n_use,
        batch_size=batch,
        n_init="auto",
        max_no_improvement=30,
        reassignment_ratio=0.01,
        random_state=random_state,
        verbose=0,
    )
    return np.asarray(km.fit_predict(X), dtype=int)


def consolidate_macros(W, labels, max_iters: int = 100) -> np.ndarray:
    """Merge fragile macro-components using volume and conductance heuristics.

    This is a heuristic post-processing step. It merges very small-volume
    clusters or conductance outliers into the cluster to which they have the
    strongest total edge weight.
    """
    W = _symmetrize_sum(W)

    shape = W.shape
    if shape is None or len(shape) != 2:
        raise ValueError("A must be a 2-D sparse matrix.")

    n_rows = int(shape[0])
    n_cols = int(shape[1])

    if n_rows != n_cols:
        raise ValueError("A must be square.")

    labels = _as_1d_labels(labels, n_rows).copy()

    max_iters = int(max_iters)
    if max_iters < 1:
        _, labels_new = np.unique(labels, return_inverse=True)
        return labels_new.astype(int)

    for _ in range(max_iters):
        uniq = np.unique(labels)
        if uniq.size <= 2:
            break

        deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)

        idx_list: list[np.ndarray] = []
        phi: list[float] = []
        vols: list[float] = []

        for g in uniq:
            idx = np.where(labels == g)[0]
            idx_list.append(idx)

            vol = float(deg[idx].sum())
            vols.append(vol)

            if idx.size == 0 or vol <= 0:
                phi.append(1.0)
                continue

            comp = W[idx, :][:, idx]
            # For symmetric adjacency with zero diagonal, comp.sum() counts
            # internal undirected edge weights twice.
            internal_twice = float(comp.sum())
            ext = max(0.0, float(vol - internal_twice))
            phi.append(ext / (vol + 1e-12))

        phi_arr = np.asarray(phi, dtype=float)
        vols_arr = np.asarray(vols, dtype=float)
        merged = False

        # Phase 1: merge tiny-volume components into their heaviest neighbour.
        med_vol = float(np.median(vols_arr)) if vols_arr.size else 0.0
        if med_vol > 0:
            for i, (idx_t, vol) in enumerate(zip(idx_list, vols_arr, strict=False)):
                if vol >= 0.6 * med_vol or idx_t.size == 0:
                    continue

                best_neighbor_pos = None
                best_w = 0.0
                for j, idx_other in enumerate(idx_list):
                    if j == i or idx_other.size == 0:
                        continue
                    w = float(W[idx_t, :][:, idx_other].sum())
                    if w > best_w:
                        best_w = w
                        best_neighbor_pos = j

                if best_neighbor_pos is not None and best_w > 0:
                    labels[idx_t] = uniq[best_neighbor_pos]
                    merged = True

        # Phase 2: conductance-based outlier merge.
        if not merged and phi_arr.size >= 3:
            q1, q3 = np.percentile(phi_arr, [25, 75])
            thr = q3 + (q3 - q1)
            mask = phi_arr > thr

            if mask.any():
                worst_pos = int(np.argmax(np.where(mask, phi_arr, -np.inf)))
                idx_worst = idx_list[worst_pos]

                best_neighbor_label = None
                best_w = 0.0
                for pos, idx_other in enumerate(idx_list):
                    if pos == worst_pos or idx_other.size == 0:
                        continue
                    w = float(W[idx_worst, :][:, idx_other].sum())
                    if w > best_w:
                        best_w = w
                        best_neighbor_label = uniq[pos]

                if best_neighbor_label is not None and best_w > 0:
                    labels[idx_worst] = best_neighbor_label
                    merged = True

        if not merged:
            break

        _, labels = np.unique(labels, return_inverse=True)

    _, labels_new = np.unique(labels, return_inverse=True)
    return labels_new.astype(int)


def louvain_micro(
    S,
    random_state=0,
    max_passes: int = 100,
    gamma: float = 0.85,
) -> np.ndarray:
    """Greedy one-level Louvain-like modularity clustering.

    This is intentionally dependency-free. It performs local node moves but does
    not implement the full multilevel Louvain aggregation cycle.
    """
    rng = np.random.RandomState(int(random_state))
    # S = _symmetrize_sum(S)
    n_rows, n_cols = S.shape
    if n_rows != n_cols:
        raise ValueError("S must be square.")
    n = int(n_rows)

    if n <= 2 or S.nnz == 0:
        return np.zeros(n, dtype=int)

    total_weight = float(S.sum())
    if total_weight <= 0:
        return np.zeros(n, dtype=int)

    ki = np.asarray(S.sum(axis=1)).ravel().astype(np.float64)
    labels = np.arange(n, dtype=int)

    indptr, indices, data = S.indptr, S.indices, S.data
    max_passes = int(max_passes)

    for _pass in range(max(1, max_passes)):
        improved = False

        # Recompute community degrees after any label compaction or previous pass.
        n_com = int(labels.max()) + 1
        com_deg = np.bincount(labels, weights=ki, minlength=n_com).astype(np.float64)

        order = np.arange(n)
        rng.shuffle(order)

        for v in order:
            v_lab = int(labels[v])
            k_v = float(ki[v])
            if k_v <= 0:
                continue

            start, end = indptr[v], indptr[v + 1]
            nbrs = indices[start:end]
            wts = data[start:end]

            com_w: dict[int, float] = {}
            for u, wvu in zip(nbrs, wts, strict=False):
                cu = int(labels[u])
                com_w[cu] = com_w.get(cu, 0.0) + float(wvu)

            best_c = v_lab
            best_dq = 0.0

            # Remove v from its current community while evaluating moves.
            com_deg[v_lab] -= k_v

            for c, k_v_in in com_w.items():
                dq = float(k_v_in) - float(gamma) * (k_v * com_deg[c] / total_weight)
                if dq > best_dq + 1e-12:
                    best_dq = dq
                    best_c = int(c)

            if best_c != v_lab:
                labels[v] = best_c
                com_deg[best_c] += k_v
                improved = True
            else:
                com_deg[v_lab] += k_v

        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(int)

        if not improved:
            break

    return labels.astype(int)


def find_components(
    P,
    random_state=0,
    consolidate: bool = True,
    max_passes: int = 100,
    gamma: float = 0.85,
) -> tuple[int, np.ndarray]:
    """Discover macro-components under the Union-of-Manifolds hypothesis.

    Parameters
    ----------
    P : sparse matrix or array-like, shape (n_samples, n_samples)
        Diffusion/operator matrix. The geometric overlap symmetrization
        ``sqrt(P_ij P_ji)`` is used for robust component discovery.
    random_state : int, default=0
        Random seed for Louvain order and MiniBatchKMeans.
    consolidate : bool, default=True
        Whether to merge fragile macro-components after spectral clustering.
    max_passes : int, default=100
        Maximum number of local-move passes in the Louvain-like micro-clustering.
    gamma : float, default=0.85
        Resolution parameter for micro-clustering.

    Returns
    -------
    n_comp : int
        Number of discovered components.
    labels : ndarray of int, shape (n_samples,)
        Component label per sample.
    """
    from scipy.sparse.linalg import eigsh

    from topo._compat.scipy_graph import graph_connected_components

    S = _symmetrize_geometric(P)
    shape = S.shape
    if shape is None or len(shape) != 2:
        raise ValueError("A must be a 2-D sparse matrix.")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if n_rows != n_cols:
        raise ValueError("A must be square.")

    if n_rows == 0:
        raise ValueError("P must contain at least one sample.")
    if S.nnz == 0:
        return 1, np.zeros(n_rows, dtype=int)

    n_cc, cc_labels = graph_connected_components(S, directed=False, return_labels=True)
    cc_labels = np.asarray(cc_labels, dtype=int)

    # If many disconnected components are already present, use them directly.
    # For <=3 components, continue to micro/macro splitting because each
    # connected component may still contain multiple weakly linked manifolds.
    if n_cc > 3:
        _, labels = np.unique(cc_labels, return_inverse=True)
        return int(labels.max() + 1), labels.astype(int)

    micro = louvain_micro(
        S,
        random_state=random_state,
        max_passes=max_passes,
        gamma=gamma,
    )
    _, micro_labels = np.unique(micro, return_inverse=True)
    micro_labels = micro_labels.astype(int)
    k = int(micro_labels.max() + 1)

    if k <= 1:
        return 1, np.zeros(n_rows, dtype=int)

    # Aggregate micro-clusters into a weighted macro graph.
    S_coo = S.tocoo()
    rows, cols, vals = S_coo.row, S_coo.col, S_coo.data
    mr, mc = micro_labels[rows], micro_labels[cols]
    upper = mr < mc

    if not np.any(upper):
        return k, micro_labels

    r, c, w = mr[upper], mc[upper], vals[upper]
    idx = r * k + c
    acc = np.bincount(idx, weights=w, minlength=k * k).astype(np.float32).reshape(k, k)
    W = csr_matrix(acc + acc.T, dtype=np.float32)
    W.setdiag(0)
    W.eliminate_zeros()

    if W.nnz == 0:
        return k, micro_labels

    if k <= 2:
        return k, micro_labels

    Lw = _normalized_laplacian(W)

    k_max = int(min(8, max(3, np.floor(np.sqrt(k) + 1))))
    k_max = int(min(k_max, k))
    if k_max < 2:
        return 1, np.zeros(n_rows, dtype=int)

    # eigsh requires k < N. For tiny macro graphs, use dense eigh.
    if k <= 3:
        vals_w, vecs_w = np.linalg.eigh(Lw.toarray())
    else:
        nev = int(min(k_max + 1, k - 1))
        vals_w, vecs_w = eigsh(Lw, k=nev, which="SM")

    order = np.argsort(vals_w)
    vals_w = np.asarray(vals_w)[order]
    vecs_w = np.asarray(vecs_w)[:, order]

    k_macro = _eigengap_k(vals_w, k_max=k_max, k_min=2)
    k_macro = int(min(max(1, k_macro), k))

    Uw = vecs_w[:, :k_macro]
    row_norm = np.linalg.norm(Uw, axis=1, keepdims=True)
    Uw = Uw / (row_norm + 1e-12)

    macro = mbkm(Uw, n_clusters=k_macro, random_state=random_state)

    if consolidate and np.unique(macro).size > 2:
        macro = consolidate_macros(W, macro)

    labels = macro[micro_labels]
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(int)
    return int(labels.max() + 1), labels


# ---------------------------------------------------------------------------
# Lightweight proxy kernel (used for tiny/aggregated blocks)
# ---------------------------------------------------------------------------


class _ProxyKernel:
    """Minimal Kernel-like object exposing ``P`` and ``K``."""

    __slots__ = ("P", "K", "L")

    def __init__(self, P):
        P = as_float32_csr(P, "P")
        self.P = P
        self.K = P
        self.L = _normalized_laplacian(_symmetrize_sum(P))


# ---------------------------------------------------------------------------
# UoMMixin — mixed into TopOGraph
# ---------------------------------------------------------------------------


class UoMMixin:
    """Union-of-Manifolds state and per-component fit pipeline.

    Provides the UoM state initialization and the per-component fit pipeline
    used by :meth:`topo.topograph.TopOGraph.fit`.
    """

    # ------------------------------------------------------------------
    # Interface contract — attributes supplied by the host class (TopOGraph).
    # ------------------------------------------------------------------

    # Core geometry
    n: int | None
    verbosity: int

    # kNN / kernel settings
    base_knn: int
    base_metric: str
    base_kernel_version: str
    graph_knn: int
    graph_metric: str
    graph_kernel_version: str

    # Eigendecomposition settings
    n_eigs: int
    n_eigs_: int | None
    eigensolver: str
    eigen_tol: float
    diff_t: int

    # Intrinsic-dimensionality estimation settings
    id_method: str
    id_ks: Any
    id_metric: str
    id_quantile: float
    id_min_components: int
    id_max_components: int
    id_headroom: float

    # Memory / caching
    low_memory: bool

    # Projection
    projection_methods: list[str]

    # Computed state
    _backend_resolved: str
    _random_state_resolved: np.random.RandomState
    _n_jobs_effective: int
    _knn_Z: Any
    _knn_msZ: Any

    # Methods provided by the host class
    def _build_kernel(self, *args: Any, **kwargs: Any) -> Any:
        """Build kernel (provided by host class)."""
        raise NotImplementedError("_build_kernel provided by host class")

    def spectral_layout(self, *args: Any, **kwargs: Any) -> Any:
        """Compute spectral layout (provided by host class)."""
        raise NotImplementedError("spectral_layout provided by host class")

    def project(self, *args: Any, **kwargs: Any) -> Any:
        """Project data (provided by host class)."""
        raise NotImplementedError("project provided by host class")

    def _init_uom_state(self) -> None:
        """Initialize all UoM-specific attributes."""
        self.uom = getattr(self, "uom", False)
        self.uom_enabled = bool(self.uom)

        self.uom_comp_labels_: np.ndarray | None = None
        self.uom_components_: list[np.ndarray] | None = None

        self.uom_knn_X_list = None
        self.knn_X_uom = None
        self.P_of_X_uom = None
        self.uom_BaseKernel_list = None
        self.uom_DMEig_list = None
        self.uom_msDMEig_list = None
        self.uom_eigenvalues_dm_list: list[np.ndarray] | None = None
        self.uom_eigenvalues_ms_list: list[np.ndarray] | None = None
        self._uom_active_mode = "msDM"
        self.uom_Z_list = None
        self.uom_msZ_list = None
        self.uom_knn_Z_list = None
        self.uom_knn_msZ_list = None
        self.uom_Kernel_Z_list = None
        self.uom_Kernel_msZ_list = None

        self.Z_uom = None
        self.msZ_uom = None
        self.knn_Z_uom = None
        self.knn_msZ_uom = None
        self.P_of_Z_uom = None
        self.P_of_msZ_uom = None
        self._uom_axis_slices = None

        self.verbosity = getattr(self, "verbosity", 0)

    # -----------------------------------------------------------------
    # Public component-finding method
    # -----------------------------------------------------------------

    def uom_find_components(
        self,
        P,
        random_state=0,
        consolidate: bool = True,
        max_passes: int = 100,
        gamma: float = 0.85,
    ) -> tuple[int, np.ndarray]:
        """Discover macro-components under the UoM hypothesis."""
        n_comp, labels = find_components(
            P,
            random_state=random_state,
            consolidate=consolidate,
            max_passes=max_passes,
            gamma=gamma,
        )
        labels = _as_1d_labels(labels)
        self.uom_comp_labels_ = labels
        self.uom_components_ = [np.where(labels == c)[0] for c in np.unique(labels)]
        return n_comp, labels

    # -----------------------------------------------------------------
    # Per-component fit pipeline
    # -----------------------------------------------------------------

    def _fit_uom(self, X):
        """Run the UoM branch of ``fit()``.

        Detects components and builds per-component scaffolds, refined graphs
        and projections, aggregating results into block-diagonal operators.
        """
        if self.verbosity >= 1:
            logger.info(
                "UoM: detecting components in P(X) and building "
                "per-component scaffolds/graphs."
            )

        n_total = self.n
        if n_total is None:
            if X is None:
                raise ValueError("UoM fitting requires known sample count.")
            n_total = int(X.shape[0])
            self.n = n_total

        if (self.uom_comp_labels_ is not None) and (
            self.uom_comp_labels_.shape[0] == n_total
        ):
            labels = _as_1d_labels(self.uom_comp_labels_, n_total)
            n_comp = int(np.unique(labels).size)
            if self.verbosity >= 1:
                logger.info("UoM: using precomputed component labels (n=%s).", n_comp)
        else:
            base_kernel = getattr(self, "base_kernel", None)
            if base_kernel is None:
                raise ValueError(
                    "Base kernel is required before UoM component detection."
                )
            n_comp, labels = self.uom_find_components(P=base_kernel.P)
            labels = _as_1d_labels(labels, n_total)
            if self.verbosity >= 1:
                logger.info("UoM: computed component labels (n=%s).", n_comp)

        self.uom_comp_labels_ = labels
        self.uom_components_ = [np.where(labels == c)[0] for c in np.unique(labels)]

        (
            self.uom_knn_X_list,
            self.uom_BaseKernel_list,
            self.uom_DMEig_list,
            self.uom_msDMEig_list,
        ) = [], [], [], []

        self.uom_Z_list, self.uom_msZ_list = [], []
        self.uom_knn_Z_list, self.uom_knn_msZ_list = [], []
        self.uom_Kernel_Z_list, self.uom_Kernel_msZ_list = [], []
        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        for comp_id, idx in enumerate(self.uom_components_):
            n_i = int(idx.size)

            if n_i < 3:
                self._fit_uom_tiny_component(idx, n_i)
                continue

            Xi = self._get_component_data(X, idx)
            if Xi is None:
                raise ValueError(
                    "Component data is unavailable for UoM kNN construction."
                )

            k_neighbors_i = min(int(self.base_knn), max(1, n_i - 1))
            knn_i = kNN(
                Xi,
                n_neighbors=k_neighbors_i,
                metric=self.base_metric,
                n_jobs=self._n_jobs_effective,
                backend=self._backend_resolved,
                verbose=False,
            )
            knn_i = as_float32_csr(knn_i, "knn_i")
            self.uom_knn_X_list.append(knn_i)

            Ki = self._build_kernel(
                knn_i,
                k_neighbors_i,
                self.base_kernel_version,
                data_for_expansion=Xi,
                base=True,
            )

            self.uom_BaseKernel_list.append(Ki)

            Ki_mat = getattr(Ki, "K", None)
            if Ki_mat is None:
                Ki_mat = getattr(Ki, "P", None)
            if Ki_mat is None:
                raise RuntimeError("UoM component kernel has neither K nor P.")

            N_i = int(Ki_mat.shape[0])
            if N_i <= 2:
                self._fit_uom_tiny_component(idx, N_i)
                continue

            k_i = self._local_uom_size(Xi, n_max=N_i - 2)

            n_eigs_eff = self.n_eigs_ if self.n_eigs_ is not None else self.n_eigs
            k_req = int(min(max(k_i, 2), N_i - 1, int(n_eigs_eff)))
            k_req = max(1, k_req)

            eig_dm_i = EigenDecomposition(
                n_components=k_req,
                method="DM",
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                t=self.diff_t,
                random_state=self._random_state_resolved,
                verbose=False,
            ).fit(Ki)

            eig_ms_i = copy.deepcopy(eig_dm_i)
            eig_ms_i.method = "msDM"

            if eig_dm_i.eigenvalues is None or eig_ms_i.eigenvalues is None:
                raise RuntimeError("UoM eigendecomposition did not expose eigenvalues.")

            self.uom_eigenvalues_dm_list.append(
                np.asarray(eig_dm_i.eigenvalues, dtype=float).copy()
            )
            self.uom_eigenvalues_ms_list.append(
                np.asarray(eig_ms_i.eigenvalues, dtype=float).copy()
            )

            Zi_all = eig_dm_i.transform()
            msZi_all = eig_ms_i.transform()
            if Zi_all is None or msZi_all is None:
                raise RuntimeError("UoM eigendecomposition transform returned None.")

            Zi_all = np.asarray(Zi_all, dtype=np.float32)
            msZi_all = np.asarray(msZi_all, dtype=np.float32)

            k_avail = min(Zi_all.shape[1], msZi_all.shape[1])
            k_use = int(min(k_i, k_avail))
            k_use = max(1, k_use)

            Zi = Zi_all[:, :k_use]
            msZi = msZi_all[:, :k_use]

            self.uom_DMEig_list.append(eig_dm_i)
            self.uom_msDMEig_list.append(eig_ms_i)
            self.uom_Z_list.append(Zi)
            self.uom_msZ_list.append(msZi)

            k_graph_i = min(int(self.graph_knn), max(1, n_i - 1))
            knn_Z_i = as_float32_csr(
                kNN(
                    Zi,
                    n_neighbors=k_graph_i,
                    metric=self.graph_metric,
                    n_jobs=self._n_jobs_effective,
                    backend=self._backend_resolved,
                    verbose=False,
                )
            )
            knn_msZ_i = as_float32_csr(
                kNN(
                    msZi,
                    n_neighbors=k_graph_i,
                    metric=self.graph_metric,
                    n_jobs=self._n_jobs_effective,
                    backend=self._backend_resolved,
                    verbose=False,
                )
            )

            self.uom_knn_Z_list.append(knn_Z_i)
            self.uom_knn_msZ_list.append(knn_msZ_i)

            KZ_i = self._build_kernel(
                knn_Z_i,
                k_graph_i,
                self.graph_kernel_version,
                data_for_expansion=Zi,
                base=False,
            )
            KmsZ_i = self._build_kernel(
                knn_msZ_i,
                k_graph_i,
                self.graph_kernel_version,
                data_for_expansion=msZi,
                base=False,
            )

            self.uom_Kernel_Z_list.append(KZ_i)
            self.uom_Kernel_msZ_list.append(KmsZ_i)

        self._aggregate_uom_blocks()

        if self.K_msZ_ is None:
            raise RuntimeError("UoM msDM scaffold affinity was not built.")

        _ = self.spectral_layout(graph=self.K_msZ_, n_components=2)

        for proj in self.projection_methods:
            for ms in (True, False):
                try:
                    self.project(projection_method=proj, multiscale=ms)
                except Exception as exc:
                    tag = "msZ" if ms else "Z/DM"
                    warnings.warn(
                        f"Projection '{proj}' on {tag} (UoM) failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if self.low_memory:
            self.uom_BaseKernel_list = None
            self.uom_DMEig_list = None
            self.uom_msDMEig_list = None
            self.uom_Kernel_Z_list = None
            self.uom_Kernel_msZ_list = None

        return self

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _local_uom_size(self, Xi_or_knn, n_max: int) -> int:
        """Estimate component-local scaffold size."""
        if Xi_or_knn is None:
            raise ValueError("Xi_or_knn must not be None.")

        n_i = int(Xi_or_knn.shape[0])
        if n_i <= 2:
            return 1

        cap = min(int(self.id_max_components), max(2, n_i - 2), int(n_max))
        min_components = int(min(self.id_min_components, cap))
        max_components = int(min(cap, n_max))
        if max_components < 1:
            return 1

        res = automated_scaffold_sizing(
            Xi_or_knn,
            method=self.id_method,
            ks=self.id_ks,
            backend=self._backend_resolved,
            metric=self.id_metric,
            n_jobs=self._n_jobs_effective,
            quantile=self.id_quantile,
            min_components=max(1, min_components),
            max_components=max(1, max_components),
            headroom=float(self.id_headroom),
            random_state=self._random_state_resolved,
            return_details=False,
        )

        k_auto = res[0] if isinstance(res, tuple) else res
        k_auto = int(k_auto)
        return int(max(1, min(k_auto, cap)))

    def _fit_uom_tiny_component(self, idx, n_i: int) -> None:
        """Fallback for components with fewer than three samples."""
        if n_i < 1:
            raise ValueError("Tiny UoM component must contain at least one sample.")

        uom_Z_list = self.uom_Z_list
        uom_msZ_list = self.uom_msZ_list
        uom_knn_Z_list = self.uom_knn_Z_list
        uom_knn_msZ_list = self.uom_knn_msZ_list
        uom_Kernel_Z_list = self.uom_Kernel_Z_list
        uom_Kernel_msZ_list = self.uom_Kernel_msZ_list
        uom_BaseKernel_list = self.uom_BaseKernel_list
        uom_knn_X_list = self.uom_knn_X_list
        uom_DMEig_list = self.uom_DMEig_list
        uom_msDMEig_list = self.uom_msDMEig_list
        uom_eigenvalues_dm_list = self.uom_eigenvalues_dm_list
        uom_eigenvalues_ms_list = self.uom_eigenvalues_ms_list

        if (
            uom_Z_list is None
            or uom_msZ_list is None
            or uom_knn_Z_list is None
            or uom_knn_msZ_list is None
            or uom_Kernel_Z_list is None
            or uom_Kernel_msZ_list is None
            or uom_BaseKernel_list is None
            or uom_knn_X_list is None
            or uom_DMEig_list is None
            or uom_msDMEig_list is None
            or uom_eigenvalues_dm_list is None
            or uom_eigenvalues_ms_list is None
        ):
            raise RuntimeError(
                "UoM component lists must be initialized before fitting."
            )

        Zi = np.zeros((n_i, 1), dtype=np.float32)
        P_block = _sparse_identity(n_i)

        uom_Z_list.append(Zi)
        uom_msZ_list.append(Zi.copy())

        uom_knn_Z_list.append(P_block.copy())
        uom_knn_msZ_list.append(P_block.copy())
        uom_Kernel_Z_list.append(_ProxyKernel(P_block))
        uom_Kernel_msZ_list.append(_ProxyKernel(P_block))
        uom_BaseKernel_list.append(_ProxyKernel(P_block))
        uom_knn_X_list.append(P_block.copy())

        uom_DMEig_list.append(None)
        uom_msDMEig_list.append(None)

        uom_eigenvalues_dm_list.append(np.ones(1, dtype=float))
        uom_eigenvalues_ms_list.append(np.ones(1, dtype=float))

    def _get_component_data(self, X, idx):
        """Slice input data for a UoM component."""
        idx = np.asarray(idx, dtype=int)

        base_kernel = getattr(self, "base_kernel", None)
        kernel_X = getattr(base_kernel, "X", None) if base_kernel is not None else None

        source = X if X is not None else kernel_X
        if source is None:
            return None

        if self.base_metric == "precomputed":
            return source[idx, :][:, idx]

        return source[idx]

    def _component_order(self) -> np.ndarray:
        """Return original sample indices in component-concatenated order."""
        components = self.uom_components_
        if components is None:
            raise ValueError("UoM components are unavailable.")
        if not components:
            raise ValueError("UoM components list is empty.")

        order = np.concatenate([np.asarray(idx, dtype=int) for idx in components])

        n = self.n
        if n is None:
            raise ValueError("UoM aggregation requires fitted sample count.")

        if order.size != n:
            raise ValueError(
                f"Component indices cover {order.size} samples, expected {n}."
            )
        if np.unique(order).size != order.size:
            raise ValueError("UoM component indices contain duplicates.")

        return order

    def _block_diag_to_original_order(self, blocks) -> csr_matrix:
        """Build block diagonal matrix and permute rows/cols to original order."""
        n = self.n
        if n is None:
            raise ValueError("UoM aggregation requires fitted sample count.")

        csr_blocks = [as_float32_csr(B, "B") for B in blocks]
        B_cat = block_diag(csr_blocks, format="csr", dtype="float32")

        order = self._component_order()
        if B_cat.shape != (order.size, order.size):
            raise ValueError(
                f"Block-diagonal shape {B_cat.shape} does not match component "
                f"order length {order.size}."
            )

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)

        return csr_matrix(B_cat[inv_order, :][:, inv_order])

    def _aggregate_scaffold_to_original_order(
        self, blocks
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Aggregate per-component scaffold blocks into original row order.

        Each component occupies its own column block. Rows outside a component
        are zero in that component's axis block.
        """
        n = self.n
        if n is None:
            raise ValueError("UoM aggregation requires fitted sample count.")

        components = self.uom_components_
        if components is None:
            raise ValueError("UoM components are unavailable.")

        total_cols = int(sum(np.asarray(B).shape[1] for B in blocks))
        out = np.zeros((int(n), total_cols), dtype=np.float32)
        slices: list[tuple[int, int]] = []

        c0 = 0
        for idx, B in zip(components, blocks, strict=True):
            B = np.asarray(B, dtype=np.float32)
            if B.ndim != 2:
                raise ValueError("Each scaffold block must be a 2-D array.")
            if B.shape[0] != len(idx):
                raise ValueError("Scaffold block row count must match component size.")

            c1 = c0 + B.shape[1]
            out[np.asarray(idx, dtype=int), c0:c1] = B
            slices.append((c0, c1))
            c0 = c1

        return out, slices

    def _aggregate_uom_blocks(self) -> None:
        """Assemble per-component results into original-order aggregates."""
        uom_Z_list = self.uom_Z_list
        uom_msZ_list = self.uom_msZ_list
        components = self.uom_components_
        uom_knn_X_list = self.uom_knn_X_list
        uom_BaseKernel_list = self.uom_BaseKernel_list
        uom_knn_Z_list = self.uom_knn_Z_list
        uom_knn_msZ_list = self.uom_knn_msZ_list
        uom_Kernel_Z_list = self.uom_Kernel_Z_list
        uom_Kernel_msZ_list = self.uom_Kernel_msZ_list

        if (
            uom_Z_list is None
            or uom_msZ_list is None
            or components is None
            or uom_knn_X_list is None
            or uom_BaseKernel_list is None
            or uom_knn_Z_list is None
            or uom_knn_msZ_list is None
            or uom_Kernel_Z_list is None
            or uom_Kernel_msZ_list is None
        ):
            raise RuntimeError(
                "UoM component lists must be initialized before aggregation."
            )

        n = self.n
        if n is None:
            raise ValueError("UoM aggregation requires fitted sample count.")

        n_comp = len(components)
        expected_lists = [
            uom_Z_list,
            uom_msZ_list,
            uom_knn_X_list,
            uom_BaseKernel_list,
            uom_knn_Z_list,
            uom_knn_msZ_list,
            uom_Kernel_Z_list,
            uom_Kernel_msZ_list,
        ]
        for lst in expected_lists:
            if len(lst) != n_comp:
                raise RuntimeError(
                    "UoM component lists are misaligned: expected "
                    f"{n_comp}, got {len(lst)}."
                )

        self.Z_uom, self._uom_axis_slices = self._aggregate_scaffold_to_original_order(
            uom_Z_list
        )

        self.msZ_uom, _ = self._aggregate_scaffold_to_original_order(uom_msZ_list)

        self.knn_X_uom = self._block_diag_to_original_order(uom_knn_X_list)
        self.P_of_X_uom = self._block_diag_to_original_order(
            [K.P for K in uom_BaseKernel_list]
        )
        self.knn_Z_uom = self._block_diag_to_original_order(uom_knn_Z_list)
        self.knn_msZ_uom = self._block_diag_to_original_order(uom_knn_msZ_list)
        self.P_of_Z_uom = self._block_diag_to_original_order(
            [K.P for K in uom_Kernel_Z_list]
        )
        self.P_of_msZ_uom = self._block_diag_to_original_order(
            [K.P for K in uom_Kernel_msZ_list]
        )

        # Canonical fitted outputs used by TopOGraph public properties.
        self.Z_ = self.Z_uom
        self.msZ_ = self.msZ_uom
        self.evals_Z_ = None
        self.evals_msZ_ = None
        self.knn_X_ = self.knn_X_uom
        self.knn_Z_ = self.knn_Z_uom
        self.knn_msZ_ = self.knn_msZ_uom
        self.P_X_ = csr_matrix(self.P_of_X_uom)
        self.P_Z_ = csr_matrix(self.P_of_Z_uom)
        self.P_msZ_ = csr_matrix(self.P_of_msZ_uom)
        self.K_Z_ = csr_matrix(self.P_of_Z_uom)
        self.K_msZ_ = csr_matrix(self.P_of_msZ_uom)

        # Internal kernel-like wrappers for downstream layout code.
        self.eigenbasis = None
        self._knn_Z = self.knn_Z_uom
        self._knn_msZ = self.knn_msZ_uom
