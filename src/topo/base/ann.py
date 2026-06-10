"""Nearest-neighbor graph construction.

Supported backends:
- sklearn.neighbors: exact kNN; correctness/reference backend.
- HNSWlib: optional fast dense-vector backend for large datasets.

Important conventions
---------------------
- ``n_neighbors`` is the user-requested number of non-self neighbors.
- Self-query paths query ``n_neighbors + 1`` internally, then remove the
  zero-distance self hit.
- Returned sparse graphs have shape ``(n_query_samples, n_fit_samples)``.
- Self-neighbor removal is centralized via
  :func:`_drop_self_and_truncate_neighbors`.

Unsupported cases for HNSWlib, such as sparse input, precomputed distances, and
explicit query matrix ``Y``, are routed through sklearn.
"""

import logging
import time
from typing import Any
from warnings import warn

import numpy as np
from joblib import cpu_count
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from topo.base.graph_matrix import as_csr_matrix

logger = logging.getLogger(__name__)

# Clamp value for genuine zero-distance neighbor edges (duplicate points).
# float32 tiny survives both float32 and float64 sparse-data round-trips,
# unlike float64 tiny which underflows to 0.0 when graphs are cast to float32.
_ZERO_DISTANCE_TINY = float(np.finfo(np.float32).tiny)


def _resolve_n_jobs(n_jobs: int | str | None) -> int:
    """Resolve sklearn/joblib-style n_jobs."""
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return cpu_count()
    return int(n_jobs)


def _validate_n_neighbors(n_neighbors: int | float | str) -> int:
    """Validate and normalize a neighbor count."""
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    return n_neighbors


def _query_k(n_neighbors: int, n_fit_samples: int) -> int:
    """Return internal query k, including possible self-neighbor."""
    n_neighbors = _validate_n_neighbors(n_neighbors)
    if n_fit_samples < 1:
        raise ValueError("Cannot query an empty index.")
    # Self-query neighbor searches usually return the query point itself first.
    return min(n_neighbors + 1, n_fit_samples)


def _drop_self_and_truncate_neighbors(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_neighbors: int,
    n_query_samples: int,
    n_fit_samples: int,
    is_self_query: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove self hits when requested and keep exactly ``n_neighbors`` per row."""
    n_neighbors = _validate_n_neighbors(n_neighbors)
    indices = np.asarray(indices)
    distances = np.asarray(distances)

    if indices.shape != distances.shape or indices.ndim != 2:
        raise ValueError("indices and distances must be matching 2-D arrays.")
    if indices.shape[0] != n_query_samples:
        raise ValueError("indices row count does not match n_query_samples.")
    if indices.size and (indices.min() < 0 or indices.max() >= n_fit_samples):
        raise ValueError("indices contain values outside [0, n_fit_samples).")

    out_indices = np.empty((n_query_samples, n_neighbors), dtype=indices.dtype)
    out_distances = np.empty((n_query_samples, n_neighbors), dtype=distances.dtype)

    for row in range(n_query_samples):
        row_indices = indices[row]
        row_distances = distances[row]

        if is_self_query:
            is_self = (row_indices == row) & np.isclose(row_distances, 0.0)
            row_indices = row_indices[~is_self]
            row_distances = row_distances[~is_self]

        if row_indices.shape[0] < n_neighbors:
            raise ValueError(
                "Neighbor search returned fewer than n_neighbors results. "
                "Use n_neighbors < n_samples for self-query graphs."
            )

        out_indices[row] = row_indices[:n_neighbors]
        out_distances[row] = row_distances[:n_neighbors]

    return out_indices, out_distances


def _as_dense_array(data: Any) -> np.ndarray:
    """Return data as a dense 2-D numpy array for HNSWlib."""
    if issparse(data):
        raise ValueError("Sparse input cannot be converted for HNSWlib backend.")

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    return arr


def _check_2d_data(data: object, name: str = "data") -> tuple[int, int]:
    """Validate that data has a 2-D shape."""
    shape = getattr(data, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2-D array-like or sparse matrix.")
    n_samples, n_features = int(shape[0]), int(shape[1])
    if n_samples < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if n_features < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    return n_samples, n_features


def _build_sparse_knn_graph(
    indices: np.ndarray,
    distances: np.ndarray,
    n_query_samples: int,
    n_fit_samples: int,
    *,
    n_neighbors: int | None = None,
    is_self_query: bool = False,
) -> csr_matrix:
    """Build a CSR neighbor-distance graph from dense index/distance arrays.

    Parameters
    ----------
    indices : ndarray of shape (n_query_samples, k)
        Neighbor column indices.
    distances : ndarray of shape (n_query_samples, k)
        Distances for the corresponding indices.
    n_query_samples : int
        Number of query samples / output rows.
    n_fit_samples : int
        Number of fitted/reference samples / output columns.
    n_neighbors : int, optional
        If provided, truncate each row to exactly this many neighbors after
        optional self-removal.
    is_self_query : bool, default=False
        Whether the query samples are the fitted samples and zero-distance
        diagonal hits should be removed.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        Sparse distance graph of shape ``(n_query_samples, n_fit_samples)``.
    """
    indices = np.asarray(indices)
    distances = np.asarray(distances)

    if indices.ndim != 2 or distances.ndim != 2:
        raise ValueError("indices and distances must be 2-D arrays.")
    if indices.shape != distances.shape:
        raise ValueError("indices and distances must have the same shape.")
    if indices.shape[0] != n_query_samples:
        raise ValueError("indices row count does not match n_query_samples.")
    if n_query_samples < 1:
        raise ValueError("n_query_samples must be >= 1.")
    if n_fit_samples < 1:
        raise ValueError("n_fit_samples must be >= 1.")
    if indices.size and (indices.min() < 0 or indices.max() >= n_fit_samples):
        raise ValueError("indices contain values outside [0, n_fit_samples).")
    if not np.issubdtype(distances.dtype, np.number):
        raise ValueError("distances must be numeric.")

    if n_neighbors is not None:
        indices, distances = _drop_self_and_truncate_neighbors(
            indices,
            distances,
            n_neighbors=n_neighbors,
            n_query_samples=n_query_samples,
            n_fit_samples=n_fit_samples,
            is_self_query=is_self_query,
        )

    k = int(indices.shape[1])
    indptr = np.arange(0, n_query_samples * k + 1, k, dtype=np.int64)

    vals = distances.ravel()
    if not np.issubdtype(vals.dtype, np.floating):
        vals = vals.astype(np.float64)

    # Genuine zero-distance neighbors (duplicate points) must survive CSR
    # storage; clamp off-diagonal zeros to the smallest positive float so
    # eliminate_zeros() below only drops self-loops.
    cols = indices.ravel()
    rows = np.repeat(np.arange(n_query_samples), k)
    off_diagonal = (
        rows != cols
        if n_query_samples == n_fit_samples
        else np.ones(vals.shape, dtype=bool)
    )
    vals = np.where((vals <= 0) & off_diagonal, _ZERO_DISTANCE_TINY, vals)

    graph = csr_matrix(
        (vals, cols, indptr),
        shape=(n_query_samples, n_fit_samples),
    )
    graph.eliminate_zeros()
    return graph


def _hnswlib_space(metric: str) -> str:
    spaces = {
        "sqeuclidean": "l2",
        "euclidean": "l2",
        "cosine": "cosine",
        "inner_product": "ip",
    }
    try:
        return spaces[metric]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported HNSWlib metric: {metric!r}. "
            "Supported metrics are 'sqeuclidean', 'euclidean', 'cosine', "
            "and 'inner_product'."
        ) from exc


def _sklearn_knn_graph(
    X,
    *,
    n_neighbors: int,
    metric: str,
    n_jobs: int = -1,
) -> csr_matrix:
    """Build a self-free kNN distance graph using sklearn."""
    graph = as_csr_matrix(
        kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="distance",
            metric=metric,
            include_self=False,
            n_jobs=n_jobs,
        ),
        "sklearn kneighbors_graph output",
    )
    # Keep genuine zero-distance neighbors (duplicate points): clamp to a
    # tiny positive float so eliminate_zeros() only drops self-loops.
    graph.data = np.where(graph.data <= 0, _ZERO_DISTANCE_TINY, graph.data)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def kNN(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix | None = None,
    n_neighbors: int | float | str = 5,
    metric: str = "euclidean",
    n_jobs: int = -1,
    backend: str = "sklearn",
    M: int = 60,
    efC: int = 200,
    efS: int = 200,
    verbose: bool = False,
    **kwargs: Any,
) -> csr_matrix:
    """Compute a k-nearest-neighbor distance graph.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n_samples, n_features)
        Reference data used to fit the neighbor index. If ``metric='precomputed'``,
        ``X`` must be a square distance matrix.
    Y : array-like or scipy.sparse matrix, optional
        Query data. If provided, the output graph has shape
        ``(Y.shape[0], X.shape[0])``.
    n_neighbors : int, default=5
        Number of non-self neighbors to return for self-query graphs.
    metric : str, default='euclidean'
        Distance metric.
    n_jobs : int, default=-1
        Number of threads. ``-1`` uses all available CPUs.
    backend : {'sklearn', 'hnswlib'}, default='sklearn'
        Neighbor-search backend.
    verbose : bool, default=False
        Emit backend/timing diagnostics through logging.

    Returns
    -------
    scipy.sparse.csr_matrix
        kNN distance graph.
    """
    n_fit_samples, n_features = _check_2d_data(X, "X")

    metric = str(metric)
    backend = str(backend).lower()
    n_neighbors = _validate_n_neighbors(n_neighbors)
    n_jobs = _resolve_n_jobs(n_jobs)

    if backend not in {"sklearn", "hnswlib"}:
        raise ValueError("backend must be one of {'sklearn', 'hnswlib'}.")

    if metric == "precomputed" and n_fit_samples != n_features:
        raise ValueError("X must be square when metric='precomputed'.")

    if Y is None and n_neighbors >= n_fit_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be smaller than n_samples={n_fit_samples} "
            "for self-query kNN graphs."
        )

    if Y is not None:
        _n_query_samples, n_query_features = _check_2d_data(Y, "Y")
        if metric == "precomputed" and n_query_features != n_fit_samples:
            raise ValueError(
                "When metric='precomputed', Y must have shape "
                "(n_query_samples, n_fit_samples)."
            )

    # HNSWlib is dense-feature only in this wrapper. Use exact sklearn for
    # unsupported cases instead of densifying sparse/precomputed inputs.
    use_hnswlib = (
        backend == "hnswlib"
        and Y is None
        and metric != "precomputed"
        and not issparse(X)
    )

    if use_hnswlib:
        X_fit = _as_dense_array(X)
        knn = HNSWlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            M=M,
            efC=efC,
            efS=efS,
            verbose=verbose,
        ).fit_transform(X_fit)
    else:
        if backend == "hnswlib":
            warn(
                "Falling back to sklearn: HNSWlib does not support this input mode "
                "in topo.base.ann.kNN.",
                UserWarning,
                stacklevel=2,
            )

        valid = set(NearestNeighbors().get_params())
        unknown = set(kwargs) - valid
        if unknown:
            raise TypeError(f"Unexpected kNN keyword argument(s): {sorted(unknown)}")
        sk_kwargs = kwargs

        if Y is None and metric != "precomputed":
            knn = _sklearn_knn_graph(
                X,
                n_neighbors=n_neighbors,
                metric=metric,
                n_jobs=n_jobs,
            )
        else:
            query_k = _query_k(n_neighbors, n_fit_samples) if Y is None else n_neighbors
            nbrs = NearestNeighbors(
                n_neighbors=query_k,
                metric=metric,
                n_jobs=n_jobs,
                **sk_kwargs,
            ).fit(X)

            if Y is None:
                distances, indices = nbrs.kneighbors(None, return_distance=True)

                knn = _build_sparse_knn_graph(
                    indices=indices,
                    distances=distances,
                    n_query_samples=n_fit_samples,
                    n_fit_samples=n_fit_samples,
                    n_neighbors=n_neighbors,
                    is_self_query=True,
                )
            else:
                knn = nbrs.kneighbors_graph(Y, mode="distance")

    knn_csr = as_csr_matrix(knn, "knn graph from kNN function")

    return knn_csr


class HNSWlibTransformer:
    """Sklearn-style wrapper around HNSWlib.

    Parameters
    ----------
    n_neighbors : int, default=30
        Number of neighbors to return.

    metric : {'sqeuclidean', 'euclidean', 'cosine', 'inner_product'}, default='cosine'
        HNSWlib metric.

    n_jobs : int, default=-1
        Number of threads.

    M : int, default=60
        HNSW graph connectivity parameter.

    efC : int, default=200
        Construction-time search parameter.

    efS : int, default=200
        Query-time search parameter.

    verbose : bool, default=False
        Print timing/backend messages.
    """

    def __init__(
        self,
        n_neighbors=30,
        metric="cosine",
        n_jobs=-1,
        M=60,
        efC=200,
        efS=200,
        verbose=False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.M = M
        self.efC = efC
        self.efS = efS
        self.verbose = verbose
        self.index_ = None
        self.n_samples_fit_: int | None = None
        self.n_features_in_: int | None = None
        self.space_: str | None = None

    def fit(self, data):
        """Fit the HNSWlib index."""
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "HNSWlib is required for HNSWlibTransformer. "
                "Install it with `pip install hnswlib`."
            ) from exc

        data = _as_dense_array(data)
        n_samples, n_features = _check_2d_data(data)

        self.n_neighbors = _validate_n_neighbors(self.n_neighbors)
        self.n_jobs = _resolve_n_jobs(self.n_jobs)
        self.n_samples_fit_ = n_samples
        self.n_features_in_ = n_features

        start = time.time()

        self.space_ = _hnswlib_space(self.metric)
        self.index_ = hnswlib.Index(space=self.space_, dim=n_features)  # type: ignore
        self.index_.init_index(
            max_elements=n_samples,
            ef_construction=self.efC,
            M=self.M,
        )
        self.index_.set_num_threads(self.n_jobs)
        self.index_.set_ef(self.efS)

        data_labels = np.arange(n_samples)
        self.index_.add_items(data, data_labels)

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "HNSWlib index-time parameters M=%s n_threads=%s efConstruction=%s",
                self.M,
                self.n_jobs,
                self.efC,
            )
            logger.info("Indexing time = %f (sec)", elapsed)

        return self

    def _prepare_query_data(self, data):
        """Convert and validate query data."""
        data = _as_dense_array(data)
        _, n_features = _check_2d_data(data)

        if self.n_features_in_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        if n_features != self.n_features_in_:
            raise ValueError(
                f"Query data has {n_features} features, but the index was "
                f"fit with {self.n_features_in_} features."
            )

        return data

    def transform(self, data, *, is_self_query: bool = False):
        """Return a CSR kNN distance graph for query data."""
        if self.index_ is None or self.n_samples_fit_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        start = time.time()
        query_data = self._prepare_query_data(data)
        n_query_samples, _ = _check_2d_data(query_data, "query_data")
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)

        self.index_.set_ef(self.efS)
        indices, distances = self.index_.knn_query(query_data, k=query_k)

        if self.metric == "euclidean":
            # HNSWlib returns squared L2 distance for space='l2'.
            distances = np.sqrt(distances)

        graph = _build_sparse_knn_graph(
            indices=indices,
            distances=distances,
            n_query_samples=n_query_samples,
            n_fit_samples=self.n_samples_fit_,
            n_neighbors=self.n_neighbors,
            is_self_query=is_self_query,
        )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "Search time = %f (sec), per query = %f (sec)",
                elapsed,
                elapsed / n_query_samples,
            )

        return graph

    def fit_transform(self, X):
        """Fit to X, then return the kNN graph for X."""
        return self.fit(X).transform(X, is_self_query=True)
