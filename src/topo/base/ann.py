#####################################
# Wrappers for approximate nearest neighbor search
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences, University of Campinas, Brazil
# contact: davisidarta@fcm.unicamp.br
######################################

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
from typing import Any, Literal, overload
from warnings import warn

import numpy as np
from joblib import cpu_count
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from topo.base.graph_matrix import as_csr_matrix

logger = logging.getLogger(__name__)


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
    # Most ANN backends return the query point itself as the first neighbor
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


def _as_dense_array(data: np.ndarray | csr_matrix | list) -> np.ndarray:
    """Convert supported array-like inputs to a dense numpy array."""
    if isinstance(data, np.ndarray):
        return data

    if issparse(data):
        return csr_matrix(data).toarray()

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
    except ImportError:
        # pandas is optional; fall back to generic numpy conversion below.
        pass

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    return arr


def _as_csr_matrix(data: np.ndarray | csr_matrix | list) -> csr_matrix:
    """Convert supported array-like inputs to CSR sparse matrix."""
    if isinstance(data, csr_matrix):
        return data

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return csr_matrix(data.to_numpy())
    except ImportError:
        # pandas is optional; fall back to generic CSR conversion below.
        pass

    return csr_matrix(data)


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

    graph = csr_matrix(
        (distances.ravel(), indices.ravel(), indptr),
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
    n_jobs: int | None,
    include_self: bool = False,
) -> csr_matrix:
    """Build kNN distance graph using sklearn kneighbors_graph."""
    graph = as_csr_matrix(
        kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="distance",
            metric=metric,
            include_self=include_self,
            n_jobs=n_jobs,
        )
    )

    if not include_self:
        graph.setdiag(0.0)
        graph.eliminate_zeros()

    return graph


@overload
def kNN(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix | None = None,
    n_neighbors: int | float | str = 5,
    metric: str = "euclidean",
    n_jobs: int | str | None = -1,
    backend: str = "hnswlib",
    low_memory: bool = True,
    M: int = 60,
    p: float = 11 / 16,
    efC: int = 200,
    efS: int = 200,
    n_trees: int = 50,
    *,
    return_instance: Literal[True],
    verbose: bool = False,
    **kwargs: Any,
) -> tuple[BaseEstimator, csr_matrix]: ...


@overload
def kNN(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix | None = None,
    n_neighbors: int | float | str = 5,
    metric: str = "euclidean",
    n_jobs: int | str | None = -1,
    backend: str = "hnswlib",
    low_memory: bool = True,
    M: int = 60,
    p: float = 11 / 16,
    efC: int = 200,
    efS: int = 200,
    n_trees: int = 50,
    return_instance: Literal[False] = False,
    verbose: bool = False,
    **kwargs: Any,
) -> csr_matrix: ...


@overload
def kNN(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix | None = None,
    n_neighbors: int | float | str = 5,
    metric: str = "euclidean",
    n_jobs: int | str | None = -1,
    backend: str = "hnswlib",
    low_memory: bool = True,
    M: int = 60,
    p: float = 11 / 16,
    efC: int = 200,
    efS: int = 200,
    n_trees: int = 50,
    return_instance: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> csr_matrix | tuple[BaseEstimator, csr_matrix]: ...


def kNN(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix | None = None,
    n_neighbors: int | float | str = 5,
    metric: str = "euclidean",
    n_jobs: int | str | None = -1,
    backend: str = "sklearn",
    low_memory: bool = True,
    M: int = 60,
    p: float = 11 / 16,
    efC: int = 200,
    efS: int = 200,
    n_trees: int = 50,
    return_instance: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> csr_matrix | tuple[BaseEstimator, csr_matrix]:
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
    return_instance : bool, default=False
        If True, return ``(estimator, graph)``.
    verbose : bool, default=False
        Emit backend/timing diagnostics through logging.

    Returns
    -------
    scipy.sparse.csr_matrix or tuple
        kNN distance graph, or ``(estimator, graph)`` if ``return_instance=True``.
    """
    del low_memory, p, n_trees  # retained only for backward-compatible signature

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

    nbrs: BaseEstimator | None

    if use_hnswlib:
        X_fit = _as_dense_array(X)
        nbrs = HNSWlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            M=M,
            efC=efC,
            efS=efS,
            verbose=verbose,
        ).fit(X_fit)
        knn = nbrs.transform(X_fit)
    else:
        if verbose and backend == "hnswlib":
            logger.info(
                "Using sklearn because HNSWlib does not support this input mode "
                "in topo.base.ann.kNN."
            )

        _valid = set(NearestNeighbors().get_params())
        sk_kwargs = {k: v for k, v in kwargs.items() if k in _valid}

        if Y is None and metric != "precomputed" and not return_instance:
            knn = _sklearn_knn_graph(
                X,
                n_neighbors=n_neighbors,
                metric=metric,
                n_jobs=n_jobs,
            )
            nbrs = None
        else:
            query_k = _query_k(n_neighbors, n_fit_samples) if Y is None else n_neighbors
            nbrs = NearestNeighbors(
                n_neighbors=query_k,
                metric=metric,
                n_jobs=n_jobs,
                **sk_kwargs,
            ).fit(X)

            if Y is None:
                if metric == "precomputed":
                    distances, indices = nbrs.kneighbors(None, return_distance=True)
                else:
                    distances, indices = nbrs.kneighbors(X, return_distance=True)

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

    if return_instance:
        if nbrs is None:
            raise ValueError(
                "Cannot return estimator instance when using the sklearn fast "
                "kneighbors_graph path. Use return_instance=False or pass "
                "metric='precomputed' to force an estimator-backed path."
            )
        return nbrs, knn_csr

    return knn_csr


def grid_search(
    X,
    n_neighbors=15,
    metric="euclidean",
    hnswlib_params=None,
    n_jobs=-1,
    verbose=False,
):
    """Evaluate approximate kNN graph quality for HNSWlib.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data used to build the neighborhood graph.
    n_neighbors : int, default=15
        Number of non-self neighbors to retrieve.
    metric : str, default='euclidean'
        Distance metric used for neighbor search.
    hnswlib_params : dict, optional
        Parameter grid for HNSWlibTransformer.
    n_jobs : int, default=-1
        Number of parallel jobs for the transformers.
    verbose : bool, default=False
        If True, log recall and timing for each parameter combination.

    Returns
    -------
    dict
        Mapping of backend names to lists of dictionaries containing parameter
        settings, recall and execution time.
    """
    n_samples, _ = _check_2d_data(X, "X")
    n_neighbors = _validate_n_neighbors(n_neighbors)
    n_jobs = _resolve_n_jobs(n_jobs)

    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be smaller than n_samples={n_samples} "
            "for self-query recall evaluation."
        )

    exact_metric = "euclidean" if metric == "sqeuclidean" else metric
    if exact_metric == "inner_product":
        exact_metric = "cosine"
        warn(
            "Using cosine brute-force neighbors as an approximate recall "
            "reference for metric='inner_product'.",
            stacklevel=2,
        )

    query_k = _query_k(n_neighbors, n_samples)
    gt = NearestNeighbors(
        n_neighbors=query_k,
        metric=exact_metric,
        algorithm="brute",
    ).fit(X)
    true_dist, true_ind = gt.kneighbors(X, return_distance=True)
    true_ind, _ = _drop_self_and_truncate_neighbors(
        true_ind,
        true_dist,
        n_neighbors=n_neighbors,
        n_query_samples=n_samples,
        n_fit_samples=n_samples,
        is_self_query=True,
    )

    results = {"hnswlib": []}

    for params in ParameterGrid(hnswlib_params) if hnswlib_params else [{}]:
        model = HNSWlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            **params,
        )

        start = time.time()
        model.fit(X)
        ind, _ = model.ind_dist_grad(
            X,
            return_grad=False,
            return_graph=False,
        )
        elapsed = time.time() - start

        recall = np.mean(
            [
                np.intersect1d(true_ind[i], ind[i]).size / n_neighbors
                for i in range(n_samples)
            ]
        )
        results["hnswlib"].append({"params": params, "recall": recall, "time": elapsed})

    if verbose:
        for backend, backend_results in results.items():
            for row in backend_results:
                logger.info(
                    "%s: params=%s, recall=%.3f, time=%.3fs",
                    backend,
                    row["params"],
                    row["recall"],
                    row["time"],
                )

    return results


class HNSWlibTransformer(TransformerMixin, BaseEstimator):
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
        self.space = metric
        self.verbose = verbose

        self.N = None
        self.m = None
        self.p = None
        self.n_samples_fit_ = None
        self.n_features_in_ = None

    def fit(self, data):
        """Fit the HNSWlib index."""
        try:
            import hnswlib  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "HNSWlib is required for HNSWlibTransformer. "
                "Install it with `pip install hnswlib`."
            ) from exc

        data = _as_dense_array(data)
        _check_2d_data(data)

        self.n_neighbors = _validate_n_neighbors(self.n_neighbors)
        self.n_jobs = _resolve_n_jobs(self.n_jobs)

        self.N, self.m = data.shape
        self.n_samples_fit_ = self.N
        self.n_features_in_ = self.m

        start = time.time()

        self.space = _hnswlib_space(self.metric)
        self.p = hnswlib.Index(space=self.space, dim=self.m)  # type: ignore
        self.p.init_index(
            max_elements=self.N,
            ef_construction=self.efC,
            M=self.M,
        )
        self.p.set_num_threads(self.n_jobs)
        self.p.set_ef(self.efS)

        data_labels = np.arange(self.N)
        self.p.add_items(data, data_labels)

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
        _check_2d_data(data)

        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Query data has {data.shape[1]} features, but the index was "
                f"fit with {self.n_features_in_} features."
            )

        return data

    def transform(self, data):
        """Return a CSR kNN distance graph for query data."""
        if self.p is None or self.n_samples_fit_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        start = time.time()
        query_data = self._prepare_query_data(data)
        n_query_samples, _ = _check_2d_data(query_data, "query_data")
        query_qty = n_query_samples
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)

        self.p.set_ef(self.efS)
        indices, distances = self.p.knn_query(query_data, k=query_k)

        if self.metric == "euclidean":
            # HNSWlib returns squared L2 distance for space='l2'.
            distances = np.sqrt(distances)

        kneighbors_graph = _build_sparse_knn_graph(
            indices=indices,
            distances=distances,
            n_query_samples=n_query_samples,
            n_fit_samples=self.n_samples_fit_,
            n_neighbors=self.n_neighbors,
            is_self_query=(n_query_samples == self.n_samples_fit_),
        )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "Search time =%f (sec), per query=%f (sec), "
                "per query adjusted for thread number=%f (sec)",
                elapsed,
                elapsed / query_qty,
                self.n_jobs * elapsed / query_qty,
            )

        return kneighbors_graph

    @overload
    def ind_dist_grad(
        self,
        data,
        return_grad: Literal[False],
        return_graph: Literal[False],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def ind_dist_grad(
        self, data, return_grad: Literal[False], return_graph: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray, csr_matrix]: ...

    def ind_dist_grad(self, data, return_grad=True, return_graph=True):
        """Return neighbor indices/distances and optionally the sparse graph.

        Gradients are intentionally not implemented. The previous implementation
        used graph row/column indices as if they were feature vectors, which is
        mathematically invalid.
        """
        if return_grad:
            raise NotImplementedError(
                "return_grad=True is not supported. "
                "Distance gradients require access to feature-space vectors and "
                "metric-specific formulas."
            )

        if self.p is None or self.n_samples_fit_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        start = time.time()
        query_data = self._prepare_query_data(data)
        n_query_samples, _ = _check_2d_data(query_data, "query_data")
        query_qty = n_query_samples
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)

        self.p.set_ef(self.efS)
        indices, distances = self.p.knn_query(query_data, k=query_k)

        if self.metric == "euclidean":
            # HNSWlib returns squared L2 distance for space='l2'.
            distances = np.sqrt(distances)

        indices, distances = _drop_self_and_truncate_neighbors(
            indices,
            distances,
            n_neighbors=self.n_neighbors,
            n_query_samples=n_query_samples,
            n_fit_samples=self.n_samples_fit_,
            is_self_query=(n_query_samples == self.n_samples_fit_),
        )

        kneighbors_graph = None
        if return_graph:
            kneighbors_graph = _build_sparse_knn_graph(
                indices=indices,
                distances=distances,
                n_query_samples=n_query_samples,
                n_fit_samples=self.n_samples_fit_,
                n_neighbors=self.n_neighbors,
                is_self_query=(n_query_samples == self.n_samples_fit_),
            )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "kNN time total=%f (sec), per query=%f (sec), "
                "per query adjusted for thread number=%f (sec)",
                elapsed,
                elapsed / query_qty,
                self.n_jobs * elapsed / query_qty,
            )

        if return_graph:
            return indices, distances, kneighbors_graph
        return indices, distances

    def test_efficiency(self, data, percent_use=0.1):
        """Estimate HNSWlib recall against sklearn brute-force nearest neighbors."""
        if self.p is None or self.n_samples_fit_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        data = _as_dense_array(data)
        _check_2d_data(data)

        _, test = train_test_split(data, test_size=percent_use)
        test = np.asarray(test)
        query_qty = test.shape[0]
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Setting query-time parameter efSearch: %s", self.efS)

        start = time.time()
        self.p.set_ef(self.efS)
        hnsw_indices, _ = self.p.knn_query(test, k=query_k)
        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "HNSWlib kNN time total=%f (sec), per query=%f (sec), "
                "per query adjusted for thread number=%f (sec)",
                elapsed,
                elapsed / query_qty,
                self.n_jobs * elapsed / query_qty,
            )

        exact_metric = "euclidean" if self.metric == "sqeuclidean" else self.metric
        if exact_metric == "inner_product":
            exact_metric = "cosine"
            warn(
                "Using cosine brute-force neighbors as an approximate recall "
                "reference for HNSWlib metric='inner_product'.",
                stacklevel=2,
            )

        start = time.time()
        nbrs = NearestNeighbors(
            n_neighbors=query_k,
            metric=exact_metric,
            algorithm="brute",
        ).fit(data)
        true_indices = nbrs.kneighbors(test, return_distance=False)
        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "Brute-force kNN time total=%f (sec), per query=%f (sec)",
                elapsed,
                elapsed / query_qty,
            )

        recall = 0.0
        for i in range(query_qty):
            correct_set = set(true_indices[i])
            ret_set = set(hnsw_indices[i])
            recall += len(correct_set.intersection(ret_set)) / len(correct_set)
        recall /= query_qty

        if self.verbose:
            logger.info("HNSWlib kNN recall %f", recall)

        return recall

    def update_search(self, n_neighbors):
        """Update number of neighbors for kNN distance computation."""
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        return self

    def fit_transform(self, X, y=None, **fit_params):  # type: ignore
        """Fit to X, then return the kNN graph for X."""
        return self.fit(X).transform(X)
