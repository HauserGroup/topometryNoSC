#####################################
# Wrappers for approximate nearest neighbor search
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences, University of Campinas, Brazil
# contact: davisidarta@fcm.unicamp.br
######################################

"""Approximate nearest-neighbor wrappers.

This module provides small sklearn-style wrappers around NMSlib and HNSWlib,
plus a convenience `kNN` function that can fall back to sklearn.

Important conventions
---------------------
- `n_neighbors` is the user-requested number of neighbors.
- For self-query compatibility, ANN backends query `n_neighbors + 1` internally
  because the sample itself is usually returned as the nearest neighbor.
- The public estimator state is never mutated during `transform`.
- Returned sparse graphs have shape `(n_query_samples, n_fit_samples)`.
"""

import logging
import time
from warnings import warn

import numpy as np
from joblib import cpu_count
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve sklearn/joblib-style n_jobs."""
    if n_jobs == -1:
        return cpu_count()
    return int(n_jobs)


def _validate_n_neighbors(n_neighbors: int) -> int:
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
    return min(n_neighbors + 1, n_fit_samples)


def _as_dense_array(data):
    """Convert supported array-like inputs to a dense numpy array."""
    if isinstance(data, np.ndarray):
        return data

    if issparse(data):
        return data.toarray()

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
    except ImportError:
        pass

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    return arr


def _as_csr_matrix(data):
    """Convert supported array-like inputs to CSR sparse matrix."""
    if issparse(data):
        return data.tocsr()

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return csr_matrix(data.to_numpy())
    except ImportError:
        pass

    return csr_matrix(data)


def _check_2d_data(data, name: str = "data"):
    """Validate that data has a 2-D shape."""
    if not hasattr(data, "shape") or len(data.shape) != 2:
        raise ValueError(f"{name} must be a 2-D array-like or sparse matrix.")
    if data.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if data.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")


def _build_sparse_knn_graph(indices, distances, n_query_samples, n_fit_samples):
    """Build a CSR neighbor-distance graph from dense index/distance arrays."""
    indices = np.asarray(indices)
    distances = np.asarray(distances)

    if indices.ndim != 2 or distances.ndim != 2:
        raise ValueError("indices and distances must be 2-D arrays.")
    if indices.shape != distances.shape:
        raise ValueError("indices and distances must have the same shape.")
    if indices.shape[0] != n_query_samples:
        raise ValueError("indices row count does not match n_query_samples.")

    k = indices.shape[1]
    indptr = np.arange(0, n_query_samples * k + 1, k, dtype=np.int64)

    return csr_matrix(
        (distances.ravel(), indices.ravel(), indptr),
        shape=(n_query_samples, n_fit_samples),
    )


def _nmslib_sparse_space(metric: str) -> str:
    spaces = {
        "sqeuclidean": "l2_sparse",
        "euclidean": "l2_sparse",
        "cosine": "cosinesimil_sparse_fast",
        "lp": "lp_sparse",
        "l1": "l1_sparse",
        "l1_sparse": "l1_sparse",
        "linf": "linf_sparse",
        "linf_sparse": "linf_sparse",
        "angular": "angulardist_sparse_fast",
        "angular_sparse": "angulardist_sparse_fast",
        "negdotprod": "negdotprod_sparse_fast",
        "negdotprod_sparse": "negdotprod_sparse_fast",
        "jaccard": "jaccard_sparse",
        "jaccard_sparse": "jaccard_sparse",
        "bit_jaccard": "bit_jaccard",
        "bit_hamming": "bit_hamming",
        "levenshtein": "leven",
        "normleven": "normleven",
    }
    try:
        return spaces[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported NMSlib sparse metric: {metric!r}") from exc


def _nmslib_dense_space(metric: str) -> str:
    spaces = {
        "sqeuclidean": "l2",
        "euclidean": "l2",
        "cosine": "cosinesimil",
        "lp": "lp",
        "l1": "l1",
        "linf": "linf",
        "angular": "angulardist",
        "negdotprod": "negdotprod",
        "inner_product": "negdotprod",
        "levenshtein": "leven",
        "jaccard_sparse": "jaccard_sparse",
        "bit_jaccard": "bit_jaccard",
        "bit_hamming": "bit_hamming",
        "jansen-shan": "jsmetrfastapprox",
    }
    try:
        return spaces[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported NMSlib dense metric: {metric!r}") from exc


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


def kNN(
    X,
    Y=None,
    n_neighbors=5,
    metric="euclidean",
    n_jobs=-1,
    backend="hnswlib",
    low_memory=True,
    M=60,
    p=11 / 16,
    efC=200,
    efS=200,
    n_trees=50,
    return_instance=False,
    verbose=False,
    **kwargs,
):
    """Compute a k-nearest-neighbor distance graph.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n_samples, n_features)
        Reference data used to fit the neighbor index.

    Y : array-like or scipy.sparse matrix, optional
        Query data. If provided, the output graph has shape
        `(Y.shape[0], X.shape[0])`. NMSlib and HNSWlib are not used for `Y`;
        the function falls back to sklearn.

    n_neighbors : int, default=5
        Number of neighbors to return.

    metric : str, default='euclidean'
        Distance metric.

    n_jobs : int, default=-1
        Number of threads. `-1` uses all available CPUs.

    backend : {'nmslib', 'hnswlib', 'sklearn'}, default='hnswlib'
        Neighbor-search backend.

    return_instance : bool, default=False
        If True, return `(estimator, graph)`.

    verbose : bool, default=False
        Print timing/backend messages.

    Returns
    -------
    scipy.sparse.csr_matrix
        kNN distance graph.
    """
    _check_2d_data(X, "X")
    n_neighbors = _validate_n_neighbors(n_neighbors)
    n_jobs = _resolve_n_jobs(n_jobs)

    if Y is not None:
        _check_2d_data(Y, "Y")
        if backend in {"nmslib", "hnswlib"}:
            warn(
                "Only the sklearn backend supports Y. Falling back to sklearn.",
                stacklevel=2,
            )
            backend = "sklearn"

    if backend == "nmslib":
        X_fit = _as_csr_matrix(X)
        nbrs = NMSlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            method="hnsw",
            n_jobs=n_jobs,
            M=M,
            efC=efC,
            efS=efS,
            verbose=verbose,
        ).fit(X_fit)
        knn = nbrs.transform(X_fit)

    elif backend == "hnswlib":
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
        if verbose and backend != "sklearn":
            logger.info("Falling back to sklearn nearest-neighbors.")
        backend = "sklearn"
        # Drop ANN-only kwargs (random_state, M, efC, efS, ...) that the
        # sklearn estimator does not accept.
        _valid = set(NearestNeighbors().get_params())
        sk_kwargs = {k: v for k, v in kwargs.items() if k in _valid}
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            **sk_kwargs,
        ).fit(X)
        if Y is None:
            knn = nbrs.kneighbors_graph(X, mode="distance")
        else:
            knn = nbrs.kneighbors_graph(Y, mode="distance")

    if return_instance:
        return nbrs, knn
    return knn


class NMSlibTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-style wrapper around NMSlib.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of neighbors to return.

    metric : str, default='cosine'
        NMSlib metric.

    method : str, default='hnsw'
        NMSlib index method.

    n_jobs : int, default=-1
        Number of threads.

    p : float, optional
        Minkowski/fractional norm parameter for `metric='lp'`.

    M : int, default=60
        HNSW graph connectivity parameter.

    efC : int, default=200
        Construction-time search parameter.

    efS : int, default=200
        Query-time search parameter.

    dense : bool, default=False
        Force dense-vector NMSlib mode.

    verbose : bool, default=False
        Print timing/backend messages.
    """

    def __init__(
        self,
        n_neighbors=15,
        metric="cosine",
        method="hnsw",
        n_jobs=-1,
        p=None,
        M=60,
        efC=200,
        efS=200,
        dense=False,
        verbose=False,
    ):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.space = metric
        self.dense = dense
        self.verbose = verbose

        self.nmslib_ = None
        self.n_samples_fit_ = None
        self.n_features_in_ = None

    def fit(self, data):
        """Fit the NMSlib index."""
        try:
            import nmslib  # type: ignore[reportMissingImports]

        except ImportError as exc:
            raise ImportError(
                "NMSlib is required for NMSlibTransformer. "
                "Install it with `pip install nmslib`."
            ) from exc

        _check_2d_data(data)
        self.n_neighbors = _validate_n_neighbors(self.n_neighbors)
        self.n_jobs = _resolve_n_jobs(self.n_jobs)

        self.n_samples_fit_, self.n_features_in_ = data.shape

        start = time.time()

        if self.dense:
            data_fit = _as_dense_array(data)
            self.space = _nmslib_dense_space(self.metric)
            data_type = nmslib.DataType.DENSE_VECTOR
        else:
            if self.metric in {"levenshtein", "normleven"}:
                data_fit = _as_dense_array(data)
                self.space = _nmslib_sparse_space(self.metric)
                data_type = nmslib.DataType.OBJECT_AS_STRING
            elif issparse(data) or not isinstance(data, np.ndarray):
                data_fit = _as_csr_matrix(data)
                self.space = _nmslib_sparse_space(self.metric)
                data_type = nmslib.DataType.SPARSE_VECTOR
            else:
                data_fit = data
                self.space = _nmslib_dense_space(self.metric)
                data_type = nmslib.DataType.DENSE_VECTOR

        if self.metric == "lp" and self.p is None:
            raise ValueError("metric='lp' requires p to be specified.")

        if self.metric == "lp" and self.p is not None and self.p < 1:
            warn(
                "Fractional Lp norms may be slower to compute. "
                "Fractions such as 0.5 or 0.25 are typically faster.",
                stacklevel=2,
            )

        init_kwargs = {
            "method": self.method,
            "space": self.space,
            "data_type": data_type,
        }
        if self.metric == "lp":
            init_kwargs["space_params"] = {"p": self.p}

        self.nmslib_ = nmslib.init(**init_kwargs)
        self.nmslib_.addDataPointBatch(data_fit)

        index_time_params = {
            "M": self.M,
            "indexThreadQty": self.n_jobs,
            "efConstruction": self.efC,
            "post": 2,
        }
        self.nmslib_.createIndex(index_time_params)

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                "NMSlib index-time parameters M=%s n_threads=%s efConstruction=%s post=2",
                self.M,
                self.n_jobs,
                self.efC,
            )
            logger.info("Indexing time = %f (sec)", elapsed)

        return self

    def _prepare_query_data(self, data):
        """Prepare query data using the same dense/sparse convention as fit."""
        _check_2d_data(data)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Query data has {data.shape[1]} features, but the index was "
                f"fit with {self.n_features_in_} features."
            )

        if self.dense:
            return _as_dense_array(data)

        if self.metric in {"levenshtein", "normleven"}:
            return _as_dense_array(data)

        if "sparse" in self.space:
            return _as_csr_matrix(data)

        return _as_dense_array(data)

    def transform(self, data):
        """Return a CSR kNN distance graph for query data."""
        if self.nmslib_ is None or self.n_samples_fit_ is None:
            raise ValueError("This NMSlibTransformer instance is not fitted yet.")

        start = time.time()
        query_data = self._prepare_query_data(data)
        n_query_samples = query_data.shape[0]
        query_qty = n_query_samples
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        query_time_params = {"efSearch": self.efS}
        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)

        results = self.nmslib_.knnQueryBatch(
            query_data,
            k=query_k,
            num_threads=self.n_jobs,
        )

        indices, distances = zip(*results)
        indices = np.vstack(indices)
        distances = np.vstack(distances)

        if self.metric == "sqeuclidean":
            distances = distances**2

        kneighbors_graph = _build_sparse_knn_graph(
            indices=indices,
            distances=distances,
            n_query_samples=n_query_samples,
            n_fit_samples=self.n_samples_fit_,
        )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"Search time ={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
            )

        return kneighbors_graph

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

        if self.nmslib_ is None or self.n_samples_fit_ is None:
            raise ValueError("This NMSlibTransformer instance is not fitted yet.")

        start = time.time()
        query_data = self._prepare_query_data(data)
        n_query_samples = query_data.shape[0]
        query_qty = n_query_samples
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        query_time_params = {"efSearch": self.efS}
        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)

        results = self.nmslib_.knnQueryBatch(
            query_data,
            k=query_k,
            num_threads=self.n_jobs,
        )

        indices, distances = zip(*results)
        indices = np.vstack(indices)
        distances = np.vstack(distances)

        if self.metric == "sqeuclidean":
            distances = distances**2

        kneighbors_graph = None
        if return_graph:
            kneighbors_graph = _build_sparse_knn_graph(
                indices=indices,
                distances=distances,
                n_query_samples=n_query_samples,
                n_fit_samples=self.n_samples_fit_,
            )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
            )

        if return_graph:
            return indices, distances, kneighbors_graph
        return indices, distances

    def test_efficiency(self, data, data_use=0.1):
        """Estimate ANN recall against sklearn brute-force nearest neighbors."""
        if self.nmslib_ is None or self.n_samples_fit_ is None:
            raise ValueError("This NMSlibTransformer instance is not fitted yet.")

        _check_2d_data(data)
        query_data = self._prepare_query_data(data)

        _, test = train_test_split(query_data, test_size=data_use)
        query_qty = test.shape[0]
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        query_time_params = {"efSearch": self.efS}
        if self.verbose:
            logger.info("Setting query-time parameters %s", query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        start = time.time()
        ann_results = self.nmslib_.knnQueryBatch(
            test,
            k=query_k,
            num_threads=self.n_jobs,
        )
        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"ANN kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
            )

        exact_metric = "euclidean" if self.metric == "sqeuclidean" else self.metric
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
                f"Brute-force kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec)"
            )

        recall = 0.0
        for i in range(query_qty):
            correct_set = set(true_indices[i])
            ret_set = set(ann_results[i][0])
            recall += len(correct_set.intersection(ret_set)) / len(correct_set)
        recall /= query_qty

        if self.verbose:
            logger.info("kNN recall %f", recall)

        return recall

    def update_search(self, n_neighbors):
        """Update number of neighbors for kNN distance computation."""
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        return self

    def fit_transform(self, X, y=None, **fit_params):  # type: ignore
        """Fit to X, then return the kNN graph for X."""
        return self.fit(X).transform(X)


def grid_search(
    X,
    n_neighbors=15,
    metric="euclidean",
    nmslib_params=None,
    hnswlib_params=None,
    n_jobs=-1,
    verbose=False,
):
    """Evaluate approximate kNN graph quality for NMSlib and HNSWlib.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data used to build the neighborhood graph.

    n_neighbors : int, default=15
        Number of neighbors to retrieve.

    metric : str, default='euclidean'
        Distance metric used for neighbor search.

    nmslib_params : dict, optional
        Parameter grid for NMSlibTransformer.

    hnswlib_params : dict, optional
        Parameter grid for HNSWlibTransformer.

    n_jobs : int, default=-1
        Number of parallel jobs for the transformers.

    verbose : bool, default=False
        If True, print recall and timing for each parameter combination.

    Returns
    -------
    dict
        Mapping of backend names to lists of dictionaries containing
        parameter settings, recall and execution time.
    """
    _check_2d_data(X)
    n_neighbors = _validate_n_neighbors(n_neighbors)
    n_jobs = _resolve_n_jobs(n_jobs)

    exact_metric = "euclidean" if metric == "sqeuclidean" else metric
    gt = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=exact_metric,
        algorithm="brute",
    ).fit(X)
    true_ind = gt.kneighbors(X, return_distance=False)

    results = {"nmslib": [], "hnswlib": []}

    for params in ParameterGrid(nmslib_params) if nmslib_params else [{}]:
        model = NMSlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            **params,
        )

        start = time.time()
        model.fit(X)
        res = model.ind_dist_grad(
            X,
            return_grad=False,
            return_graph=False,
        )
        ind = res[0]
        elapsed = time.time() - start

        # Drop self-neighbor if present.
        ind_eval = ind[:, 1:] if ind.shape[1] > n_neighbors else ind

        recall = np.mean(
            [
                np.intersect1d(true_ind[i], ind_eval[i]).size / n_neighbors
                for i in range(X.shape[0])
            ]
        )
        results["nmslib"].append({"params": params, "recall": recall, "time": elapsed})

    for params in ParameterGrid(hnswlib_params) if hnswlib_params else [{}]:
        model = HNSWlibTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            **params,
        )

        start = time.time()
        model.fit(X)
        res = model.ind_dist_grad(
            X,
            return_grad=False,
            return_graph=False,
        )
        ind = res[0]
        elapsed = time.time() - start

        # Drop self-neighbor if present.
        ind_eval = ind[:, 1:] if ind.shape[1] > n_neighbors else ind

        recall = np.mean(
            [
                np.intersect1d(true_ind[i], ind_eval[i]).size / n_neighbors
                for i in range(X.shape[0])
            ]
        )
        results["hnswlib"].append({"params": params, "recall": recall, "time": elapsed})

    if verbose:
        for backend, backend_results in results.items():
            for row in backend_results:
                logger.info(
                    f"{backend}: params={row['params']}, "
                    f"recall={row['recall']:.3f}, "
                    f"time={row['time']:.3f}s"
                )

    return results


class HNSWlibTransformer(TransformerMixin, BaseEstimator):
    """Sklearn-style wrapper around HNSWlib.

    Parameters
    ----------
    n_neighbors : int, default=30
        Number of neighbors to return.

    metric : {'sqeuclidean', 'euclidean', 'cosine', 'inner_product'}, default='cosine'
        HNSWlib metric. For additional metrics, use NMSlibTransformer.

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
            import hnswlib
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
        n_query_samples = query_data.shape[0]
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
        )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"Search time ={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
            )

        return kneighbors_graph

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
        n_query_samples = query_data.shape[0]
        query_qty = n_query_samples
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Query-time parameter efSearch: %s", self.efS)

        self.p.set_ef(self.efS)
        indices, distances = self.p.knn_query(query_data, k=query_k)

        if self.metric == "euclidean":
            # HNSWlib returns squared L2 distance for space='l2'.
            distances = np.sqrt(distances)

        kneighbors_graph = None
        if return_graph:
            kneighbors_graph = _build_sparse_knn_graph(
                indices=indices,
                distances=distances,
                n_query_samples=n_query_samples,
                n_fit_samples=self.n_samples_fit_,
            )

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
            )

        if return_graph:
            return indices, distances, kneighbors_graph
        return indices, distances

    def test_efficiency(self, data, percent_use=0.1):
        """Estimate ANN recall against sklearn brute-force nearest neighbors."""
        if self.p is None or self.n_samples_fit_ is None:
            raise ValueError("This HNSWlibTransformer instance is not fitted yet.")

        data = _as_dense_array(data)
        _check_2d_data(data)

        _, test = train_test_split(data, test_size=percent_use)
        query_qty = test.shape[0]
        query_k = _query_k(self.n_neighbors, self.n_samples_fit_)

        if self.verbose:
            logger.info("Setting query-time parameter efSearch: %s", self.efS)

        start = time.time()
        self.p.set_ef(self.efS)
        ann_indices, _ = self.p.knn_query(test, k=query_k)
        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"ANN kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec), "
                f"per query adjusted for thread number={self.n_jobs * elapsed / query_qty:f} (sec)"
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
                f"Brute-force kNN time total={elapsed:f} (sec), per query={elapsed / query_qty:f} (sec)"
            )

        recall = 0.0
        for i in range(query_qty):
            correct_set = set(true_indices[i])
            ret_set = set(ann_indices[i])
            recall += len(correct_set.intersection(ret_set)) / len(correct_set)
        recall /= query_qty

        if self.verbose:
            logger.info("kNN recall %f", recall)

        return recall

    def update_search(self, n_neighbors):
        """Update number of neighbors for kNN distance computation."""
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        return self

    def fit_transform(self, X, y=None, **fit_params):  # type: ignore
        """Fit to X, then return the kNN graph for X."""
        return self.fit(X).transform(X)
