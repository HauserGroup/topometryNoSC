"""Continuous k-nearest-neighbors graph construction.

CkNN, following Berry and Sauer, is an unweighted graph: samples ``i`` and
``j`` are adjacent when ``d(i, j) < delta * sqrt(rho_i * rho_j)``, where
``rho_i`` is the distance from sample ``i`` to its ``scale_k``-th nearest
neighbor.
"""

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from topo.base.ann import kNN
from topo.utils._utils import get_indices_distances_from_sparse_matrix

SymmetrizeMode = Literal["or", "and"]


def _validate_cknn_inputs(n_samples: int, scale_k: int, delta: float) -> None:
    if scale_k < 1:
        raise ValueError("scale_k must be >= 1.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if scale_k >= n_samples:
        raise ValueError("scale_k must be smaller than n_samples.")


def _validate_positive_radii(rho: np.ndarray) -> None:
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError(
            "CkNN requires positive finite k-th-neighbor radii. "
            "Check for duplicate samples, self-neighbors, or invalid distances."
        )


def _dense_distances(X, metric: str) -> np.ndarray:
    if metric == "precomputed":
        D = X.toarray() if issparse(X) else np.asarray(X)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("precomputed CkNN distances must be a square matrix.")
        return np.asarray(D, dtype=float).copy()
    return np.asarray(sklearn_pairwise_distances(X, metric=metric), dtype=float)


def _brute_force_ratio_matrix(
    X,
    *,
    scale_k: int,
    metric: str,
) -> csr_matrix:
    D = _dense_distances(X, metric)
    n_samples = D.shape[0]
    _validate_cknn_inputs(n_samples, scale_k, delta=1.0)

    np.fill_diagonal(D, np.inf)
    rho = np.partition(D, scale_k - 1, axis=1)[:, scale_k - 1]
    _validate_positive_radii(rho)

    denom = np.sqrt(rho[:, None] * rho[None, :])
    ratio = D / denom
    np.fill_diagonal(ratio, 0.0)
    graph = csr_matrix(ratio.astype(np.float32))
    graph.eliminate_zeros()
    return graph


def _brute_force_cknn_graph(
    X,
    *,
    scale_k: int,
    delta: float,
    metric: str,
    include_self: bool,
) -> csr_matrix:
    D = _dense_distances(X, metric)
    n_samples = D.shape[0]
    _validate_cknn_inputs(n_samples, scale_k, delta)

    np.fill_diagonal(D, np.inf)
    rho = np.partition(D, scale_k - 1, axis=1)[:, scale_k - 1]
    _validate_positive_radii(rho)

    threshold = delta * np.sqrt(rho[:, None] * rho[None, :])
    adjacency = threshold > D
    np.fill_diagonal(adjacency, bool(include_self))
    graph = csr_matrix(adjacency.astype(np.float32))
    graph.eliminate_zeros()
    return graph


def _as_symmetric_binary_graph(
    graph: csr_matrix,
    *,
    include_self: bool,
    symmetrize: SymmetrizeMode,
) -> csr_matrix:
    if symmetrize == "or":
        graph = graph.maximum(graph.T)
    elif symmetrize == "and":
        graph = graph.minimum(graph.T)
    else:
        raise ValueError("symmetrize must be 'or' or 'and'.")
    graph.setdiag(1.0 if include_self else 0.0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data = np.ones_like(graph.data, dtype=np.float32)
    return graph.tocsr()


def _cknn_from_knn_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_samples: int,
    scale_k: int,
    delta: float,
    include_self: bool = False,
    symmetrize: SymmetrizeMode = "or",
    return_ratio: bool = False,
) -> csr_matrix:
    """Build a CkNN graph or ratio matrix from fixed-width candidate arrays."""
    _validate_cknn_inputs(n_samples, scale_k, delta)
    if symmetrize not in {"or", "and"}:
        raise ValueError("symmetrize must be 'or' or 'and'.")

    indices = np.asarray(indices)
    distances = np.asarray(distances, dtype=float)
    if indices.shape != distances.shape:
        raise ValueError("indices and distances must have the same shape.")
    if indices.ndim != 2:
        raise ValueError("indices and distances must be 2-D arrays.")
    if indices.shape[0] != n_samples:
        raise ValueError("indices/distances row count must equal n_samples.")
    if indices.shape[1] < scale_k:
        raise ValueError("Need at least scale_k neighbors per sample.")

    rho = distances[:, scale_k - 1]
    _validate_positive_radii(rho)

    candidate_k = indices.shape[1]
    rows = np.repeat(np.arange(n_samples), candidate_k)
    cols = indices.reshape(-1)
    d_ij = distances.reshape(-1)

    valid = (cols >= 0) & (cols < n_samples) & np.isfinite(d_ij)
    rows = rows[valid]
    cols = cols[valid]
    d_ij = d_ij[valid]

    if not include_self:
        nonself = rows != cols
        rows = rows[nonself]
        cols = cols[nonself]
        d_ij = d_ij[nonself]

    denom = np.sqrt(rho[rows] * rho[cols])
    if return_ratio:
        ratio_data = (d_ij / denom).astype(np.float32)
        graph = csr_matrix((ratio_data, (rows, cols)), shape=(n_samples, n_samples))
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        if symmetrize == "or":
            graph = graph.minimum(graph.T)
        else:
            mutual = graph.multiply(graph.T.astype(bool))
            graph = mutual.minimum(mutual.T)
        graph.eliminate_zeros()
        return graph.tocsr()

    edge_mask = d_ij < delta * denom
    graph = csr_matrix(
        (
            np.ones(np.count_nonzero(edge_mask), dtype=np.float32),
            (rows[edge_mask], cols[edge_mask]),
        ),
        shape=(n_samples, n_samples),
    )
    return _as_symmetric_binary_graph(
        graph, include_self=include_self, symmetrize=symmetrize
    )


def _drop_self_and_truncate_candidate_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_samples: int,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_indices = np.empty((n_samples, candidate_k), dtype=np.int64)
    out_distances = np.empty((n_samples, candidate_k), dtype=float)
    for row in range(n_samples):
        row_indices = np.asarray(indices[row])
        row_distances = np.asarray(distances[row], dtype=float)
        keep = row_indices != row
        row_indices = row_indices[keep]
        row_distances = row_distances[keep]
        if row_indices.shape[0] < candidate_k:
            raise ValueError(
                "Candidate neighbor search returned fewer than candidate_k "
                "non-self neighbors."
            )
        out_indices[row] = row_indices[:candidate_k]
        out_distances[row] = row_distances[:candidate_k]
    return out_indices, out_distances


def _complete_candidate_arrays(X, metric: str) -> tuple[np.ndarray, np.ndarray]:
    D = _dense_distances(X, metric)
    n_samples = D.shape[0]
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)[:, : n_samples - 1]
    distances = np.take_along_axis(D, order, axis=1)
    return order.astype(np.int64), distances.astype(float)


def _candidate_knn_arrays(
    X,
    *,
    scale_k: int,
    candidate_k: int | None,
    metric: str,
    backend: str,
    n_jobs: int | None,
    verbose: bool,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(X.shape[0])
    if candidate_k is None:
        candidate_k = max(3 * scale_k, scale_k + 15)
    candidate_k = min(int(candidate_k), n_samples - 1)
    if candidate_k < scale_k:
        raise ValueError("candidate_k must be >= scale_k.")

    if candidate_k == n_samples - 1:
        return _complete_candidate_arrays(X, metric)

    if metric == "precomputed":
        graph = X.tocsr() if issparse(X) else csr_matrix(X)
        query_k = min(candidate_k + 1, n_samples)
    else:
        query_k = min(candidate_k + 1, n_samples - 1)
        graph = kNN(
            X,
            n_neighbors=query_k,
            metric=metric,
            backend=backend,
            n_jobs=1 if n_jobs is None else n_jobs,
            verbose=verbose,
            **kwargs,
        )
    indices, distances = get_indices_distances_from_sparse_matrix(graph, query_k)
    return _drop_self_and_truncate_candidate_arrays(
        indices,
        distances,
        n_samples=n_samples,
        candidate_k=candidate_k,
    )


def cknn_ratio_matrix(
    X,
    *,
    scale_k: int = 10,
    metric: str = "euclidean",
    candidate_k: int | None = None,
    exact: bool = False,
    symmetrize: SymmetrizeMode = "or",
    backend: str = "sklearn",
    n_jobs: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> csr_matrix:
    """Return sparse CkNN normalized distance ratios.

    Entry ``(i, j)`` is ``d(i, j) / sqrt(rho_i * rho_j)``. This matrix is useful
    for persistence/order diagnostics; it is not the binary CkNN adjacency used
    for the unnormalized graph Laplacian. Ratios are always computed without
    self-loops.
    """
    n_samples = int(X.shape[0])
    _validate_cknn_inputs(n_samples, scale_k, delta=1.0)
    if exact:
        return _brute_force_ratio_matrix(X, scale_k=scale_k, metric=metric)

    indices, distances = _candidate_knn_arrays(
        X,
        scale_k=scale_k,
        candidate_k=candidate_k,
        metric=metric,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    return _cknn_from_knn_arrays(
        indices,
        distances,
        n_samples=n_samples,
        scale_k=scale_k,
        delta=1.0,
        include_self=False,
        symmetrize=symmetrize,
        return_ratio=True,
    )


def cknn_graph(
    X,
    *,
    scale_k: int = 10,
    delta: float = 1.0,
    metric: str = "euclidean",
    candidate_k: int | None = None,
    exact: bool = False,
    include_self: bool = False,
    symmetrize: SymmetrizeMode = "or",
    backend: str = "sklearn",
    n_jobs: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> csr_matrix:
    """Build the unweighted Continuous k-Nearest Neighbors graph.

    Two samples ``i`` and ``j`` are connected when
    ``d(i, j) < delta * sqrt(rho_i * rho_j)``, where ``rho_i`` is the distance
    from sample ``i`` to its ``scale_k``-th nearest neighbor. Exact mode
    thresholds all pairs. Candidate-neighbor mode is scalable but may miss edges
    if ``candidate_k`` is too small.
    """
    n_samples = int(X.shape[0])
    _validate_cknn_inputs(n_samples, scale_k, delta)
    if exact:
        return _brute_force_cknn_graph(
            X,
            scale_k=scale_k,
            delta=delta,
            metric=metric,
            include_self=include_self,
        )

    indices, distances = _candidate_knn_arrays(
        X,
        scale_k=scale_k,
        candidate_k=candidate_k,
        metric=metric,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    return _cknn_from_knn_arrays(
        indices,
        distances,
        n_samples=n_samples,
        scale_k=scale_k,
        delta=delta,
        include_self=include_self,
        symmetrize=symmetrize,
        return_ratio=False,
    )
