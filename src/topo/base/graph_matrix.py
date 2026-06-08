"""Sparse graph matrix utilities for KNN graph conversions.

Helpers to convert between sparse neighbor graphs and dense index/distance
array representations. These functions enable efficient manipulation of
sparse k-nearest-neighbor graph matrices.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix


def get_sparse_matrix_from_indices_distances(
    knn_indices, knn_dists, n_obs, n_neighbors
):
    """Build sparse CSR matrix from KNN indices and distances.

    Converts dense index and distance arrays (e.g., from sklearn kneighbors)
    into a sparse CSR matrix representation suitable for graph operations.

    Parameters
    ----------
    knn_indices : ndarray of shape (n_samples, k)
        Neighbor indices.
    knn_dists : ndarray of shape (n_samples, k)
        Distances to neighbors.
    n_obs : int
        Number of samples (rows in output matrix).
    n_neighbors : int
        Number of neighbors per sample.

    Returns
    -------
    graph : scipy.sparse.csr_matrix of shape (n_obs, n_obs)
        Sparse KNN graph with distances as weights.
    """
    rows = np.zeros((n_obs * n_neighbors), dtype=int)
    cols = np.zeros((n_obs * n_neighbors), dtype=int)
    vals = np.zeros((n_obs * n_neighbors), dtype=float)
    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def get_indices_distances_from_sparse_matrix(X, n_neighbors):
    """Extract KNN indices and distances from sparse matrix.

    Converts a sparse k-nearest-neighbors distance matrix into dense
    arrays of neighbor indices and distances.

    Parameters
    ----------
    X : scipy.sparse matrix
        Input KNN distance matrix.
    n_neighbors : int
        Number of neighbors per sample.

    Returns
    -------
    knn_indices : ndarray of shape (n_samples, n_neighbors)
        Indices of nearest neighbors.
    knn_dists : ndarray of shape (n_samples, n_neighbors)
        Distances to nearest neighbors.
    """
    _knn_indices = np.zeros((X.shape[0], n_neighbors), dtype=int)
    _knn_dists = np.zeros(_knn_indices.shape, dtype=float)
    for row_id in range(X.shape[0]):
        # Find KNNs row-by-row
        row_data = X[row_id].data
        row_indices = X[row_id].indices
        if len(row_data) < n_neighbors:
            raise ValueError("Some rows contain fewer than n_neighbors distances!")
        row_nn_data_indices = np.argsort(row_data)[:n_neighbors]
        _knn_indices[row_id] = row_indices[row_nn_data_indices]
        _knn_dists[row_id] = row_data[row_nn_data_indices]
    return _knn_indices, _knn_dists


def as_csr_matrix(
    value: Any,
    name: str = "matrix",
    *,
    dtype: Any | None = None,
    copy: bool = False,
) -> csr_matrix:
    """Return value as a scipy.sparse.csr_matrix.

    This is a typing/runtime boundary helper. It should not change graph
    semantics beyond CSR conversion and optional dtype conversion.
    """
    if value is None:
        raise ValueError(f"{name} must not be None.")

    try:
        out = csr_matrix(value, copy=copy)
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to a CSR sparse matrix.") from exc

    if dtype is not None and out.dtype != np.dtype(dtype):
        out = out.astype(dtype, copy=False)

    return csr_matrix(out)


def as_float32_csr(
    value: Any,
    name: str = "matrix",
    *,
    copy: bool = False,
) -> csr_matrix:
    """Return value as CSR float32."""
    return as_csr_matrix(value, name=name, dtype=np.float32, copy=copy)


def sparse_identity(n: int, *, dtype: Any = np.float32) -> csr_matrix:
    """Return an n-by-n CSR identity matrix."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")
    diag = np.ones(n, dtype=dtype)
    return csr_matrix(sp.diags(diag, offsets=0, shape=(n, n), format="csr"))
