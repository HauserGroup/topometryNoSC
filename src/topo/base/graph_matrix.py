"""Sparse graph matrix utilities for KNN graph conversions.

Helpers to convert between sparse neighbor graphs and dense index/distance
array representations. These functions enable efficient manipulation of
sparse k-nearest-neighbor graph matrices.
"""

import numpy as np
from scipy.sparse import coo_matrix


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
