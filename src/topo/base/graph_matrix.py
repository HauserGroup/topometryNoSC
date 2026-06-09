"""Sparse graph matrix utilities for KNN graph conversions.

Helpers to convert between sparse neighbor graphs and dense index/distance
array representations. These functions enable efficient manipulation of
sparse k-nearest-neighbor graph matrices.
"""

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity

CSRMatrix = csr_matrix


def get_sparse_matrix_from_indices_distances(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
) -> csr_matrix:
    """Build a CSR kNN distance graph from dense neighbor index/distance arrays."""
    indices = np.asarray(knn_indices)
    dists = np.asarray(knn_dists)

    n_obs = int(n_obs)
    n_neighbors = int(n_neighbors)

    if indices.ndim != 2 or dists.ndim != 2:
        raise ValueError("knn_indices and knn_dists must be 2-D arrays.")
    if indices.shape != dists.shape:
        raise ValueError("knn_indices and knn_dists must have the same shape.")
    if indices.shape[0] != n_obs:
        raise ValueError(f"Expected {n_obs} rows, got {indices.shape[0]}.")
    if indices.shape[1] < n_neighbors:
        raise ValueError(
            f"Expected at least {n_neighbors} neighbors, got {indices.shape[1]}."
        )

    indices = indices[:, :n_neighbors]
    dists = dists[:, :n_neighbors]

    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = indices.reshape(-1)
    vals = dists.reshape(-1)

    valid = cols >= 0
    rows = rows[valid]
    cols = cols[valid]
    vals = vals[valid]

    if np.any(cols >= n_obs):
        raise ValueError("knn_indices contains indices outside n_obs.")
    if not np.isfinite(vals).all():
        raise ValueError("knn_dists must be finite.")
    if np.any(vals < 0):
        raise ValueError("knn_dists must be non-negative.")

    graph = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    graph.eliminate_zeros()
    return csr_matrix(graph)


def get_indices_distances_from_sparse_matrix(X, n_neighbors):
    """Extract sorted kNN index/distance arrays from a sparse distance graph."""
    graph = as_csr_matrix(X, "X")
    n_neighbors = int(n_neighbors)

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")

    n_rows, _ = matrix_shape(graph, "X")
    knn_indices = np.empty((n_rows, n_neighbors), dtype=np.int64)
    knn_dists = np.empty((n_rows, n_neighbors), dtype=float)

    for row_id in range(n_rows):
        start, end = graph.indptr[row_id], graph.indptr[row_id + 1]
        row_indices = graph.indices[start:end]
        row_data = graph.data[start:end]

        if row_data.size < n_neighbors:
            raise ValueError(
                f"Row {row_id} contains {row_data.size} distances, "
                f"expected at least {n_neighbors}."
            )

        order = np.argsort(row_data, kind="stable")[:n_neighbors]
        knn_indices[row_id] = row_indices[order]
        knn_dists[row_id] = row_data[order]

    return knn_indices, knn_dists


def matrix_shape(value: Any, name: str = "matrix") -> tuple[int, int]:
    """Return a validated 2-D matrix shape."""
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")
    return int(shape[0]), int(shape[1])


def n_rows(value: Any, name: str = "matrix") -> int:
    """Return the number of rows in a validated 2-D matrix."""
    return matrix_shape(value, name)[0]


def as_csr_matrix(
    value: Any,
    name: str = "matrix",
    *,
    dtype: Any | None = None,
    copy: bool = False,
) -> csr_matrix:
    """Return value as a 2-D scipy.sparse.csr_matrix."""
    if value is None:
        raise ValueError(f"{name} must not be None.")

    try:
        out = csr_matrix(value, copy=copy)
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to a CSR sparse matrix.") from exc

    matrix_shape(out, name)

    if dtype is not None and out.dtype != np.dtype(dtype):
        out = out.astype(dtype, copy=False)

    return out


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
    return csr_matrix(identity(n, dtype=dtype, format="csr"))
