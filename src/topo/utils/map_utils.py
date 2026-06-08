# These are some utility functions implemented in UMAP, added here as module
# Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
# License: BSD 3 clause
#
# For more information on the original UMAP implementation, please see:
# https://umap-learn.readthedocs.io/
#
# BSD 3-Clause License
#
# Copyright (c) 2017, Leland McInnes
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utility functions adapted from UMAP.

A small grab-bag of numba-accelerated helpers reused across the package:
fast nearest-neighbor indices from distance rows, the Tausworthe RNG, vector
norms, submatrix extraction, Gaussian-density evaluation, and sparse-row
uniquing.
"""

import time

import numba
import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import KDTree


@numba.njit(fastmath=True)
def eval_gaussian(x, pos, cov):
    """Evaluate a 2-D Gaussian with mean ``pos`` and covariance ``cov`` at ``x``."""
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det <= 1e-16:
        return 0.0

    inv00 = cov[1, 1] / det
    inv01 = -cov[0, 1] / det
    inv10 = -cov[1, 0] / det
    inv11 = cov[0, 0] / det

    dx0 = x[0] - pos[0]
    dx1 = x[1] - pos[1]

    # Correct quadratic form: diff.T @ inv(cov) @ diff.
    m_dist = inv00 * dx0**2 + (inv01 + inv10) * dx0 * dx1 + inv11 * dx1**2

    return np.exp(-0.5 * m_dist) / (2.0 * np.pi * np.sqrt(det))


@numba.njit(fastmath=True)
def eval_density_at_point(x, embedding):
    """Sum per-point Gaussian densities of ``embedding`` evaluated at ``x``.

    ``embedding`` must have columns ``x, y, width, height, angle``.
    """
    result = 0.0
    for i in range(embedding.shape[0]):
        pos = embedding[i, :2]
        t = embedding[i, 4]
        c = np.cos(t)
        s = np.sin(t)
        U = np.array([[c, s], [s, -c]])
        cov = U @ np.diag(embedding[i, 2:4]) @ U
        result += eval_gaussian(x, pos, cov)
    return result


def create_density_plot(X, Y, embedding):
    """Evaluate a normalized Gaussian-mixture density over a meshgrid ``(X, Y)``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    embedding = np.asarray(embedding, dtype=float)

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    if X.ndim != 2:
        raise ValueError("X and Y must be 2-D meshgrid arrays.")
    if embedding.ndim != 2 or embedding.shape[1] < 5:
        raise ValueError("embedding must have shape (n_samples, >=5).")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X and Y must contain only finite values.")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding must contain only finite values.")

    Z = np.zeros_like(X, dtype=float)
    tree = KDTree(embedding[:, :2])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            query = np.array([[X[i, j], Y[i, j]]], dtype=float)
            nearby_idx = tree.query_radius(query, r=2.0)[0]
            nearby_points = embedding[nearby_idx]
            point = np.array([X[i, j], Y[i, j]], dtype=float)
            Z[i, j] = eval_density_at_point(point, nearby_points)

    total = Z.sum()
    if total <= 0.0 or not np.isfinite(total):
        return Z
    return Z / total


@numba.njit(fastmath=True)
def torus_euclidean_grad(x, y, torus_dimensions=(2 * np.pi, 2 * np.pi)):
    r"""Compute Euclidean distance and gradient on a torus.

    Distance and gradient for points on a torus with periodic boundary
    conditions.

    .. math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    distance_sqr = 0.0
    g = np.zeros_like(x)

    for i in range(x.shape[0]):
        diff = x[i] - y[i]
        a = abs(diff)
        period = torus_dimensions[i]

        if 2.0 * a < period or a <= 1e-12:
            distance_sqr += a**2
            g[i] = diff
        else:
            wrapped = period - a
            distance_sqr += wrapped**2
            g[i] = diff * (a - period) / a

    distance = np.sqrt(distance_sqr)
    return distance, g / (1e-6 + distance)


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """Compute nearest-neighbor indices from a dense distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_candidates)
        Distance matrix or distance-like scores. Each row is sorted
        independently.
    n_neighbors : int
        Number of smallest entries to return from each row.

    Returns
    -------
    knn_indices : ndarray of shape (n_samples, n_neighbors)
        Indices of the ``n_neighbors`` smallest entries in each row.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """Generate pseudorandom int32 from tau stream."""
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])")
def tau_rand(state):
    """Generate pseudorandom float32 in [0, 1] from tau stream."""
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit()
def norm(vec):
    """Compute the standard L2 norm of a vector."""
    result = 0.0
    for i in range(vec.shape[0]):
        result += vec[i] ** 2
    return np.sqrt(result)


@numba.njit(parallel=True)
def submatrix(dmat, indices_col, n_neighbors):
    """Return row-wise selected entries from a matrix.

    Parameters
    ----------
    dmat : ndarray of shape (n_samples, n_features)
        Original matrix.
    indices_col : ndarray of shape (n_samples, n_neighbors)
        Column indices to keep for each row.
    n_neighbors : int
        Number of selected columns per row.

    Returns
    -------
    submat : ndarray of shape (n_samples, n_neighbors)
        Row-wise selected entries.
    """
    n_samples_transform, _n_samples_fit = dmat.shape
    submat = np.zeros((n_samples_transform, n_neighbors), dtype=dmat.dtype)
    for i in numba.prange(n_samples_transform):
        for j in range(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat


def ts():
    """Return a human-readable timestamp for verbose logging messages."""
    return time.ctime(time.time())


def csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True):
    """Find unique rows in a sparse CSR matrix.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        Input sparse matrix.
    return_index : bool, default=True
        If True, return row indices in the original matrix.
    return_inverse : bool, default=True
        If True, return indices to reconstruct original rows.
    return_counts : bool, default=True
        If True, return count of each unique row.

    Returns
    -------
    tuple
        The requested outputs from :func:`numpy.unique`, excluding the unique
        row keys themselves. This preserves the historical behavior of returning
        only index/inverse/count arrays.
    """
    if not issparse(matrix):
        raise TypeError("matrix must be a scipy sparse matrix.")

    matrix = matrix.tocsr()
    row_keys = []
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        cols = tuple(matrix.indices[start:end].tolist())
        vals = tuple(matrix.data[start:end].tolist())
        row_keys.append((cols, vals))

    unique_result = np.unique(
        np.asarray(row_keys, dtype=object),
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    # np.unique returns (unique, index?, inverse?, counts?). Historically this
    # helper returned only the optional arrays, not the unique row objects.
    if not isinstance(unique_result, tuple):
        return ()

    return unique_result[1:]
