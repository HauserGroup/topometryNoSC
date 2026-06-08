# Some other utility functions
"""Internal landmark-selection helpers.

Functions to select landmark points for the layout and evaluation routines.
Sparse graph matrix utilities have been moved to topo.base.graph_matrix.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state


def get_landmark_indices(
    data,
    n_landmarks: int = 1000,
    method: str = "random",
    random_state=None,
    **kwargs,
) -> np.ndarray:
    """Select landmark row indices from data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features) or sparse matrix
        Input data. For ``method='kmeans'``, this must be a feature matrix, not a
        precomputed graph.
    n_landmarks : int, default=1000
        Number of landmarks to select. Must be between 1 and ``n_samples``.
    method : {'random', 'kmeans'}, default='random'
        Landmark selection strategy.

        * ``'random'``: uniform random sample of row indices.
        * ``'kmeans'``: MiniBatchKMeans clustering; for each centroid the
          nearest actual data point is returned.
    random_state : int, numpy.random.RandomState, or None
        RNG seed/state.
    **kwargs
        Extra keyword arguments forwarded to ``MiniBatchKMeans``.

    Returns
    -------
    indices : ndarray of int, shape (n_landmarks,)
        Row indices of selected landmarks.
    """
    if not hasattr(data, "shape") or len(data.shape) != 2:
        raise ValueError("data must be a 2-D array-like or sparse matrix.")

    n_samples = int(data.shape[0])
    if n_samples < 1:
        raise ValueError("data must contain at least one sample.")

    method = str(method).lower()
    if method not in {"random", "kmeans"}:
        raise ValueError("Unknown landmark selection method; use 'random' or 'kmeans'.")

    n_landmarks = int(n_landmarks)
    if n_landmarks < 1:
        raise ValueError("n_landmarks must be at least 1.")
    if n_landmarks > n_samples:
        raise ValueError(f"n_landmarks={n_landmarks} must be <= n_samples={n_samples}.")

    rng = check_random_state(random_state)

    if method == "random":
        return rng.choice(n_samples, size=n_landmarks, replace=False).astype(int)

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin

        if issparse(data):
            # MiniBatchKMeans accepts sparse input. Avoid densifying large matrices.
            data_arr = data.tocsr()
        else:
            data_arr = np.asarray(data)

        if not issparse(data_arr) and not np.isfinite(data_arr).all():
            raise ValueError("data contains NaN or infinite values.")

        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            random_state=rng,
            **kwargs,
        ).fit(data_arr)

        indices = pairwise_distances_argmin(kmeans.cluster_centers_, data_arr)
        return np.asarray(indices, dtype=int)

    raise ValueError("Unknown landmark selection method; use 'random' or 'kmeans'.")
