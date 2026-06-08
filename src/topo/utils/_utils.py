# Some other utility functions
"""Internal landmark-selection helpers.

Functions to select landmark points for the layout and evaluation routines.
Sparse graph matrix utilities have been moved to topo.base.graph_matrix.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state


def get_landmark_indices(
    data, n_landmarks=1000, method="random", random_state=None, **kwargs
):
    """
    Select landmark indices from data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features) or sparse matrix
        Input data. For ``method='kmeans'``, must be a feature matrix (not a
        precomputed graph).
    n_landmarks : int, default 1000
        Number of landmarks to select.
    method : {'random', 'kmeans'}, default 'random'
        Landmark selection strategy.
        * ``'random'``: uniform random sample of row indices.
        * ``'kmeans'``: MiniBatchKMeans clustering; for each centroid the
          nearest actual data point is returned (so the result is always a
          valid index array into ``data``).
    random_state : int or numpy.random.RandomState, optional
        RNG seed / state.
    **kwargs
        Extra keyword arguments forwarded to ``MiniBatchKMeans``.

    Returns
    -------
    indices : ndarray of int, shape (n_landmarks,)
        Row indices of the selected landmarks.
    """
    random_state = check_random_state(random_state)
    if method == "random":
        all_idx = np.arange(np.shape(data)[0])
        return random_state.choice(all_idx, size=n_landmarks, replace=False)
    elif method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin

        data_arr = np.asarray(data.toarray() if issparse(data) else data)
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks, random_state=random_state, **kwargs
        ).fit(data_arr)
        # Return the index of the nearest actual data point to each centroid.
        indices = pairwise_distances_argmin(kmeans.cluster_centers_, data_arr)
        return indices
    else:
        raise ValueError("Unknown landmark selection method; use 'random' or 'kmeans'.")
