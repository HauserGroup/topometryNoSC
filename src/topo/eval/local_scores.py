"""Local geometry-preservation scores.

Geodesic-distance computation and neighborhood-based correlation scores
(Spearman/Kendall) that quantify how well an embedding preserves local structure.
"""

import logging

import numpy as np
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr

from topo.base.ann import kNN
from topo.utils._utils import get_landmark_indices

logger = logging.getLogger(__name__)


def _matrix_shape(A) -> tuple[int, ...]:
    shape = A.shape if isinstance(A, csr_matrix) else np.shape(A)
    if shape is None:
        raise ValueError("Input must expose a valid shape.")
    return tuple(int(dim) for dim in shape)


def _is_square_matrix(A) -> bool:
    shape = _matrix_shape(A)
    return len(shape) == 2 and shape[0] == shape[1]


def geodesic_distance(
    A,
    method="D",
    unweighted=False,
    directed=False,
    indices=None,
    n_jobs=-1,
    random_state=None,
):
    """Compute the geodesic distance matrix from an adjacency (or affinity) matrix.

    The default behavior is to subset the geodesic distance matrix to only include distances up
    to the k-th nearest neighbor distance for each point. This is to ensure we are only assessing
    the performance of the embedding on the local structure of the data.

    Parameters
    ----------
    A : array-like, shape (n_vertices, n_vertices)
        Adjacency or affinity matrix of a graph.

    method : string, optional, default: 'D'
        Method to compute the shortest path.
        - 'D': Dijkstra's algorithm.
        - 'FW': Floyd-Warshall algorithm.
        - 'B': Bellman-Ford algorithm.
        - 'J': Johnson algorithm.
        - 'F': Floyd algorithm.

    unweighted : bool, optional, default: False
        If True, the adjacency matrix is considered as unweighted.

    directed : bool, optional, default: True
        If True, the adjacency matrix is considered as directed.

    indices : array-like, shape (n_indices, ), optional, default: None
        Indices of the vertices to compute the geodesic distance matrix.

    n_jobs : int, optional, default: 1
        The number of parallel jobs to use during search.

    Returns
    -------
    geodesic_distance : array-like, shape (n_vertices, n_vertices)

    """
    if n_jobs == 1:
        G = csgraph.shortest_path(
            A, method=method, unweighted=unweighted, directed=directed, indices=None
        )
        if indices is not None:
            G = G.T[indices].T
        # guarantee symmetry
        G = (G + G.T) / 2
        # zero diagonal
        G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    else:
        import multiprocessing as mp
        from functools import partial

        if n_jobs == -1:
            from joblib import cpu_count

            n_jobs = cpu_count()
        if not isinstance(n_jobs, int):
            n_jobs = 1
        if method == "FW":
            raise ValueError(
                "The Floyd-Warshall algorithm cannot be used with parallel computations."
            )
        if indices is None:
            indices = np.arange(A.shape[0])
        elif np.issubdtype(type(indices), np.integer):
            indices = np.array([indices])
        n = len(indices)
        local_function = partial(
            csgraph.shortest_path, A, method, directed, False, unweighted, False
        )
        if n_jobs == 1 or n == 1:
            try:
                G = csgraph.shortest_path(
                    A, method, directed, False, unweighted, False, indices
                )
            except csgraph.NegativeCycleError as err:
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present."
                ) from err
        else:
            try:
                with mp.Pool(n_jobs) as pool:
                    G = np.array(pool.map(local_function, indices))
            except csgraph.NegativeCycleError as err:
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present."
                ) from err
        if n == 1:
            G = G.ravel()
        # guarantee symmetry
        G = (G + G.T) / 2
        #
        G[np.where(G == 0)] = np.inf
        # zero diagonal
        G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    return G


def knn_spearman_r(
    data_graph,
    embedding_graph,
    path_method="D",
    subsample_idx=None,
    unweighted=False,
    n_jobs=1,
):
    """Spearman correlation between reference and embedding geodesic distances."""
    # data_graph is a (N,N) similarity matrix from the reference high-dimensional data
    # embedding_graph is a (N,N) similarity matrix from the lower dimensional embedding
    geodesic_dist = geodesic_distance(
        data_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs
    )
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(
        embedding_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs
    )
    if subsample_idx is not None:
        embedded_dist = embedded_dist[subsample_idx, :][:, subsample_idx]
    res, _ = spearmanr(squareform(geodesic_dist), squareform(embedded_dist))
    return res


def knn_kendall_tau(
    data_graph,
    embedding_graph,
    path_method="D",
    subsample_idx=None,
    unweighted=False,
    n_jobs=1,
):
    """Kendall's tau between reference and embedding geodesic distances."""
    geodesic_dist = geodesic_distance(
        data_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs
    )
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(
        embedding_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs
    )
    if subsample_idx is not None:
        embedded_dist = embedded_dist[subsample_idx, :][:, subsample_idx]
    res, _ = kendalltau(squareform(geodesic_dist), squareform(embedded_dist))
    return res


def geodesic_correlation(
    data,
    emb,
    landmarks=None,
    landmark_method="random",
    metric="euclidean",
    n_neighbors=3,
    n_jobs=-1,
    cor_method="spearman",
    random_state=None,
    return_graphs=False,
    verbose=False,
    **kwargs,
):
    """Correlate geodesic distances of ``data`` and embedding ``emb``.

    Builds neighborhood graphs for both, computes geodesic distances (optionally
    on landmarks) and returns their rank correlation under ``cor_method``.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, np.random.RandomState):
        pass
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        raise TypeError(
            "random_state must be None, an int, or a numpy RandomState, "
            f"got {type(random_state)!r}."
        )
    DATA_IS_GRAPH = _is_square_matrix(data)
    EMB_IS_GRAPH = _is_square_matrix(emb)
    if not DATA_IS_GRAPH:
        data_graph = kNN(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            return_instance=False,
            verbose=False,
            **kwargs,
        )
    else:
        data_graph = data.copy()
    if not EMB_IS_GRAPH:
        emb_graph = kNN(
            emb,
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
            return_instance=False,
            verbose=False,
            **kwargs,
        )
    else:
        emb_graph = emb.copy()
    # Define landmarks if applicable

    landmark_indices = None

    if landmarks is not None:
        if isinstance(landmarks, int):
            landmark_indices = get_landmark_indices(
                data_graph,
                n_landmarks=landmarks,
                method=landmark_method,
                random_state=random_state,
            )
            if landmark_indices.shape[0] == _matrix_shape(data)[0]:
                landmark_indices = None
        elif isinstance(landmarks, np.ndarray):
            landmark_indices = landmarks
        else:
            raise ValueError("'landmarks' must be either an integer or a numpy array.")

    if landmark_indices is not None:
        data_graph = csr_matrix(data_graph)
        data_graph = data_graph[landmark_indices, :][:, landmark_indices]

        emb_graph = csr_matrix(emb_graph)
        emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]

    base_geodesics = squareform(
        geodesic_distance(data_graph, directed=False, n_jobs=n_jobs)
    )
    embedding_geodesics = squareform(
        geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
    )
    if cor_method == "spearman":
        if verbose:
            logger.info("Computing Spearman R...")
        corr, _pvalue = spearmanr(base_geodesics, embedding_geodesics)
        results = float(np.asarray(corr).squeeze())

    else:
        if verbose:
            logger.info("Computing Kendall Tau for eigenbasis...")
        corr, _pvalue = kendalltau(base_geodesics, embedding_geodesics)
        results = float(np.asarray(corr).squeeze())

    if return_graphs:
        return results, data_graph, emb_graph
    return results
