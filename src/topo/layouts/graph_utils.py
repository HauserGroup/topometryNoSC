"""Graph utilities for MAP/UMAP-style layout optimization.

This module keeps only TopoMetry-specific layout orchestration: graph
preprocessing, initialization, optional checkpoint capture, and optional density
outputs. The numerical SGD optimizer is delegated to ``umap-learn``.
"""

import logging
from typing import Any

import numpy as np
from sklearn.neighbors import KDTree
from umap.layouts import optimize_layout_euclidean, optimize_layout_generic
from umap.umap_ import make_epochs_per_sample

from topo._compat.umap import find_umap_ab_params, fuzzy_graph_from_data
from topo.base import dists as dist
from topo.spectral import LE

find_ab_params = find_umap_ab_params

logger = logging.getLogger(__name__)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def _as_embedding_array(embedding: Any) -> np.ndarray:
    """Return an optimizer result as a 2-D float32 embedding array."""
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Optimizer returned a non-2-D embedding.")
    return arr


def _spectral_initialization(graph, n_components: int, random_state):
    """Return a spectral initialization for MAP/UMAP-style layout optimization.

    This replaces the old ``topo.spectral.eigen.spectral_layout`` helper. The
    initialization is intentionally local to this module because disconnected
    graph layout policy belongs with layout optimization, not with the generic
    eigendecomposition transformer.
    """
    try:
        init = LE(
            graph,
            n_eigs=int(n_components),
            laplacian_type="normalized",
            drop_first=True,
            return_evals=False,
        )
    except Exception as exc:
        logger.warning(
            "Spectral initialization failed; falling back to random initialization: %s",
            exc,
        )
        return random_state.uniform(
            low=-10.0,
            high=10.0,
            size=(graph.shape[0], int(n_components)),
        ).astype(np.float32)

    return np.asarray(init, dtype=np.float32)


def simplicial_set_embedding(
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    densmap,
    densmap_kwds=None,
    output_dens=False,
    output_metric=dist.euclidean_grad,
    output_metric_kwds=None,
    euclidean_output=True,
    parallel=True,
    verbose=False,
    save_every=None,
    save_limit=None,
    save_callback=None,
    include_init_snapshot=True,
):
    """Perform a fuzzy simplicial-set embedding using UMAP's layout optimizer.

    This function keeps TopoMetry-specific orchestration around graph
    preprocessing, initialization, optional density outputs, and checkpoint
    metadata. The numerical SGD optimizer itself is delegated to upstream
    ``umap-learn``.

    Parameters
    ----------
    graph : sparse matrix
        Weighted adjacency matrix of the high-dimensional fuzzy 1-skeleton.
    n_components : int
        Target embedding dimensionality.
    initial_alpha : float
        Initial learning rate for SGD.
    a, b : float
        Parameters of the low-dimensional UMAP attraction curve.
    gamma : float
        Negative-sample repulsion weight.
    negative_sample_rate : float
        Number of negative samples drawn per positive edge.
    n_epochs : int or None
        Number of optimization epochs. If ``None`` or ``<= 0``, uses the UMAP
        heuristic: 1000 epochs for small graphs and 300 for larger graphs, plus
        200 extra epochs for densMAP.
    init : {'spectral', 'random'} or ndarray
        Initialization strategy or explicit initial coordinates.
    random_state : numpy.random.RandomState
        Random-number generator.
    metric, metric_kwds
        Metric information used only for optional embedding-density outputs.
    densmap : bool
        Whether to use the density-augmented densMAP objective.
    densmap_kwds : dict or None
        densMAP auxiliary data. Required when ``densmap`` or ``output_dens`` is
        enabled.
    output_dens : bool, default=False
        Whether to compute original and embedding radii.
    output_metric, output_metric_kwds
        Output metric and keyword arguments for non-Euclidean layout
        optimization.
    euclidean_output : bool, default=True
        Whether to use UMAP's Euclidean layout optimizer. If ``False``, use the
        generic metric optimizer.
    parallel : bool, default=True
        Whether to allow UMAP's numba layout optimizer to run in parallel.
    verbose : bool, default=False
        Whether to log progress information.
    save_every : int or None, optional
        Retained for API compatibility. In the simplified upstream-UMAP path,
        per-epoch checkpointing is not implemented. If provided and ``> 0``, the
        final embedding is stored in ``aux_data["checkpoints"]``.
    save_limit : int or None, optional
        Maximum number of snapshots to keep in memory.
    save_callback : callable or None, optional
        Optional callback called as ``save_callback(epoch, embedding)`` for each
        stored snapshot.
    include_init_snapshot : bool, default=True
        Whether to store a snapshot at epoch 0 after initialization and before
        SGD.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        Final optimized embedding.
    aux_data : dict
        Auxiliary outputs. Contains ``"initialization"`` and, when requested,
        ``"checkpoints"``, ``"rad_orig"``, and/or ``"rad_emb"``.
    """
    densmap_kwds = {} if densmap_kwds is None else dict(densmap_kwds)
    output_metric_kwds = {} if output_metric_kwds is None else dict(output_metric_kwds)

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = int(graph.shape[1])

    if n_epochs is None or n_epochs <= 0:
        n_epochs = 1000 if graph.shape[0] <= 10000 else 300
        if densmap:
            n_epochs += 200
    n_epochs = int(n_epochs)

    if graph.nnz == 0:
        raise ValueError("Cannot optimize an empty fuzzy graph.")

    max_weight = float(graph.data.max())
    if max_weight <= 0:
        raise ValueError("Fuzzy graph must contain at least one positive edge weight.")

    graph.data[graph.data < (max_weight / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if graph.nnz == 0:
        raise ValueError(
            "All fuzzy graph edges were pruned before layout optimization."
        )

    if isinstance(init, np.ndarray):
        embedding = np.asarray(init, dtype=np.float32)
        if embedding.ndim != 2:
            raise ValueError("Explicit init array must be 2-D.")
        if embedding.shape != (graph.shape[0], int(n_components)):
            raise ValueError(
                "Explicit init array must have shape "
                f"{(graph.shape[0], int(n_components))}; got {embedding.shape}."
            )
        initialisation = embedding.copy()

    elif isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0,
            high=10.0,
            size=(graph.shape[0], int(n_components)),
        ).astype(np.float32)
        initialisation = embedding.copy()

    elif isinstance(init, str) and init == "spectral":
        initialisation = _spectral_initialization(
            graph,
            n_components=int(n_components),
            random_state=random_state,
        )

        scale = float(np.abs(initialisation).max())
        expansion = 10.0 / scale if scale > 0 else 1.0

        embedding = (initialisation * expansion).astype(np.float32)
        embedding += random_state.normal(
            scale=0.0001,
            size=(graph.shape[0], int(n_components)),
        ).astype(np.float32)

    else:
        init_data = np.asarray(init, dtype=np.float32)
        if init_data.ndim != 2:
            raise ValueError("init must be 'random', 'spectral', or a 2-D array.")
        if init_data.shape != (graph.shape[0], int(n_components)):
            raise ValueError(
                "Explicit init array must have shape "
                f"{(graph.shape[0], int(n_components))}; got {init_data.shape}."
            )

        if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
            tree = KDTree(init_data)
            dist_arr, _ = tree.query(init_data, k=2)
            nndist = float(np.mean(dist_arr[:, 1]))
            embedding = init_data + random_state.normal(
                scale=0.001 * nndist,
                size=init_data.shape,
            ).astype(np.float32)
        else:
            embedding = init_data.copy()

        initialisation = embedding.copy()

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data: dict[str, Any] = {}
    checkpoints: list[dict[str, Any]] = []

    def _store_snapshot(epoch: int, Y: np.ndarray) -> None:
        """Store a snapshot in memory and/or stream it through a callback."""
        if save_callback is not None:
            try:
                save_callback(int(epoch), Y)
            except Exception as exc:
                if verbose:
                    logger.warning(
                        "save_callback failed at epoch %s: %s",
                        epoch,
                        exc,
                    )

        checkpoints.append({"epoch": int(epoch), "embedding": Y.copy()})

        if save_limit is not None and len(checkpoints) > int(save_limit):
            del checkpoints[0]

    if densmap or output_dens:
        if "graph_dists" not in densmap_kwds:
            raise ValueError(
                "densmap_kwds must contain 'graph_dists' when densmap or "
                "output_dens is enabled."
            )

        if verbose:
            logger.info("Computing original densities")

        dists = densmap_kwds["graph_dists"]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        ro = np.zeros(n_vertices, dtype=np.float32)

        for i in range(len(head)):
            j = head[i]
            k = tail[i]
            D = dists[j, k] * dists[j, k]
            mu = graph.data[i]
            ro[j] += mu * D
            ro[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        mu_sum_safe = mu_sum.copy()
        mu_sum_safe[mu_sum_safe == 0.0] = 1.0
        ro = np.log(epsilon + (ro / mu_sum_safe))

        if densmap:
            ro_std = float(np.std(ro))
            if ro_std == 0.0:
                R = np.zeros_like(ro, dtype=np.float32)
            else:
                R = ((ro - np.mean(ro)) / ro_std).astype(np.float32)

            densmap_kwds["mu"] = graph.data
            densmap_kwds["mu_sum"] = mu_sum
            densmap_kwds["R"] = R

        if output_dens:
            aux_data["rad_orig"] = ro

    coord_min = np.min(embedding, axis=0)
    coord_range = np.max(embedding, axis=0) - coord_min
    coord_range[coord_range == 0.0] = 1.0

    embedding = (10.0 * (embedding - coord_min) / coord_range).astype(
        np.float32,
        order="C",
    )

    if include_init_snapshot:
        _store_snapshot(epoch=0, Y=embedding)

    epochs_per_sample = make_epochs_per_sample(weight, n_epochs)

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma=gamma,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            move_other=True,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma=gamma,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            output_metric=output_metric,
            output_metric_kwds=tuple(output_metric_kwds.values()),
            verbose=verbose,
            move_other=True,
        )

    embedding = _as_embedding_array(embedding)

    if save_every is not None and int(save_every) > 0:
        _store_snapshot(epoch=n_epochs, Y=embedding)

    if output_dens:
        if verbose:
            logger.info("Computing embedding densities")

        fss_result = fuzzy_graph_from_data(
            embedding,
            n_neighbors=int(densmap_kwds["n_neighbors"]),
            random_state=random_state,
            metric=metric,
            verbose=verbose,
            return_dists=True,
        )
        if len(fss_result) != 4:
            raise RuntimeError(
                "Expected fuzzy_graph_from_data(..., return_dists=True) to return "
                "(graph, sigmas, rhos, dists)."
            )

        emb_graph, _emb_sigmas, _emb_rhos, emb_dists_raw = fss_result
        emb_dists = np.asarray(emb_dists_raw)

        emb_graph = emb_graph.tocoo()
        emb_graph.sum_duplicates()
        emb_graph.eliminate_zeros()

        emb_shape = emb_graph.shape
        if emb_shape is None:
            raise ValueError("Embedding graph must have a valid shape.")

        n_emb_vertices = int(emb_shape[1])
        mu_sum = np.zeros(n_emb_vertices, dtype=np.float32)
        re = np.zeros(n_emb_vertices, dtype=np.float32)

        head_e = emb_graph.row
        tail_e = emb_graph.col

        for i in range(len(head_e)):
            j = head_e[i]
            k = tail_e[i]
            D = emb_dists[j, k]
            mu = emb_graph.data[i]
            re[j] += mu * D
            re[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        mu_sum_safe = mu_sum.copy()
        mu_sum_safe[mu_sum_safe == 0.0] = 1.0
        aux_data["rad_emb"] = np.log(epsilon + (re / mu_sum_safe))

    aux_data["initialization"] = initialisation

    if checkpoints:
        aux_data["checkpoints"] = checkpoints

    return embedding, aux_data
