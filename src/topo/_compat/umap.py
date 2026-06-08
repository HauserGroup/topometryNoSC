"""Adapter for UMAP-specific graph and layout internals.

Delegation strategy:
- Fuzzy simplicial graph construction: delegated to upstream umap-learn
- Standard UMAP layout: available via umap.UMAP estimator
- MAP (Manifold Approximation & Projection): custom TopoMetry implementation
  with checkpoint support (save_every, save_callback, include_init_snapshot)

Only this module imports from :mod:`umap.umap_`. The rest of the package uses
these wrappers so kNN-array validation and return contracts stay centralized.
"""

from inspect import signature
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse


def find_umap_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Return UMAP's fitted low-dimensional membership-curve parameters."""
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread=spread, min_dist=min_dist)
    return float(a), float(b)


def validate_knn_for_umap(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    *,
    n_samples: int,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate kNN arrays and return the first ``n_neighbors`` columns."""
    indices = np.asarray(knn_indices)
    dists = np.asarray(knn_dists)
    if indices.ndim != 2:
        raise ValueError("knn_indices must be a 2D array.")
    if dists.ndim != 2:
        raise ValueError("knn_dists must be a 2D array.")
    if indices.shape != dists.shape:
        raise ValueError("knn_indices and knn_dists must have the same shape.")
    if indices.shape[0] != int(n_samples):
        raise ValueError(
            f"knn_indices must have {int(n_samples)} rows; got {indices.shape[0]}."
        )
    if indices.shape[1] < int(n_neighbors):
        raise ValueError(
            "knn_indices and knn_dists must have at least "
            f"{int(n_neighbors)} columns; got {indices.shape[1]}."
        )
    indices = indices[:, : int(n_neighbors)]
    dists = dists[:, : int(n_neighbors)]
    if np.any(indices < 0):
        raise ValueError("knn_indices contains missing neighbors (-1).")
    if np.any(indices >= n_samples):
        raise ValueError("knn_indices contains indices outside n_samples.")
    if not np.isfinite(dists).all():
        raise ValueError("knn_dists must be finite.")
    if np.any(dists < 0):
        raise ValueError("knn_dists must be nonnegative.")
    if np.any(np.diff(dists, axis=1) < -1e-7):
        raise ValueError("knn_dists rows must be sorted in nondecreasing order.")
    return indices.astype(np.int32, copy=False), dists.astype(np.float32, copy=False)


def _call_fuzzy_simplicial_set(**kwargs: Any):
    from umap.umap_ import fuzzy_simplicial_set

    if "low_memory" not in signature(fuzzy_simplicial_set).parameters:
        kwargs.pop("low_memory", None)
    return fuzzy_simplicial_set(**kwargs)


def _as_csr_matrix(graph: Any) -> csr_matrix:
    return cast(csr_matrix, graph.tocsr())


def _as_optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    return None if arr.ndim == 0 else arr


def fuzzy_graph_from_data(
    X: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int | np.random.RandomState | None,
    metric: str = "euclidean",
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
    low_memory: bool = True,
    metric_kwds: dict[str, Any] | None = None,
    angular: bool = False,
    verbose: bool = False,
    return_dists: bool = False,
) -> (
    tuple[csr_matrix, np.ndarray, np.ndarray]
    | tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]
):
    """Build a UMAP fuzzy simplicial-set graph directly from data."""
    if metric == "precomputed" and issparse(X):
        from topo.base.graph_matrix import get_indices_distances_from_sparse_matrix

        knn_indices, knn_dists = get_indices_distances_from_sparse_matrix(
            X, n_neighbors
        )
        return fuzzy_graph_from_knn(
            np.empty((X.shape[0], 1), dtype=np.float32),
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            verbose=verbose,
            return_dists=return_dists,
        )

    result = cast(
        tuple[Any, ...],
        _call_fuzzy_simplicial_set(
            X=X,
            n_neighbors=int(n_neighbors),
            random_state=random_state,
            metric=metric,
            metric_kwds={} if metric_kwds is None else metric_kwds,
            angular=angular,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            apply_set_operations=True,
            verbose=verbose,
            return_dists=return_dists,
            low_memory=low_memory,
        ),
    )
    graph, sigmas, rhos = result[:3]
    dists = result[3] if len(result) > 3 else None
    if return_dists:
        dists_array = _as_optional_array(dists)
        return _as_csr_matrix(graph), np.asarray(sigmas), np.asarray(rhos), dists_array
    return _as_csr_matrix(graph), np.asarray(sigmas), np.asarray(rhos)


def fuzzy_graph_from_knn(
    X: np.ndarray,
    *,
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    n_neighbors: int,
    random_state: int | np.random.RandomState | None,
    metric: str = "precomputed",
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
    verbose: bool = False,
    return_dists: bool = False,
) -> (
    tuple[csr_matrix, np.ndarray, np.ndarray]
    | tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]
):
    """Build a UMAP fuzzy simplicial-set graph from validated kNN arrays."""
    n_samples = int(np.shape(X)[0])
    indices, dists = validate_knn_for_umap(
        knn_indices,
        knn_dists,
        n_samples=n_samples,
        n_neighbors=n_neighbors,
    )
    result = cast(
        tuple[Any, ...],
        _call_fuzzy_simplicial_set(
            X=X,
            n_neighbors=int(n_neighbors),
            random_state=random_state,
            metric=metric,
            knn_indices=indices,
            knn_dists=dists,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            apply_set_operations=True,
            verbose=verbose,
            return_dists=return_dists,
        ),
    )
    graph, sigmas, rhos = result[:3]
    if return_dists:
        returned_dists = result[3] if len(result) > 3 else dists
        dists_array = _as_optional_array(returned_dists)
        return (
            _as_csr_matrix(graph),
            np.asarray(sigmas),
            np.asarray(rhos),
            dists_array,
        )
    return _as_csr_matrix(graph), np.asarray(sigmas), np.asarray(rhos)
