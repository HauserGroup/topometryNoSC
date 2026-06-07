"""Adapter for UMAP-specific graph and layout internals.

Only this module imports from :mod:`umap.umap_`. The rest of the package uses
these wrappers so version checks and kNN-array validation stay centralized.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
from packaging.version import Version
from scipy.sparse import csr_matrix

MIN_UMAP = Version("0.5.8")
MAX_UMAP = Version("0.6")


def check_umap_version() -> None:
    """Raise if the installed ``umap-learn`` version is unsupported."""
    installed = Version(version("umap-learn"))
    if not (MIN_UMAP <= installed < MAX_UMAP):
        raise RuntimeError(
            f"topometry-nosc requires umap-learn>={MIN_UMAP},<{MAX_UMAP}; "
            f"found {installed}."
        )


def find_umap_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Return UMAP's fitted low-dimensional membership-curve parameters."""
    check_umap_version()
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
    """Validate and normalize fixed-width kNN arrays for ``umap-learn``."""
    indices = np.asarray(knn_indices)
    dists = np.asarray(knn_dists)
    expected_shape = (int(n_samples), int(n_neighbors))
    if indices.shape != expected_shape:
        raise ValueError(
            f"knn_indices must have shape {expected_shape}; got {indices.shape}."
        )
    if dists.shape != expected_shape:
        raise ValueError(
            f"knn_dists must have shape {expected_shape}; got {dists.shape}."
        )
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
    check_umap_version()
    from umap.umap_ import fuzzy_simplicial_set

    result = fuzzy_simplicial_set(
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
    )
    graph, sigmas, rhos = result[:3]
    dists = result[3] if len(result) > 3 else None
    del low_memory  # kept for a stable adapter signature
    if return_dists:
        return graph.tocsr(), np.asarray(sigmas), np.asarray(rhos), dists
    return graph.tocsr(), np.asarray(sigmas), np.asarray(rhos)


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
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a UMAP fuzzy simplicial-set graph from fixed-width kNN arrays."""
    check_umap_version()
    from umap.umap_ import fuzzy_simplicial_set

    n_samples = int(np.shape(X)[0])
    indices, dists = validate_knn_for_umap(
        knn_indices,
        knn_dists,
        n_samples=n_samples,
        n_neighbors=n_neighbors,
    )
    result = fuzzy_simplicial_set(
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
        return_dists=False,
    )
    graph, sigmas, rhos = result[:3]
    return graph.tocsr(), np.asarray(sigmas), np.asarray(rhos)
