"""Deprecated compatibility access to UMAP fuzzy-graph helpers.

UMAP-specific fuzzy graph construction is delegated to ``umap-learn`` through
``topo._compat.umap``. This module remains only to avoid abruptly breaking
direct imports during the transition.
"""

from __future__ import annotations

import warnings

from topo._compat.umap import fuzzy_graph_from_data

__all__ = ["fuzzy_simplicial_set"]


def _warn(name: str) -> None:
    warnings.warn(
        f"topo.tpgraph.fuzzy.{name} is deprecated; use umap.umap_.{name} "
        "or topo._compat.umap instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _fuzzy_simplicial_set_adapter(
    X,
    n_neighbors=15,
    metric="cosine",
    backend="nmslib",
    n_jobs=1,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    return_dists=False,
    verbose=False,
    **kwargs,
):
    _warn("fuzzy_simplicial_set")
    if not apply_set_operations:
        raise ValueError("umap-learn adapter requires apply_set_operations=True.")
    del backend, n_jobs
    return fuzzy_graph_from_data(
        X,
        n_neighbors=n_neighbors,
        random_state=kwargs.pop("random_state", None),
        metric=metric,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        metric_kwds=kwargs.pop("metric_kwds", None),
        angular=kwargs.pop("angular", False),
        verbose=verbose,
        return_dists=return_dists,
    )


fuzzy_simplicial_set = _fuzzy_simplicial_set_adapter


def __getattr__(name: str):
    if name in {"smooth_knn_dist", "compute_membership_strengths"}:
        _warn(name)
        from umap import umap_ as umap_internal

        return getattr(umap_internal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
