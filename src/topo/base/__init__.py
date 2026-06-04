"""Foundational primitives: neighbor search and distances.

Approximate/exact k-nearest-neighbor search (:func:`~topo.base.ann.kNN`) and the
numba-accelerated distance and sparse-graph helpers the rest of the package
builds on. The numba-dependent helpers are imported only when ``numba`` is
available.
"""

import importlib.util

from .ann import grid_search, kNN

__all__ = ["kNN", "grid_search"]

_have_numba = importlib.util.find_spec("numba") is not None

if _have_numba:
    from .dists import pairwise_distances as pairwise_distances
    from .sparse import sparse_diff as sparse_diff
