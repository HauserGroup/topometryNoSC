import importlib.util

from .ann import grid_search, kNN

__all__ = ["kNN", "grid_search"]

_have_numba = importlib.util.find_spec("numba") is not None

if _have_numba:
    from .dists import pairwise_distances as pairwise_distances
    from .sparse import sparse_diff as sparse_diff
