"""General-purpose helpers shared across the package.

Exposes small utilities for converting between sparse neighbor graphs and
dense index/distance arrays, plus landmark selection used by the layout and
evaluation routines.
"""

from ._utils import (
    get_indices_distances_from_sparse_matrix,
    get_landmark_indices,
    get_sparse_matrix_from_indices_distances,
)

__all__ = [
    "get_indices_distances_from_sparse_matrix",
    "get_landmark_indices",
    "get_sparse_matrix_from_indices_distances",
]
