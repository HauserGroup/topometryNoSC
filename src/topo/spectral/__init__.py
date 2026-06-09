"""Spectral graph operators and eigendecomposition utilities."""

from ._spectral import (
    LE,
    degree,
    degree_matrix,
    degree_vector,
    diffusion_operator,
    graph_laplacian,
)
from .eigen import EigenDecomposition, eigendecompose

__all__ = [
    "graph_laplacian",
    "diffusion_operator",
    "LE",
    "degree",
    "degree_vector",
    "degree_matrix",
    "EigenDecomposition",
    "eigendecompose",
]
