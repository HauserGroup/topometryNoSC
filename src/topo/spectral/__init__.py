"""Spectral operators and eigendecomposition."""

from ._spectral import LE, degree, diffusion_operator, graph_laplacian
from .eigen import EigenDecomposition, eigendecompose

__all__ = [
    "graph_laplacian",
    "diffusion_operator",
    "LE",
    "degree",
    "EigenDecomposition",
    "eigendecompose",
]
