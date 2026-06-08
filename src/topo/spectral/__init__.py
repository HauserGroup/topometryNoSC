"""Spectral operators and eigendecomposition.

Graph Laplacians, diffusion operators and Laplacian-eigenmap layouts, plus the
:class:`~topo.spectral.eigen.EigenDecomposition` transformer that turns a kernel
or operator into a spectral embedding. Members are imported lazily.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "graph_laplacian",
    "diffusion_operator",
    "EigenDecomposition",
    "eigendecompose",
]

_EXPORTS = {
    "graph_laplacian": ("._spectral", "graph_laplacian"),
    "diffusion_operator": ("._spectral", "diffusion_operator"),
    "EigenDecomposition": (".eigen", "EigenDecomposition"),
    "eigendecompose": (".eigen", "eigendecompose"),
}

if TYPE_CHECKING:
    from ._spectral import diffusion_operator as diffusion_operator
    from ._spectral import graph_laplacian as graph_laplacian
    from .eigen import EigenDecomposition as EigenDecomposition
    from .eigen import eigendecompose as eigendecompose


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
