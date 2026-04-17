from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
	"graph_laplacian",
	"diffusion_operator",
	"LE",
	"degree",
	"optimize_layout_euclidean",
	"optimize_layout_generic",
	"optimize_layout_inverse",
	"optimize_layout_aligned_euclidean",
	"EigenDecomposition",
	"eigendecompose",
]

_EXPORTS = {
	"graph_laplacian": ("._spectral", "graph_laplacian"),
	"diffusion_operator": ("._spectral", "diffusion_operator"),
	"LE": ("._spectral", "LE"),
	"degree": ("._spectral", "degree"),
	"optimize_layout_euclidean": (
		".umap_layouts",
		"optimize_layout_euclidean",
	),
	"optimize_layout_generic": (".umap_layouts", "optimize_layout_generic"),
	"optimize_layout_inverse": (".umap_layouts", "optimize_layout_inverse"),
	"optimize_layout_aligned_euclidean": (
		".umap_layouts",
		"optimize_layout_aligned_euclidean",
	),
	"EigenDecomposition": (".eigen", "EigenDecomposition"),
	"eigendecompose": (".eigen", "eigendecompose"),
}

if TYPE_CHECKING:
	from ._spectral import LE as LE
	from ._spectral import degree as degree
	from ._spectral import diffusion_operator as diffusion_operator
	from ._spectral import graph_laplacian as graph_laplacian
	from .eigen import EigenDecomposition as EigenDecomposition
	from .eigen import eigendecompose as eigendecompose
	from .umap_layouts import optimize_layout_aligned_euclidean as optimize_layout_aligned_euclidean
	from .umap_layouts import optimize_layout_euclidean as optimize_layout_euclidean
	from .umap_layouts import optimize_layout_generic as optimize_layout_generic
	from .umap_layouts import optimize_layout_inverse as optimize_layout_inverse


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