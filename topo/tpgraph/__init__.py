from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
	"compute_kernel",
	"Kernel",
	"cknn_graph",
	"fuzzy_simplicial_set",
	"IntrinsicDim",
	"automated_scaffold_sizing",
]

_EXPORTS = {
	"compute_kernel": (".kernels", "compute_kernel"),
	"Kernel": (".kernels", "Kernel"),
	"cknn_graph": (".cknn", "cknn_graph"),
	"fuzzy_simplicial_set": (".fuzzy", "fuzzy_simplicial_set"),
	"IntrinsicDim": (".intrinsic_dim", "IntrinsicDim"),
	"automated_scaffold_sizing": (
		".intrinsic_dim",
		"automated_scaffold_sizing",
	),
}

if TYPE_CHECKING:
	from .cknn import cknn_graph as cknn_graph
	from .fuzzy import fuzzy_simplicial_set as fuzzy_simplicial_set
	from .intrinsic_dim import IntrinsicDim as IntrinsicDim
	from .intrinsic_dim import automated_scaffold_sizing as automated_scaffold_sizing
	from .kernels import Kernel as Kernel
	from .kernels import compute_kernel as compute_kernel


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
