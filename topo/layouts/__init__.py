from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
	"make_epochs_per_sample",
	"simplicial_set_embedding",
	"find_ab_params",
	"fuzzy_embedding",
	"Isomap",
	"Projector",
]

_EXPORTS = {
	"make_epochs_per_sample": (".graph_utils", "make_epochs_per_sample"),
	"simplicial_set_embedding": (".graph_utils", "simplicial_set_embedding"),
	"find_ab_params": (".graph_utils", "find_ab_params"),
	"fuzzy_embedding": (".map", "fuzzy_embedding"),
	"Isomap": (".isomap", "Isomap"),
	"Projector": (".projector", "Projector"),
}

if TYPE_CHECKING:
	from .graph_utils import find_ab_params as find_ab_params
	from .graph_utils import make_epochs_per_sample as make_epochs_per_sample
	from .graph_utils import simplicial_set_embedding as simplicial_set_embedding
	from .isomap import Isomap as Isomap
	from .map import fuzzy_embedding as fuzzy_embedding
	from .projector import Projector as Projector


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
