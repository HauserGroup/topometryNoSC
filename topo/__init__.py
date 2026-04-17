from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
	"TopOGraph",
	"load_topograph",
	"save_topograph",
	"layouts",
	"plot",
	"spectral",
	"tpgraph",
	"eval",
	"utils",
	"analysis",
	"uom",
	"__version__",
]

_MODULE_EXPORTS = {
	"layouts": ".layouts",
	"plot": ".plot",
	"spectral": ".spectral",
	"tpgraph": ".tpgraph",
	"eval": ".eval",
	"utils": ".utils",
	"analysis": ".analysis",
	"uom": ".uom",
}

_ATTRIBUTE_EXPORTS = {
	"TopOGraph": (".topograph", "TopOGraph"),
	"load_topograph": (".topograph", "load_topograph"),
	"save_topograph": (".topograph", "save_topograph"),
	"__version__": (".version", "__version__"),
}

if TYPE_CHECKING:
	from . import analysis as analysis
	from . import eval as eval
	from . import layouts as layouts
	from . import plot as plot
	from . import spectral as spectral
	from . import tpgraph as tpgraph
	from . import uom as uom
	from . import utils as utils
	from .topograph import TopOGraph as TopOGraph
	from .topograph import load_topograph as load_topograph
	from .topograph import save_topograph as save_topograph
	from .version import __version__ as __version__


def __getattr__(name):
	if name in _MODULE_EXPORTS:
		module = import_module(_MODULE_EXPORTS[name], __name__)
		globals()[name] = module
		return module

	if name in _ATTRIBUTE_EXPORTS:
		module_name, attr_name = _ATTRIBUTE_EXPORTS[name]
		module = import_module(module_name, __name__)
		value = getattr(module, attr_name)
		globals()[name] = value
		return value

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return sorted(set(globals()) | set(__all__))
