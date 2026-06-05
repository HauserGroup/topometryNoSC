"""TopOMetry — geometry-first topological dimensionality reduction.

`topo` learns the latent geometry of high-dimensional data through a pipeline
of neighborhood graphs, Laplace-Beltrami-type operators, spectral scaffolds and
refined graphs, then builds low-dimensional layouts for analysis and
visualization. The high-level entry point is the :class:`~topo.topograph.TopOGraph`
orchestrator, which wraps the scikit-learn-style transformers in
:mod:`topo.tpgraph`, :mod:`topo.spectral` and :mod:`topo.layouts`::

    import topo

    tg = topo.TopOGraph()
    embedding = tg.fit_transform(X)

Submodules are imported lazily, so importing :mod:`topo` is cheap and optional
dependencies are only required when the features that use them are called.
"""

from importlib import import_module
from typing import TYPE_CHECKING


def _ensure_no_upstream_conflict() -> None:
    """Fail fast if the upstream ``topometry`` distribution is co-installed.

    This fork ships the same import package name (``topo``) as upstream
    TopOMetry. Installing both ``topometry`` and ``topometry-nosc`` into one
    environment lets them silently overwrite each other's ``topo/`` files, so a
    mixed install is never valid. Detect it and raise immediately with
    actionable instructions.
    """
    import importlib.metadata as _metadata

    try:
        _metadata.distribution("topometry")
    except _metadata.PackageNotFoundError:
        return

    raise ImportError(
        "Conflicting installation detected: the upstream 'topometry' "
        "distribution is installed in the same environment as this fork "
        "('topometry-nosc'). Both provide the import package 'topo' and will "
        "overwrite each other. Keep exactly one:\n"
        "    pip uninstall topometry        # keep this fork (topometry-nosc)\n"
        "  or\n"
        "    pip uninstall topometry-nosc   # keep upstream TopOMetry"
    )


_ensure_no_upstream_conflict()

__all__ = [
    "TopOGraph",
    "load_topograph",
    "save_topograph",
    "configure_logging",
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
    "configure_logging": ("._logging", "configure"),
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
    from ._logging import configure as configure_logging
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
