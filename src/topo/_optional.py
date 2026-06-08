"""Centralised handling of optional dependencies.

The core package depends on the numerical stack and ``umap-learn``. Everything
else (plotting, dataframe I/O, the AMG eigensolver, approximate-nearest-neighbour
backends and the third-party layout libraries) is optional and gated through the
helpers in this module so that:

* a missing optional dependency raises a single, actionable error message that
  names the ``pip install topometry-nosc[...]`` extra to install, and
* detection of available backends lives in *one* place instead of being
  duplicated across :mod:`topo.tpgraph.kernels` and :mod:`topo.layouts.projector`.
"""

import importlib
import importlib.util
from types import ModuleType

__all__ = [
    "has",
    "optional_import",
    "require",
    "available_ann_backends",
    "best_ann_backend",
]

# Map an importable module name to the pip extra that provides it. Modules not
# listed here fall back to a ``pip install <name>`` hint.
_EXTRA_FOR: dict[str, str] = {
    "matplotlib": "plot",
    "pandas": "pandas",
    "pyamg": "amg",
    "hnswlib": "ann",
    "nmslib": "ann",
    "annoy": "ann",
    "faiss": "ann",
    "pacmap": "layouts",
    "pymde": "layouts",
    "trimap": "layouts",
}

# Preference order for approximate nearest-neighbour backends.
_ANN_BACKENDS: tuple[str, ...] = ("hnswlib", "nmslib", "annoy", "faiss")


def has(name: str) -> bool:
    """Return ``True`` if importable module ``name`` is installed."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def optional_import(name: str) -> ModuleType | None:
    """Import and return module ``name``, or ``None`` if it is not installed."""
    if not has(name):
        return None
    return importlib.import_module(name)


def require(name: str, *, purpose: str | None = None) -> ModuleType:
    """Import module ``name`` or raise a friendly :class:`ImportError`.

    Parameters
    ----------
    name : str
        Importable module name, e.g. ``"matplotlib"``.
    purpose : str, optional
        Human-readable description of what the dependency is needed for, used in
        the error message.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed, with a hint naming the relevant
        ``pip install topometry-nosc[<extra>]`` command.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        extra = _EXTRA_FOR.get(name)
        hint = (
            f"pip install topometry-nosc[{extra}]" if extra else f"pip install {name}"
        )
        reason = f" ({purpose})" if purpose else ""
        raise ImportError(
            f"Optional dependency '{name}' is required{reason} but is not "
            f"installed. Install it with `{hint}`."
        ) from exc


def available_ann_backends() -> list[str]:
    """Return installed approximate nearest-neighbour backends, in preference order."""
    return [name for name in _ANN_BACKENDS if has(name)]


def best_ann_backend(preferred: str | None = None) -> str:
    """Pick an ANN backend, honouring ``preferred`` when it is installed.

    Falls back to the first available backend in :data:`_ANN_BACKENDS`, and
    finally to ``"sklearn"`` (always available) when no ANN backend is present.
    """
    if preferred and has(preferred):
        return preferred
    available = available_ann_backends()
    if available:
        return available[0]
    return "sklearn"
