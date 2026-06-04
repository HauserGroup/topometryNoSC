"""Package version."""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version("topometry")
except PackageNotFoundError:  # local editable/tree import
    __version__ = "2.1.0"
