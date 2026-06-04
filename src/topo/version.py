"""Package version.

``__version__`` below is the single static source of truth read at build time
by Hatchling. At runtime, if the package is installed, the value is refreshed
from the installed distribution metadata so editable/source trees and built
wheels always agree.
"""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__version__ = "2.1.0"

with suppress(PackageNotFoundError):  # local editable / source-tree import
    __version__ = _version("topometry")
