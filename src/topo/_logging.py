"""Logging configuration for TopoMetry.

Library code emits diagnostics through the ``"topo"`` logger hierarchy (each
module uses ``logging.getLogger(__name__)``) instead of printing to stdout.
:func:`configure` maps the estimators' ``verbose`` / ``verbosity`` flags onto a
log level and attaches a handler the first time it is called, so the familiar
``verbose=True`` behaviour is preserved without polluting stdout by default.
"""

import logging

#: Root logger for the package; every module logger is a child of this one.
logger = logging.getLogger("topo")


def configure(verbose: bool | int = False, level: int | None = None) -> None:
    """Set the package log level and ensure a handler is attached.

    Parameters
    ----------
    verbose : bool or int, optional
        When truthy, the package logs at ``INFO`` (or ``DEBUG`` for values >= 2);
        otherwise it stays at ``WARNING``. Ignored if ``level`` is given.
    level : int, optional
        Explicit :mod:`logging` level overriding ``verbose``.
    """
    if level is None:
        if int(verbose) >= 2:
            level = logging.DEBUG
        elif verbose:
            level = logging.INFO
        else:
            level = logging.WARNING
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(handler)
    # Let the application control propagation/formatting if it wants to.
    logger.propagate = False
