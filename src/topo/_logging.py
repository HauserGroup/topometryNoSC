"""Logging configuration for TopoMetry.

Library code emits diagnostics through the ``"topo"`` logger hierarchy. Modules
should use ``logging.getLogger(__name__)`` rather than printing to stdout.

The :func:`configure` helper maps estimator ``verbose`` / ``verbosity`` flags
onto package log levels and attaches a default stream handler the first time it
is called. Applications may still configure logging themselves by replacing
handlers or changing propagation on the ``"topo"`` logger.
"""

import logging

#: Root logger for the package; every module logger is a child of this one.
logger = logging.getLogger("topo")


def configure(
    verbose: bool | int = False,
    level: int | None = None,
    *,
    propagate: bool = False,
) -> None:
    """Set the package log level and ensure a default handler is attached.

    Parameters
    ----------
    verbose : bool or int, default=False
        When truthy, log at ``INFO``. Values >= 2 log at ``DEBUG``. Ignored if
        ``level`` is provided.
    level : int, optional
        Explicit :mod:`logging` level overriding ``verbose``.
    propagate : bool, default=False
        Whether ``"topo"`` log records should propagate to ancestor loggers.
        Keep this ``False`` to avoid duplicate messages when the default handler
        is attached.
    """
    if level is None:
        if isinstance(verbose, bool):
            level = logging.INFO if verbose else logging.WARNING
        elif int(verbose) >= 2:
            level = logging.DEBUG
        elif int(verbose) >= 1:
            level = logging.INFO
        else:
            level = logging.WARNING

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(handler)

    logger.propagate = bool(propagate)
