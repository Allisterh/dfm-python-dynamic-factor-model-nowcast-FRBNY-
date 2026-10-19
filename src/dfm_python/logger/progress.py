"""Fit-scoped progress output.

``fit_verbosity(verbose)`` maps a model's ``verbose`` hyperparameter to log
output that exists only while ``fit`` runs:

- ``verbose=0``: nothing is attached; the package stays silent.
- ``verbose=1``: a temporary ``StreamHandler(sys.stdout)`` shows INFO records.
- ``verbose=2``: the same handler shows DEBUG records.

On exit, also when the body raises, the temporary handler is removed and the
``'dfm_python'`` logger's level is restored. If :func:`set_verbosity` already
installed its handler, no second handler is added (records would otherwise be
printed twice); only the level is lowered for the duration of the block.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Iterator, Optional

from .logger import (
    DEFAULT_DATEFMT,
    DEFAULT_FORMAT,
    PACKAGE_LOGGER_NAME,
    _verbosity_handler,
)

# Attribute set on the temporary handler, so tests and callers can recognise it.
_FIT_HANDLER_MARK = "_dfm_python_fit_handler"


def _verbose_to_level(verbose: int) -> Optional[int]:
    """Return the logging level for ``verbose``, or ``None`` for silent."""
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        # Lazy import: dfm_python.utils imports dfm_python.logger.
        from ..utils.errors import ConfigurationError
        raise ConfigurationError(
            f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {verbose!r}"
        )
    if verbose == 0:
        return None
    if verbose == 1:
        return logging.INFO
    return logging.DEBUG


@contextmanager
def fit_verbosity(verbose: int) -> Iterator[None]:
    """Show progress records from the ``'dfm_python'`` logger inside the block.

    Parameters
    ----------
    verbose : int
        0 is silent, 1 shows INFO progress, 2 (or more) shows DEBUG records.

    Raises
    ------
    ConfigurationError
        If ``verbose`` is not a non-negative integer.
    """
    level = _verbose_to_level(verbose)
    if level is None:
        yield
        return

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    saved_level = package_logger.level
    handler: Optional[logging.Handler] = None
    if _verbosity_handler(package_logger) is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
        setattr(handler, _FIT_HANDLER_MARK, True)
        package_logger.addHandler(handler)
    # Only ever make the logger more verbose; a user who set DEBUG keeps DEBUG.
    if package_logger.getEffectiveLevel() > level:
        package_logger.setLevel(level)
    try:
        yield
    finally:
        if handler is not None:
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(saved_level)
