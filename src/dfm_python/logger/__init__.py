"""Logging utilities.

The package is silent by default: the ``'dfm_python'`` logger has only a
``NullHandler``.

- ``logger``: ``get_logger``, ``setup_logging``, ``configure_logging`` and the
  opt-in verbosity (``set_verbosity``, ``get_verbosity`` and the
  ``DFM_PYTHON_VERBOSITY`` environment variable).
- ``progress``: the fit-scoped contexts ``fit_verbosity`` and
  ``quiet_engine_loggers``, and the engines' progress reporting
  (``log_em_iteration``, ``log_convergence``, ``iVDFMTrainLogger``).
"""

from .logger import (
    get_logger,
    setup_logging,
    configure_logging,
    set_verbosity,
    get_verbosity,
)

from .progress import fit_verbosity, quiet_engine_loggers

__all__ = [
    'get_logger',
    'setup_logging',
    'configure_logging',
    'set_verbosity',
    'get_verbosity',
    'fit_verbosity',
    'quiet_engine_loggers',
]
