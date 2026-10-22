"""Logging configuration: the package logger, the opt-in verbosity and ``configure_logging``."""

import logging
import os
import sys
import warnings
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime


class CoreDebugFilter(logging.Filter):
    """Keep the core records for the ``log_dfm`` files.

    Passes warnings and errors, INFO from the core loggers (minus detailed
    timing breakdowns), INFO whose message holds a core keyword, and DEBUG
    from the core loggers about errors or initialization.
    """

    CORE_KEYWORDS = [
        'training', 'iteration', 'converged', 'convergence',
        'initialization', 'initialized', 'initializing',
        'failed', 'error', 'warning', 'exception',
        'max_iter', 'threshold', 'loglik', 'delta',
        'completed', 'starting', 'block structure',
        'mixed-frequency', 'dataset ready', 'model saved',
        'eigenval', 'max |Q|', 'max |C|', 'max |A|',
        'e-step:', 'm-step:', 'completed in'
    ]

    CORE_LOGGERS = [
        'dfm_python.models.dfm',
        'dfm_python.ssm.kalman',
        'dfm_python.logger.progress'
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        from_core = any(record.name.startswith(core) for core in self.CORE_LOGGERS)
        if record.levelno == logging.DEBUG:
            if from_core:
                msg_lower = record.getMessage().lower()
                if any(kw in msg_lower for kw in ['error', 'failed', 'exception', 'initialization']):
                    return True
            return False

        if record.levelno == logging.INFO:
            msg_lower = record.getMessage().lower()
            if from_core:
                if any(skip in msg_lower for skip in [
                    'detailed breakdown', 'timing breakdown',
                    'smooth_pair', 'filter timing', 'smooth timing'
                ]):
                    return False
                return True
            if any(kw in msg_lower for kw in self.CORE_KEYWORDS):
                return True

        return False


PACKAGE_LOGGER_NAME = "dfm_python"
"""Name of the package logger that every module logger propagates to."""

VERBOSITY_ENV_VAR = "DFM_PYTHON_VERBOSITY"
"""Environment variable read at import; its value is passed to :func:`set_verbosity`."""

DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATEFMT = '%Y-%m-%d %H:%M:%S'

_VERBOSITY_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# Attribute set on the one handler that set_verbosity installs, so it is never duplicated.
_VERBOSITY_HANDLER_MARK = "_dfm_python_verbosity_handler"


def get_logger(name: str) -> logging.Logger:
    """``logging.getLogger(name)``, the module logger convention of the package.

    No handler is attached and propagation is left untouched: records reach
    the ``'dfm_python'`` logger, which only has a ``NullHandler`` until the
    user opts in with :func:`set_verbosity` (or configures ``logging``).
    """
    return logging.getLogger(name)


def _resolve_level(level) -> int:
    """Turn a verbosity name or a ``logging`` level number into a level number."""
    if isinstance(level, bool):
        raise _verbosity_error(level)
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        key = level.strip().lower()
        if key in _VERBOSITY_LEVELS:
            return _VERBOSITY_LEVELS[key]
        if key.isdigit():
            return int(key)
    raise _verbosity_error(level)


def _verbosity_error(level) -> Exception:
    # Lazy import: dfm_python.utils imports this module through dfm_python.logger.
    from ..utils.errors import ConfigurationError
    names = ", ".join(repr(k) for k in _VERBOSITY_LEVELS)
    return ConfigurationError(
        f"Invalid verbosity {level!r}: pass one of {names} or a logging level number "
        f"such as logging.INFO."
    )


def _verbosity_handler(package_logger: logging.Logger) -> Optional[logging.Handler]:
    """Return the handler installed by :func:`set_verbosity`, if any."""
    for handler in package_logger.handlers:
        if getattr(handler, _VERBOSITY_HANDLER_MARK, False):
            return handler
    return None


def set_verbosity(level) -> None:
    """Show dfm_python log records on standard output from ``level`` upwards.

    The package is silent by default. This is the opt-in: it installs one
    ``StreamHandler(sys.stdout)`` on the ``'dfm_python'`` logger (never more
    than one, however often it is called) and sets the logger's level.

    Parameters
    ----------
    level : {'debug', 'info', 'warning', 'error'} or int
        Verbosity name (case-insensitive) or a ``logging`` level number.

    Raises
    ------
    ConfigurationError
        If ``level`` is not a known name or an integer.
    """
    resolved = _resolve_level(level)
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _verbosity_handler(package_logger) is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
        setattr(handler, _VERBOSITY_HANDLER_MARK, True)
        package_logger.addHandler(handler)
    package_logger.setLevel(resolved)


def get_verbosity() -> int:
    """Return the effective level of the ``'dfm_python'`` logger as a ``logging`` number."""
    return logging.getLogger(PACKAGE_LOGGER_NAME).getEffectiveLevel()


def _apply_env_verbosity() -> None:
    """Call :func:`set_verbosity` with ``$DFM_PYTHON_VERBOSITY`` when it is set; an invalid value warns instead of failing the import."""
    value = os.environ.get(VERBOSITY_ENV_VAR)
    if value is None or not value.strip():
        return
    try:
        set_verbosity(value)
    except ValueError as exc:
        warnings.warn(f"Ignoring {VERBOSITY_ENV_VAR}={value!r}: {exc}", UserWarning, stacklevel=2)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """Alias of :func:`configure_logging` with ``level`` and ``format_string``."""
    configure_logging(level=level, format_string=format_string)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None,
    log_dfm_dir: Optional[Path] = None,
    enable_core_log: bool = True,
    log_file_prefix: Optional[str] = None
) -> None:
    """Replace the ``'dfm_python'`` logger's handlers with a stdout handler and optional files.

    Propagation to the root logger is switched off so records are not
    printed twice.

    Parameters
    ----------
    level : int, default logging.INFO
    format_string : str, optional
        Record format; the default shows time, logger name, level and message.
    log_file : str, optional
        Unfiltered log file, written in addition to the console.
    module_levels : dict, optional
        Levels per logger name, e.g. ``{'dfm_python.models': logging.DEBUG}``.
    log_dfm_dir : Path, optional
        Directory of the timestamped core log file (:class:`CoreDebugFilter`),
        written when ``enable_core_log`` is True.
    enable_core_log : bool, default True
    log_file_prefix : str, optional
        Core log file name prefix; by default ``"ddfm_core"`` for a directory
        named ``log_ddfm`` and ``"dfm_core"`` otherwise.
    """
    if format_string is None:
        format_string = DEFAULT_FORMAT
    formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATEFMT)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if enable_core_log and log_dfm_dir is not None:
        log_dfm_dir = Path(log_dfm_dir)
        log_dfm_dir.mkdir(parents=True, exist_ok=True)
        if log_file_prefix is None:
            log_file_prefix = "ddfm_core" if log_dfm_dir.name == "log_ddfm" else "dfm_core"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        core_log_file = log_dfm_dir / f"{log_file_prefix}_{timestamp}.log"

        core_file_handler = logging.FileHandler(core_log_file)
        core_file_handler.setLevel(logging.DEBUG)  # the filter decides
        core_file_handler.setFormatter(formatter)
        core_file_handler.addFilter(CoreDebugFilter())
        logger.addHandler(core_file_handler)
        logger.info(f"Core debugging logs enabled: {core_log_file}")

    if module_levels:
        for module_name, module_level in module_levels.items():
            logging.getLogger(module_name).setLevel(module_level)
