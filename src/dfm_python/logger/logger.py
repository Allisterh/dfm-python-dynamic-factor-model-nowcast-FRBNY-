"""Custom logging configuration for dfm-python.

This module provides standard Python logging setup and configuration utilities.
All logging uses Python's standard logging module with hierarchical configuration.
"""

import logging
import os
import sys
import warnings
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime


class CoreDebugFilter(logging.Filter):
    """Filter to capture only core debugging information for log_dfm files.
    
    Captures:
    - EM iterations and convergence
    - Training start/end
    - Initialization summaries
    - Warnings and errors
    - Key parameter updates
    - Skips: DEBUG messages, verbose timing logs, detailed breakdowns
    """
    
    # Keywords that indicate important/core messages
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
    
    # Logger names that should always be included (even DEBUG)
    CORE_LOGGERS = [
        'dfm_python.models.dfm',
        'dfm_python.ssm.kalman',
        'dfm_python.logger.progress'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if record should be logged.
        
        Returns True if:
        - Level is WARNING or higher
        - Level is INFO and message contains core keywords
        - Logger is in CORE_LOGGERS and level is INFO or higher
        - Message starts with important patterns
        """
        # Always include warnings and errors
        if record.levelno >= logging.WARNING:
            return True
        
        # Skip DEBUG messages unless from core loggers
        if record.levelno == logging.DEBUG:
            # Allow DEBUG from core loggers if they're critical
            logger_name = record.name
            if any(logger_name.startswith(core) for core in self.CORE_LOGGERS):
                # Only allow DEBUG for critical initialization/error messages
                msg_lower = record.getMessage().lower()
                if any(kw in msg_lower for kw in ['error', 'failed', 'exception', 'initialization']):
                    return True
            return False
        
        # For INFO messages, check if they contain core keywords
        if record.levelno == logging.INFO:
            msg_lower = record.getMessage().lower()
            
            # Check if from core logger
            logger_name = record.name
            if any(logger_name.startswith(core) for core in self.CORE_LOGGERS):
                # Include INFO from core loggers, but filter verbose timing
                # Skip very detailed breakdown logs
                if any(skip in msg_lower for skip in [
                    'detailed breakdown', 'timing breakdown',
                    'smooth_pair', 'filter timing', 'smooth timing'
                ]):
                    return False
                # Include everything else from core loggers
                return True
            
            # For other loggers, check keywords
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
    """Get a logger instance for a module.
    
    This is the standard way to get a logger in the DFM package.
    All modules should use: _logger = get_logger(__name__)
    
    The returned logger is a plain ``logging.getLogger(name)``. No handler is
    attached and propagation is left untouched: records from module loggers
    (e.g., ``'dfm_python.models.dfm'``) reach the package logger
    ``'dfm_python'``, which only has a ``NullHandler`` until the user opts in
    with :func:`set_verbosity` (or configures the ``logging`` module).
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Logger instance for the module
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
    """Call :func:`set_verbosity` with ``$DFM_PYTHON_VERBOSITY`` when it is set.
    
    An invalid value emits a ``UserWarning`` instead of failing the import.
    """
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
    """Setup package-wide logging configuration.
    
    Alias for configure_logging().
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
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
    """Configure package-wide logging.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level for the package
    format_string : str, optional
        Custom format string. If None, uses default format.
    log_file : str, optional
        Optional file path to write logs to. If provided, logs will be
        written to both console and file.
    module_levels : dict, optional
        Dictionary mapping module names to specific log levels.
        Example: {'dfm_python.models': logging.DEBUG, 'dfm_python.trainer': logging.WARNING}
    log_dfm_dir : Path, optional
        Directory for core debugging logs. If provided and enable_core_log=True,
        creates a timestamped log file with filtered core debugging information.
    enable_core_log : bool, default True
        Whether to enable core debugging log file in log_dfm_dir.
    log_file_prefix : str, optional
        Prefix for log file name. If None, defaults to "dfm_core" or infers from directory name.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure package logger
    logger = logging.getLogger('dfm_python')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    # Clear the configuration flag so it can be reconfigured
    if hasattr(logger, '_dfm_configured'):
        delattr(logger, '_dfm_configured')
    
    # Disable propagation to root logger to prevent duplicate messages
    # If root logger also has handlers, messages would be logged twice
    logger.propagate = False
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified (unfiltered)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add core debugging log file (filtered) in log_dfm_dir
    if enable_core_log and log_dfm_dir is not None:
        log_dfm_dir = Path(log_dfm_dir)
        log_dfm_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log file prefix
        if log_file_prefix is None:
            # Infer from directory name (log_dfm -> dfm_core, log_ddfm -> ddfm_core)
            dir_name = log_dfm_dir.name
            if dir_name == "log_ddfm":
                log_file_prefix = "ddfm_core"
            elif dir_name == "log_dfm":
                log_file_prefix = "dfm_core"
            else:
                log_file_prefix = "dfm_core"  # Default fallback
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        core_log_file = log_dfm_dir / f"{log_file_prefix}_{timestamp}.log"
        
        core_file_handler = logging.FileHandler(core_log_file)
        core_file_handler.setLevel(logging.DEBUG)  # Set to DEBUG so filter can decide
        core_file_handler.setFormatter(formatter)
        
        # Add filter to capture only core debugging info
        core_filter = CoreDebugFilter()
        core_file_handler.addFilter(core_filter)
        
        logger.addHandler(core_file_handler)
        
        # Log that core logging is enabled
        logger.info(f"Core debugging logs enabled: {core_log_file}")
    
    # Set module-specific log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)

