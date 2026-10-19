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

``quiet_engine_loggers(verbose, names)`` covers the other half of
``verbose=0``: an engine that logs progress at INFO stays silent during
``fit`` even when the user opted in with :func:`set_verbosity`, because the
engine's loggers are raised to WARNING (or another level) for the duration of
the block.

The engines' progress reporting lives here too (it was in the removed
``train_logger`` and ``ivdfm_logger`` modules; messages and levels are
unchanged), logged to the ``'dfm_python.logger.progress'`` logger:

- :func:`log_em_iteration` and :func:`log_convergence`, called by the DFM EM
  loop (``functional.em``);
- :class:`iVDFMTrainLogger`, the iVDFM engine's training reporter.
"""

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from .logger import (
    DEFAULT_DATEFMT,
    DEFAULT_FORMAT,
    PACKAGE_LOGGER_NAME,
    _verbosity_handler,
    get_logger,
)

_logger = get_logger(__name__)

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


@contextmanager
def quiet_engine_loggers(verbose: int, names: Iterable[str], level: int = logging.WARNING) -> Iterator[None]:
    """With ``verbose=0``, raise the loggers ``names`` (and their children) to ``level`` inside the block.

    Engines log their progress (epochs, initialization) at INFO. Inside
    ``fit`` with ``verbose=0`` those records are dropped even when the package
    logger is at INFO (for example after ``set_verbosity("info")``); with the
    default ``level`` warnings and errors still pass. With ``verbose >= 1``
    nothing changes, so :func:`fit_verbosity` shows the records. A logger that
    is already at ``level`` or above keeps its level. Levels are restored on
    exit, also when the body raises.

    Parameters
    ----------
    verbose : int
        The model's ``verbose`` hyperparameter (0 silent, 1 progress, 2 debug).
    names : iterable of str
        Logger names, for example ``"dfm_python.models.ivdfm"``.
    level : int, default logging.WARNING
        The level the loggers are raised to. A progress reporter whose
        summaries are logged as warnings (such as "did not converge") can be
        raised to ``logging.ERROR``.

    Raises
    ------
    ConfigurationError
        If ``verbose`` is not a non-negative integer.
    """
    if _verbose_to_level(verbose) is not None:
        yield
        return
    saved = []
    for name in names:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        if logger.getEffectiveLevel() < level:
            logger.setLevel(level)
    try:
        yield
    finally:
        for logger, level in reversed(saved):
            logger.setLevel(level)


# ---------------------------------------------------------------------------
# Engine progress reporting
# ---------------------------------------------------------------------------


def log_em_iteration(
    iteration: int,
    loglik: Optional[float] = None,
    delta: Optional[float] = None,
    **kwargs
) -> None:
    """Log EM algorithm iteration (convenience function for DFM).
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    loglik : float, optional
        Log-likelihood value
    delta : float, optional
        Convergence delta
    **kwargs
        Additional metrics to log
    """
    if loglik is not None:
        msg = f"EM iteration {iteration:4d} | Log-likelihood: {loglik:12.6f}"
    else:
        msg = f"EM iteration {iteration:4d}"
    
    if delta is not None:
        msg += f" | Delta: {delta:10.6e}"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            msg += f" | {key}: {value:.6f}"
        elif isinstance(value, np.ndarray):
            msg += f" | {key}: shape {value.shape}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


def log_convergence(
    converged: bool,
    num_iter: Optional[int] = None,
    num_epochs: Optional[int] = None,
    final_loglik: Optional[float] = None,
    final_loss: Optional[float] = None,
    reason: Optional[str] = None,
    model_type: str = "dfm"
) -> None:
    """Log convergence status (convenience function).
    
    Parameters
    ----------
    converged : bool
        Whether training converged
    num_iter : int, optional
        Number of EM iterations completed (for DFM)
    num_epochs : int, optional
        Number of epochs completed (for DDFM)
    final_loglik : float, optional
        Final log-likelihood value (for DFM)
    final_loss : float, optional
        Final training loss (for DDFM)
    reason : str, optional
        Reason for stopping
    model_type : str, default "dfm"
        Type of model: "dfm" or "ddfm"
    """
    model_type = model_type.lower()
    
    if model_type == "dfm":
        if converged:
            _logger.info(f"✓ EM algorithm converged after {num_iter} iterations")
        else:
            _logger.warning(f"⚠ EM algorithm did not converge after {num_iter} iterations")
    else:  # ddfm
        if converged:
            _logger.info(f"✓ Training converged after {num_epochs} epochs")
        else:
            _logger.warning(f"⚠ Training did not converge after {num_epochs} epochs")
    
    if reason:
        _logger.info(f"  Reason: {reason}")
    
    if final_loglik is not None:
        _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")
    if final_loss is not None:
        _logger.info(f"  Final training loss: {final_loss:.6f}")


class iVDFMTrainLogger:
    """Training logger for iVDFM models (variational inference).

    The old ``BaseTrainLogger`` merged with its iVDFM subclass: ``start``,
    ``log_convergence`` and ``end`` are the base's methods and ``log_epoch``
    is the iVDFM one. The base's EM ``log_iteration`` and DDFM ``log_epoch``
    had no caller and were dropped.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize iVDFM training logger.
        
        Parameters
        ----------
        verbose : bool, default True
            Whether to log detailed information
        """
        self.model_name = "iVDFM"
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.iterations: int = 0
        self.epochs: int = 0
        self.converged: bool = False
        self.metrics_history: List[Dict[str, Any]] = []
        
    def start(
        self, 
        config: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log training start.
        
        Parameters
        ----------
        config : dict, optional
            Training configuration to log (e.g., max_iter, threshold, epochs, learning_rate)
        data_info : dict, optional
            Data information to log (e.g., shape, num_series, num_timepoints)
        """
        self.start_time = datetime.now()
        self.iterations = 0
        self.epochs = 0
        self.converged = False
        self.metrics_history = []
        
        _logger.info(f"{'='*70}")
        _logger.info(f"Starting {self.model_name} training")
        _logger.info(f"{'='*70}")
        
        if config and self.verbose:
            _logger.info("Training configuration:")
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, (list, tuple)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
        
        if data_info and self.verbose:
            _logger.info("Data information:")
            for key, value in data_info.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
        
        _logger.info("")
    
    def log_epoch(
        self,
        epoch: int,
        elbo: float = None,
        recon_loss: float = None,
        kl_loss: float = None,
        learning_rate: float = None,
        **kwargs
    ) -> None:
        """Log training epoch information for iVDFM.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        elbo : float, optional
            ELBO value (negative, to minimize)
        recon_loss : float, optional
            Reconstruction loss component
        kl_loss : float, optional
            KL divergence component
        learning_rate : float, optional
            Current learning rate
        **kwargs
            Additional metrics to log
        """
        self.epochs = epoch
        
        # Store metrics in history
        metrics = {
            "epoch": epoch,
            "elbo": elbo,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "learning_rate": learning_rate
        }
        metrics.update(kwargs)
        self.metrics_history.append(metrics)
        
        if self.verbose:
            msg = f"Epoch {epoch:4d}"
            if elbo is not None:
                msg += f" | ELBO: {elbo:12.6f}"
            if recon_loss is not None:
                msg += f" | Recon: {recon_loss:12.6f}"
            if kl_loss is not None:
                msg += f" | KL: {kl_loss:12.6f}"
            if learning_rate is not None:
                msg += f" | LR: {learning_rate:.6e}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
        else:
            # Less verbose: log every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                msg = f"Epoch {epoch:4d}"
                if elbo is not None:
                    msg += f" | ELBO: {elbo:12.6f}"
                _logger.info(msg)
    
    def log_convergence(
        self,
        converged: bool,
        num_iter: Optional[int] = None,
        num_epochs: Optional[int] = None,
        final_loglik: Optional[float] = None,
        final_loss: Optional[float] = None,
        reason: Optional[str] = None
    ) -> None:
        """Log convergence status.
        
        Parameters
        ----------
        converged : bool
            Whether training converged
        num_iter : int, optional
            Number of EM iterations completed (for DFM)
        num_epochs : int, optional
            Number of epochs completed (for DDFM)
        final_loglik : float, optional
            Final log-likelihood value (for DFM)
        final_loss : float, optional
            Final training loss (for DDFM)
        reason : str, optional
            Reason for stopping (e.g., "converged", "max_iterations", "max_epochs", "early_stopping")
        """
        self.converged = converged
        if num_iter is not None:
            self.iterations = num_iter
        if num_epochs is not None:
            self.epochs = num_epochs
        
        _logger.info("")
        if num_iter is not None:
            if converged:
                _logger.info(f"✓ Training converged after {num_iter} EM iterations")
            else:
                _logger.warning(f"⚠ Training did not converge after {num_iter} EM iterations")
        elif num_epochs is not None:
            if converged:
                _logger.info(f"✓ Training converged after {num_epochs} epochs")
            else:
                _logger.warning(f"⚠ Training did not converge after {num_epochs} epochs")
        
        if reason:
            _logger.info(f"  Reason: {reason}")
        
        if final_loglik is not None:
            _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")
        if final_loss is not None:
            _logger.info(f"  Final training loss: {final_loss:.6f}")
    
    def end(self, success: bool = True, **kwargs) -> None:
        """Log training end.
        
        Parameters
        ----------
        success : bool, default True
            Whether training completed successfully
        **kwargs
            Additional information to log (e.g., metrics, warnings, model info)
        """
        self.end_time = datetime.now()
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            _logger.info("")
            _logger.info(f"{'='*70}")
            if success:
                _logger.info("Training completed successfully")
            else:
                _logger.error("Training failed")
            
            _logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            if self.iterations > 0:
                _logger.info(f"  EM Iterations: {self.iterations}")
            if self.epochs > 0:
                _logger.info(f"  Epochs: {self.epochs}")
            
            _logger.info(f"  Converged: {self.converged}")
            
            # Log summary metrics from history
            if self.metrics_history:
                logliks = [m.get("loglik") for m in self.metrics_history if m.get("loglik") is not None]
                if logliks:
                    _logger.info(f"  Initial log-likelihood: {logliks[0]:.6f}")
                    _logger.info(f"  Final log-likelihood: {logliks[-1]:.6f}")
                    if len(logliks) > 1:
                        improvement = logliks[-1] - logliks[0]
                        _logger.info(f"  Improvement: {improvement:.6f}")
                
                losses = [m.get("train_loss") for m in self.metrics_history if m.get("train_loss") is not None]
                if losses:
                    _logger.info(f"  Initial train loss: {losses[0]:.6f}")
                    _logger.info(f"  Final train loss: {losses[-1]:.6f}")
                    if len(losses) > 1:
                        improvement = losses[0] - losses[-1]  # Loss decreases
                        _logger.info(f"  Improvement: {improvement:.6f}")
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
            
            _logger.info(f"{'='*70}")
            _logger.info("")
