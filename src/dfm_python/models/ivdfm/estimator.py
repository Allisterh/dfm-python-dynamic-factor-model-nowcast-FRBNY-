"""``iVDFM``: the identifiable variational dynamic factor model estimator (sklearn conventions, PanelData input).

The estimator is a thin layer over the private engine
:class:`~dfm_python.models.ivdfm._engine.iVDFMEngine`. ``fit`` builds the
engine's sliding-window dataset from :class:`~dfm_python.PanelData` (targets
from ``y``, the context u_t from ``exog`` plus generated time features),
constructs the engine with every hyperparameter passed explicitly (the engine
seeds, then builds its networks) and trains it. The engine's numbers are
unchanged: same data and same hyperparameters give the same factors,
innovations and forecasts.

``predict(h)`` returns forecasts in data units (the engine's model-space
forecast inverse-transformed with ``scaler_``), and ``predict_factors(h)`` the
forecast factors.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ...base import BaseFactorModel
from ...config.constants import VALID_OPTIMIZERS
from ...config.types import to_numpy
from ...dataset.ivdfm_dataset import iVDFMDataset
from ...dataset.panel import PanelData
from ...logger import get_logger
from ...logger.progress import quiet_engine_loggers
from ...utils.errors import ConfigurationError, DataValidationError
from ._engine import iVDFMEngine

__all__ = ["iVDFM"]

_logger = get_logger(__name__)

#: Scaler names accepted by the dataset, and the class each one builds.
_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
}
_INNOVATION_DISTRIBUTIONS = ("laplace", "gaussian", "student_t", "gamma", "beta", "exponential")
_DECODER_TYPES = ("linear", "residual", "mlp")
_ACTIVATIONS = ("relu", "lrelu", "tanh", "sigmoid")
_SCHEDULERS = ("step", "plateau", "cosine", "exponential", None)
_POSITIVE_INTS = (
    "n_factors", "time_context", "encoder_n_hidden_layers", "decoder_n_hidden_layers", "prior_n_hidden_layers",
    "factor_order", "n_regimes", "batch_size", "max_epochs",
)
_HIDDEN_DIMS = ("encoder_hidden_dim", "decoder_hidden_dim", "prior_hidden_dim")
# Loggers the engine (and the code it calls) writes progress to; raised to WARNING during fit with verbose=0.
_ENGINE_LOGGERS = (
    "dfm_python.models.ivdfm",
    "dfm_python.dataset.ivdfm_dataset",
    "dfm_python.numeric",
    "dfm_python.layer",
    "dfm_python.ssm",
)
# The engine's training reporter, which also logs its end-of-training summary
# ("did not converge after N epochs") as a warning; raised to ERROR during fit
# with verbose=0. converged_ carries the same information.
_PROGRESS_LOGGERS = ("dfm_python.logger",)
# Name of the adapter's time column; a suffix is added if a series or exog column has this name.
_TIME_COLUMN = "__dfm_python_time__"


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


class _RecordingDataset(iVDFMDataset):
    """The engine's dataset, recording the window index of every ``__getitem__`` call.

    On CPU the engine trains and extracts its final factors through a shuffled
    ``DataLoader``, so its factor arrays are in shuffled window order. The last
    ``len(dataset)`` recorded indices are the order of that final extraction.
    Recording reads no random numbers, so the engine's numbers are unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: list[int] = []

    def __getitem__(self, idx: int):
        self.accessed.append(int(idx))
        return super().__getitem__(idx)


class iVDFM(BaseFactorModel):
    """Identifiable variational dynamic factor model.

    Latent factors follow AR(``factor_order``) dynamics driven by innovations
    whose conditionally independent prior given the context u_t makes the
    factors identifiable (iVAE conditions on the innovations). Training
    maximizes the ELBO on sliding windows of the panel. The context u_t is the
    ``exog`` columns of :class:`~dfm_python.PanelData` plus ``time_context``
    generated time features.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int, default 3
        Number of latent factors r.
    window : int, optional
        Length of the training windows. None uses the whole series as one
        window. A window longer than the series is clipped to its length.
    stride : int, optional
        Step between window starts. None means ``window // 2`` (at least 1).
    time_context : int, default 1
        Number of generated time features in u_t: the normalized time step,
        plus ``time_context - 1`` sine features. A ``DatetimeIndex`` (or a
        ``PeriodIndex``, through its start times) gives the time step; a
        ``RangeIndex`` gives the position.
    encoder_hidden_dim, decoder_hidden_dim, prior_hidden_dim : int or list of int
        Hidden widths of the innovation encoder (default 200), the decoder
        (default 200) and the prior network (default 100).
    encoder_n_hidden_layers, decoder_n_hidden_layers, prior_n_hidden_layers : int
        Hidden layers of the same networks (defaults 2, 2 and 1).
    decoder_type : {"mlp", "residual", "linear"}, default "mlp"
    activation : {"lrelu", "relu", "tanh", "sigmoid"}, default "lrelu"
    slope : float, default 0.1
        Leaky ReLU slope.
    factor_order : int, default 1
        AR order p of the factor dynamics.
    innovation_distribution : str, default "laplace"
        One of "laplace", "gaussian", "student_t", "gamma", "beta",
        "exponential".
    decoder_var : float, default 0.01
        Observation noise variance of the decoder.
    n_regimes : int, default 1
        Number of regimes K; K > 1 adds regime-indexed priors, decoders and AR
        coefficients mixed by a regime network on u_t.
    regime_temperature : float, default 1.0
        Softmax temperature of the regime weights (> 0).
    beta_kl : float, default 1.0
        Weight of the KL term in the ELBO (>= 0).
    use_layer_norm : bool, default False
        Layer normalization in the encoder and decoder.
    mixing : bool, default False
        Global post-factor mixing f -> M f (identity initialized) before
        decoding.
    learning_rate : float, default 1e-3
    optimizer : {"Adam", "AdamW", "SGD"}, default "Adam"
    optimizer_weight_decay : float, default 0.0
    optimizer_momentum : float, default 0.9
        Momentum of SGD.
    batch_size : int, default 32
        Windows per batch.
    max_epochs : int, default 100
    tol : float, default 5e-4
        Training stops when the mean ELBO of the last five batches changes by
        less than ``tol``.
    patience : int, optional, default 20
        Early stopping: stop after this many epochs without a lower ELBO and
        restore the best epoch. None disables it.
    scheduler_type : {"step", "plateau", "cosine", "exponential", None}, default "step"
        Learning-rate scheduler; None disables it.
    scheduler_step_size : int, optional
        StepLR step; None means ``max_epochs // 3``.
    scheduler_gamma : float, default 0.5
        StepLR / ExponentialLR factor, in (0, 1].
    scheduler_patience : int, default 10
        ReduceLROnPlateau patience.
    scheduler_factor : float, default 0.1
        ReduceLROnPlateau factor, in (0, 1].
    scheduler_min_lr : float, default 0.0
        ReduceLROnPlateau minimum learning rate.
    scaler : str or sklearn transformer, optional
        Scaler of the targets (not of exog): "standard", "robust", "minmax",
        "maxabs", "quantile", or an unfitted transformer, which is cloned. None
        trains on the data as given. Forecasts are always in data units.
    random_state : int, optional, default 3
        Seed of the global torch and numpy RNGs, applied just before the
        networks are built. None does not seed, so runs differ.
    device : str or torch.device, default "auto"
        "auto" uses CUDA if available, else CPU.
    verbose : int, default 0
        0 is silent; 1 logs the engine's progress (one record per epoch); 2
        also logs debug records.

    Attributes
    ----------
    result_ : iVDFMResult
        Factors, innovations, reconstructions and training diagnostics of the
        engine (factors in the engine's window order; see ``factors_``).
    factors_ : pandas.DataFrame
        Smoothed factors of every training window, float64, with a
        ``(window, time)`` MultiIndex in chronological order and columns
        ``factor_0`` .. ``factor_{r-1}``.
    network_ : torch.nn.Module
        The trained engine.
    device_ : torch.device
        The resolved device.
    scaler_ : sklearn transformer or None
        The target scaler fitted on the training data.
    n_features_in_, feature_names_in_, exog_names_in_, index_
        Number and names of the targets, the exog names and the training index.
    n_iter_ : int
        Training epochs (the best epoch when early stopping restored it).
    converged_ : bool
        True when training stopped before ``max_epochs`` (tolerance or
        patience).
    n_factors_ : int
        Number of factors.
    random_state_ : int or None
        The seed the run used.
    window_, stride_ : int
        The window length and stride used for training.
    time_context_ : int
        Number of generated time features in u_t.
    context_dim_ : int
        Width of u_t: ``time_context_`` plus the number of exog columns.
    conditioning_index_ : pandas.Index
        Time index of the data the forecasts start from; ``predict(h)``
        returns the ``h`` stamps after its last entry.

    Examples
    --------
    >>> from dfm_python import iVDFM, PanelData
    >>> data = PanelData(y=targets_df, exog=context_df)  # doctest: +SKIP
    >>> model = iVDFM(n_factors=5, factor_order=3, window=200, max_epochs=10, scaler="standard").fit(data)  # doctest: +SKIP
    >>> model.predict(6, futr_exog=future_context_df)  # doctest: +SKIP
    """

    _tags = {
        "task": "forecast",
        "backend": "torch",
        "capability:exog": True,
        "capability:missing_values": True,
        "capability:update": True,
        "capability:factor_forecast": True,
        "capability:device": True,
        "update_kind": "encoder_forward",
        "predict_data": "window",
        # The class default (random_state=3); an instance reports random_state is None.
        "non_deterministic": False,
    }

    def __init__(
        self,
        *,
        n_factors: int = 3,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        time_context: int = 1,
        encoder_hidden_dim: Any = 200,
        encoder_n_hidden_layers: int = 2,
        decoder_type: str = "mlp",
        decoder_hidden_dim: Any = 200,
        decoder_n_hidden_layers: int = 2,
        prior_hidden_dim: Any = 100,
        prior_n_hidden_layers: int = 1,
        activation: str = "lrelu",
        slope: float = 0.1,
        factor_order: int = 1,
        innovation_distribution: str = "laplace",
        decoder_var: float = 0.01,
        n_regimes: int = 1,
        regime_temperature: float = 1.0,
        beta_kl: float = 1.0,
        use_layer_norm: bool = False,
        mixing: bool = False,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_weight_decay: float = 0.0,
        optimizer_momentum: float = 0.9,
        batch_size: int = 32,
        max_epochs: int = 100,
        tol: float = 5e-4,
        patience: Optional[int] = 20,
        scheduler_type: Optional[str] = "step",
        scheduler_step_size: Optional[int] = None,
        scheduler_gamma: float = 0.5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.1,
        scheduler_min_lr: float = 0.0,
        scaler: Any = None,
        random_state: Optional[int] = 3,
        device: Any = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.window = window
        self.stride = stride
        self.time_context = time_context
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_hidden_layers = encoder_n_hidden_layers
        self.decoder_type = decoder_type
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_hidden_layers = decoder_n_hidden_layers
        self.prior_hidden_dim = prior_hidden_dim
        self.prior_n_hidden_layers = prior_n_hidden_layers
        self.activation = activation
        self.slope = slope
        self.factor_order = factor_order
        self.innovation_distribution = innovation_distribution
        self.decoder_var = decoder_var
        self.n_regimes = n_regimes
        self.regime_temperature = regime_temperature
        self.beta_kl = beta_kl
        self.use_layer_norm = use_layer_norm
        self.mixing = mixing
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_momentum = optimizer_momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.scheduler_type = scheduler_type
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.scaler = scaler
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _instance_tags(self) -> dict:
        return {"non_deterministic": self.random_state is None}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small CPU parameter sets: one regime with a scaler, and two regimes with AR(2) factors."""
        tiny = {
            "n_factors": 2, "window": 8, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
            "batch_size": 4, "max_epochs": 2, "random_state": 0, "device": "cpu",
        }
        return [
            {**tiny, "scaler": "standard"},
            {**tiny, "n_regimes": 2, "factor_order": 2, "time_context": 2},
        ]

    # -- validation ----------------------------------------------------------

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` naming the first invalid hyperparameter.

        The rules are those of the old ``iVDFMConfig`` validation, under the
        canonical names, plus type checks.
        """
        name = type(self).__name__

        def fail(message: str) -> None:
            raise ConfigurationError(f"{name}: {message}")

        for param in _POSITIVE_INTS:
            value = getattr(self, param)
            if not _is_int(value) or value < 1:
                fail(f"{param} must be a positive integer, got {value!r}.")
        for param in ("window", "stride", "patience", "scheduler_step_size"):
            value = getattr(self, param)
            if value is not None and (not _is_int(value) or value < 1):
                fail(f"{param} must be None or a positive integer, got {value!r}.")
        for param in _HIDDEN_DIMS:
            value = getattr(self, param)
            widths = list(value) if isinstance(value, (list, tuple)) else [value]
            if not widths or not all(_is_int(w) and w >= 1 for w in widths):
                fail(f"{param} must be a positive integer or a non-empty list of them, got {value!r}.")
        if self.decoder_type not in _DECODER_TYPES:
            fail(f"decoder_type must be one of {_DECODER_TYPES}, got {self.decoder_type!r}.")
        if self.activation not in _ACTIVATIONS:
            fail(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}.")
        if self.innovation_distribution not in _INNOVATION_DISTRIBUTIONS:
            fail(f"innovation_distribution must be one of {_INNOVATION_DISTRIBUTIONS}, got {self.innovation_distribution!r}.")
        if self.optimizer not in VALID_OPTIMIZERS:
            fail(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got {self.optimizer!r}.")
        if self.scheduler_type not in _SCHEDULERS:
            fail(f"scheduler_type must be one of {_SCHEDULERS}, got {self.scheduler_type!r}.")
        positive = ("slope", "decoder_var", "regime_temperature", "learning_rate")
        for param in positive:
            value = getattr(self, param)
            if param == "slope":
                ok = _is_real(value) and value >= 0
            else:
                ok = _is_real(value) and value > 0
            if not ok:
                bound = ">= 0" if param == "slope" else "> 0"
                fail(f"{param} must be a finite number {bound}, got {value!r}.")
        for param in ("beta_kl", "optimizer_weight_decay", "optimizer_momentum", "tol", "scheduler_min_lr"):
            value = getattr(self, param)
            if not _is_real(value) or value < 0:
                fail(f"{param} must be a finite number >= 0, got {value!r}.")
        for param in ("scheduler_gamma", "scheduler_factor"):
            value = getattr(self, param)
            if not _is_real(value) or not 0 < value <= 1:
                fail(f"{param} must be in (0, 1], got {value!r}.")
        if not _is_int(self.scheduler_patience) or self.scheduler_patience < 0:
            fail(f"scheduler_patience must be an integer >= 0, got {self.scheduler_patience!r}.")
        for param in ("use_layer_norm", "mixing"):
            if not isinstance(getattr(self, param), (bool, np.bool_)):
                fail(f"{param} must be True or False, got {getattr(self, param)!r}.")
        self._scaler_name()
        if self.random_state is not None and not _is_int(self.random_state):
            fail(f"random_state must be None or an integer, got {self.random_state!r}.")
        if not _is_int(self.verbose) or self.verbose < 0:
            fail(f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}.")
        self._resolve_device()

    def _scaler_name(self) -> Optional[str]:
        """The scaler as the dataset's string name (``scaler`` field of the engine config), validating it."""
        scaler = self.scaler
        if scaler is None:
            return None
        if isinstance(scaler, str):
            if scaler.lower() not in _SCALERS:
                raise ConfigurationError(
                    f"{type(self).__name__}: scaler must be None, one of {sorted(_SCALERS)} or a transformer "
                    f"instance, got {scaler!r}."
                )
            return scaler
        if not (hasattr(scaler, "fit_transform") and hasattr(scaler, "inverse_transform")):
            raise ConfigurationError(
                f"{type(self).__name__}: scaler must have fit_transform and inverse_transform, got "
                f"{type(scaler).__name__}."
            )
        for key, cls in _SCALERS.items():
            if type(scaler) is cls:
                return key
        return type(scaler).__name__

    def _resolve_device(self) -> torch.device:
        """``device`` as a ``torch.device``; ``"auto"`` means CUDA if available, else CPU."""
        device = self.device
        if isinstance(device, str) and device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return torch.device(device)
        except (RuntimeError, TypeError, ValueError) as err:
            raise ConfigurationError(
                f"{type(self).__name__}: device must be 'auto', a device string such as 'cpu' or 'cuda:0', "
                f"or a torch.device; got {device!r} ({err})."
            ) from err

    # -- adapter and engine --------------------------------------------------

    def _fresh_scaler(self) -> Any:
        """The scaler handed to the dataset: the name, or a clone of the instance (warns if it was fitted)."""
        scaler = self.scaler
        if scaler is None or isinstance(scaler, str):
            return scaler
        try:
            check_is_fitted(scaler)
        except (SklearnNotFittedError, TypeError):
            pass
        else:
            warnings.warn(
                f"{type(self).__name__}: the scaler {type(scaler).__name__} passed in is already fitted; fit uses an "
                "unfitted clone and refits it on the training targets, so its statistics are discarded. The fitted "
                "copy is scaler_.",
                UserWarning,
                stacklevel=6,  # _fresh_scaler < _adapter < _fit < BaseFactorModel.fit < iVDFM.fit < caller
            )
        return clone(scaler)

    def _adapter(self, panel: PanelData, device: torch.device) -> _RecordingDataset:
        """The engine's dataset built from ``panel``: a frame ``[time | y | exog]`` on a RangeIndex.

        The time column holds the ``DatetimeIndex`` (a ``PeriodIndex`` through
        ``to_timestamp()``); a ``RangeIndex`` gives no time column, so the time
        features use the position. The dataset interpolates missing values,
        scales the targets and cuts the windows exactly as before.
        """
        frame = panel.y.reset_index(drop=True)
        if panel.exog is not None:
            frame = pd.concat([frame, panel.exog.reset_index(drop=True)], axis=1)
        index = panel.index
        time_col = None
        if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            stamps = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
            if stamps.tz is not None:
                stamps = stamps.tz_convert("UTC").tz_localize(None)
            # The dataset reads the stamps as int64 nanoseconds, whatever resolution the index has.
            stamps = stamps.as_unit("ns")
            time_col = _TIME_COLUMN
            while time_col in set(frame.columns):
                time_col += "_"
            frame.insert(0, time_col, np.asarray(stamps))
        return _RecordingDataset(
            data=frame,
            window=self.window,
            stride=self.stride,
            time_idx=time_col,
            context=list(panel.exog_names),
            time_context=self.time_context,
            scaler=self._fresh_scaler(),
            device=device,
        )

    def _engine_params(self, dataset: iVDFMDataset) -> dict:
        """Every engine hyperparameter, passed explicitly under the engine's names."""

        def widths(value: Any) -> Any:
            return list(value) if isinstance(value, tuple) else value

        return {
            "num_factors": self.n_factors,
            "window": dataset.window,
            "encoder_hidden_dim": widths(self.encoder_hidden_dim),
            "encoder_n_hidden_layers": self.encoder_n_hidden_layers,
            "decoder_type": self.decoder_type,
            "decoder_hidden_dim": widths(self.decoder_hidden_dim),
            "decoder_n_hidden_layers": self.decoder_n_hidden_layers,
            "prior_hidden_dim": widths(self.prior_hidden_dim),
            "prior_n_hidden_layers": self.prior_n_hidden_layers,
            "activation": self.activation,
            "slope": self.slope,
            "factor_order": self.factor_order,
            "innovation_distribution": self.innovation_distribution,
            "decoder_var": self.decoder_var,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "tolerance": self.tol,
            "seed": self.random_state,
            # The rest reach the engine's config through its **kwargs.
            "stride": dataset.stride,
            "time_context": self.time_context,
            "scaler": self._scaler_name(),
            "num_regimes": self.n_regimes,
            "regime_temperature": self.regime_temperature,
            "beta_kl": self.beta_kl,
            "use_layer_norm": self.use_layer_norm,
            "mixing": self.mixing,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_momentum": self.optimizer_momentum,
            "patience": self.patience,
            "scheduler_type": self.scheduler_type,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
        }

    @staticmethod
    def _restore_none_params(engine: iVDFMEngine, params: dict) -> None:
        """Give the engine the None values its constructor dropped.

        The engine drops None values before building its config, so
        ``patience=None`` would become 20 and ``scheduler_type=None`` would
        become ``"step"``. The engine reads ``patience``, ``scheduler_type``
        and ``scheduler_step_size`` from its attributes during ``fit``.
        """
        fields = {f.name for f in dataclasses.fields(engine._config)}
        dropped = {k: v for k, v in params.items() if v is None and k in fields}
        if dropped:
            engine._config = dataclasses.replace(engine._config, **dropped)
        for attr in ("patience", "scheduler_type", "scheduler_step_size"):
            setattr(engine, attr, params[attr])

    # -- fit -----------------------------------------------------------------

    def fit(self, data: Any, **fit_kwargs: Any) -> iVDFM:
        """Train on ``data`` and return ``self``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            Targets in ``y`` and, optionally, context variables in ``exog``.
            Missing values are interpolated linearly, then filled forward,
            backward and with zeros (in targets and exog). Mixed frequency and
            ``series_exog`` are rejected.

        Raises
        ------
        ConfigurationError
            For an invalid hyperparameter.
        UnsupportedInputError
            For ``series_exog`` or a mixed-frequency panel.

        Warns
        -----
        UserWarning
            When ``scaler`` is an already fitted transformer (it is cloned and
            refitted).

        A failed fit leaves the model unfitted (earlier fitted attributes are
        removed).
        """
        return super().fit(data, **fit_kwargs)

    def _fit(self, data: PanelData, **fit_kwargs: Any) -> None:
        if fit_kwargs:
            raise TypeError(f"iVDFM.fit() got unexpected keyword arguments {sorted(fit_kwargs)}.")
        self._validate_params()
        device = self._resolve_device()
        dataset = self._adapter(data, device)
        params = self._engine_params(dataset)
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _PROGRESS_LOGGERS, logging.ERROR):
            # The engine seeds the global torch and numpy RNGs (random_state) and then builds its networks.
            engine = iVDFMEngine(dataset=dataset, device=device, **params)
            self._restore_none_params(engine, params)
            if self.verbose >= 2:
                _logger.debug("iVDFM: engine hyperparameters %s", params)
            engine.fit()
        engine._dataset = None  # the engine keeps no reference to the training data

        self.network_ = engine
        self.device_ = device
        self.scaler_ = dataset.scaler
        self.result_ = engine.get_result()
        self.n_iter_ = int(engine._num_iter)
        self.converged_ = bool(engine._converged)
        self.n_factors_ = int(engine.latent_dim)
        self.random_state_ = self.random_state
        self.window_ = int(dataset.window)
        self.stride_ = int(dataset.stride)
        self.time_context_ = int(self.time_context)
        self.context_dim_ = int(dataset.context_dim)
        self.conditioning_index_ = data.index
        self.factors_ = self._factors_frame(engine.factors, dataset.accessed, data.index)

    def _factors_frame(self, factors: np.ndarray, accessed: list, index: pd.Index) -> pd.DataFrame:
        """``factors_``: the engine's window factors in chronological order on a ``(window, time)`` MultiIndex.

        The engine's last ``len(windows)`` dataset accesses are its final
        factor extraction, so window ``order[k]`` is row ``k`` of the engine's
        factors. Without recorded accesses (the single-batch GPU path) the
        windows are in natural order.
        """
        n_windows, window, r = factors.shape
        order = np.asarray(accessed[-n_windows:] if len(accessed) >= n_windows else range(n_windows))
        chronological = np.asarray(factors)[np.argsort(order, kind="stable")]
        starts = np.arange(n_windows) * self.stride_
        positions = (starts[:, None] + np.arange(window)[None, :]).reshape(-1)
        multi = pd.MultiIndex.from_arrays(
            [np.repeat(np.arange(n_windows), window), index[positions]],
            names=["window", index.name if index.name is not None else "time"],
        )
        return pd.DataFrame(
            chronological.reshape(n_windows * window, r).astype(np.float64),
            index=multi,
            columns=[f"factor_{i}" for i in range(r)],
        )

    # -- forecasting ---------------------------------------------------------

    def _context_data(self, h: int, futr_exog: Any) -> Optional[np.ndarray]:
        """The engine's ``context_data`` for ``h`` steps: ``[time features | futr_exog]``, or None."""
        exog_names = list(self.exog_names_in_)
        if futr_exog is None:
            if exog_names:
                warnings.warn(
                    f"iVDFM was fitted with exog {exog_names} but predict got no futr_exog; the forecast context "
                    "is generated time features in every column, as the engine does without context_data. Pass "
                    "futr_exog (h rows, one column per exog name) for the context the model was trained on.",
                    UserWarning,
                    stacklevel=4,
                )
            return None
        if not exog_names:
            raise DataValidationError(
                "iVDFM was fitted without exog, so predict takes no futr_exog; drop the argument."
            )
        if isinstance(futr_exog, pd.DataFrame):
            missing = [c for c in exog_names if c not in futr_exog.columns]
            if missing:
                raise DataValidationError(
                    f"futr_exog lacks the exog columns {missing}; the model was fitted on {exog_names}."
                )
            values = futr_exog[exog_names].to_numpy(dtype=np.float64)
        else:
            values = np.asarray(futr_exog, dtype=np.float64)
        if values.shape != (h, len(exog_names)):
            raise DataValidationError(
                f"futr_exog must have shape (h, n_exog) = ({h}, {len(exog_names)}), got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise DataValidationError("futr_exog contains NaN or infinite values.")
        time_features = self.network_._future_time_features(h, self.time_context_)
        return np.hstack([time_features, values])

    def _run_forecast(
        self, method: str, h: Any, data: Any, futr_exog: Any, deterministic: Any, n_samples: Any
    ) -> tuple[np.ndarray, np.ndarray, int]:
        self._check_is_fitted(method)
        h = self._validate_horizon(h)
        if data is not None:
            self._validate_data(data)  # inputs the tags exclude raise UnsupportedInputError first
            raise NotImplementedError(f"iVDFM.{method}(data=...) is not implemented yet; call {method}(h).")
        if not isinstance(deterministic, (bool, np.bool_)):
            raise TypeError(f"deterministic must be True or False, got {deterministic!r}.")
        if not _is_int(n_samples) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        context_data = self._context_data(h, futr_exog)
        factors, y_pred = self.network_._forecast(h, context_data, bool(deterministic), int(n_samples))
        return to_numpy(factors.squeeze(0)), to_numpy(y_pred.squeeze(0)), h

    def _frame(self, values: np.ndarray, h: int, columns: list, kind: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(values, dtype=np.float64),
            index=self._forecast_index(self.conditioning_index_, h),
            columns=columns,
        )
        frame.attrs = {"model": type(self).__name__, "kind": kind, "origin": self.conditioning_index_[-1]}
        return frame

    def predict(
        self,
        h: int,
        data: Any = None,
        *,
        futr_exog: Any = None,
        deterministic: bool = True,
        n_samples: int = 1,
    ) -> pd.DataFrame:
        """Forecast the targets ``h`` steps ahead, in data units.

        The engine's closed-loop forecast starts from the last training factor
        state; the decoded observations are inverse-transformed with
        ``scaler_``.

        Parameters
        ----------
        h : int
            Forecast horizon (a positive integer; anything else raises
            ``TypeError``).
        data : None
            Conditioning on new data is not implemented yet.
        futr_exog : DataFrame or array of shape (h, n_exog), optional
            Future values of the exog columns (a DataFrame is matched by column
            name). They follow the generated time features in the context u_t.
            None generates time features for every context column, as the
            engine does without context; that warns for a model with exog.
        deterministic : bool, default True
            True uses zero innovations; False samples them from the prior.
        n_samples : int, default 1
            Prior samples averaged when ``deterministic=False``.

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows after the last conditioning time stamp, one
            column per target in the user's order; ``attrs`` holds ``model``,
            ``kind="forecast"`` and ``origin``.
        """
        _, y_pred, h = self._run_forecast("predict", h, data, futr_exog, deterministic, n_samples)
        values = np.asarray(y_pred, dtype=np.float64)
        if self.scaler_ is not None:
            values = self.scaler_.inverse_transform(values)
        return self._frame(values, h, list(self.feature_names_in_), "forecast")

    def predict_factors(
        self,
        h: int,
        data: Any = None,
        *,
        futr_exog: Any = None,
        deterministic: bool = True,
        n_samples: int = 1,
    ) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (same arguments as :meth:`predict`).

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows, columns ``factor_0`` .. ``factor_{r-1}``;
            ``attrs["kind"]`` is ``"factors"``.
        """
        factors, _, h = self._run_forecast("predict_factors", h, data, futr_exog, deterministic, n_samples)
        return self._frame(factors, h, [f"factor_{i}" for i in range(factors.shape[1])], "factors")

    def update(self, data: Any, *, refit: bool = False) -> iVDFM:
        """Not implemented yet; refit with ``fit(data)``."""
        self._check_is_fitted("update")
        raise NotImplementedError("iVDFM.update is not implemented yet; call fit(data) to refit on the new data.")
