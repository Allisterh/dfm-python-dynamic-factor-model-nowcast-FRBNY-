"""``DDFM``: the deep dynamic factor model estimator (sklearn conventions, PanelData input).

The estimator is a thin layer over the private engine
:class:`~dfm_python.models.ddfm._engine.DDFMEngine`. ``fit`` builds the
engine's dataset from :class:`~dfm_python.PanelData` as the frame
``[exog | y]`` (the covariates of the encoder, then the targets), constructs
the engine with every hyperparameter passed explicitly, trains the autoencoder
with MCMC denoising and builds the linear state space on its factors. The
engine's numbers are unchanged: same data and same hyperparameters give the
same weights, factors, state-space matrices and forecasts.

``predict(h)`` returns the target forecasts in data units (the engine's
forecast, inverse-transformed with ``scaler_``; the identity with
``scaler=None``), and ``predict_factors(h)`` the AR(1) factor forecast.
"""

from __future__ import annotations

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
from ...dataset.ddfm_dataset import DDFMDataset
from ...dataset.panel import PanelData
from ...logger import get_logger
from ...logger.progress import quiet_engine_loggers
from ...utils.errors import ConfigurationError
from ._engine import DDFMEngine

__all__ = ["DDFM"]

_logger = get_logger(__name__)

#: Scaler names accepted by ``scaler``, and the class each one builds.
_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
}
_DECODER_TYPES = ("linear", "mlp")
_ACTIVATIONS = ("relu", "tanh", "sigmoid")
_LIMIT_DIRECTIONS = ("forward", "backward", "both")
_POSITIVE_INTS = ("n_factors", "n_mc_samples", "window_size", "max_iter")
# The engine's progress interval (disp): it logs a summary every 10 MCMC iterations, at INFO.
_ENGINE_DISP = 10
# Loggers the engine (and the code it calls) writes progress to; raised to WARNING during fit with verbose=0.
_ENGINE_LOGGERS = (
    "dfm_python.models.ddfm",
    "dfm_python.dataset.ddfm_dataset",
    "dfm_python.numeric",
    "dfm_python.utils",
)
# The MCMC loop's variance-collapse diagnostics, logged as warnings on every
# iteration they apply to; raised to ERROR during fit with verbose=0.
_DIAGNOSTIC_LOGGERS = ("dfm_python.models.ddfm.sampling",)
# Name of the adapter's time column; it is never in the frame (a suffix is added
# if a series or exog column has this name), so the dataset keeps the frame's RangeIndex.
_TIME_COLUMN = "__dfm_python_time__"
# Engine attributes that only training reads (the MCMC state and the training
# frames); fit drops them. _update and _forecast read none of them.
_TRAINING_BUFFERS = (
    "_last_iter_datasets",
    "data_imputed",
    "data_denoised",
    "data_denoised_interpolated",
    "y_actual",
    "rng",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


class DDFM(BaseFactorModel):
    """Deep dynamic factor model.

    A nonlinear encoder maps the covariates and targets at each time step to
    ``n_factors`` latent factors, and a linear (or MLP) decoder reconstructs
    the targets from them. The autoencoder is pre-trained, then trained with
    MCMC denoising: each iteration draws ``n_mc_samples`` corrupted copies of
    the targets from the idiosyncratic residuals, trains on them and imputes
    missing targets. After training, an AR(1) transition and the decoder's
    last linear layer form a linear state space used for forecasting.

    The covariates are the ``exog`` columns of :class:`~dfm_python.PanelData`:
    they enter the encoder but are not reconstructed or forecast.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int, default 4
        Number of latent factors r (the width of the encoder's output layer).
    encoder_layers : tuple of int, default (16,)
        Widths of the encoder's hidden layers. The engine's encoder is
        ``(*encoder_layers, n_factors)``. May be empty with the linear decoder;
        the MLP decoder mirrors the hidden layers, so it needs at least one.
    decoder_type : {"linear", "mlp"}, default "linear"
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
    learning_rate : float, default 0.005
    optimizer : {"Adam", "AdamW", "SGD"}, default "Adam"
    n_mc_samples : int, default 10
        Monte Carlo samples (corrupted copies of the targets) per MCMC
        iteration; pre-training also runs this many epochs.
    window_size : int, default 100
        Time steps per training batch.
    max_iter : int, default 200
        Maximum MCMC iterations.
    tol : float, default 5e-4
        Training stops when the relative change of the reconstruction MSE
        between iterations falls below ``tol``.
    interpolation_method : str, default "linear"
        ``pandas.DataFrame.interpolate`` method for missing values (the
        frame has a RangeIndex, so index-based methods use positions).
    interpolation_limit : int, optional, default 10
        Maximum number of consecutive missing values filled; None fills all.
    interpolation_limit_direction : {"forward", "backward", "both"}, default "both"
    scaler : str or sklearn transformer, optional
        Scaler of the targets: "standard", "robust", "minmax", "maxabs",
        "quantile", or an unfitted transformer, which is cloned. The
        covariates are scaled with a fresh instance of the same class. None
        trains on the data as given. Forecasts are always in data units.
    random_state : int, optional, default 3
        Seed of the autoencoder's initialization (global torch RNG) and of the
        local generator of the MCMC noise. None does not seed, so runs differ.
    device : str or torch.device, default "auto"
        "auto" uses CUDA if available, else CPU.
    verbose : int, default 0
        0 is silent; 1 logs the engine's progress; 2 also logs debug records.

    Attributes
    ----------
    result_ : DDFMResult
        Factors ``Z`` (T x r, averaged over the MC samples), the
        reconstruction ``x_sm`` (T x N, model space) and the state space:
        ``A`` (the AR(1) transition F), ``C`` (the loadings H, N x r, the DFM
        orientation), ``Q``, ``R``, ``Z_0``, ``V_0``.
    factors_ : pandas.DataFrame
        ``result_.Z`` as float64 on ``index_``, columns ``factor_0`` ..
        ``factor_{r-1}``.
    network_ : torch.nn.Module
        The trained engine.
    device_ : torch.device
        The resolved device.
    scaler_ : sklearn transformer or None
        The target scaler fitted on the training data.
    n_features_in_, feature_names_in_, exog_names_in_, index_
        Number and names of the targets, the exog names and the training index.
    n_iter_ : int
        MCMC iterations run.
    converged_ : bool
        True when training stopped on ``tol`` before ``max_iter``.
    n_factors_ : int
        Number of factors.
    random_state_ : int or None
        The seed the run used.
    conditioning_index_ : pandas.Index
        Time index of the data the forecasts start from; ``predict(h)``
        returns the ``h`` stamps after its last entry.

    Examples
    --------
    >>> from dfm_python import DDFM, PanelData
    >>> data = PanelData(y=targets_df, exog=covariates_df)  # doctest: +SKIP
    >>> model = DDFM(n_factors=4, encoder_layers=(16,), max_iter=200, scaler="standard").fit(data)  # doctest: +SKIP
    >>> model.predict(6)  # doctest: +SKIP
    >>> model.predict_factors(6)  # doctest: +SKIP
    """

    _tags = {
        "task": "forecast",
        "backend": "torch",
        "capability:exog": True,
        "capability:missing_values": True,
        "capability:update": True,
        "capability:factor_forecast": True,
        "capability:device": True,
        "update_kind": "encoder_forward_rescale",
        "predict_data": "full_panel",
        # The class default (random_state=3); an instance reports random_state is None.
        "non_deterministic": False,
    }

    def __init__(
        self,
        *,
        n_factors: int = 4,
        encoder_layers: Any = (16,),
        decoder_type: str = "linear",
        activation: str = "relu",
        learning_rate: float = 0.005,
        optimizer: str = "Adam",
        n_mc_samples: int = 10,
        window_size: int = 100,
        max_iter: int = 200,
        tol: float = 5e-4,
        interpolation_method: str = "linear",
        interpolation_limit: Optional[int] = 10,
        interpolation_limit_direction: str = "both",
        scaler: Any = None,
        random_state: Optional[int] = 3,
        device: Any = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.encoder_layers = encoder_layers
        self.decoder_type = decoder_type
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_iter = max_iter
        self.tol = tol
        self.interpolation_method = interpolation_method
        self.interpolation_limit = interpolation_limit
        self.interpolation_limit_direction = interpolation_limit_direction
        self.scaler = scaler
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _instance_tags(self) -> dict:
        return {"non_deterministic": self.random_state is None}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small CPU parameter sets: a linear decoder, and an MLP decoder with a scaler."""
        tiny = {
            "n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 2,
            "random_state": 0, "device": "cpu",
        }
        return [tiny, {**tiny, "decoder_type": "mlp", "scaler": "standard"}]

    # -- validation ----------------------------------------------------------

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` naming the first invalid hyperparameter."""
        name = type(self).__name__

        def fail(message: str) -> None:
            raise ConfigurationError(f"{name}: {message}")

        for param in _POSITIVE_INTS:
            value = getattr(self, param)
            if not _is_int(value) or value < 1:
                fail(f"{param} must be a positive integer, got {value!r}.")
        if self.decoder_type not in _DECODER_TYPES:
            fail(f"decoder_type must be one of {_DECODER_TYPES}, got {self.decoder_type!r}.")
        layers = self.encoder_layers
        if not isinstance(layers, (list, tuple)) or not all(_is_int(w) and w >= 1 for w in layers):
            fail(f"encoder_layers must be a list or tuple of positive integers, got {layers!r}.")
        if self.decoder_type == "mlp" and len(layers) == 0:
            fail(
                "encoder_layers must not be empty with decoder_type='mlp': the MLP decoder mirrors the encoder's "
                "hidden layers. Give at least one hidden width, or use decoder_type='linear'."
            )
        if self.activation not in _ACTIVATIONS:
            fail(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}.")
        if self.optimizer not in VALID_OPTIMIZERS:
            fail(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got {self.optimizer!r}.")
        if not _is_real(self.learning_rate) or self.learning_rate <= 0:
            fail(f"learning_rate must be a finite number > 0, got {self.learning_rate!r}.")
        if not _is_real(self.tol) or self.tol < 0:
            fail(f"tol must be a finite number >= 0, got {self.tol!r}.")
        if not isinstance(self.interpolation_method, str) or not self.interpolation_method:
            fail(f"interpolation_method must be a non-empty string, got {self.interpolation_method!r}.")
        limit = self.interpolation_limit
        if limit is not None and (not _is_int(limit) or limit < 1):
            fail(f"interpolation_limit must be None or a positive integer, got {limit!r}.")
        if self.interpolation_limit_direction not in _LIMIT_DIRECTIONS:
            fail(
                f"interpolation_limit_direction must be one of {_LIMIT_DIRECTIONS}, got "
                f"{self.interpolation_limit_direction!r}."
            )
        self._check_scaler()
        if self.random_state is not None and not _is_int(self.random_state):
            fail(f"random_state must be None or an integer, got {self.random_state!r}.")
        if not _is_int(self.verbose) or self.verbose < 0:
            fail(f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}.")
        self._resolve_device()

    def _check_scaler(self) -> None:
        scaler = self.scaler
        if scaler is None:
            return
        if isinstance(scaler, str):
            if scaler.lower() not in _SCALERS:
                raise ConfigurationError(
                    f"{type(self).__name__}: scaler must be None, one of {sorted(_SCALERS)} or a transformer "
                    f"instance, got {scaler!r}."
                )
            return
        if not (hasattr(scaler, "fit_transform") and hasattr(scaler, "inverse_transform")):
            raise ConfigurationError(
                f"{type(self).__name__}: scaler must have fit_transform and inverse_transform, got "
                f"{type(scaler).__name__}."
            )

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
        """The scaler handed to the dataset: a new instance for a name, or a clone of the instance (warns if fitted)."""
        scaler = self.scaler
        if scaler is None:
            return None
        if isinstance(scaler, str):
            return _SCALERS[scaler.lower()]()
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
                stacklevel=6,  # _fresh_scaler < _adapter < _fit < BaseFactorModel.fit < DDFM.fit < caller
            )
        return clone(scaler)

    def _adapter(self, panel: PanelData, *, fitted: bool = False) -> DDFMDataset:
        """The engine's dataset built from ``panel``: the frame ``[exog | y]`` on a RangeIndex.

        The exog columns are the dataset's covariates and the ``y`` columns its
        targets, in that order (the column order of the engine's input). The
        time column name is absent from the frame, so the dataset keeps the
        RangeIndex. The dataset fits the target scaler and scales the
        covariates with a fresh instance of the same class.

        With ``fitted=True`` (the data of ``update`` and ``predict(data=...)``)
        the scaler is an unfitted clone of ``scaler_`` (None without one).
        """
        y = panel.y.reset_index(drop=True)
        if panel.exog is None:
            frame = y
        else:
            frame = pd.concat([panel.exog.reset_index(drop=True), y], axis=1)
        time_col = _TIME_COLUMN
        while time_col in set(frame.columns):
            time_col += "_"
        if fitted:
            scaler = None if self.scaler_ is None else clone(self.scaler_)
        else:
            scaler = self._fresh_scaler()
        return DDFMDataset(data=frame, time_idx=time_col, covariates=list(panel.exog_names), scaler=scaler)

    def _engine_params(self) -> dict:
        """Every engine hyperparameter, passed explicitly under the engine's names."""
        return {
            "encoder_size": tuple(int(w) for w in self.encoder_layers) + (int(self.n_factors),),
            "decoder_type": self.decoder_type,
            "seed": self.random_state,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "n_mc_samples": self.n_mc_samples,
            "window_size": self.window_size,
            "max_iter": self.max_iter,
            "tolerance": self.tol,
            "disp": _ENGINE_DISP,
            "interpolation_method": self.interpolation_method,
            "interpolation_limit": self.interpolation_limit,
            "interpolation_limit_direction": self.interpolation_limit_direction,
        }

    # -- fit -----------------------------------------------------------------

    def fit(self, data: Any, **fit_kwargs: Any) -> DDFM:
        """Train on ``data`` and return ``self``.

        Pre-trains the autoencoder, runs the MCMC denoising loop and builds
        the linear state space (AR(1) factors, decoder loadings) used by
        ``predict``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            Targets in ``y`` and, optionally, encoder covariates in ``exog``.
            Missing values are allowed (interpolated for the encoder, and
            missing targets imputed during training). Mixed frequency and
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
            raise TypeError(f"DDFM.fit() got unexpected keyword arguments {sorted(fit_kwargs)}.")
        self._validate_params()
        device = self._resolve_device()
        dataset = self._adapter(data)
        params = self._engine_params()
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _DIAGNOSTIC_LOGGERS, logging.ERROR):
            if self.verbose >= 2:
                _logger.debug("DDFM: engine hyperparameters %s", params)
            engine = DDFMEngine(dataset=dataset, config=None, device=device, **params)
            engine.fit()
            engine.build_state_space()
            result = engine.get_result()
        # The engine keeps no reference to the training data and drops the MCMC
        # buffers; it keeps its scaled training frame (engine.data), which update reads.
        engine._dataset = None
        for name in _TRAINING_BUFFERS:
            engine.__dict__.pop(name, None)

        self.network_ = engine
        self.device_ = device
        self.scaler_ = engine.scaler
        self.result_ = result
        self.n_iter_ = int(engine._num_iter)
        self.converged_ = bool(getattr(engine, "_converged", False))
        self.n_factors_ = int(self.n_factors)
        self.random_state_ = self.random_state
        self.conditioning_index_ = data.index
        self.factors_ = pd.DataFrame(
            np.asarray(result.Z, dtype=np.float64),
            index=data.index,
            columns=[f"factor_{i}" for i in range(self.n_factors_)],
        )

    # -- forecasting ---------------------------------------------------------

    def _run_forecast(self, method: str, h: Any, data: Any) -> tuple[np.ndarray, np.ndarray, int, pd.Index]:
        self._check_is_fitted(method)
        h = self._validate_horizon(h)
        if data is not None:
            raise NotImplementedError(
                f"DDFM.{method}(h, data=...) is not implemented yet; call {method}(h) for the forecast from the "
                "training data."
            )
        y_forecast, z_forecast = self.network_._forecast(h)
        return y_forecast, z_forecast, h, self.conditioning_index_

    def _frame(self, values: np.ndarray, h: int, columns: list, kind: str, origin_index: pd.Index) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(values, dtype=np.float64),
            index=self._forecast_index(origin_index, h),
            columns=columns,
        )
        frame.attrs = {"model": type(self).__name__, "kind": kind, "origin": origin_index[-1]}
        return frame

    def predict(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the targets ``h`` steps ahead, in data units.

        The factors follow the fitted AR(1) transition from the last factor
        state, and the targets are their image under the loadings,
        inverse-transformed with ``scaler_`` (the identity with
        ``scaler=None``).

        Parameters
        ----------
        h : int
            Forecast horizon (a positive integer; anything else raises
            ``TypeError``).
        data : optional
            Not implemented yet; must be None.

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows after the last conditioning time stamp, one
            column per target in the user's order; ``attrs`` holds ``model``,
            ``kind="forecast"`` and ``origin``.
        """
        y_forecast, _, h, origin = self._run_forecast("predict", h, data)
        return self._frame(y_forecast, h, list(self.feature_names_in_), "forecast", origin)

    def predict_factors(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (same arguments as :meth:`predict`).

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows, columns ``factor_0`` .. ``factor_{r-1}``;
            ``attrs["kind"]`` is ``"factors"``.
        """
        _, z_forecast, h, origin = self._run_forecast("predict_factors", h, data)
        return self._frame(z_forecast, h, [f"factor_{i}" for i in range(z_forecast.shape[1])], "factors", origin)
