"""Deep Dynamic Factor Model (DDFM) using PyTorch.

Implements the original DDFM algorithm with MCMC-based denoising training
and sequential MC sample processing.

``DDFMEngine`` is the private engine behind the public estimator
:class:`dfm_python.models.ddfm.estimator.DDFM`, which builds the engine's
dataset from ``PanelData`` and passes every hyperparameter explicitly. Its
training code is the pre-redesign ``DDFM`` class unchanged, apart from three
edits that change no math and no RNG order:

(a) a ``device`` keyword: ``None`` keeps the automatic choice (CUDA when
    available, else CPU) and anything else becomes ``torch.device(device)``;
    the dataset tensors built in ``run_mcmc_iteration`` follow it;
(b) a pure extraction from ``predict``: :meth:`DDFMEngine._forecast` runs the
    AR(1) factor forecast and maps it to the targets, inverse-transforming with
    ``self.scaler`` when it is set (identity otherwise); ``predict(horizon)``
    keeps its checks and returns ``_forecast``'s ``(y, Z)``;
(c) ``target_series``, ``covariates`` and the fitted ``covariate_scaler``
    are copied from the dataset in ``__init__``, so an engine needs no
    dataset to update.

``_update`` was rewritten before the 0.8.0 release (it changes the update
numbers, not the fit's): it takes only the rows after the conditioning data,
in data units, scales them with the scalers fitted at fit time (the old
update refitted the target scaler on the training rows plus the whole update
panel and scaled the covariates with a scaler fitted on that panel),
interpolates them as fit interpolates its data and appends only their factors.

``get_result`` reports ``C = H`` (series x factors, the ``DFMResult.C``
orientation) instead of ``H.T``; nothing else reads it.

:meth:`DDFMEngine._from_saved` rebuilds a trained engine for the estimator's
``load`` without ``__init__``: no dataset and no seeding (it replaces the
minimal-dataset rebuild of the old ``load``).

Removed, because the estimator covers them: the ``config`` parameter (the
engine read only its interpolation values; they are passed directly) and
``_get_interpolation_params``; the ``torch.save`` checkpoint methods
``save``/``load`` with the ``load_state_dict`` override and
``_create_minimal_dataset_from_metadata`` (the estimator saves a directory
and loads through ``_from_saved``); the ``data=``/``update=`` and
``return_series``/``return_factors`` branches of ``predict`` and the public
``update`` wrapper (the estimator calls ``_update``). Without ``save``/``load``
the engine no longer inherits the old abstract base
``dfm_python.models.base.BaseFactorModel`` (since deleted); :func:`_init_engine_state` sets the
attributes that base's ``__init__`` set. The never-read ``max_epoch_pre_train``
and ``min_target_interporate_ratio`` stay as attributes with their defaults.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Tuple
import pandas as pd

from ...logger import get_logger
from ...numeric.stability import convergence_checker
from ...numeric.builder import build_ddfm_state_space
from ...numeric.statistic import (
    compute_variance_mean, compute_array_stats,
    average_3d_array
)
from ...numeric.builder import build_ddfm_optimizer
from .encoder import SimpleAutoencoder
from .sampling import run_mcmc_iteration
from ...config.schema.params import DDFMModelState
from ...config.schema.results import DDFMResult
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPS,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_DTYPE,
    DEFAULT_DDFM_LEARNING_RATE,
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_DDFM_WINDOW_SIZE,
    DEFAULT_MAX_MCMC_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_DISP,
    DEFAULT_SEED,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_MULT_EPOCH_PRETRAIN,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_LOSS_LOG_PRECISION,
    DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
    DEFAULT_FACTOR_COLLAPSE_THRESHOLD,
    DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX,
    DEFAULT_SCALE_RATIO_MIN,
    DEFAULT_STANDARDIZATION_MEAN_THRESHOLD,
    DEFAULT_STANDARDIZATION_STD_MIN,
    DEFAULT_STANDARDIZATION_STD_MAX,
    DEFAULT_STANDARDIZED_TARGET_STD,
    DEFAULT_TARGET_PREDICTION_STD,
    DEFAULT_VARIANCE_COLLAPSE_STD,
    DEFAULT_TARGET_CONVERGENCE_ITERATIONS,
    DEFAULT_TARGET_DDFM_LOSS,
    DEFAULT_DDFM_LOSS_MULTIPLIER,
)
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError
from ...utils.validation import check_condition
from ...numeric.validator import validate_no_nan_inf
from ...utils.helper import interpolate_array, interpolate_dataframe
from ...config.types import to_tensor, to_numpy

from ...dataset.ddfm_dataset import DDFMDataset

_logger = get_logger(__name__)


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class DDFMEngine(nn.Module):
    """Deep Dynamic Factor Model using PyTorch."""

    def __init__(
        self,
        dataset: DDFMDataset,
        encoder_size: Optional[tuple] = None,
        decoder_type: str = "linear",
        seed: int = DEFAULT_SEED,
        activation: str = 'relu',
        learning_rate: float = DEFAULT_DDFM_LEARNING_RATE,
        optimizer: str = 'Adam',
        n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
        window_size: int = DEFAULT_DDFM_WINDOW_SIZE,
        max_epoch_pre_train: int = DEFAULT_PRETRAIN_EPOCHS,
        max_iter: int = DEFAULT_MAX_MCMC_ITER,
        tolerance: float = DEFAULT_TOLERANCE,
        disp: int = DEFAULT_DISP,
        min_target_interporate_ratio: Optional[float] = DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
        interpolation_method: str = 'linear',
        interpolation_limit: Optional[int] = 10,
        interpolation_limit_direction: str = 'both',
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize DDFM model."""
        _init_engine_state(self)
        nn.Module.__init__(self)

        if not isinstance(dataset, DDFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DDFMDataset, got {type(dataset).__name__}"
            )

        self._dataset = dataset
        self.target_series = list(dataset.target_series)
        self.covariates = list(dataset.covariates)
        
        if encoder_size is None:
            encoder_size = tuple(DEFAULT_ENCODER_LAYERS)
        self.encoder_size = encoder_size
        self.decoder_size = None
        
        # Validate decoder_type
        if decoder_type not in ("linear", "mlp"):
            raise ConfigurationError(f"decoder_type must be 'linear' or 'mlp', got '{decoder_type}'")
        self.decoder_type = decoder_type
        if decoder_type == "mlp":
            self.decoder_size = tuple(reversed(self.encoder_size[:-1])) if len(self.encoder_size) > 1 else None
            
        self.activation = activation
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_epoch_pre_train = max_epoch_pre_train
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        
        self.scaler = getattr(dataset, 'scaler', None)
        self.covariate_scaler = getattr(dataset, 'covariate_scaler', None)
        self.min_target_interporate_ratio = min_target_interporate_ratio

        # Interpolation parameters
        self.interpolation_method = interpolation_method
        self.interpolation_limit = interpolation_limit
        self.interpolation_limit_direction = interpolation_limit_direction

        # Calculate dimensions using dataset shape properties
        # input_dim = X features + y targets (full_input concatenates them)
        # output_dim = only y targets (decoder only reconstructs targets, not features)
        self.num_series = self._dataset.data_shape[1]  # Total number of series
        self.input_dim = self._dataset.data_shape[1]  # full_input = X + y_corrupted
        self.output_dim = self._dataset.target_shape[1]  # Decoder outputs only targets

        self.initializer_seed = seed
        
        # Optimizer setup
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None  # Built in _build_optimizer()
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None  # Built in _build_optimizer()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.training_state = DDFMModelState()

        self.lags_input = 0

    @classmethod
    def _from_saved(cls, params: dict, dims: dict, device: Any) -> 'DDFMEngine':
        """Rebuild a trained engine without ``__init__``: no dataset, no seeding.

        ``params`` holds the keyword arguments the estimator passes to
        ``__init__`` (``encoder_size``, ``decoder_type``, ``seed``,
        ``activation``, ``learning_rate``, ``optimizer``, ``n_mc_samples``,
        ``window_size``, ``max_iter``, ``tolerance``, ``disp`` and the
        interpolation values); a keyword left out takes ``__init__``'s default.
        ``dims`` holds what ``__init__`` reads from the dataset: ``input_dim``,
        ``output_dim``, ``num_series``, ``target_series`` and ``covariates``.
        Every attribute ``__init__`` sets is set the same way, except
        ``scaler`` and ``covariate_scaler``, which are None (the caller sets
        the fitted ones), and the device, which is ``device``.

        The autoencoder is built as ``fit`` builds it, without its seed and
        inside ``torch.random.fork_rng`` (its initialization draws from the
        global torch RNG, which is restored), moved to ``device`` and put in
        eval mode. There is no optimizer, scheduler or dataset, and
        ``training_state`` is an empty ``DDFMModelState``: load the saved
        weights, factors and state-space matrices over the engine.
        """
        import inspect

        defaults = {
            name: p.default for name, p in inspect.signature(cls.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        defaults.pop("device")  # a separate argument
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            raise TypeError(f"DDFMEngine._from_saved got unexpected engine parameters {unknown}.")
        values = {**defaults, **params}

        self = cls.__new__(cls)
        _init_engine_state(self)
        nn.Module.__init__(self)

        self._dataset = None
        self.target_series = list(dims["target_series"])
        self.covariates = list(dims["covariates"])

        encoder_size = values["encoder_size"]
        if encoder_size is None:
            encoder_size = tuple(DEFAULT_ENCODER_LAYERS)
        self.encoder_size = encoder_size
        self.decoder_size = None
        decoder_type = values["decoder_type"]
        if decoder_type not in ("linear", "mlp"):
            raise ConfigurationError(f"decoder_type must be 'linear' or 'mlp', got '{decoder_type}'")
        self.decoder_type = decoder_type
        if decoder_type == "mlp":
            self.decoder_size = tuple(reversed(self.encoder_size[:-1])) if len(self.encoder_size) > 1 else None

        self.activation = values["activation"]
        self.n_mc_samples = values["n_mc_samples"]
        self.window_size = values["window_size"]
        self.max_epoch_pre_train = values["max_epoch_pre_train"]
        self.max_iter = values["max_iter"]
        self.tolerance = values["tolerance"]
        self.disp = values["disp"]

        self.scaler = None
        self.covariate_scaler = None
        self.min_target_interporate_ratio = values["min_target_interporate_ratio"]
        self.interpolation_method = values["interpolation_method"]
        self.interpolation_limit = values["interpolation_limit"]
        self.interpolation_limit_direction = values["interpolation_limit_direction"]

        self.num_series = int(dims["num_series"])
        self.input_dim = int(dims["input_dim"])
        self.output_dim = int(dims["output_dim"])

        self.initializer_seed = values["seed"]
        self.learning_rate = values["learning_rate"]
        self.optimizer_type = values["optimizer"]
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(device)
        self.training_state = DDFMModelState()
        self.lags_input = 0

        # The layer initializers draw from the global torch RNG (the CUDA one too on CUDA).
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.autoencoder = SimpleAutoencoder.build(
                input_dim=self.input_dim,
                encoder_size=self.encoder_size,
                decoder_size=self.decoder_size,
                decoder_type=self.decoder_type,
                output_dim=self.output_dim,
                activation=self.activation,
                seed=None,
            )
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        return self

    def _get_averaged_factors(self) -> np.ndarray:
        """Get factors averaged across MC samples if 3D, otherwise return as-is."""
        return average_3d_array(self.factors, axis=0)
    
    
    @property
    def _has_factors(self) -> bool:
        """Check if factors attribute exists and is not None."""
        return getattr(self, 'factors', None) is not None
    
    def _impute_missing_targets(self, y_pred_full: np.ndarray) -> None:
        """Impute missing target values with predictions.

        Only when every column is a target; with covariates this is a no-op
        (see the comment in the body).

        Parameters
        ----------
        y_pred_full : np.ndarray
            Predictions for target series (T, num_target_series) or full data (T, num_series)
        """
        missing_y = self._dataset.missing_y
        if not missing_y.any():
            return
        
        if self._dataset.all_columns_are_targets:
            # y_pred_full has full shape (T, num_series). Fill a copy and write it
            # back through the frame: under pandas copy-on-write (the default
            # from pandas 3) ``.values`` is read-only.
            imputed = self.data_imputed.to_numpy(copy=True)
            imputed[missing_y] = y_pred_full[missing_y]
            self.data_imputed.iloc[:, :] = imputed
        # With covariates (y_pred_full holds the target columns only), nothing is
        # imputed. Up to 0.8 this branch was the chained write
        # ``data_imputed.values[:, target_indices][missing_y] = ...``: the integer
        # array ``target_indices`` makes a fancy-index copy, so the write never
        # reached ``data_imputed``. The no-op is kept so that DDFM's numbers do not
        # change. TODO: impute the missing targets here too in a release that may
        # change DDFM's results (it changes ``data_imputed``, the residuals and
        # the denoised targets when covariates are given).
    
    def _update_imputed_and_eps(self, y_pred_full: np.ndarray) -> None:
        """Update data_imputed with predictions and compute eps (idiosyncratic residuals).
        
        Only target series (y) are imputed - features (X) are not imputed since they're only
        used for encoder input. Uses dataset's missing_y to identify missing target values.
        """
        # Impute missing target values
        self._impute_missing_targets(y_pred_full)
        
        # Compute eps: y_actual - y_pred
        if self._dataset.all_columns_are_targets:
            eps_full = self.data_imputed.values - y_pred_full
            self.eps = eps_full[:, self.target_indices]
        else:
            # y_pred_full only has target columns, extract target columns from data_imputed
            self.eps = self.data_imputed.values[:, self.target_indices] - y_pred_full
    
    def _update_previous_predictions(
        self, 
        y_pred: np.ndarray, 
        y_pred_full: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Update previous prediction state for convergence checking.
        
        Consolidates duplicate pattern of copying predictions for next iteration.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Target prediction for current iteration
        y_pred_full : np.ndarray
            Full prediction for current iteration
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (y_pred_prev, y_pred_prev_full)
        """
        return (
            y_pred.copy(),
            y_pred_full.copy() if self._dataset.all_columns_are_targets else None
        )
    
    def _build_optimizer(self) -> None:
        """Build optimizer and scheduler for training."""
        self.optimizer, self.scheduler = build_ddfm_optimizer(
            model=self.autoencoder,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            n_mc_samples=self.n_mc_samples
        )
    
    def _build_inputs_for_pretrain(self, interpolate: bool = True) -> pd.DataFrame:
        """Build inputs for pre-training.
        
        Returns the data as-is (user provides data with lagged features if needed).
        Optionally interpolates missing values.
        
        Parameters
        ----------
        interpolate : bool
            Whether to interpolate missing values using configured interpolation method
            
        Returns
        -------
        pd.DataFrame
            Input data, optionally interpolated
        """
        full_input_data = self._dataset.data.copy()
        
        if interpolate and full_input_data.isna().sum().sum() > 0:
            full_input_data = interpolate_dataframe(
                full_input_data,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        
        return full_input_data
    
    def _select_convergence_predictions(
        self,
        y_pred_prev_full: Optional[np.ndarray],
        y_pred_prev: np.ndarray,
        y_pred_full: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select appropriate predictions for convergence checking."""
        if self._dataset.all_columns_are_targets:
            return y_pred_prev_full, y_pred_full
        else:
            return y_pred_prev, y_pred
    
    def _extract_target_predictions(self, y_pred_full_tensor: torch.Tensor) -> torch.Tensor:
        """Extract target predictions from full prediction tensor.
        
        When covariates exist, decoder output_dim is already num_target_series,
        so y_pred_full_tensor already contains only target predictions.
        """
        # Decoder output_dim is always num_target_series, so predictions are already target-only
        return y_pred_full_tensor
    
    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize MCMC state: interpolate data, make initial prediction, compute initial eps."""
        self.data_denoised_interpolated = interpolate_dataframe(
            self.data_denoised,
            method=self.interpolation_method,
            limit=self.interpolation_limit,
            limit_direction=self.interpolation_limit_direction
        )
        self.data_imputed = self.data_denoised_interpolated.copy()
        
        # Match TensorFlow: build_inputs() then predict on full input data
        # For lags_input=0, _build_inputs_for_pretrain returns self.data (no lags)
        # Use interpolate=True to match TensorFlow's build_inputs() behavior
        full_input_data = self._build_inputs_for_pretrain(interpolate=True)
        full_input = to_numpy(full_input_data)
        full_input_tensor = to_tensor(full_input, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_pred_full_tensor = self.autoencoder.predict(full_input_tensor)
        y_pred_tensor = y_pred_full_tensor
        y_pred_full = to_numpy(y_pred_full_tensor)
        y_pred = to_numpy(y_pred_tensor)
        
        # Update imputed data with predictions (fill missing values)
        # Only target series (y) are imputed - features (X) are not imputed
        self._impute_missing_targets(y_pred_full)
        
        # Compute eps: idiosyncratic residuals on *targets only*.
        # When covariates exist, the decoder predicts targets only (output_dim = num_target_series),
        # so we must subtract predictions from the target slice, not from the full (X+y) matrix.
        if self.lags_input == 0:
            # IMPORTANT: use interpolated data for residuals so eps does not contain NaNs
            # Mixed-frequency data can have large missing blocks; eps NaNs propagate into AR-idio
            # estimation and can cause SVD failures when building the state space.
            if self._dataset.all_columns_are_targets:
                y_actual_full = to_numpy(self.data_denoised_interpolated)
            else:
                y_actual_full = to_numpy(self.data_denoised_interpolated[self._dataset.target_series])
        else:
            # If lags_input > 0, match target-only shape when covariates exist
            if self._dataset.all_columns_are_targets:
                y_actual_full = full_input_data[self._dataset.data.columns].values
            else:
                y_actual_full = full_input_data[self._dataset.target_series].values
        eps_full = y_actual_full - y_pred_full
        if self._dataset.all_columns_are_targets:
            self.eps = eps_full[:, self.target_indices]
        else:
            # eps_full is already target-only
            self.eps = eps_full
        y_pred_prev, y_pred_prev_full = self._update_previous_predictions(
            y_pred, y_pred_full
        )
        
        return y_pred_prev, y_pred_prev_full
                
    def fit(self) -> None:
        """Fit DDFM: builds model, pre-trains, and trains in one method."""
        start_time = time.time()
        self.autoencoder = SimpleAutoencoder.build(
            input_dim=self.input_dim,
            encoder_size=self.encoder_size,
            decoder_size=self.decoder_size,
            decoder_type=self.decoder_type,
            output_dim=self.output_dim,
            activation=self.activation,
            seed=self.initializer_seed
        )
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.autoencoder.to(self.device)
        
        
        self._build_optimizer()
        
        # Pre-train autoencoder on clean data
        min_obs = 50
        mult_epoch_pre = 1
        pretrain_epochs = self.n_mc_samples * mult_epoch_pre
        
        data_pre_train = self._build_inputs_for_pretrain(interpolate=False)
        data_pre_train_dropped = data_pre_train.dropna()
        use_mse_loss = len(data_pre_train_dropped) >= min_obs
        
        if not use_mse_loss:
            data_pre_train = self._build_inputs_for_pretrain(interpolate=True)
            data_pre_train_dropped = data_pre_train.dropna()
        
        full_input_pre_train = data_pre_train_dropped.values
        
        if self._dataset.all_columns_are_targets:
            y_pre_train = data_pre_train_dropped.values
        else:
            y_pre_train = data_pre_train_dropped[self._dataset.target_series].values
        
        assert full_input_pre_train.shape[1] == self.input_dim, \
            f"Input dimension mismatch: {full_input_pre_train.shape[1]} != {self.input_dim}"
        assert y_pre_train.shape[1] == self.output_dim, \
            f"Output dimension mismatch: {y_pre_train.shape[1]} != {self.output_dim}"
        
        full_input_tensor = torch.from_numpy(full_input_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_tensor = torch.from_numpy(y_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        final_epoch_losses = self.autoencoder.pretrain(
            full_input=full_input_tensor,
            y=y_tensor,
            epochs=pretrain_epochs,
            batch_size=self.window_size,
            optimizer=self.optimizer,
            use_mse_loss=use_mse_loss
        )
        
        if final_epoch_losses:
            _logger.info(f'Pre-training completed: final loss={final_epoch_losses[-1]:.{DEFAULT_LOSS_LOG_PRECISION}f}')
        
        self.data = self._dataset.data.copy()
        self.data_denoised = self.data.copy()
        # Note: missing_y is already available from dataset (target series only)
        # We don't need to store missing_mask separately - use dataset.missing_y directly
        self.target_indices = self._dataset.target_indices
        if not self._dataset.all_columns_are_targets:
            self._target_col_tensor = torch.tensor(self.target_indices, device=self.device, dtype=torch.long)
        self.rng = np.random.RandomState(self.initializer_seed)
        y_pred_prev, y_pred_prev_full = self._initialize_mcmc_state()
        if self._dataset.all_columns_are_targets:
            # All columns are targets
            # Use interpolated data to avoid NaNs in convergence and diagnostics
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:]
        else:
            # Only some columns are targets: use target columns from original data starting from lags_input
            # For non-target case, we need to extract target columns from self.data
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:, self.target_indices]
        
        converged = False
        self._num_iter = 0
        self.prediction_std = None
        self.factor_std = None
        while not converged and self._num_iter < self.max_iter:
            # Run MCMC iteration: denoise, sample, train, predict
            (
                self.factors, y_pred, y_pred_full, self.prediction_std, 
                self.factor_std, self.data_denoised_interpolated, autoencoder_datasets
            ) = run_mcmc_iteration(
                eps=self.eps,
                data_imputed=self.data_imputed,
                data_denoised=self.data_denoised,
                dataset=self._dataset,
                encoder=self.encoder,
                decoder=self.decoder,
                autoencoder=self.autoencoder,
                y_actual=self.y_actual,
                lags_input=self.lags_input,
                n_mc_samples=self.n_mc_samples,
                window_size=self.window_size,
                learning_rate=self.learning_rate,
                optimizer_type=self.optimizer_type,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extract_target_predictions=self._extract_target_predictions,
                interpolation_method=self.interpolation_method,
                interpolation_limit=self.interpolation_limit,
                interpolation_limit_direction=self.interpolation_limit_direction,
                target_scaler=self.scaler,
                num_iter=self._num_iter,
                disp=self.disp,
                device=self.device,
                rng=self.rng
            )
            
            self._update_imputed_and_eps(y_pred_full)
            
            if self._num_iter > 1:
                y_pred_prev, y_pred = self._select_convergence_predictions(
                    y_pred_prev_full, y_pred_prev,
                    y_pred_full, y_pred
                )
                delta, self.loss_now = convergence_checker(y_pred_prev, y_pred, self.y_actual)
                
                _logger.info(f'iteration: {self._num_iter} - delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f} - loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                
                if self._num_iter % self.disp == 0:
                    prediction_std_mean = compute_variance_mean(self.prediction_std)
                    factor_std_mean = compute_variance_mean(self.factor_std) if self.factor_std is not None else None
                    log_parts = [f'iteration: {self._num_iter}', f'loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}', f'delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f}']
                    if prediction_std_mean is not None:
                        log_parts.append(f'pred_std: {prediction_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    if factor_std_mean is not None:
                        log_parts.append(f'factor_std: {factor_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    _logger.info(' - '.join(log_parts))
                
                if delta < self.tolerance:
                    converged = True
                    self._converged = True
                    _logger.info(f'Convergence achieved in {self._num_iter + 1} iterations')
            
            y_pred_prev, y_pred_prev_full = self._update_previous_predictions(
                y_pred, y_pred_full
            )
            
            # Store last iteration datasets for MLP decoder (needed for last_neurons extraction)
            if self.decoder_type == "mlp":
                self._last_iter_datasets = autoencoder_datasets
            
            self._num_iter += 1
        
        # Extract last neurons (for MLP decoder: second-to-last layer output)
        if self.decoder_type == "linear":
            self.last_neurons = self.factors
        else:
            decoder_intermediate = self.decoder.get_intermediate()
            if decoder_intermediate is None:
                raise ConfigurationError(
                    f"Decoder {type(self.decoder).__name__} has no intermediate layers for last_neurons extraction"
                )
            self.autoencoder.eval()
            with torch.no_grad():
                last_neurons_list = [
                    decoder_intermediate(self.encoder(ae_dataset.full_input))
                    for ae_dataset in self._last_iter_datasets
                ]
            self.last_neurons = np.array([to_numpy(ln) for ln in last_neurons_list])
        
        # Store training time
        self._training_time = time.time() - start_time
    
    def build_state_space(self) -> None:
        """Build state-space model from trained autoencoder."""
        f_t = self._get_averaged_factors()
        eps_t = self.eps
        
        # Get decoder weight
        linear_layer = self.decoder.get_last_linear_layer()
        decoder_weight = to_numpy(linear_layer.weight.data)
        
        # Build state-space parameters
        F, Q, mu_0, Sigma_0, H, R = build_ddfm_state_space(
            factors=f_t,
            eps=eps_t,
            decoder_weight=decoder_weight,
            observed_y=self._dataset.observed_y,
            model_name=self.__class__.__name__
        )
        
        # Store state-space parameters in training_state (single source of truth)
        self.training_state.F = F
        self.training_state.Q = Q
        self.training_state.mu_0 = mu_0
        self.training_state.Sigma_0 = Sigma_0
        self.training_state.H = H
        self.training_state.R = R
    
    def predict(self, horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast the targets and factors ``horizon`` steps ahead (default ``DEFAULT_FORECAST_HORIZON``).

        Requires ``fit`` and ``build_state_space``. Returns ``(y_forecast,
        Z_forecast)`` from :meth:`_forecast` after checking that the model is
        trained, the state space is built, the horizon is valid and a scaler is
        set. To condition on new rows, call ``_update(new_rows)`` first.
        """
        
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        check_condition(
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} prediction failed: state-space model has not been built",
            details="Please call build_state_space() after training to enable prediction"
        )
        
        # Validate and resolve horizon
        from ...numeric.validator import validate_horizon
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        horizon = validate_horizon(horizon)
        
        # Inverse transform target series to original scale
        if self.scaler is None:
            raise ConfigurationError(
                f"{self.__class__.__name__} forecast failed: scaler is None",
                details="Dataset must provide scaler for proper forecast scaling"
            )
        
        return self._forecast(horizon)
    
    def _forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast the targets and factors ``horizon`` steps ahead.

        Pure: reads the state-space parameters and the last factor state and
        changes nothing. Returns ``(y_forecast, Z_forecast)`` of shapes
        ``(horizon, num_target_series)`` and ``(horizon, num_factors)``;
        ``y_forecast`` is on the original scale when ``self.scaler`` is set.
        """
        # Get state-space parameters from training_state (single source of truth)
        F = self.training_state.F  # Transition matrix (m x m)
        H = self.training_state.H  # Observation matrix (N x m) where N is num_target_series
        # Defensive: training instability can introduce NaNs/Infs in state-space params
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get last factor state from training (average across MC samples)
        factors_avg = self._get_averaged_factors()
        
        # Get last factor state: (num_factors,)
        # Use last row if factors exist, otherwise use initial state
        Z_last = factors_avg[-1, :] if len(factors_avg) > 0 else self.training_state.mu_0
        # Defensive: factors can contain NaNs if upstream training/update was numerically unstable.
        # Replace with 0s so forecasting can still proceed (reporting/evaluation use-case).
        Z_last = np.nan_to_num(Z_last, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate factor state
        validate_no_nan_inf(Z_last, name="last factor state Z_last")
        validate_no_nan_inf(F, name="transition matrix F")
        validate_no_nan_inf(H, name="observation matrix H")
        
        # Forecast factors forward using AR(1) dynamics
        from ...numeric.estimator import forecast_ar1_factors
        Z_forecast = forecast_ar1_factors(Z_last, F, horizon, dtype=DEFAULT_DTYPE)
        
        # Transform factors to observations (target series only)
        # H shape: (num_target_series, num_factors)
        # Z_forecast shape: (horizon, num_factors)
        # y_forecast shape: (horizon, num_target_series)
        y_forecast_std = Z_forecast @ H.T
        
        # Inverse transform target series to original scale (identity without
        # a scaler). Scaler was fitted on target series only, so
        # inverse_transform directly
        if self.scaler is not None:
            y_forecast = self.scaler.inverse_transform(y_forecast_std)
        else:
            y_forecast = y_forecast_std
        y_forecast = np.nan_to_num(y_forecast, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure numpy array and validate
        y_forecast = np.asarray(y_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(y_forecast, name="forecast y_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        return y_forecast, Z_forecast
    
    def get_result(self):
        """Extract DDFMResult from trained model."""
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} get_result failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        check_condition(
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} get_result failed: state-space model has not been built",
            details="Please call build_state_space() after training"
        )
        
        # Get state-space parameters from training_state (single source of truth)
        F = self.training_state.F  # Transition matrix
        H = self.training_state.H  # Observation matrix
        Q = self.training_state.Q  # Process noise covariance
        R = self.training_state.R  # Observation noise covariance
        mu_0 = self.training_state.mu_0  # Initial state mean
        Sigma_0 = self.training_state.Sigma_0  # Initial state covariance
        
        # Get factors (average across MC samples if 3D)
        Z = self._get_averaged_factors()
        
        # Compute smoothed data: x_sm = Z @ H.T
        # Use get_x_sm_original_scale() on result to get unscaled values if needed
        # x_sm contains only target series (features X are only for encoder input, not in results)
        x_sm = Z @ H.T  # (T, num_target_series)
        
        # Create result
        # For DDFM: A = F (transition), C = H (observation), r = [num_factors], p = 1
        # Z already computed above, reuse it instead of calling _get_averaged_factors() again
        num_factors = Z.shape[1]
        return DDFMResult(
            x_sm=x_sm,
            Z=Z,
            # H is (num_target_series x num_factors), the DFMResult.C orientation
            # (it was H.T, factors x series, before the estimator).
            C=H,
            R=R,
            A=F,
            Q=Q,
            Z_0=mu_0,
            V_0=Sigma_0,
            r=np.array([num_factors]),  # Single block with num_factors
            p=DEFAULT_FACTOR_ORDER,  # AR(1) dynamics
            target_scaler=self.scaler,  # Store scaler in target_scaler field (DFMResult API)
            converged=getattr(self, '_converged', False),
            num_iter=getattr(self, '_num_iter', self.max_iter),  # Use actual iteration count if available
            loglik=-float('inf')  # DDFM doesn't compute log-likelihood
        )

    def _scale_new_rows(self, new_rows: pd.DataFrame) -> pd.DataFrame:
        """``new_rows`` (data units, columns ``[covariates | targets]``) scaled with the fitted scalers.

        The targets go through ``self.scaler`` and the covariates through
        ``self.covariate_scaler``, the instances fitted on the training data;
        neither is refitted. Without a scaler the values pass through. Missing
        values stay missing.
        """
        values = new_rows.to_numpy(dtype=np.float64, copy=True)
        scaled = pd.DataFrame(values, columns=list(new_rows.columns))
        if self.scaler is not None:
            if self.covariates and self.covariate_scaler is None:
                raise ModelNotInitializedError(
                    f"{self.__class__.__name__} update failed: the covariate scaler fitted at fit time is missing",
                    details="Refit the model; runs saved before the covariate scaler was kept cannot be updated",
                )
            targets = list(self.target_series)
            scaled[targets] = self.scaler.transform(scaled[targets].to_numpy())
            if self.covariates:
                covariates = list(self.covariates)
                scaled[covariates] = self.covariate_scaler.transform(scaled[covariates].to_numpy())
        return scaled

    def _update(self, new_rows: pd.DataFrame) -> None:
        """Condition the factors on ``new_rows``, the rows after the conditioning data, in data units.

        ``new_rows`` has the columns of ``self.data`` (``[covariates |
        targets]``) in that order. The rows are scaled with the scalers fitted
        at fit time (:meth:`_scale_new_rows`, never refitted) and appended to
        ``self.data``, the scaled conditioning frame (the training rows and
        the rows earlier updates appended). Missing values are filled as fit
        fills its data: ``interpolate_dataframe`` with the interpolation
        settings, run over the whole conditioning frame so that the new rows
        are interpolated across the boundary with the earlier ones (gaps the
        interpolation leaves are carried forward, then backward). The encoder
        maps the filled new rows to factors, which are appended to
        ``self.factors`` (repeated over the MC samples when the factors are
        3-D); the forecast then starts from the last new row. The network,
        the scalers and the state space are unchanged.
        """
        from ...utils.errors import DataValidationError

        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} update failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        check_condition(
            getattr(self, 'autoencoder', None) is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} update failed: model has not been built",
            details="Please call fit() first"
        )

        columns = list(self.data.columns)
        if list(new_rows.columns) != columns:
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): the new rows have columns {list(new_rows.columns)[:5]}..., "
                f"expected {columns[:5]}... (the training columns [covariates | targets], in that order)."
            )
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(new_rows[col])]
        if non_numeric:
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): non-numeric columns {non_numeric[:5]}; all columns must be numeric."
            )
        n_new = len(new_rows)
        if n_new < 1:
            raise DataValidationError(f"{self.__class__.__name__}._update(): no new rows.")

        scaled = self._scale_new_rows(new_rows)
        conditioning = pd.concat([self.data.reset_index(drop=True), scaled], axis=0, ignore_index=True)
        filled = conditioning
        if conditioning.isna().to_numpy().any():
            filled = interpolate_dataframe(
                conditioning,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        new_input = filled.to_numpy(dtype=np.float64)[-n_new:]
        if not np.isfinite(new_input).all():
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): the new rows hold infinite values (after scaling)."
            )

        new_input_tensor = to_tensor(new_input, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            new_factors_np = to_numpy(self.encoder(new_input_tensor))  # (T_new, num_factors)

        if self.factors.ndim == 3:
            new_factors_expanded = np.repeat(np.expand_dims(new_factors_np, axis=0), self.factors.shape[0], axis=0)
            self.factors = np.concatenate([self.factors, new_factors_expanded], axis=1)
        else:
            self.factors = np.vstack([self.factors, new_factors_np])
        self.data = conditioning
