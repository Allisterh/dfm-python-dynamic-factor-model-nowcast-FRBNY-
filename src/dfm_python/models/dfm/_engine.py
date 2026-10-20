"""Linear Dynamic Factor Model (DFM) engine: EM with NumPy and an in-house Kalman filter/smoother.

``DFMEngine`` is the private engine of the linear DFM. Its code is the
pre-redesign ``DFM`` class (``models/dfm/dfm.py``) moved here unchanged,
apart from these edits, none of which changes math or numbers:

(a) the class is renamed from ``DFM`` to ``DFMEngine`` (with the return
    annotations that named it);
(b) ``update()`` imports ``validate_and_convert_update_data`` from
    ``dfm_python.numeric.validator`` (``...numeric.validator``). The old
    ``..numeric.validator`` resolved to the non-existent
    ``dfm_python.models.numeric``, so ``update()`` always raised
    ``ModuleNotFoundError``;
(c) the engine's config class is the private ``_DFMEngineConfig`` (the old
    ``DFMConfig``; ``dfm_python.DFMConfig`` now mirrors the estimator);
(d) ``log_blocks_diagnostics`` is imported from ``dfm_python.numeric.builder``
    (it moved there, unchanged, when the ``logger.dfm_logger`` module was
    removed).

:meth:`DFMEngine._from_saved` rebuilds a trained engine for the estimator's
``load`` without ``__init__`` (no dataset): it sets what ``__init__`` and
``fit`` set from the saved state.

The public :class:`~dfm_python.models.dfm.estimator.DFM` estimator builds
this engine from ``PanelData`` (``dfm_python.DFM`` names the estimator) with
the ``dataset=``/``config=`` constructor.

Removed, because the estimator covers them (none of these ran in ``fit``,
``predict`` or ``update``): ``load_config`` (it called the missing
``_load_config_common`` and could not run); the pickle ``save`` and the
pickle/joblib ``load``, which needed a dataset (the estimator saves a
pickle-free directory and loads through ``_from_saved``); ``reset``; the
``result`` property (``get_result()`` returns the same cached result); and
``predict``'s ``horizon=None`` default (one year of clock periods through
``utils.misc.compute_default_horizon``, removed with it): ``horizon`` is
required. Without ``save``/``load`` the engine no longer inherits the old
abstract base ``dfm_python.models.base.BaseFactorModel`` (since deleted):
:func:`_init_engine_state` sets the attributes that base's ``__init__`` set, at
the same point in ``__init__`` and ``_from_saved``, and
:meth:`DFMEngine._ensure_result` is that base's method, unchanged.
"""

# Standard library imports
import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# NumPy Kalman filter and Durbin-Koopman smoother (ssm/kalman.py)
from ...ssm.kalman import DFMKalmanFilter

# Local imports
from ...config import DFMResult
from ...config.schema.model import _DFMEngineConfig
from ...config.schema.params import DFMModelState
# get_agg_structure and get_slower_freq_tent_weights are now used in dfm.mixed_frequency and dfm.initialization modules
from ...config.constants import (
    DEFAULT_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_ZERO_VALUE,
    FREQUENCY_HIERARCHY,
    DEFAULT_HIERARCHY_VALUE,
    MAX_STD_UNSCALED,
    MIN_STD_SCALED,
)
from ...logger import get_logger
from ...dataset.dfm_dataset import DFMDataset
from ...utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    ConfigurationError,
    DataError,
    DataValidationError,
    PredictionError,
    NumericalError
)
from ...utils.validation import check_condition, has_shape_with_min_dims
from ...numeric.validator import validate_dfm_initialization

# Import EM config from functional module
from ...functional.em import _DEFAULT_EM_CONFIG as _EM_CONFIG

# Import initialization and mixed-frequency functions from dfm submodule
from .initialization import (
    initialize_parameters,
)
from .mixed_freq import (
    setup_mixed_frequency_params,
)
from ...numeric.builder import build_dfm_structure, build_dfm_blocks, log_blocks_diagnostics
from ...numeric.validator import validate_column_order

_logger = get_logger(__name__)


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class DFMEngine:
    """Linear Dynamic Factor Model using the EM algorithm (NumPy E-step and M-step)."""
    
    def __init__(
        self,
        dataset: DFMDataset,
        config: Optional[_DFMEngineConfig] = None,
        scaler: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Initialize DFM instance.
        
        At initialization, variables and time_index are extracted from DFMDataset:
        - `dataset.variables`: DataFrame with numeric columns (used for training)
        - `dataset.time_index`: TimeIndex object (used for time alignment)
        
        Parameters
        ----------
        dataset : DFMDataset
            DFMDataset instance (required). Variables and time_index are extracted during fit().
        config : _DFMEngineConfig, optional
            DFM configuration. If None, config will be created from kwargs using _DFMEngineConfig.from_dict().
            All _DFMEngineConfig parameters can be passed directly as kwargs when config=None.
        scaler : StandardScaler, RobustScaler, MinMaxScaler, or None, optional
            Sklearn scaler instance for data standardization. If provided, it's used directly via isinstance() check.
            If None, a scaler will be created during fit() based on config.scaler or defaults to StandardScaler.
            If provided, it will be fitted during fit() if not already fitted.
        **kwargs : Any
            If config is None: All kwargs are used to create _DFMEngineConfig via _DFMEngineConfig.from_dict(kwargs).
            If config is provided: kwargs are used as parameter overrides (keys should match _DFMEngineConfig attribute names).
            Example: `DFM(dataset=ds, blocks={...}, threshold=1e-6)` creates config from kwargs.
            Example: `DFM(dataset=ds, config=cfg, threshold=1e-6)` overrides config.threshold.
            Precedence: kwargs > config > defaults.
            
        Returns
        -------
        None
            Initializes DFM instance in-place.
            
        Raises
        ------
        ConfigurationError
            If config validation fails or required parameters are missing.
        ValueError
            If mixed_freq=True and tent_weights are not specified in config.
            
        Examples
        --------
        >>> # Create config from kwargs
        >>> model = DFM(
        ...     dataset=my_dataset,
        ...     blocks={"global": {"num_factors": 1, "series": ["GDP", "CPI"]}},
        ...     threshold=1e-6,
        ...     max_iter=100
        ... )
        >>> 
        >>> # Or pass existing config
        >>> model = DFM(
        ...     dataset=my_dataset,
        ...     config=my_config,
        ...     threshold=1e-6  # Override config.threshold
        ... )
        """
        _init_engine_state(self)
        
        # Dataset is required
        if not isinstance(dataset, DFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DFMDataset, got {type(dataset).__name__}"
            )
        self._dataset = dataset
        
        # Create config from kwargs if not provided
        if config is None:
            if not kwargs:
                raise ConfigurationError(
                    "DFM: Either config or kwargs must be provided",
                    details="Provide a _DFMEngineConfig object, or pass all config parameters as kwargs (e.g., blocks, threshold, max_iter, etc.)"
                )
            # Remove dataset and scaler from kwargs (they're not part of _DFMEngineConfig)
            config_kwargs = {k: v for k, v in kwargs.items() if k not in ('dataset', 'scaler')}
            if not config_kwargs:
                raise ConfigurationError(
                    "DFM: No config parameters provided in kwargs",
                    details="Provide _DFMEngineConfig parameters like blocks, threshold, max_iter, etc."
                )
            # Create config from kwargs
            self._config = _DFMEngineConfig.from_dict(config_kwargs)
            # Clear kwargs since they were used to create config
            kwargs = {}
        else:
            self._config = config
        
        # Apply kwargs overrides to config (if any)
        # This allows runtime parameter overrides while keeping config as source of truth
        if kwargs:
            for key, value in kwargs.items():
                # Skip dataset and scaler (already handled)
                if key in ('dataset', 'scaler'):
                    continue
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                    _logger.debug(f"DFM: Overrode config.{key} with {value}")
                else:
                    _logger.warning(f"DFM: Unknown parameter '{key}' in kwargs, ignoring")
        
        # Mixed frequency: auto-detected from Dataset or config during fit()
        self._mixed_freq: Optional[bool] = None  # Internal property, auto-detected
        
        # Cache for smoothed factors from last EM iteration (avoids recomputation during save())
        self._cached_smoothed_factors: Optional[np.ndarray] = None
        
        # Mixed frequency parameters (set during fit)
        self._constraint_matrix = None  # R_mat: constraint matrix for tent kernel aggregation
        self._constraint_vector = None  # q: constraint vector for tent kernel aggregation
        self._n_slower_freq = 0  # Number of slower-frequency series
        self._tent_weights_dict = None
        self._frequencies = None
        self._idio_indicator = None  # i_idio: indicator for idiosyncratic components
        
        # Build model structure from config
        # Allow "minimal" configs that omit frequency by late-binding to dataset columns.
        # If config.frequency is missing, get_blocks_array(columns=...) will default all series to clock frequency.
        dataset_columns = list(self._dataset.variables.columns) if self._dataset.variables is not None else None
        self.blocks, self.r, self.num_factors, self.p = build_dfm_structure(self._config, columns=dataset_columns)
        
        # Create empty training state with structure parameters only
        # State-space parameters (A, C, Q, R, Z_0, V_0) will be set during fit()
        self.training_state = DFMModelState(
            num_factors=self.num_factors,
            r=self.r,
            p=self.p,
            blocks=self.blocks,
            mixed_freq=None,  # Set during fit()
            constraint_matrix=None,
            constraint_vector=None,
            n_slower_freq=0,
            n_clock_freq=None,
            tent_weights_dict=None,
            frequencies=None,
            idio_indicator=None,
            max_lag_size=None,
            A=None,
            C=None,
            Q=None,
            R=None,
            Z_0=None,
            V_0=None
        )
        
        # Training state
        self.data_processed: Optional[np.ndarray] = None
        self.scaler: Optional[Any] = scaler  # Sklearn scaler for data standardization and inverse transformation
    
    @classmethod
    def _from_saved(
        cls,
        engine_config: Dict[str, Any],
        state: DFMModelState,
        data_processed: np.ndarray,
        smoothed_state: np.ndarray,
        meta: Dict[str, Any],
        scaler: Optional[Any],
        result: DFMResult,
    ) -> 'DFMEngine':
        """Rebuild a trained engine without ``__init__``: no dataset is needed.

        Sets every attribute ``__init__`` and ``fit`` set:

        - ``_config``: ``_DFMEngineConfig.from_dict(engine_config)``, the dict
          the estimator built the engine config from;
        - ``blocks``, ``r``, ``num_factors``, ``p`` and ``training_state``
          from ``state``, a complete ``DFMModelState`` (structure,
          mixed-frequency parameters and ``A`` .. ``V_0``), and the
          mixed-frequency attributes (``_mixed_freq``, ``_constraint_matrix``,
          ``_constraint_vector``, ``_n_slower_freq``, ``_n_clock_freq``,
          ``_tent_weights_dict``, ``_frequencies``, ``_idio_indicator``,
          ``_max_lag_size``) from the same fields;
        - ``data_processed`` (the scaled data the filter reads, update rows
          included), ``scaler``, ``_dataset=None`` and
          ``_cached_smoothed_factors=None``;
        - ``_training_columns``, ``_training_num_iter``,
          ``_training_converged`` and ``_training_loglik`` from ``meta``
          (keys ``training_columns``, ``num_iter``, ``converged``, ``loglik``);
        - ``_result``: a copy of ``result`` with ``Z=smoothed_state`` (the
          smoothed state ``update`` extends) and, when ``smoothed_state`` has
          more rows than ``result.Z``, ``x_sm = smoothed_state @ C.T`` as
          ``update`` computes it.
        """
        engine = cls.__new__(cls)
        _init_engine_state(engine)
        # What __init__ sets.
        engine._config = _DFMEngineConfig.from_dict(dict(engine_config))
        engine._dataset = None
        engine.training_state = state
        engine.blocks = state.blocks
        engine.r = state.r
        engine.num_factors = state.num_factors
        engine.p = state.p
        engine.data_processed = data_processed
        engine.scaler = scaler
        engine._cached_smoothed_factors = None
        # What fit sets.
        engine._mixed_freq = state.mixed_freq
        engine._constraint_matrix = state.constraint_matrix
        engine._constraint_vector = state.constraint_vector
        engine._n_slower_freq = state.n_slower_freq
        engine._n_clock_freq = state.n_clock_freq
        engine._tent_weights_dict = state.tent_weights_dict
        engine._frequencies = state.frequencies
        engine._idio_indicator = state.idio_indicator
        engine._max_lag_size = state.max_lag_size
        engine._training_columns = list(meta["training_columns"])
        engine._training_num_iter = meta["num_iter"]
        engine._training_converged = meta["converged"]
        engine._training_loglik = meta["loglik"]
        result = copy.deepcopy(result)
        if smoothed_state.shape[0] == result.Z.shape[0]:
            x_sm = result.x_sm
        else:
            x_sm = smoothed_state @ result.C.T
        engine._result = dataclasses.replace(result, Z=smoothed_state, x_sm=x_sm)
        return engine

    @property
    def A(self) -> Optional[np.ndarray]:
        """Transition matrix (m x m) - VAR dynamics for factors."""
        return self.training_state.A if self.training_state else None
    
    @property
    def C(self) -> Optional[np.ndarray]:
        """Observation matrix (N x m) - factor loadings."""
        return self.training_state.C if self.training_state else None
    
    @property
    def Q(self) -> Optional[np.ndarray]:
        """Process noise covariance (m x m) - innovation covariance."""
        return self.training_state.Q if self.training_state else None
    
    @property
    def R(self) -> Optional[np.ndarray]:
        """Observation noise covariance (N x N) - typically diagonal."""
        return self.training_state.R if self.training_state else None
    
    @property
    def Z_0(self) -> Optional[np.ndarray]:
        """Initial state mean (m,) - initial factor values."""
        return self.training_state.Z_0 if self.training_state else None
    
    @property
    def V_0(self) -> Optional[np.ndarray]:
        """Initial state covariance (m x m) - initial uncertainty."""
        return self.training_state.V_0 if self.training_state else None
    
    def _check_parameters_initialized(self) -> None:
        """Check if model parameters are initialized (required for prediction).
        
        Raises
        ------
        ModelNotInitializedError
            If parameters are not initialized
        """
        from ...numeric.validator import validate_parameters_initialized
        validate_parameters_initialized(
            {
                'A': self.A, 'C': self.C, 'Q': self.Q,
                'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
            },
            model_name=self.__class__.__name__
        )
    
    def _update_parameters(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
                          R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> None:
        """Update model parameters from NumPy arrays.
        
        Parameters
        ----------
        A, C, Q, R, Z_0, V_0 : np.ndarray
            Parameter arrays
        """
        # Update training_state directly (single source of truth)
        self.training_state.A = np.asarray(A, dtype=DEFAULT_DTYPE) if A is not None else None
        self.training_state.C = np.asarray(C, dtype=DEFAULT_DTYPE) if C is not None else None
        self.training_state.Q = np.asarray(Q, dtype=DEFAULT_DTYPE) if Q is not None else None
        self.training_state.R = np.asarray(R, dtype=DEFAULT_DTYPE) if R is not None else None
        self.training_state.Z_0 = np.asarray(Z_0, dtype=DEFAULT_DTYPE) if Z_0 is not None else None
        self.training_state.V_0 = np.asarray(V_0, dtype=DEFAULT_DTYPE) if V_0 is not None else None
    
    
    def fit(
        self,
        X: Optional[Union[np.ndarray, Any]] = None,
        checkpoint_callback: Optional[Any] = None
    ) -> DFMModelState:
        """Fit model using the EM algorithm.
        
        Uses DFMKalmanFilter for the E-step (Kalman filter/smoother) and a custom M-step
        that preserves block structure and mixed-frequency constraints.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Standardized data (T x N). If None, data is extracted from dataset.
        checkpoint_callback : callable, optional
            Callback function for checkpointing during training.
            
        Returns
        -------
        DFMModelState
            Complete model state including structure, mixed-frequency parameters, and fitted state-space parameters
        """

        # Clear cached smoothed factors on retrain
        self._cached_smoothed_factors = None
        self._result = None  # Also clear result cache
        
        # Clear all caches for fresh training run (ensures no stale data from previous runs)
        if self._config is not None and hasattr(self._config, '_cached_blocks'):
            self._config._cached_blocks = None
        
        # Use X parameter if provided, otherwise extract from dataset
        # Extract data from DFMDataset (variables and time_index are extracted at initialization)
        # Dataset provides: variables (DataFrame) and time_index (TimeIndex)
        dataset = self._dataset
        if X is not None:
            # Use provided X - convert to numpy if needed
            if isinstance(X, np.ndarray):
                X_np = X
                # Get columns from dataset for validation
                data_df = dataset.variables
                columns = list(data_df.columns)
                if X_np.shape[1] != len(columns):
                    raise DataValidationError(
                        f"Provided X has {X_np.shape[1]} columns but dataset has {len(columns)} columns. "
                        f"Column count must match.",
                        details=f"X shape: {X_np.shape}, dataset columns: {columns[:5]}..."
                    )
            else:
                # Convert DataFrame to numpy
                X_np = np.asarray(X)
                if hasattr(X, 'columns'):
                    columns = list(X.columns)
                else:
                    data_df = dataset.variables
                    columns = list(data_df.columns)
        else:
            # Extract from dataset (original behavior)
            data_df = dataset.variables  # Extract variables DataFrame
            X_np = data_df.values  # Convert to numpy array
            columns = list(data_df.columns)

        # Get clock frequency from config
        clock = getattr(self._config, 'clock', DEFAULT_CLOCK_FREQUENCY)
        
        # Store column names for validation (column order validation happens after standardization)
        # Store column names for later validation in update()
        self._training_columns = columns
        
        # Check if scaler is already fitted before fit_transform
        # Standardize data if scaler is available
        if self.scaler is not None:
            # Check if scaler is already fitted (has mean_ or scale_ attributes)
            is_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
            if is_fitted:
                # Already fitted - only transform
                X_np_std = self.scaler.transform(X_np)
            else:
                # Not fitted - fit and transform
                X_np_std = self.scaler.fit_transform(X_np)
        else:
            X_np_std = X_np
        
        # GUARDRAIL: Validate data scaling after preprocessing
        # Make preprocessing a contract, not a suggestion
        # Unscaled data (e.g., raw FX levels) causes numerical instability and slow runs
        # Check AFTER scaler is applied (if provided) to catch cases where scaler wasn't applied
        data_std = np.nanstd(X_np_std, axis=0)
        unscaled_series = np.where((data_std > MAX_STD_UNSCALED) | (data_std < MIN_STD_SCALED))[0]
        if len(unscaled_series) > 0:
            unscaled_names = [columns[i] for i in unscaled_series[:5]]  # Show first 5
            raise DataValidationError(
                f"Input data appears unscaled after preprocessing. Series with problematic std: {unscaled_names}... "
                f"(std range: [{np.min(data_std):.2e}, {np.max(data_std):.2e}]). "
                f"Please apply a scaler (e.g., StandardScaler) before fitting the model. "
                f"Unscaled data causes numerical instability, excessive stabilization, and slow convergence.",
                details=f"Expected std range: [{MIN_STD_SCALED:.2e}, {MAX_STD_UNSCALED:.2e}]. "
                       f"Found {len(unscaled_series)}/{len(columns)} series outside this range. "
                       f"Scaler provided: {self.scaler is not None}"
            )
        
        # Build blocks array to match actual data dimensions
        N_actual = X_np_std.shape[1]
        self.blocks = build_dfm_blocks(self.blocks, self._config, columns, N_actual)
        log_blocks_diagnostics(self.blocks, columns, N_actual)
        
        # Setup mixed-frequency parameters
        mf_params = setup_mixed_frequency_params(self._config, clock, columns, N_actual)
        
        # Column ordering validation is enforced in setup_mixed_frequency_params()
        # It will raise DataValidationError if [clock series | slower series] ordering is violated
        # This is a hard requirement for correct mixed-frequency handling

        self._mixed_freq        = mf_params['mixed_freq']
        self._constraint_matrix = mf_params['R_mat']
        self._constraint_vector = mf_params['q']
        self._n_slower_freq     = mf_params['n_slower_freq']
        self._n_clock_freq      = mf_params['n_clock_freq']
        self._tent_weights_dict = mf_params['tent_weights_dict']
        self._frequencies       = mf_params['frequencies_np']
        self._idio_indicator    = mf_params['idio_indicator']
        self._max_lag_size      = max(self.p + 1, mf_params['tent_kernel_size'])
        
        # Initialize parameters (required for EM algorithm)
        _logger.info("Initializing DFM parameters...")
        _logger.info(f"  Data: {X_np_std.shape[0]} time steps × {X_np_std.shape[1]} series")
        _logger.info(f"  Blocks: {self.blocks.shape[1]}, Factors: {self.num_factors}, "
                    f"Mixed freq: {self._mixed_freq}")
        A_np, C_np, Q_np, R_np, Z_0_np, V_0_np = initialize_parameters(
            X_np_std, self.r, self.p, self.blocks, 
            self._constraint_matrix, self._constraint_vector, self._n_slower_freq, 
            self._idio_indicator, clock, self._tent_weights_dict
        )
        
        self.data_processed = X_np_std
        
        # Validate numerical stability before proceeding
        validate_dfm_initialization(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        self._update_parameters(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        _logger.info(f"Initialization complete: state_dim={self.A.shape[0]}, obs_dim={self.C.shape[0]}, "
                    f"factors={self.num_factors}, max_lag={self._max_lag_size}, mixed_freq={self._mixed_freq}")
        
        # Update training_state with mixed-frequency parameters before passing to EM
        self.training_state.mixed_freq = self._mixed_freq
        self.training_state.constraint_matrix = self._constraint_matrix
        self.training_state.constraint_vector = self._constraint_vector
        self.training_state.n_slower_freq = self._n_slower_freq
        self.training_state.n_clock_freq = self._n_clock_freq
        self.training_state.tent_weights_dict = self._tent_weights_dict
        self.training_state.frequencies = self._frequencies
        self.training_state.idio_indicator = self._idio_indicator
        self.training_state.max_lag_size = self._max_lag_size
        
        initial_state = self.training_state
        
        # Create EMConfig from _DFMEngineConfig (uses consolidated parameters)
        em_config = self._config.to_em_config() if self._config is not None else _EM_CONFIG
        
        # Run EM algorithm using run_em_algorithm() directly
        from ...functional.em import run_em_algorithm
        try:
            final_state, training_metadata = run_em_algorithm(
                X=X_np_std,  # Use standardized data
                initial_state=initial_state,
                max_iter=self._config.max_iter,
                threshold=self._config.threshold,
                config=em_config,
                checkpoint_callback=checkpoint_callback
            )
        except Exception as e:
            _logger.error(f"EM algorithm failed: {e}", exc_info=True)
            _logger.error("  Initialization parameters:")
            _logger.error(f"    A shape: {self.A.shape}, C shape: {self.C.shape}")
            _logger.error(f"    Q shape: {self.Q.shape}, R shape: {self.R.shape}")
            _logger.error(f"    Z_0 shape: {self.Z_0.shape}, V_0 shape: {self.V_0.shape}")
            _logger.error(f"    Blocks shape: {self.blocks.shape}, r: {self.r}")
            _logger.error(f"    p: {self.p}, max_lag_size: {self._max_lag_size}")
            raise
        
        # Update training_state with all parameters from final_state
        self.training_state = final_state
        
        # Store training metadata separately (not in state-space params)
        self._training_loglik = training_metadata['loglik']
        self._training_num_iter = training_metadata['num_iter']
        self._training_converged = training_metadata['converged']
        
        # Cache smoothed factors from last EM iteration (avoids recomputation during save())
        if 'smoothed_factors' in training_metadata and training_metadata['smoothed_factors'] is not None:
            self._cached_smoothed_factors = training_metadata['smoothed_factors']
        
        return self.training_state
    
    def _create_kalman_filter(
        self,
        initial_state_mean: Optional[np.ndarray] = None,
        initial_state_covariance: Optional[np.ndarray] = None
    ) -> DFMKalmanFilter:
        """Create Kalman filter with current training state parameters.
        
        Parameters
        ----------
        initial_state_mean : np.ndarray, optional
            Initial state mean. If None, uses training_state.Z_0
        initial_state_covariance : np.ndarray, optional
            Initial state covariance. If None, uses training_state.V_0
            
        Returns
        -------
        DFMKalmanFilter
            Configured Kalman filter instance
        """
        if initial_state_mean is None:
            initial_state_mean = self.training_state.Z_0
        if initial_state_covariance is None:
            initial_state_covariance = self.training_state.V_0
        
        return DFMKalmanFilter(
            transition_matrices=self.training_state.A,
            observation_matrices=self.training_state.C,
            transition_covariance=self.training_state.Q,
            observation_covariance=self.training_state.R,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )
    
    def _compute_smoothed_factors(self) -> np.ndarray:
        """Compute smoothed factors using Kalman filter.
        
        Uses cached factors from last EM iteration if available (avoids expensive recomputation).
        Otherwise, computes them using filter_and_smooth().
        
        Returns
        -------
        np.ndarray
            Smoothed factors (T x m)
        """
        check_condition(
            self.training_state is not None and self.data_processed is not None,
            ModelNotTrainedError,
            "Model not fitted or data not available",
            details="Please call fit() method before computing smoothed factors"
        )
        
        # Use cached smoothed factors from last EM iteration if available
        # This avoids expensive recomputation during save() (saves 100-300s for large datasets)
        if self._cached_smoothed_factors is not None:
            _logger.debug("Using cached smoothed factors from last EM iteration (skipping recomputation)")
            return self._cached_smoothed_factors
        
        # Fallback: compute smoothed factors (should rarely happen if EM completed successfully).
        # The filter raises NumericalError when the parameters make it non-finite.
        _logger.debug("Computing smoothed factors (cache not available)")
        kalman_final = self._create_kalman_filter()
        smoothed_state_means, _, _, _ = kalman_final.filter_and_smooth(self.data_processed)
        return smoothed_state_means

    def _filter_new_rows(self, data_new: np.ndarray) -> np.ndarray:
        """Filtered states of ``data_new``, continuing the training filter.

        Runs the Kalman filter with the fitted parameters over all rows the
        model conditions on (``data_processed``) followed by ``data_new``, so the
        prior of the first new row is ``(A x_T|T, A P_T|T A' + Q)``, and returns
        the filtered means ``x_t|t`` of the new rows (T_new x m).
        """
        X_all = np.vstack([self.data_processed, data_new])
        filtered_means, _ = self._create_kalman_filter().filter(X_all)
        return np.asarray(filtered_means[-data_new.shape[0]:], dtype=DEFAULT_DTYPE)

    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        # Return cached result if available (avoids expensive recomputation)
        if self._result is not None:
            return self._result
        
        # Compute smoothed factors (validates training_state and data_processed internally)
        Z = self._compute_smoothed_factors()
        
        # Get parameters
        A = self.training_state.A
        C = self.training_state.C
        Q = self.training_state.Q
        R = self.training_state.R
        Z_0 = self.training_state.Z_0
        V_0 = self.training_state.V_0
        
        # Compute smoothed data
        x_sm = Z @ C.T
        
        # Get training metadata from instance attributes
        converged = getattr(self, '_training_converged', False)
        num_iter = getattr(self, '_training_num_iter', 0)
        loglik = getattr(self, '_training_loglik', 0.0)
        
        result = DFMResult(
            x_sm=x_sm, Z=Z, C=C, R=R, A=A, Q=Q,
            target_scaler=None,  # DFM uses model.scaler, not target_scaler
            Z_0=Z_0, V_0=V_0, r=self.r, p=self.p,
            converged=converged,
            num_iter=num_iter,
            loglik=loglik
        )
        
        # Cache result to avoid recomputation
        self._result = result
        return result
    
    def _ensure_result(self) -> DFMResult:
        """Ensure result exists, computing it if necessary."""
        if self._result is None:
            if self.training_state is None:
                raise ModelNotTrainedError("Model has not been trained yet")
            self._result = self.get_result()
        return self._result
    
    def _get_last_state(self, result: DFMResult, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """Get last state from result.Z with safe fallback.
        
        Safely extracts the last row from result.Z, falling back to result.Z_0
        if result.Z is empty or invalid.
        
        Parameters
        ----------
        result : DFMResult
            Model result containing Z and Z_0
        fallback : np.ndarray, optional
            Fallback state if result.Z is invalid. If None, uses result.Z_0.
        
        Returns
        -------
        np.ndarray
            Last state vector (1D array)
        """
        if fallback is None:
            fallback = result.Z_0
        
        if has_shape_with_min_dims(result.Z, min_dims=1) and result.Z.shape[0] > 0:
            return result.Z[-1, :]
        return fallback
    
    
    def update(self, data: Union[np.ndarray, Any], rescale: bool = False, retrain: bool = False, **kwargs) -> None:
        """Update model state with new observations.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations (T_new x N). Must be preprocessed but NOT standardized.
        rescale : bool, default False
            If True, fit scaler on new data and transform. If False, only transform.
        retrain : bool, default False
            If True, run EM (``run_em_algorithm``, the driver ``fit`` uses) on
            the training rows plus the new rows, starting from the fitted
            parameters; the reported loglik is the exact loglik of the returned
            parameters on all rows, and an iteration that would lower it is not
            kept. If False, only update the state via the Kalman filter.
        **kwargs
            Additional parameters for retraining:
            - max_iter : int, default 1
                Number of EM iterations when retrain=True
            - threshold : float, optional
                Convergence threshold for EM
            
        Notes
        -----
        - Data must be preprocessed (transformations applied) but NOT standardized
        - Columns must be in the same order as training data
        """
        # Validate column order and format if data is DataFrame
        validate_column_order(data, self.scaler)
        
        # Validate and convert data
        from ...numeric.validator import validate_and_convert_update_data
        data_new = validate_and_convert_update_data(
            data, 
            self.data_processed, 
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        
        # Scale data
        if self.scaler is not None:
            if rescale:
                self.scaler.fit(data_new)
            data_new = self.scaler.transform(data_new)
        
        if retrain:
            # Run full EM
            X_combined = np.vstack([self.data_processed, data_new])
            self.data_processed = X_combined
            
            # Update training_state with current parameters
            initial_state = self.training_state
            
            # Create EMConfig
            em_config = self._config.to_em_config() if self._config is not None else _EM_CONFIG
            
            # Get max_iter and threshold from kwargs or defaults
            max_iter = kwargs.get('max_iter', 1)
            threshold = kwargs.get('threshold', self._config.threshold)
            
            # Run EM
            from ...functional.em import run_em_algorithm
            final_state, training_metadata = run_em_algorithm(
                X=X_combined,
                initial_state=initial_state,
                max_iter=max_iter,
                threshold=threshold,
                config=em_config,
                checkpoint_callback=None
            )
            
            # Update training_state
            self.training_state = final_state
            
            # Update training metadata
            self._training_loglik = training_metadata['loglik']
            self._training_num_iter = training_metadata['num_iter']
            self._training_converged = training_metadata['converged']
            
            # Invalidate cached smoothed factors on retrain
            # Cached factors become stale after parameter updates
            self._result = None
            self._cached_smoothed_factors = None
            _logger.debug("Invalidated cached smoothed factors due to retrain")
        else:
            # Only update the state: continue the training filter over the new rows
            # (one code path for one and for many rows) and append their filtered states.
            result = self._ensure_result()
            Z_new = self._filter_new_rows(data_new)

            # Update model state
            result.Z = np.vstack([result.Z, Z_new])
            self.data_processed = np.vstack([self.data_processed, data_new])
            result.x_sm = result.Z @ result.C.T
            
            # Invalidate cached smoothed factors after update
            # Cached factors are now stale because result.Z was mutated
            self._cached_smoothed_factors = None
            self._result = result  # Update cached result with new Z
            _logger.debug("Invalidated cached smoothed factors after update() - result.Z was mutated")
    
    def predict(
        self,
        horizon: int,
        *,
        data: Optional[Union[np.ndarray, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int
            Number of periods ahead to forecast.
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame, optional
            New preprocessed observations (T_new x N). If None, uses latest training data.
            Data must be preprocessed but NOT standardized.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_forecast, Z_forecast) where:
            - X_forecast: (horizon x N) forecasted series
            - Z_forecast: (horizon x m) forecasted factors
        """
        # Validate model is trained
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() first"
        )
        
        self._check_parameters_initialized()
        result = self._ensure_result()
        
        check_condition(
            result.Z is not None,
            ModelNotTrainedError,
            "DFM prediction failed: result.Z is not available"
        )
        
        # Validate horizon
        from ...numeric.validator import validate_horizon
        horizon = validate_horizon(horizon)
        A = result.A
        C = result.C
        
        # Determine initial factor state
        if data is not None:
            # Validate and convert data
            from ...numeric.validator import validate_and_convert_update_data
            data_new = validate_and_convert_update_data(
                data,
                self.data_processed,
                dtype=DEFAULT_DTYPE,
                model_name=self.__class__.__name__
            )
            
            # Always apply scaler transform
            if self.scaler is not None:
                data_new = self.scaler.transform(data_new)
            
            # Continue the training filter over the new rows; forecast from x_T+n|T+n.
            Z_last = self._filter_new_rows(data_new)[-1]
        else:
            # Use latest training state
            fallback = np.zeros(result.A.shape[0], dtype=DEFAULT_DTYPE) if result.A is not None else None
            Z_last = self._get_last_state(result, fallback=fallback)
        
        # Validate
        from ...numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(Z_last, name="factor state Z_last")
        validate_no_nan_inf(A, name="transition matrix A")
        validate_no_nan_inf(C, name="observation matrix C")
        
        # Forecast factors forward
        from ...numeric.estimator import forecast_ar1_factors
        Z_forecast = forecast_ar1_factors(Z_last, A, horizon, dtype=DEFAULT_DTYPE)
        
        # Transform to series
        X_forecast_std = Z_forecast @ C.T
        
        # Log extreme predictions
        max_abs_value = np.abs(X_forecast_std).max()
        if max_abs_value > 5.0:
            _logger.warning(
                f"DFM predictions in standardized space are extreme (max abs: {max_abs_value:.2f}). "
                f"This indicates model instability or poor training."
            )
        
        # Unstandardize forecasts
        if self.scaler is not None:
            X_forecast = self.scaler.inverse_transform(X_forecast_std)
        else:
            X_forecast = X_forecast_std
        
        # Validate
        from ...config.types import to_numpy
        X_forecast = to_numpy(X_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(X_forecast, name="forecast X_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        
        return X_forecast, Z_forecast
