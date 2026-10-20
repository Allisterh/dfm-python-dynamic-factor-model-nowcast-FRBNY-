"""Identifiable Variational Dynamic Factor Model (iVDFM).

Implements the iVDFM framework that combines identifiable latent-variable modeling
with explicit stochastic dynamics. Identifiability is achieved by applying iVAE
conditions to the innovation process driving dynamics.

``iVDFMEngine`` is the private engine behind the public estimator
:class:`dfm_python.models.ivdfm.estimator.iVDFM`, which builds the engine's
dataset from ``PanelData`` and passes every hyperparameter explicitly. The
engine's code is the pre-redesign ``iVDFM`` class unchanged, apart from two
pure extractions from ``predict`` (:meth:`iVDFMEngine._future_time_features`
and :meth:`iVDFMEngine._forecast`, which ``predict`` calls), the private
config class ``_iVDFMEngineConfig`` (the old ``iVDFMConfig``),
:meth:`iVDFMEngine._from_saved`, which rebuilds a saved run without
``__init__`` (no dataset, no seeding), and these removals: ``set_dataset``
(pass ``dataset=`` to ``__init__``), the ``torch.save`` checkpoint methods
``save``/``load`` (the estimator saves the weights to ``weights.pt``), the
raw-data branch of ``update`` (it takes an ``iVDFMDataset`` only) and
``hyperparameter_optimization`` (see :func:`dfm_python.tuning.tune_hyperparameters`).
Without ``save``/``load`` the engine no longer inherits the old abstract base
``dfm_python.models.base.BaseFactorModel`` (since deleted); :func:`_init_engine_state` sets the
attributes that base's ``__init__`` set.
"""

import time
from typing import Optional, Any, Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ...logger import get_logger
from ...logger.progress import iVDFMTrainLogger
from ...config.constants import DEFAULT_LOSS_LOG_PRECISION
from ...config.types import to_numpy
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_SEED,
    DEFAULT_DTYPE,
    DEFAULT_TOLERANCE,
    DEFAULT_IVDFM_LATENT_DIM,
    DEFAULT_IVDFM_AUX_DIM,
    DEFAULT_REGULARIZATION,
)
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError, DataValidationError
from ...utils.seed import apply_seed
from ...utils.loss import compute_elbo_loss
from ...layer.pca import extract_pca_factors
from .encoder import iVDFMInnovationEncoder
from .decoder import iVDFMDecoder
from .prior import iVDFMPriorNetwork
from ...layer.regime import RegimeNet
from ...layer.mixing import build_factor_mixing
from ...ssm.companion import iVDFMCompanionSSM
from ...dataset.ivdfm_dataset import iVDFMDataset
from ...config.schema.model import _iVDFMEngineConfig
from ...config.schema.params import iVDFMModelState
from ...config.schema.results import iVDFMResult
from ...numeric.builder import build_ivdfm_optimizer

_logger = get_logger(__name__)


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class iVDFMEngine(nn.Module):
    """Identifiable Variational Dynamic Factor Model.
    
    Combines identifiable latent-variable modeling (iVAE) with explicit
    stochastic dynamics. Identifiability is achieved by applying conditional
    exponential-family priors to innovations rather than states.
    """
    
    def __init__(
        self,
        dataset: Optional["iVDFMDataset"] = None,
        data_dim: Optional[int] = None,
        num_factors: Optional[int] = None,  # Aligned with config: num_factors (not latent_dim)
        context_dim: Optional[int] = None,
        window: Optional[int] = None,
        config: Optional[_iVDFMEngineConfig] = None,
        # Network architecture (used if config is None)
        encoder_hidden_dim: Union[int, list] = 200,
        encoder_n_hidden_layers: int = 2,
        decoder_type: Optional[str] = None,
        decoder_hidden_dim: Union[int, list] = 200,
        decoder_n_hidden_layers: int = 2,
        prior_hidden_dim: Union[int, list] = 100,
        prior_n_hidden_layers: int = 1,
        activation: str = 'lrelu',
        slope: float = 0.1,
        # Dynamics parameters
        factor_order: int = 1,  # AR order for factors (p in AR(p))
        # Prior/innovation parameters
        innovation_distribution: str = 'laplace',  # 'laplace', 'student_t', etc.
        decoder_var: float = 0.01,
        # Training parameters
        # NOTE: these default to None so they don't override config values
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        # Device
        device: Optional[torch.device] = None,
        seed: int = DEFAULT_SEED,
        **kwargs  # Allow additional parameters to override config
    ):
        """Initialize iVDFM model.
        
        Parameters
        ----------
        dataset : Optional[iVDFMDataset]
            If provided, data_dim, context_dim, and window are inferred from it
            and fit() will use this dataset.
        data_dim : int
            Dimension of observed data (N in paper)
        num_factors : int
            Number of latent factors (r in paper)
        context_dim : int
            Dimension of context variable u_t
        window : int
            Length of sliding windows (T in paper)
        config : Optional[Any]
            Configuration object
        encoder_hidden_dim : Union[int, list]
            Hidden dimensions for innovation encoder
        encoder_n_hidden_layers : int
            Number of hidden layers in innovation encoder
        decoder_hidden_dim : Union[int, list]
            Hidden dimensions for decoder
        decoder_n_hidden_layers : int
            Number of hidden layers in decoder
        prior_hidden_dim : Union[int, list]
            Hidden dimensions for prior network
        prior_n_hidden_layers : int
            Number of hidden layers in prior network
        activation : str
            Activation function ('lrelu', 'relu', 'tanh')
        slope : float
            Slope for leaky ReLU
        factor_order : int
            AR order for factor dynamics (p in AR(p))
        innovation_distribution : str
            Distribution for innovations ('laplace', 'student_t', etc.)
        decoder_var : float
            Decoder variance (observation noise)
        learning_rate : float
            Learning rate for optimizer
        optimizer : str
            Optimizer type ('Adam', 'AdamW', 'SGD')
        batch_size : int
            Batch size for training
        max_epochs : int
            Maximum number of training epochs
        tolerance : float
            Convergence tolerance
        device : Optional[torch.device]
            Device for computation (None for auto-detect)
        seed : int
            Random seed
        """
        _init_engine_state(self)
        nn.Module.__init__(self)
        
        # Build config: start with provided config or defaults, then override with parameters and kwargs
        if config is not None:
            if isinstance(config, _iVDFMEngineConfig):
                from dataclasses import asdict
                config_dict = asdict(config)
            elif isinstance(config, dict):
                config_dict = config.copy()
            else:
                raise ConfigurationError(
                    f"config must be _iVDFMEngineConfig instance or dict, got {type(config)}"
                )
        else:
            config_dict = {}
        
        # Directly update config_dict with explicit parameters (aligned with config keys)
        # Only add if parameter is not None (to allow using config defaults)
        if num_factors is not None:
            config_dict['num_factors'] = num_factors
        if encoder_hidden_dim is not None:
            config_dict['encoder_hidden_dim'] = encoder_hidden_dim
        if encoder_n_hidden_layers is not None:
            config_dict['encoder_n_hidden_layers'] = encoder_n_hidden_layers
        if decoder_type is not None:
            config_dict['decoder_type'] = decoder_type
        if decoder_hidden_dim is not None:
            config_dict['decoder_hidden_dim'] = decoder_hidden_dim
        if decoder_n_hidden_layers is not None:
            config_dict['decoder_n_hidden_layers'] = decoder_n_hidden_layers
        if prior_hidden_dim is not None:
            config_dict['prior_hidden_dim'] = prior_hidden_dim
        if prior_n_hidden_layers is not None:
            config_dict['prior_n_hidden_layers'] = prior_n_hidden_layers
        if activation is not None:
            config_dict['activation'] = activation
        if slope is not None:
            config_dict['slope'] = slope
        if factor_order is not None:
            config_dict['factor_order'] = factor_order
        if innovation_distribution is not None:
            config_dict['innovation_distribution'] = innovation_distribution
        if decoder_var is not None:
            config_dict['decoder_var'] = decoder_var
        if learning_rate is not None:
            config_dict['learning_rate'] = learning_rate
        if optimizer is not None:
            config_dict['optimizer'] = optimizer
        if batch_size is not None:
            config_dict['batch_size'] = batch_size
        if max_epochs is not None:
            config_dict['max_epochs'] = max_epochs
        if tolerance is not None:
            config_dict['tolerance'] = tolerance
        if seed is not None:
            config_dict['seed'] = seed
        
        # Special handling for dimensions (can be None)
        # Only add to config_dict if explicitly provided (not None)
        if data_dim is not None:
            config_dict['data_dim'] = data_dim
        if context_dim is not None:
            config_dict['context_dim'] = context_dim
        # Note: if context_dim is None, don't add to config_dict - will use None directly
        if window is not None:
            config_dict['window'] = window
        
        #         # Override with kwargs (highest precedence)
        config_dict.update(kwargs)
        
        # If dataset is provided, infer dimensions and store for fit()
        if dataset is not None:
            self._dataset = dataset
            if data_dim is None:
                data_dim = dataset.data_dim
            if context_dim is None:
                context_dim = dataset.context_dim
            if window is None:
                window = dataset.window
        else:
            self._dataset = None
        
        # Remove None values to use defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Create config object (structural params: num_factors, factor_order, num_regimes, etc. are model selection, not tuning)
        self._config = _iVDFMEngineConfig.from_dict(config_dict) if config_dict else _iVDFMEngineConfig()
        
        # Extract all parameters from config (simplified, unified approach)
        self.data_dim = data_dim  # Can be None, inferred during fit
        # Keep latent_dim as internal attribute name (more intuitive), but use num_factors from config
        self.latent_dim = self._config.num_factors or DEFAULT_IVDFM_LATENT_DIM
        # context_dim: preserve None if not provided
        # Note: context_dim is NOT in _iVDFMEngineConfig schema (replaced by time_context + context columns).
        # context_dim is inferred from dataset during fit() based on time_context + custom context columns.
        # If user explicitly passes context_dim parameter, use it; otherwise None (will be inferred from dataset).
        self.context_dim = context_dim
        # time_context: dimension of time-based features (separate from custom context columns)
        self.time_context = self._get_config_attr('time_context', 1)
        # window: if None, use full T (full series as one sequence)
        # Priority: explicit parameter > config value > None (which dataset will convert to T)
        if window is not None:
            self.window = window
        elif self._config is not None and hasattr(self._config, 'window'):
            self.window = self._config.window
        else:
            self.window = None  # Will be converted to T by dataset
        # Extract config attributes (network architecture)
        self.encoder_hidden_dim = self._config.encoder_hidden_dim
        self.encoder_n_hidden_layers = self._config.encoder_n_hidden_layers
        self.decoder_type = getattr(self._config, "decoder_type", "mlp")
        self.decoder_hidden_dim = self._config.decoder_hidden_dim
        self.decoder_n_hidden_layers = self._config.decoder_n_hidden_layers
        self.prior_hidden_dim = self._config.prior_hidden_dim
        self.prior_n_hidden_layers = self._config.prior_n_hidden_layers
        self.activation = self._config.activation
        self.slope = self._config.slope
        
        # Extract config attributes (dynamics and distribution)
        self.factor_order = self._config.factor_order
        self.innovation_distribution = self._config.innovation_distribution
        self.decoder_var = self._config.decoder_var
        self.num_regimes = self._get_config_attr('num_regimes', 1)
        self.regime_temperature = self._get_config_attr('regime_temperature', 1.0)
        self.beta_kl = self._get_config_attr('beta_kl', 1.0)
        self.use_layer_norm = self._get_config_attr('use_layer_norm', False)
        self.mixing = self._get_config_attr('mixing', False)
        
        # Extract config attributes (training)
        self.learning_rate = self._config.learning_rate
        self.optimizer_type = self._config.optimizer
        self.optimizer_weight_decay = self._config.optimizer_weight_decay
        self.optimizer_momentum = self._config.optimizer_momentum
        self.batch_size = self._config.batch_size
        self.max_epochs = self._config.max_epochs
        self.tolerance = self._config.tolerance
        self.patience = self._get_config_attr('patience', None)  # Early stopping patience (None = disabled)
        
        # Extract config attributes (scheduler)
        self.scheduler_type = self._config.scheduler_type
        self.scheduler_step_size = self._config.scheduler_step_size
        self.scheduler_gamma = self._config.scheduler_gamma
        self.scheduler_patience = self._config.scheduler_patience
        self.scheduler_factor = self._config.scheduler_factor
        self.scheduler_min_lr = self._config.scheduler_min_lr
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Random seed
        self.seed = seed
        if seed is not None:
            apply_seed(seed, torch_global=True, numpy_global=True)
        
        # Initialize components (only if dimensions are available)
        # If data_dim or context_dim is None, components will be built during fit
        if self.data_dim is not None and self.context_dim is not None:
            self._build_components()
        else:
            # Components will be built during fit when dimensions are known
            self.innovation_encoder = None
            self.prior_network = None
            self.prior_networks = None
            self.decoder = None
            self.ssm = None
        
        # Optimizer (built during training)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Training state
        self.training_state: Optional[Dict] = None
        self.factors: Optional[np.ndarray] = None
        self.innovations: Optional[np.ndarray] = None
        # _dataset set above when dataset= is passed; otherwise None (fit needs dataset=)

        # Move to device
        self.to(self.device)
    
    @classmethod
    def _from_saved(cls, params: Dict[str, Any], dims: Dict[str, Any], device: Any) -> 'iVDFMEngine':
        """Rebuild an engine without ``__init__``: no dataset, no seeding.
        
        ``params`` holds the keyword arguments the estimator passes to
        ``__init__`` (``num_factors``, ``window``, the architecture, training
        and scheduler values, ``tolerance``, ``seed``, ``stride``,
        ``time_context``, ``scaler``, ``num_regimes``, ...). ``_config`` is
        built from them as ``__init__`` builds it, except that None values are
        kept (the estimator sets them on a fitted engine too). ``dims`` holds
        the values training sets: ``data_dim``, ``context_dim``,
        ``time_context`` (after fit, the width of u_t) and ``window``. Every
        attribute ``__init__`` sets is set; the components are built on
        ``device`` inside ``torch.random.fork_rng`` (their builders seed and
        draw from the global torch RNG, which is restored afterwards). The
        engine has no optimizer, scheduler or training state and is in eval
        mode, as after ``fit``; load the saved weights over the components.
        """
        import dataclasses

        device = torch.device(device)
        config_dict = {k: v for k, v in params.items() if v is not None}
        config = _iVDFMEngineConfig.from_dict(config_dict) if config_dict else _iVDFMEngineConfig()
        fields = {f.name for f in dataclasses.fields(config)}
        dropped = {k: v for k, v in params.items() if v is None and k in fields}
        if dropped:
            config = dataclasses.replace(config, **dropped)

        self = cls.__new__(cls)
        _init_engine_state(self)
        nn.Module.__init__(self)
        self._config = config
        self._dataset = None
        self.data_dim = dims["data_dim"]
        self.latent_dim = config.num_factors or DEFAULT_IVDFM_LATENT_DIM
        self.context_dim = dims["context_dim"]
        self.time_context = dims["time_context"]
        self.window = dims["window"]
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encoder_n_hidden_layers = config.encoder_n_hidden_layers
        self.decoder_type = getattr(config, "decoder_type", "mlp")
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.decoder_n_hidden_layers = config.decoder_n_hidden_layers
        self.prior_hidden_dim = config.prior_hidden_dim
        self.prior_n_hidden_layers = config.prior_n_hidden_layers
        self.activation = config.activation
        self.slope = config.slope
        self.factor_order = config.factor_order
        self.innovation_distribution = config.innovation_distribution
        self.decoder_var = config.decoder_var
        self.num_regimes = self._get_config_attr('num_regimes', 1)
        self.regime_temperature = self._get_config_attr('regime_temperature', 1.0)
        self.beta_kl = self._get_config_attr('beta_kl', 1.0)
        self.use_layer_norm = self._get_config_attr('use_layer_norm', False)
        self.mixing = self._get_config_attr('mixing', False)
        self.learning_rate = config.learning_rate
        self.optimizer_type = config.optimizer
        self.optimizer_weight_decay = config.optimizer_weight_decay
        self.optimizer_momentum = config.optimizer_momentum
        self.batch_size = config.batch_size
        self.max_epochs = config.max_epochs
        self.tolerance = config.tolerance
        self.patience = self._get_config_attr('patience', None)
        self.scheduler_type = config.scheduler_type
        self.scheduler_step_size = config.scheduler_step_size
        self.scheduler_gamma = config.scheduler_gamma
        self.scheduler_patience = config.scheduler_patience
        self.scheduler_factor = config.scheduler_factor
        self.scheduler_min_lr = config.scheduler_min_lr
        self.device = device
        self.seed = params.get("seed")

        # The builders call torch.manual_seed (all devices), so every CUDA generator is forked too.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self._build_components()
        self.to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.training_state = None
        self.factors = None
        self.innovations = None
        self.regime_weights = None
        self._num_iter = 0
        self.loss_now = None
        self._elbo = None
        self._converged = False
        self.eval()
        return self

    def _get_config_attr(self, attr: str, default: Any = None) -> Any:
        """Helper to safely get config attribute with default."""
        return getattr(self._config, attr, default) if self._config is not None else default
    
    
    def _set_random_f0(self) -> None:
        """Set f0 to random values (fallback for initialization failures)."""
        with torch.no_grad():
            self.ssm.f0.data = torch.randn(self.latent_dim, device=self.device) * 0.1
    
    def _normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """Normalize f0 vector: pad if needed, apply sign convention.
        
        Parameters
        ----------
        f0 : np.ndarray
            Raw f0 vector (may be shorter than latent_dim)
            
        Returns
        -------
        np.ndarray
            Normalized f0 vector (latent_dim,)
        """
        # Pad if needed
        if f0.shape[0] < self.latent_dim:
            f0 = np.pad(f0, (0, self.latent_dim - f0.shape[0]))
        
        # Deterministic sign convention for reproducibility
        if np.sum(f0) < 0:
            f0 = -f0
        
        return f0
    
    def _normalize_factors_shape(self, factors: np.ndarray) -> np.ndarray:
        """Normalize factors shape to 2D (T, r) by averaging over batch dimension if needed.
        
        Parameters
        ----------
        factors : np.ndarray
            Factors array, shape (batch, T, r) or (T, r)
            
        Returns
        -------
        np.ndarray
            Normalized factors, shape (T, r)
        """
        if factors.ndim == 3:
            return np.mean(factors, axis=0)
        return factors
    
    def _compute_full_state(self, z_np: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Compute full augmented state for companion form.
        
        For p=1: full_state = factors (T, r)
        For p>1: full_state = augmented state (T, r*p) with lags
        
        Parameters
        ----------
        z_np : Optional[np.ndarray]
            Factors array, shape (T, r)
            
        Returns
        -------
        Optional[np.ndarray]
            Full state array, shape (T, r) for p=1 or (T, r*p) for p>1
        """
        if z_np is None or not isinstance(z_np, np.ndarray):
            return None
        
        T, r = z_np.shape
        if self.factor_order == 1:
            return z_np  # (T, r)
        
        # Construct augmented state: s_t[i*p : (i+1)*p] = [f_t[i], f_{t-1}[i], ..., f_{t-p+1}[i]]
        full_state = np.zeros((T, r * self.factor_order), dtype=z_np.dtype)
        f0_np = self.ssm.f0.data.cpu().numpy() if hasattr(self.ssm.f0, 'data') else self.ssm.f0
        
        for t in range(T):
            for i in range(r):
                for lag in range(self.factor_order):
                    idx = t - lag
                    if idx >= 0:
                        full_state[t, i * self.factor_order + lag] = z_np[idx, i]
                    else:
                        # Use f0 for negative indices
                        if f0_np is not None and i < len(f0_np):
                            full_state[t, i * self.factor_order + lag] = f0_np[i]
        
        return full_state
    
    def _set_ssm_ar_params(self, ar_coeffs: np.ndarray, B_diag: np.ndarray) -> None:
        """Set SSM AR parameters from numpy arrays. When K>1, set A_diag (K,p,r), B_diag (K,r) from OLS (regime 0) and copy+noise for others."""
        with torch.no_grad():
            if self.factor_order == 1:
                A_flat = ar_coeffs[:, 0] if ar_coeffs.ndim > 1 else ar_coeffs
                self.ssm.A.data = torch.tensor(A_flat, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            else:
                self.ssm.ar_coeffs.data = torch.tensor(
                    ar_coeffs, dtype=DEFAULT_TORCH_DTYPE, device=self.device
                )
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            if self.A_diag is not None and self.B_diag is not None:
                K = self.A_diag.shape[0]
                p, r = self.factor_order, self.latent_dim
                # ar_coeffs from OLS: (r, p); B_diag (r,). Store regime 0 then copy+noise.
                ar_t = torch.tensor(ar_coeffs, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                if ar_t.dim() == 1:
                    ar_t = ar_t.unsqueeze(0).expand(r, -1)
                B_t = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                self.A_diag.data[0] = ar_t.permute(1, 0)  # (p, r)
                self.B_diag.data[0] = B_t
                for k in range(1, K):
                    self.A_diag.data[k] = self.A_diag.data[0].clone() + torch.randn_like(self.A_diag.data[0], device=self.device) * 0.02
                    self.B_diag.data[k] = self.B_diag.data[0].clone() + torch.randn_like(self.B_diag.data[0], device=self.device) * 0.02
    
    def _build_components(self):
        """Build model components: encoder, decoder, prior, SSM."""
        # Validate dimensions are available
        if self.data_dim is None or self.context_dim is None:
            raise ModelNotInitializedError(
                "Cannot build components: data_dim and context_dim must be set. "
                "Call fit() first to infer dimensions from data."
            )
        
        # Innovation encoder: q(η_t | y_t, u_t)
        self.innovation_encoder = iVDFMInnovationEncoder(
            data_dim=self.data_dim,
            latent_dim=self.latent_dim,
            aux_dim=self.context_dim,
            hidden_dim=self.encoder_hidden_dim,
            n_hidden_layers=self.encoder_n_hidden_layers,
            activation=self.activation,
            slope=self.slope,
            use_layer_norm=self.use_layer_norm,
            device=self.device,
            seed=self.seed,
        )
        
        K = self.num_regimes
        # Prior: p(η_t | u_t). K=1 single network; K>1 regime-indexed priors p_k(η_t | u_t)
        if K == 1:
            self.prior_network = iVDFMPriorNetwork(
                aux_dim=self.context_dim,
                latent_dim=self.latent_dim,
                hidden_dim=self.prior_hidden_dim,
                n_hidden_layers=self.prior_n_hidden_layers,
                activation=self.activation,
                slope=self.slope,
                innovation_distribution=self.innovation_distribution,
                device=self.device,
                seed=self.seed,
            )
            self.prior_networks = None
        else:
            self.prior_network = None
            self.prior_networks = nn.ModuleList([
                iVDFMPriorNetwork(
                    aux_dim=self.context_dim,
                    latent_dim=self.latent_dim,
                    hidden_dim=self.prior_hidden_dim,
                    n_hidden_layers=self.prior_n_hidden_layers,
                    activation=self.activation,
                    slope=self.slope,
                    innovation_distribution=self.innovation_distribution,
                    device=self.device,
                    seed=self.seed,
                )
                for _ in range(K)
            ])

        # Decoder(s)
        if K == 1:
            self.regime_net = None
            self.decoders = None
            self.decoder = iVDFMDecoder(
                latent_dim=self.latent_dim,
                data_dim=self.data_dim,
                decoder_type=self.decoder_type,
                hidden_dim=self.decoder_hidden_dim,
                n_hidden_layers=self.decoder_n_hidden_layers,
                activation=self.activation,
                slope=self.slope,
                use_layer_norm=self.use_layer_norm,
                decoder_var=self.decoder_var,
                device=self.device,
                seed=self.seed,
            )
        else:
            self.regime_net = RegimeNet(
                input_dim=self.context_dim,
                num_regimes=K,
                hidden_dim=32,
                n_hidden_layers=1,
                activation=self.activation,
                slope=self.slope,
                device=self.device,
                seed=self.seed,
            )
            self.decoder = None
            self.decoders = nn.ModuleList([
                iVDFMDecoder(
                    latent_dim=self.latent_dim,
                    data_dim=self.data_dim,
                    decoder_type=self.decoder_type,
                    hidden_dim=self.decoder_hidden_dim,
                    n_hidden_layers=self.decoder_n_hidden_layers,
                    activation=self.activation,
                    slope=self.slope,
                    use_layer_norm=self.use_layer_norm,
                    decoder_var=self.decoder_var,
                    device=self.device,
                    seed=self.seed,
                )
                for _ in range(K)
            ])

        # Post-factor mixing: optional global only (mixing=True -> one M, identity init; mixing=False -> none).
        self.factor_mixing = (
            build_factor_mixing(self.latent_dim, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            if self.mixing
            else None
        )

        # SSM: maps innovations to factors via deterministic dynamics
        self.ssm = iVDFMCompanionSSM(
            latent_dim=self.latent_dim,
            factor_order=self.factor_order,
            device=self.device,
        )

        # Regime-indexed AR(p): A_diag (K, p, r), B_diag (K, r). Effective per window: A_eff = Σ_k π_k A_diag[k], B_eff = Σ_k π_k B_diag[k].
        if K > 1:
            p = self.factor_order
            init_scale = 0.1
            self.A_diag = nn.Parameter(torch.randn(K, p, self.latent_dim, device=self.device) * init_scale)
            self.B_diag = nn.Parameter(torch.randn(K, self.latent_dim, device=self.device) * init_scale)
        else:
            self.A_diag = None
            self.B_diag = None

    def _effective_regime_coeffs(self, pi_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Effective AR(p) coefficients per window: A_eff = Σ_k π_k A_diag[k], B_eff = Σ_k π_k B_diag[k].
        
        pi_w : (batch, K)
        Returns A_eff (batch, r, p), B_eff (batch, r).
        """
        # A_diag (K, p, r) -> (batch, K, p, r); pi_w (batch, K, 1, 1); sum over K -> (batch, p, r) -> (batch, r, p)
        A_eff = (pi_w.unsqueeze(2).unsqueeze(3) * self.A_diag.unsqueeze(0)).sum(dim=1)  # (batch, p, r)
        A_eff = A_eff.permute(0, 2, 1)  # (batch, r, p)
        B_eff = (pi_w.unsqueeze(2) * self.B_diag.unsqueeze(0)).sum(dim=1)  # (batch, r)
        return A_eff, B_eff

    def _decode(
        self,
        factors: torch.Tensor,
        u_1T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode factors to observations. Optional global mixing f -> M f when mixing=True; then K=1 single decoder or K>1 π-weighted decoders."""
        squeeze_out = factors.dim() == 2
        if squeeze_out:
            factors = factors.unsqueeze(0)
        mixed = self.factor_mixing(factors) if self.factor_mixing is not None else factors
        if self.num_regimes == 1:
            out = self.decoder(mixed)
        else:
            if u_1T is not None:
                if u_1T.dim() == 2:
                    u_1T = u_1T.unsqueeze(0)
                pi = self.regime_net(u_1T, temperature=self.regime_temperature)
            else:
                batch, T, _ = mixed.shape
                pi = torch.full(
                    (batch, T, self.num_regimes),
                    1.0 / self.num_regimes,
                    device=mixed.device,
                    dtype=mixed.dtype,
                )
            out = torch.zeros(
                mixed.shape[0],
                mixed.shape[1],
                self.data_dim,
                device=mixed.device,
                dtype=mixed.dtype,
            )
            for k in range(self.num_regimes):
                out = out + pi[:, :, k : k + 1] * self.decoders[k](mixed)
        if squeeze_out:
            out = out.squeeze(0)
        return out

    def _sample_prior(self, u: torch.Tensor) -> torch.Tensor:
        """Sample innovations from prior p(η|u). K=1: single prior; K>1: sample r~π then η~p_r(·|u)."""
        if self.num_regimes == 1:
            return self.prior_network.sample(u)
        # K>1: π = regime_net(u), for each t sample r_t ~ Cat(π_t), η_t ~ p_{r_t}(u_t)
        pi = self.regime_net(u, temperature=self.regime_temperature)  # (batch, T, K)
        batch, T, _ = u.shape
        eta_list = []
        for t in range(T):
            pi_t = pi[:, t, :]  # (batch, K)
            r = torch.multinomial(pi_t + 1e-8, 1).squeeze(-1)  # (batch,)
            eta_t = torch.zeros(batch, self.latent_dim, device=u.device, dtype=u.dtype)
            for k in range(self.num_regimes):
                mask = r == k
                if mask.any():
                    u_t = u[mask, t : t + 1, :]  # (n, 1, context_dim)
                    eta_t[mask] = self.prior_networks[k].sample(u_t).squeeze(1)
            eta_list.append(eta_t)
        return torch.stack(eta_list, dim=1)  # (batch, T, r)

    def _initialize_f0_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Initialize f_0 (initial factor state) using single-window PCA on data.
        
        Parameters
        ----------
        dataset : iVDFMDataset
            Dataset containing training data
        """
        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total
        if T_init < 2:
            self._set_random_f0()
            _logger.warning(f"Insufficient data for PCA init (T={T_init}). Using random f_0.")
            return

        max_components = min(self.data_dim, T_init, self.latent_dim)
        f0_mean = None
        
        try:
            # Single window PCA
            y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
            f_init, _ = extract_pca_factors(y_win, n_components=max_components)
            f0_mean = np.mean(f_init, axis=0)  # (max_components,)
                
        except Exception as e:
            self._set_random_f0()
            _logger.warning(f"PCA init failed: {e}. Using random f_0.")
            return

        # Normalize f0 (pad, sign convention)
        if f0_mean is None:
            f0_mean = np.zeros(self.latent_dim, dtype=DEFAULT_DTYPE)
        f0_mean = self._normalize_f0(f0_mean)

        # Set f0
        with torch.no_grad():
            self.ssm.f0.data = torch.tensor(f0_mean, dtype=DEFAULT_TORCH_DTYPE, device=self.device)

        _logger.info(
            f"Initialized f_0 using PCA. "
            f"f_0 range: [{f0_mean.min():.4f}, {f0_mean.max():.4f}]"
        )
    
    def _initialize_ar_coeffs_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Initialize AR coefficients using OLS estimation from PCA factors.
        
        Extracts PCA factors from data, estimates AR(p) coefficients for each factor,
        and initializes SSM parameters.

        The initialization window has min(data_dim, T_init, latent_dim) PCA
        factors. With ``factor_order > 1``, the factors beyond those keep the
        SSM's own (non-data) initialization of their AR coefficients and
        innovation scale; an OLS fit on all-zero padded factors would give
        all-zero coefficient rows. With ``factor_order == 1`` the padded
        factors get the OLS values of a zero series (A = 0, a stable AR(1)
        after the sigmoid), as before.
        
        Parameters
        ----------
        dataset : iVDFMDataset
            Dataset containing training data
        """
        from ...numeric.estimator import estimate_arp_ols
        
        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total
        
        if T_init < self.factor_order + 1:
            _logger.warning(f"Insufficient data for AR init (T={T_init}, order={self.factor_order}). Skipping.")
            return
        
        try:
            # Extract data window and compute PCA factors
            y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
            max_components = min(self.data_dim, T_init, self.latent_dim)
            factors, _ = extract_pca_factors(y_win, n_components=max_components)
            n_data = factors.shape[1]
            keep_default = self.factor_order > 1 and n_data < self.latent_dim

            # Pad if needed (factor_order == 1 only; see the docstring)
            if factors.shape[1] < self.latent_dim and not keep_default:
                factors = np.pad(factors, ((0, 0), (0, self.latent_dim - factors.shape[1])))
            
            # Estimate AR(p) coefficients using OLS
            ar_coeffs, B_diag = estimate_arp_ols(
                factors=factors,
                order=self.factor_order,
                regularization=DEFAULT_REGULARIZATION,
                dtype=DEFAULT_DTYPE
            )

            if keep_default:
                # The factors without a PCA factor keep the SSM's non-data initialization.
                ar_default = self.ssm.ar_coeffs.detach().cpu().numpy().astype(ar_coeffs.dtype)
                B_default = self.ssm.B.detach().cpu().numpy().astype(B_diag.dtype)
                ar_coeffs = np.concatenate([ar_coeffs, ar_default[n_data:]], axis=0)
                B_diag = np.concatenate([B_diag, B_default[n_data:]], axis=0)
                _logger.info(
                    f"AR({self.factor_order}) initialization: {n_data} PCA factors for {self.latent_dim} "
                    f"factors; factors {n_data}..{self.latent_dim - 1} keep the default initialization."
                )
            
            # Initialize SSM parameters
            self._set_ssm_ar_params(ar_coeffs, B_diag)
            
            _logger.info(
                f"Initialized AR({self.factor_order}) coefficients using OLS from PCA factors. "
                f"AR coeffs range: [{ar_coeffs.min():.4f}, {ar_coeffs.max():.4f}], "
                f"B range: [{B_diag.min():.4f}, {B_diag.max():.4f}]"
            )
        except Exception as e:
            _logger.warning(f"AR coefficient initialization failed: {e}. Using random initialization.")
    
    def _build_optimizer(self):
        """Build optimizer and scheduler using builder utility."""
        self.optimizer, self.scheduler = build_ivdfm_optimizer(
            model=self,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            max_epochs=self.max_epochs,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_momentum=self.optimizer_momentum,
            scheduler_type=self.scheduler_type,
            scheduler_step_size=self.scheduler_step_size,
            scheduler_gamma=self.scheduler_gamma,
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            scheduler_min_lr=self.scheduler_min_lr,
        )
    
    
    def forward(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through iVDFM.
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Context variable sequence, shape (batch, T, context_dim)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - y_pred: predicted observations, shape (batch, T, N)
            - eta: innovations, shape (batch, T, r)
            - factors: latent factors, shape (batch, T, r)
            - encoder_params: encoder parameters (list of dicts)
            - prior_params: prior parameters (list of dicts)
        """
        batch_size, T, _ = y_1T.shape
        
        # Encode innovations: q(η_t | y_t, u_t) for all time steps
        # Encoder processes (batch, T, N) and (batch, T, context_dim) directly
        mu_all, logvar_all = self.innovation_encoder.forward(y_1T, u_1T)
        # mu_all, logvar_all: (batch, T, r)
        
        # Sample innovations using reparameterization trick
        std_all = torch.exp(0.5 * logvar_all)
        eps = torch.randn_like(mu_all)
        eta_1T = mu_all + eps * std_all  # (batch, T, r)
        
        # Prior parameters: p(η_t | u_t) or mixture Σ_k π_{t,k} p_k(η_t | u_t)
        encoder_params_list = [
            {'mu': mu_all[:, t, :], 'logvar': logvar_all[:, t, :]}
            for t in range(T)
        ]
        if self.num_regimes == 1:
            prior_params_all = self.prior_network(u_1T)
            prior_keys = list(prior_params_all.keys())
            prior_params_list = [
                {key: prior_params_all[key][:, t, :] for key in prior_keys}
                for t in range(T)
            ]
            regime_weights = None
            prior_params_per_regime = None
        else:
            pi = self.regime_net(u_1T, temperature=self.regime_temperature)  # (batch, T, K)
            prior_params_per_k = [self.prior_networks[k](u_1T) for k in range(self.num_regimes)]
            prior_keys = list(prior_params_per_k[0].keys())
            prior_params_per_regime = [
                [{key: prior_params_per_k[k][key][:, t, :] for key in prior_keys} for t in range(T)]
                for k in range(self.num_regimes)
            ]
            prior_params_list = prior_params_per_regime[0]  # fallback for any path that only reads prior_params
            regime_weights = pi

        # Compute factors: K=1 use global SSM; K>1 use effective AR(p) per window (π at origin)
        if self.num_regimes > 1 and self.A_diag is not None:
            pi_w = regime_weights[:, 0, :]  # (batch, K) at window origin
            A_eff, B_eff = self._effective_regime_coeffs(pi_w)  # (batch, r, p), (batch, r)
            group_id = pi_w.argmax(dim=-1)  # (batch,) dominant regime per window for grouped Krylov
            factors_1T = self.ssm.forward_with_coeffs(eta_1T, None, A_eff, B_eff, group_id=group_id)
        else:
            factors_1T = self.ssm.forward(eta_1T)  # (batch, T, r)

        # Decode observations (mixture of decoders when K > 1)
        y_pred = self._decode(factors_1T, u_1T)
        
        out = {
            'y_pred': y_pred,
            'eta': eta_1T,
            'factors': factors_1T,
            'encoder_params': encoder_params_list,
            'prior_params': prior_params_list,
        }
        if regime_weights is not None:
            out['regime_weights'] = regime_weights
            out['prior_params_per_regime'] = prior_params_per_regime
        return out
    
    def elbo(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor,
        N: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Evidence Lower Bound (ELBO).
        
        ELBO = E[log p(y_t | f_t)] - Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Context variable sequence, shape (batch, T, context_dim)
        N : int
            Total number of samples in dataset (for TC computation, currently unused)
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss (ELBO) and dictionary of component losses
        """
        # Single forward pass
        outputs = self.forward(y_1T, u_1T)
        y_pred = outputs['y_pred']
        encoder_params = outputs['encoder_params']
        prior_params = outputs['prior_params']
        regime_weights = outputs.get('regime_weights')
        prior_params_per_regime = outputs.get('prior_params_per_regime')

        # ELBO: KL = Σ_t Σ_k π_{t,k} KL(q(η_t) || p_k(η_t|u_t)) when K>1
        elbo, loss_dict = compute_elbo_loss(
            y_true=y_1T,
            y_pred=y_pred,
            encoder_params=encoder_params,
            prior_params=prior_params,
            innovation_distribution=self.innovation_distribution,
            decoder_variance=self.decoder_var,
            beta_kl=self.beta_kl,
            reduction='mean',
            regime_weights=regime_weights,
            prior_params_per_regime=prior_params_per_regime,
        )

        return elbo, loss_dict

    def fit(self, *args, **kwargs) -> 'iVDFMEngine':
        """Fit on the dataset passed to ``__init__`` as ``dataset=``."""
        dataset = self._dataset
        if dataset is None or not isinstance(dataset, iVDFMDataset):
            raise ModelNotTrainedError(
                "fit() requires a dataset. Pass an iVDFMDataset to the engine as dataset=."
            )

        actual_data_dim = dataset.target_length
        actual_context_dim = dataset.context_length
        self.time_context = actual_context_dim
        
        # Update dimensions
        need_rebuild = False
        if self.data_dim is None:
            self.data_dim = actual_data_dim
            need_rebuild = True
        elif self.data_dim != actual_data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, dataset has {actual_data_dim}",
                details="If data_dim was set in __init__, it must match the observation dimension after extracting context columns"
            )
        
        if self.context_dim is None:
            # Infer from dataset (this is the final dimension: time_context + custom context columns)
            self.context_dim = actual_context_dim
            need_rebuild = True
        elif self.context_dim != actual_context_dim:
            # If context_dim was explicitly set, it must match
            # But if dataset inferred a different dimension from context columns, use dataset's value
            _logger.warning(
                f"context_dim mismatch: model expects {self.context_dim}, dataset has {actual_context_dim}. "
                f"Using dataset's inferred dimension {actual_context_dim}."
            )
            self.context_dim = actual_context_dim
            need_rebuild = True
        
        # Rebuild components if dimensions were updated
        if need_rebuild:
            self._build_components()
        
        # Initialize f_0 (initial factor state) using PCA on initial data
        self._initialize_f0_from_data(dataset)
        
        # Initialize AR coefficients using OLS from data (always enabled)
        self._initialize_ar_coeffs_from_data(dataset)
        
        # Build optimizer
        self._build_optimizer()
        
        # When dataset is pre-loaded to GPU and fits in one batch, train on single batch (no DataLoader)
        # This matches the old "raw data" path: one bulk tensor, one batch per epoch, minimal overhead.
        all_tensors = dataset.get_all_tensors_on_device() if hasattr(dataset, "get_all_tensors_on_device") else None
        use_single_batch = (
            all_tensors is not None
            and len(dataset) <= self.batch_size
        )
        if use_single_batch:
            y_all, u_all = all_tensors
            dataloader = None
        else:
            dataloader = dataset.get_dataloader(
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
        
        # Initialize training logger
        train_logger = iVDFMTrainLogger(verbose=True)
        train_logger.start(
            config={
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_type,
            },
            data_info={
                'num_sequences': len(dataset),
                'data_dim': self.data_dim,
                'context_dim': self.context_dim,
                'latent_dim': self.latent_dim,
            }
        )
        
        # Training loop
        self.train()
        N_total = len(dataset)  # Total number of sequences
        
        self._num_iter = 0
        self.loss_now = None
        self._elbo = None
        self._converged = False
        
        # Early stopping setup
        best_elbo = float('inf')  # Track best ELBO (lower is better, since it's a loss to minimize)
        epochs_without_improvement = 0
        best_model_state = None
        
        early_stop_enabled = self.patience is not None and self.patience > 0
        patience_str = f"early stopping patience={self.patience}" if early_stop_enabled else "early stopping disabled"
        batches_per_epoch = 1 if use_single_batch else len(dataloader)
        _logger.info(f"Starting training: {self.max_epochs} epochs, {batches_per_epoch} batches/epoch, {patience_str}")
        
        start_time = time.time()
        t_fwd, t_bwd, t_clip, t_step, t_data = 0.0, 0.0, 0.0, 0.0, 0.0
        _device = next(self.parameters()).device
        _sync = lambda: torch.cuda.synchronize() if _device.type == "cuda" else None
        t_data_start = time.perf_counter()
        
        for epoch in range(self.max_epochs):
            epoch_recon_losses = []
            epoch_kl_losses = []
            epoch_elbos = []
            
            if use_single_batch:
                batch_iter = [(y_all, u_all)]
            else:
                batch_iter = dataloader
            
            for batch_idx, (y_batch, u_batch) in enumerate(batch_iter):
                _sync()
                t_data += time.perf_counter() - t_data_start
                self.optimizer.zero_grad()
                
                # Forward pass
                if y_batch.is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                elbo, loss_dict = self.elbo(y_batch, u_batch, N_total)
                _sync()
                t_fwd += time.perf_counter() - t0
                
                # Check for NaN/Inf
                if torch.isnan(elbo) or torch.isinf(elbo):
                    _logger.error(
                        f"NaN/Inf detected in ELBO at epoch {epoch}, batch {batch_idx}. "
                        f"ELBO={elbo.item()}, Recon={loss_dict['reconstruction'].item()}, "
                        f"KL={loss_dict['kl'].item()}"
                    )
                    raise ValueError("Training failed: NaN/Inf in loss")
                
                # Backward pass
                t0 = time.perf_counter()
                elbo.backward()
                _sync()
                t_bwd += time.perf_counter() - t0
                
                # Gradient clipping for stability
                t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                _sync()
                t_clip += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                self.optimizer.step()
                _sync()
                t_step += time.perf_counter() - t0
                
                # Store losses (ensure KL is non-negative)
                recon_val = loss_dict['reconstruction'].item()
                kl_val = loss_dict['kl'].item()
                elbo_val = elbo.item()
                
                # KL should never be negative; fail fast if it happens.
                if kl_val < 0:
                    raise ValueError(
                        f"Training failed: negative KL at epoch {epoch}, batch {batch_idx}: "
                        f"KL={kl_val:.6f} (Recon={recon_val:.6f}, ELBO={elbo_val:.6f})"
                    )
                
                epoch_recon_losses.append(recon_val)
                epoch_kl_losses.append(kl_val)
                epoch_elbos.append(elbo_val)
                t_data_start = time.perf_counter() if not use_single_batch else time.perf_counter()
            
            # Update state
            self._num_iter = epoch + 1
            self.loss_now = np.mean(epoch_recon_losses) if epoch_recon_losses else None
            kl_loss_mean = np.mean(epoch_kl_losses) if epoch_kl_losses else None
            self._elbo = np.mean(epoch_elbos) if epoch_elbos else None
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler (different schedulers need different inputs)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self._elbo if self._elbo is not None else float('inf'))
                else:
                    self.scheduler.step()
            
            # Logging with detailed information
            elapsed_time = time.time() - start_time
            # Prepare log kwargs
            log_kwargs = {
                'learning_rate': current_lr,
                'time_elapsed': f"{elapsed_time:.2f}s"
            }
            
            train_logger.log_epoch(
                epoch=epoch + 1,
                elbo=self._elbo,
                recon_loss=self.loss_now,
                kl_loss=kl_loss_mean,
                **log_kwargs
            )
            
            # Progress indicator every 10 epochs (redundant with train_logger, but provides percentage)
            # Only log if train_logger is not verbose or we want periodic progress updates
            if (epoch + 1) % 10 == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.max_epochs
                _logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"Recon: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"KL: {kl_loss_mean:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Early stopping: track best ELBO and check for improvement
            if early_stop_enabled and self._elbo is not None:
                # ELBO is a loss to minimize, so lower is better
                if self._elbo < best_elbo:
                    best_elbo = self._elbo
                    epochs_without_improvement = 0
                    # Save best model state (deep copy)
                    import copy
                    best_model_state = {
                        'model_state_dict': copy.deepcopy(self.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
                        'epoch': epoch + 1,
                        'elbo': self._elbo,
                    }
                    _logger.debug(f"New best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} at epoch {epoch + 1}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        _logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}: "
                            f"no improvement for {epochs_without_improvement} epochs "
                            f"(best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f}, "
                            f"current ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f})"
                        )
                        # Restore best model state
                        if best_model_state is not None:
                            self.load_state_dict(best_model_state['model_state_dict'])
                            if best_model_state['optimizer_state_dict'] is not None and self.optimizer is not None:
                                self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                            self._elbo = best_model_state['elbo']
                            self._num_iter = best_model_state['epoch']
                            _logger.info(f"Restored best model from epoch {best_model_state['epoch']}")
                        
                        self._converged = True
                        train_logger.log_convergence(
                            converged=True,
                            num_epochs=epoch + 1,
                            final_loss=best_elbo,
                            reason="early_stopping"
                        )
                        break
            
            # Check convergence (tolerance-based)
            if epoch > 0 and self._check_convergence(epoch_elbos):
                self._converged = True
                train_logger.log_convergence(
                    converged=True,
                    num_epochs=epoch + 1,
                    final_loss=self._elbo,
                    reason="converged"
                )
                _logger.info(f"Converged at epoch {epoch + 1}")
                break
        
        total_step = t_fwd + t_bwd + t_clip + t_step + t_data
        if total_step > 0:
            _logger.info(
                "Training time breakdown: data_load=%.2fs (%.0f%%), forward=%.2fs (%.0f%%), backward=%.2fs (%.0f%%), "
                "clip_grad=%.2fs (%.0f%%), optimizer_step=%.2fs (%.0f%%)",
                t_data, 100 * t_data / total_step,
                t_fwd, 100 * t_fwd / total_step,
                t_bwd, 100 * t_bwd / total_step,
                t_clip, 100 * t_clip / total_step,
                t_step, 100 * t_step / total_step,
            )
        
        # Final logging
        if not self._converged:
            train_logger.log_convergence(
                converged=False,
                num_epochs=self.max_epochs,
                final_loss=self._elbo,
                reason="max_epochs"
            )
        
        # Extract factors and innovations (and regime_weights when K > 1)
        self.eval()
        with torch.no_grad():
            # Use full dataset for final extraction
            all_factors = []
            all_innovations = []
            all_regime_weights = []
            iter_extract = [(y_all, u_all)] if use_single_batch else dataloader
            for y_batch, u_batch in iter_extract:
                outputs = self.forward(y_batch, u_batch)
                all_factors.append(to_numpy(outputs['factors']))
                all_innovations.append(to_numpy(outputs['eta']))
                if outputs.get('regime_weights') is not None:
                    all_regime_weights.append(to_numpy(outputs['regime_weights']))
            
            # Concatenate all batches
            if all_factors:
                self.factors = np.concatenate(all_factors, axis=0)
                self.innovations = np.concatenate(all_innovations, axis=0)
                if all_regime_weights:
                    rw = np.concatenate(all_regime_weights, axis=0)  # (batch, T, K)
                    if rw.ndim == 3:
                        self.regime_weights = rw[:, -1, :]  # (batch, K) one per window-end
                    else:
                        self.regime_weights = rw
                else:
                    self.regime_weights = None
        
        # Store training state
        self.training_state = iVDFMModelState.from_model(self)
        
        return self
    
    def _check_convergence(self, elbo_history: list, window: int = 5) -> bool:
        """Check if training has converged.
        
        Parameters
        ----------
        elbo_history : list
            History of ELBO values
        window : int
            Window size for convergence check
        
        Returns
        -------
        bool
            True if converged
        """
        if len(elbo_history) < window + 1:
            return False
        
        recent = elbo_history[-window:]
        previous = elbo_history[-window-1:-1]
        
        # Check if improvement is below tolerance
        improvement = np.mean(recent) - np.mean(previous)
        return abs(improvement) < self.tolerance
    
    def predict(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        context_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Union[List[str], List[int]]] = None,
        horizon: int = 1,
        deterministic: bool = True,
        num_samples: int = 1,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Predict future values.
        
        Parameters
        ----------
        data : Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]]
            Historical data for prediction. If provided, will extract last factor state.
            If DataFrame, can include context variables via context.
        context_data : Optional[Union[np.ndarray, torch.Tensor]]
            Context variables for prediction horizon, shape (horizon, context_dim).
            If None, will be generated as time-based context.
        context : Optional[Union[List[str], List[int]]]
            Column names (DataFrame) or indices (array) for context variables in data.
            Used only if data is provided and is DataFrame/array with context columns.
        horizon : int
            Prediction horizon (number of steps ahead)
        deterministic : bool, default True
            If True, uses zero innovations (deterministic forecast).
            If False, samples innovations from prior network using context variables.
        num_samples : int, default 1
            Number of Monte Carlo samples when deterministic=False.
            Multiple samples are averaged to reduce variance and approximate E[p(y|u)].
            Ignored when deterministic=True.
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        np.ndarray
            Predictions, shape (horizon, data_dim)
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before prediction")
        
        if self.factors is None:
            raise ModelNotTrainedError("Factors not available. Model must be trained with fit()")
        
        # Validate horizon
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        
        factors_future, y_pred = self._forecast(horizon, context_data, deterministic, num_samples)
        
        # Convert to numpy and remove batch dimension
        y_pred_np = to_numpy(y_pred.squeeze(0))  # (horizon, data_dim)
        
        return y_pred_np

    def _future_time_features(self, horizon: int, time_context: int) -> np.ndarray:
        """Time-based context for the ``horizon`` steps after training, ``time_context`` columns.
        
        Extracted unchanged from ``predict``: the normalized time step continues
        from ``T_train`` (the length of the window-averaged factor path), and
        ``time_context > 1`` adds periodic sine features.
        
        Returns
        -------
        np.ndarray
            float32 array, shape (horizon, time_context)
        """
        if self.factors.ndim == 3:
            factors_avg = np.mean(self.factors, axis=0)  # (T, r)
        else:
            factors_avg = self.factors  # (T, r)
        
        # Generate time-based context: continue from last training time step
        T_train = factors_avg.shape[0]
        time_indices = np.arange(T_train, T_train + horizon, dtype=np.float32)
        if T_train > 1:
            # Normalize by training length
            time_indices = time_indices / (T_train - 1)
        else:
            time_indices = time_indices / T_train
        
        # Create time features using time_context (not full context_dim which may include custom context)
        if time_context == 1:
            u_future = time_indices.reshape(-1, 1)
        else:
            features = [time_indices.reshape(-1, 1)]
            for i in range(1, time_context):
                freq = 2 * np.pi * (i + 1) / T_train if T_train > 1 else 2 * np.pi * (i + 1)
                periodic = np.sin(freq * np.arange(T_train, T_train + horizon, dtype=np.float32))
                features.append(periodic.reshape(-1, 1))
            u_future = np.hstack(features)
        return u_future

    def _forecast(
        self,
        horizon: int,
        context_data: Optional[Union[np.ndarray, torch.Tensor]],
        deterministic: bool,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-loop forecast from the last training factor state (extracted unchanged from ``predict``).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(factors_future, y_pred)``: forecast factors, shape (1, horizon, r),
            and decoded observations in model space, shape (1, horizon, data_dim).
        """
        self.eval()
        with torch.no_grad():
            # Get last factor state from training
            # factors shape: (batch, T, r) or (T, r)
            if self.factors.ndim == 3:
                # Average over batches if needed
                factors_avg = np.mean(self.factors, axis=0)  # (T, r)
            else:
                factors_avg = self.factors  # (T, r)
            
            # Get last factor state
            f_last = factors_avg[-1, :]  # (r,)
            f_last_tensor = torch.from_numpy(f_last).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, r) for batch dimension
            
            # Generate context variables for forecast horizon
            if context_data is None:
                # Generate time-based context: continue from last training time step
                u_future = self._future_time_features(horizon, self.time_context)
                
                # If model has custom context columns, user must provide context_data for prediction
                # (time features alone won't match full context_dim)
                if self.context_dim > self.time_context:
                    _logger.warning(
                        f"Model has custom context columns (context_dim={self.context_dim} > time_context={self.time_context}). "
                        f"Only time features generated. Provide context_data parameter for full context."
                    )
            else:
                # Use provided context_data
                if isinstance(context_data, torch.Tensor):
                    u_future = context_data.cpu().numpy()
                else:
                    u_future = context_data
                
                if u_future.shape[0] != horizon:
                    raise ValueError(
                        f"context_data shape[0] ({u_future.shape[0]}) must match horizon ({horizon})"
                    )
                if u_future.shape[1] != self.context_dim:
                    raise ValueError(
                        f"context_data shape[1] ({u_future.shape[1]}) must match context_dim ({self.context_dim})"
                    )
            
            u_future_tensor = torch.from_numpy(u_future).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, horizon, context_dim)
            
            # Generate innovations for forecast horizon
            if deterministic:
                # Deterministic forecast: zero innovations
                eta_future = torch.zeros(1, horizon, self.latent_dim, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            else:
                # Sample innovations from prior p(η|u); regime-weighted when K>1
                if num_samples == 1:
                    eta_future = self._sample_prior(u_future_tensor)
                else:
                    samples = []
                    for _ in range(num_samples):
                        eta_sample = self._sample_prior(u_future_tensor)  # (1, horizon, r)
                        samples.append(eta_sample)
                    eta_future = torch.stack(samples, dim=0).mean(dim=0)  # (1, horizon, r)
            
            # Forecast factors: K>1 use effective AR(p) per window (π at first future step); K=1 use global SSM
            if self.num_regimes > 1 and self.A_diag is not None:
                pi_w = self.regime_net(u_future_tensor[:, 0:1, :], temperature=self.regime_temperature).squeeze(1)  # (1, K)
                A_eff, B_eff = self._effective_regime_coeffs(pi_w)  # (1, r, p), (1, r)
                group_id = pi_w.argmax(dim=-1)  # (1,) for grouped path
                factors_future = self.ssm.forward_closed_loop_with_coeffs(
                    f_current=f_last_tensor,
                    eta_future=eta_future,
                    ar_coeffs=A_eff,
                    B_diag=B_eff,
                    group_id=group_id,
                )
            else:
                factors_future = self.ssm.forward_closed_loop(
                    f_current=f_last_tensor,
                    eta_future=eta_future,
                    horizon=horizon,
                )

            # Decode factors to observations (u_future_tensor always available in predict)
            y_pred = self._decode(factors_future, u_future_tensor)
            
            return factors_future, y_pred
    
    def update(
        self,
        dataset: iVDFMDataset,
        *,
        append: bool = True,
        store_full_history: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Update model state with new observations (online learning).
        Accepts an iVDFMDataset only (the estimator builds it with the fitted window, stride and scaler).
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before update")
        if not isinstance(dataset, iVDFMDataset):
            raise TypeError(f"update() takes an iVDFMDataset, got {type(dataset).__name__}")

        # Validate dimensions match
        if dataset.target_length != self.data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, new data has {dataset.target_length}"
            )
        if dataset.context_length != self.context_dim:
            raise DataValidationError(
                f"context_dim mismatch: model expects {self.context_dim}, new data has {dataset.context_length}"
            )
        
        # Create data loader using dataset's method (no shuffling for update)
        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Forward pass to get new factors and innovations
        self.eval()
        with torch.no_grad():
            if store_full_history:
                all_factors: list[np.ndarray] = []
                all_innovations: list[np.ndarray] = []
                all_regime_weights: list[np.ndarray] = []
            else:
                last_factor_state: Optional[np.ndarray] = None  # (r,)
                last_innovation_state: Optional[np.ndarray] = None  # (r,)
            
            for y_batch, u_batch in dataloader:
                outputs = self.forward(y_batch, u_batch)
                if store_full_history:
                    all_factors.append(to_numpy(outputs['factors']))
                    all_innovations.append(to_numpy(outputs['eta']))
                    if outputs.get('regime_weights') is not None:
                        all_regime_weights.append(to_numpy(outputs['regime_weights']))
                else:
                    # Keep only the last state (batch, time step) for memory efficiency
                    f_btr = to_numpy(outputs["factors"])
                    e_btr = to_numpy(outputs["eta"])
                    if f_btr.size > 0:
                        last_factor_state = f_btr[-1, -1, :].copy()
                    if e_btr.size > 0:
                        last_innovation_state = e_btr[-1, -1, :].copy()
            
            if store_full_history:
                # Concatenate all batches
                if not all_factors:
                    _logger.warning("No data processed in update")
                    return

                new_factors = np.concatenate(all_factors, axis=0)  # (num_windows, T, r)
                new_innovations = np.concatenate(all_innovations, axis=0)
                new_regime_weights = np.concatenate(all_regime_weights, axis=0) if all_regime_weights else None  # (num_windows, T, K) or empty

                # One factor per window-end timestep (num_windows, r): each window has dynamics (T steps),
                # we keep the state at the end of each window for alignment with full-series models (e.g. DDFM).
                if new_factors.ndim == 3:
                    new_factors = new_factors[:, -1, :]   # (num_windows, r)
                    new_innovations = new_innovations[:, -1, :]
                    if new_regime_weights is not None and new_regime_weights.ndim == 3:
                        new_regime_weights = new_regime_weights[:, -1, :]  # (num_windows, K)
            else:
                if last_factor_state is None or last_innovation_state is None:
                    _logger.warning("No data processed in update")
                    return
                new_factors = last_factor_state.reshape(1, -1)
                new_innovations = last_innovation_state.reshape(1, -1)
                new_regime_weights = None

            # Update model state: append or overwrite
            if append and self.factors is not None and self.innovations is not None:
                # Normalize existing state to 2D if needed
                factors_existing = self._normalize_factors_shape(self.factors)
                innovations_existing = self._normalize_factors_shape(self.innovations)
                self.factors = np.concatenate([factors_existing, new_factors], axis=0)
                self.innovations = np.concatenate([innovations_existing, new_innovations], axis=0)
                if new_regime_weights is not None and getattr(self, 'regime_weights', None) is not None:
                    self.regime_weights = np.concatenate([self.regime_weights, new_regime_weights], axis=0)
                elif new_regime_weights is not None:
                    self.regime_weights = new_regime_weights
            else:
                self.factors = new_factors
                self.innovations = new_innovations
                if new_regime_weights is not None:
                    self.regime_weights = new_regime_weights

            # Update training state
            self.training_state = iVDFMModelState.from_model(self)

            if verbose:
                _logger.info(
                    f"Model state updated with {len(dataset)} new sequences. "
                    f"Factors shape: {self.factors.shape}, Innovations shape: {self.innovations.shape}"
                )

    def get_result(self) -> iVDFMResult:
        """Extract result from trained model.
        
        Returns
        -------
        iVDFMResult
            Model result object with all state-space parameters
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        
        # Extract factors and innovations
        if self.factors is None or self.innovations is None:
            raise ModelNotTrainedError("Factors and innovations not available")

        # Decode factors to reconstructions (no u at result time: use uniform π when K > 1)
        with torch.no_grad():
            z_tensor = torch.from_numpy(self.factors).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            y_hat = self._decode(z_tensor, u_1T=None)
            recon = to_numpy(y_hat)

        # Minimal config snapshot for reproducibility (plain python)
        cfg_snapshot = None
        try:
            from dataclasses import asdict
            cfg_snapshot = asdict(self._config) if self._config is not None else None
        except Exception:
            cfg_snapshot = None

        # Training diagnostics
        num_epochs = int(self._num_iter) if self._num_iter is not None else None
        converged = bool(self._converged)
        training_loss = self.loss_now
        training_elbo = self._elbo

        # Provide weights-only payload (recommended by PyTorch)
        weights = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        # Shared BaseResult fields
        # - use Z for factors (common convention in the codebase)
        # - use x_sm for reconstructions in model space
        z_np = self.factors
        x_sm = recon
        if isinstance(z_np, np.ndarray) and z_np.ndim == 3:
            # (batch, T, r) -> (T, r) for summary/interop
            z_np = np.mean(z_np, axis=0)
        if isinstance(x_sm, np.ndarray) and x_sm.ndim == 3:
            # (batch, T, N) -> (T, N)
            x_sm = np.mean(x_sm, axis=0)

        # Compute full_state (augmented state for companion form)
        full_state = self._compute_full_state(z_np)

        # Regime and mixing (for result summary and inspection)
        num_regimes = getattr(self, "num_regimes", None)
        regime_temperature = getattr(self, "regime_temperature", None)
        mixing = getattr(self, "mixing", None)
        mixing_matrix = None
        if self.factor_mixing is not None and hasattr(self.factor_mixing, "M"):
            mixing_matrix = to_numpy(self.factor_mixing.M.weight.detach())

        regime_weights_out = getattr(self, "regime_weights", None)
        return iVDFMResult(
            innovations=self.innovations,
            reconstructions=recon,
            full_state=full_state,
            x_sm=x_sm,
            Z=z_np,
            r=np.array([self.latent_dim], dtype=int),
            p=int(self.factor_order),
            target_scaler=getattr(self._config, "target_scaler", None) if self._config else None,
            num_iter=int(self._num_iter) if self._num_iter is not None else 0,
            objective=training_elbo,
            training_elbo=training_elbo,
            training_loss=training_loss,
            num_epochs=num_epochs,
            converged=converged,
            config=cfg_snapshot,
            model_state_dict=weights,
            regime_weights=regime_weights_out,
            num_regimes=num_regimes,
            regime_temperature=regime_temperature,
            mixing=mixing,
            mixing_matrix=mixing_matrix,
        )
