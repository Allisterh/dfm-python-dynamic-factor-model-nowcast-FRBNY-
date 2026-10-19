"""Attention Factor Model (AFM) for statistical arbitrage.

Faithful implementation of Epstein, Wang, Choi and Pelger (2025), "Attention
Factors for Statistical Arbitrage" (arXiv:2510.11616). The model jointly learns
a conditional factor model whose residuals are traded portfolios and a
residual-history trading policy, under an after-cost Sharpe objective with an
explained-variance regularizer:

    max_{omega_F, omega_port}  Sharpe(R_net) + lambda_var * EV,   s.t. ||omega||_1 = 1.

Three model classes from the paper are covered by two switches:
  * ``factor_model='attention', trading='longconv'`` -> Attention Factors;
  * ``factor_model='pca',       trading='longconv'`` -> PCA + LongConv benchmark;
  * ``factor_model='pca',       trading='ou'``       -> PCA + Avellaneda--Lee.

This is a from-the-paper implementation (no reference code exists) intended to be
correct and runnable on modest data; it is not a reproduction of the paper's
empirical CRSP study, which requires licensed point-in-time data.

``AFMEngine`` is the private engine behind the public estimator
:class:`dfm_python.models.afm.estimator.AFM`, which builds its inputs from
``PanelData`` and passes every hyperparameter explicitly. The engine's code is
the pre-redesign ``AFM`` class unchanged, apart from the ``device`` keyword,
:meth:`AFMEngine._from_saved` and these removals:

- It no longer inherits the old abstract base ``dfm_python.models.base.BaseFactorModel``
  (deleted); :func:`_init_engine_state` sets the attributes that base's ``__init__``
  set, and :meth:`AFMEngine._ensure_result` is the base's method copied unchanged.
- The ``torch.save`` ``save``/``load`` pair and the inherited ``reset`` and
  ``_restore_checkpoint_metadata`` are gone; the estimator saves and loads through
  :meth:`AFMEngine._from_saved`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ...config.schema.results import AFMResult
from ...config.schema.params import AFMModelState
from ...utils.errors import (ConfigurationError, ModelNotTrainedError,
                             DataValidationError)
from ...utils.seed import apply_seed
from .factors import AttentionFactors, PCAFactors, residuals, rolling_pca_weights
from .longconv import LongConv1d
from .baselines import avellaneda_lee_weights

_EPS = 1e-8


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class AFMEngine(nn.Module):
    """Attention Factor Model engine.

    Parameters
    ----------
    dataset : AFMDataset, optional
        Data source exposing ``returns`` (T, N) and, for attention,
        ``characteristics`` (T, N, M).
    config : AFMConfig, optional
        Configuration; individual keyword arguments override its fields.
    n_char, n_factors, embed_dim, hist_len, n_kernels, ridge : int/float
        Architecture. ``hist_len`` is the residual history length ``s``.
    factor_model : {'attention', 'pca'}
    trading : {'longconv', 'ou'}
    lambda_var, turnover_cost, short_cost, risk_free : float
        Objective and cost parameters (defaults are the paper's 5bps/1bp).
    learning_rate, max_epochs, patience, seed
        Training controls.
    device : torch.device or str, optional
        Device for the network and tensors. None (default) uses CUDA if
        available, else CPU.
    """

    def __init__(self, dataset: Optional[Any] = None, config: Optional[Any] = None,
                 *, n_char: Optional[int] = None, n_factors: Optional[int] = None,
                 embed_dim: Optional[int] = None, hist_len: Optional[int] = None,
                 n_kernels: Optional[int] = None, ridge: Optional[float] = None,
                 squash_lambda: Optional[float] = None, pca_window: Optional[int] = None,
                 reestim: Optional[int] = None,
                 factor_model: Optional[str] = None, trading: Optional[str] = None,
                 lambda_var: Optional[float] = None, turnover_cost: Optional[float] = None,
                 short_cost: Optional[float] = None, risk_free: Optional[float] = None,
                 learning_rate: Optional[float] = None, max_epochs: Optional[int] = None,
                 patience: Optional[int] = None, seed: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 **kwargs: Any) -> None:
        _init_engine_state(self)
        nn.Module.__init__(self)

        # Precedence: explicit kwarg > config field > hardcoded default.
        self._config = config

        def pick(explicit, name, default):
            if explicit is not None:
                return explicit
            if config is not None:
                v = getattr(config, name, None)
                if v is not None:
                    return v
            return default

        self.factor_model = pick(factor_model, "factor_model", "attention")
        self.trading = pick(trading, "trading", "longconv")
        self.n_factors = pick(n_factors, "num_factors", None)
        self.embed_dim = pick(embed_dim, "embed_dim", 32)
        self.hist_len = pick(hist_len, "hist_len", 20)
        self.n_kernels = pick(n_kernels, "n_kernels", 32)
        self.ridge = pick(ridge, "ridge", 1e-2)
        self.squash_lambda = pick(squash_lambda, "squash_lambda", 1e-3)
        self.pca_window = pick(pca_window, "pca_window", 252)
        self.reestim = pick(reestim, "reestim", 21)
        self.lambda_var = pick(lambda_var, "lambda_var", 0.1)
        self.turnover_cost = pick(turnover_cost, "turnover_cost", 5e-4)
        self.short_cost = pick(short_cost, "short_cost", 1e-4)
        self.risk_free = pick(risk_free, "risk_free", 0.0)
        self.learning_rate = pick(learning_rate, "learning_rate", 1e-3)
        self.max_epochs = pick(max_epochs, "max_epochs", 50)
        self.patience = pick(patience, "patience", None)
        self.seed = pick(seed, "seed", None)

        if self.factor_model not in ("attention", "pca"):
            raise ConfigurationError(
                f"factor_model must be 'attention' or 'pca', got '{self.factor_model}'")
        if self.trading not in ("longconv", "ou"):
            raise ConfigurationError(
                f"trading must be 'longconv' or 'ou', got '{self.trading}'")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.seed is not None:
            apply_seed(int(self.seed))

        self._dataset = dataset
        if dataset is not None:
            self._infer_dims(dataset, n_char)
        else:
            self.n_char = n_char
        self._built = False
        if self.n_factors is not None and (self.factor_model == "pca" or
                                            self.n_char is not None):
            self._build_components()

        # Training diagnostics (read by AFMModelState.from_model).
        self.loss_now: Optional[float] = None
        self._num_iter: int = 0
        self._converged: bool = False
        self._sharpe: Optional[float] = None
        self._ev: Optional[float] = None

    @classmethod
    def _from_saved(cls, params: dict, n_char: Optional[int],
                    device: Any) -> AFMEngine:
        """Rebuild an engine without ``__init__``: no dataset, no seeding.

        ``params`` holds the hyperparameters that ``__init__`` assigns
        (``factor_model``, ``trading``, ``n_factors``, ``embed_dim``,
        ``hist_len``, ``n_kernels``, ``ridge``, ``squash_lambda``,
        ``pca_window``, ``reestim``, ``lambda_var``, ``turnover_cost``,
        ``short_cost``, ``risk_free``, ``learning_rate``, ``max_epochs``,
        ``patience`` and ``seed``). The components are built on ``device``
        inside ``torch.random.fork_rng``, so their initial draws do not consume
        the caller's CPU RNG; load the saved weights over them.
        """
        with torch.random.fork_rng(devices=[]):
            self = cls.__new__(cls)
            _init_engine_state(self)
            nn.Module.__init__(self)
            self._config = None
            for name in ("factor_model", "trading", "n_factors", "embed_dim",
                         "hist_len", "n_kernels", "ridge", "squash_lambda",
                         "pca_window", "reestim", "lambda_var", "turnover_cost",
                         "short_cost", "risk_free", "learning_rate", "max_epochs",
                         "patience", "seed"):
                setattr(self, name, params[name])
            self.device = torch.device(device)
            self._dataset = None
            self.n_char = n_char
            self._built = False
            self.loss_now = None
            self._num_iter = 0
            self._converged = False
            self._sharpe = None
            self._ev = None
            self._build_components()
        return self

    # ------------------------------------------------------------------ setup
    def _infer_dims(self, dataset: Any, n_char: Optional[int]) -> None:
        self.n_char = n_char if n_char is not None else getattr(dataset, "n_char", None)
        if self.n_factors is None:
            self.n_factors = getattr(dataset, "n_factors", None)

    def _build_components(self) -> None:
        if self.n_factors is None or self.n_factors < 1:
            raise ConfigurationError("n_factors must be a positive integer")
        if self.factor_model == "attention":
            if self.n_char is None:
                raise ConfigurationError("attention factors require n_char")
            self.factor_block: Any = AttentionFactors(
                self.n_char, self.n_factors, self.embed_dim, self.ridge)
        else:
            self.factor_block = PCAFactors(self.n_factors, self.ridge)
        if self.trading == "longconv":
            self.filter: Optional[nn.Module] = LongConv1d(
                self.hist_len, self.n_kernels, self.squash_lambda)
        else:
            self.filter = None
        self.to(self.device)
        self._built = True

    def set_dataset(self, dataset: Any) -> AFMEngine:
        """Attach a dataset and (re)build components (iVDFM pattern)."""
        self._dataset = dataset
        self._infer_dims(dataset, getattr(self, "n_char", None))
        self._build_components()
        return self

    # --------------------------------------------------------------- forward
    def _factor_weights(self, returns: torch.Tensor,
                        characteristics: Optional[torch.Tensor]):
        """Per-date factor weights/loadings for a batch window.

        Returns ``omega_f`` (B, W, K, N) and ``beta_t`` (B, W, N, K). For PCA a
        single covariance is estimated per window and broadcast across dates.
        """
        if self.factor_model == "attention":
            if characteristics is None:
                raise DataValidationError("attention factors require characteristics")
            return self.factor_block(characteristics)
        # PCA: re-estimated on a rolling trailing window (not one static covariance).
        return rolling_pca_weights(returns, self.n_factors, self.pca_window,
                                   self.reestim, self.ridge)

    def forward(self, returns: torch.Tensor,
                characteristics: Optional[torch.Tensor] = None) -> dict:
        """Compute residuals, book returns, and the training objective.

        Parameters
        ----------
        returns : torch.Tensor
            Asset returns, shape ``(B, W, N)``.
        characteristics : torch.Tensor, optional
            Firm characteristics ``(B, W, N, M)`` (required for attention).

        Returns
        -------
        dict
            ``loss`` (scalar), ``sharpe``, ``ev``, ``residuals`` (B, W, N),
            ``book_returns`` (B, n_eval), ``weights`` (B, n_eval, N).
        """
        if returns.dim() != 3:
            raise DataValidationError(
                f"returns must be (B, W, N), got shape {tuple(returns.shape)}")
        B, W, N = returns.shape
        if W <= self.hist_len:
            raise DataValidationError(
                f"window length {W} must exceed hist_len {self.hist_len}")

        returns = returns.to(self.device)
        if characteristics is not None:
            characteristics = characteristics.to(self.device)

        omega_f, beta_t = self._factor_weights(returns, characteristics)
        eps = residuals(omega_f, beta_t, returns)                  # (B, W, N)

        book, weights = [], []
        prev_w = torch.zeros(B, N, device=returns.device, dtype=returns.dtype)
        for w in range(self.hist_len, W):
            hist = eps[:, w - self.hist_len:w, :].transpose(1, 2)  # (B, N, s)
            if self.trading == "longconv":
                omega_port = self.filter(hist)                     # (B, N)
            else:
                omega_port = avellaneda_lee_weights(hist)          # (B, N)
            of, bt = omega_f[:, w - 1], beta_t[:, w - 1]           # (B,K,N),(B,N,K)
            # omega_eps^T omega_port = omega_port - omega_F^T (beta^T omega_port)
            z = (bt.transpose(-1, -2) @ omega_port.unsqueeze(-1)).squeeze(-1)  # (B,K)
            omega_asset = omega_port - (of.transpose(-1, -2) @ z.unsqueeze(-1)).squeeze(-1)
            norm = omega_asset.abs().sum(-1, keepdim=True) + _EPS
            omega_asset = omega_asset / norm                       # ||omega||_1 = 1
            r_port = (returns[:, w] * omega_asset).sum(-1)         # (B,)
            cost = (self.turnover_cost * (omega_asset - prev_w).abs().sum(-1)
                    + self.short_cost * torch.relu(-omega_asset).sum(-1))
            book.append(r_port - cost)
            weights.append(omega_asset)
            prev_w = omega_asset

        book_returns = torch.stack(book, dim=1)                    # (B, n_eval)
        weights = torch.stack(weights, dim=1)                      # (B, n_eval, N)

        mean = book_returns.mean(-1) - self.risk_free
        std = book_returns.std(-1) + _EPS
        sharpe = (mean / std).mean()
        var_eps = eps.var(dim=1)                                   # (B, N)
        var_r = returns.var(dim=1) + _EPS
        ev = (1.0 - var_eps / var_r).mean()
        loss = -(sharpe + self.lambda_var * ev)
        return {"loss": loss, "sharpe": sharpe, "ev": ev, "residuals": eps,
                "book_returns": book_returns, "weights": weights}

    # -------------------------------------------------------------------- fit
    def fit(self, X: Optional[Any] = None) -> AFMModelState:
        """Train by maximizing the after-cost Sharpe + explained-variance objective."""
        if self._dataset is None:
            raise ConfigurationError("no dataset attached; pass one to __init__ or set_dataset")
        if not self._built:
            self._build_components()
        returns, characteristics = self._batch_from_dataset()
        params = [p for p in self.parameters() if p.requires_grad]
        from ...numeric.builder import build_afm_optimizer
        optimizer = build_afm_optimizer(params, self.learning_rate) if params else None

        best = float("inf")
        no_improve = 0
        last = None
        for epoch in range(int(self.max_epochs)):
            out = self.forward(returns, characteristics)
            loss = out["loss"]
            if optimizer is not None and torch.is_grad_enabled() and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
            last = out
            self.loss_now = float(loss.detach())
            self._num_iter = epoch + 1
            if self.loss_now < best - 1e-6:
                best = self.loss_now
                no_improve = 0
            else:
                no_improve += 1
            if optimizer is None:
                break                                              # nothing to learn
            if self.patience is not None and no_improve >= self.patience:
                self._converged = True
                break

        self._sharpe = float(last["sharpe"].detach())
        self._ev = float(last["ev"].detach())
        self._residuals = last["residuals"].detach().cpu().numpy()
        self._book_returns = last["book_returns"].detach().cpu().numpy()
        self.training_state = AFMModelState.from_model(self)
        self._result = None
        return self.training_state

    def _batch_from_dataset(self):
        ds = self._dataset
        returns, characteristics = ds.get_tensors(self.device)
        if returns.dim() == 2:                                     # (T, N) -> (1, T, N)
            returns = returns.unsqueeze(0)
        if characteristics is not None and characteristics.dim() == 3:
            characteristics = characteristics.unsqueeze(0)
        return returns, characteristics

    # ---------------------------------------------------------------- predict
    def predict(self, horizon: Optional[int] = None, *, data: Optional[Any] = None):
        """Return residuals, portfolio weights, and book returns for the data.

        AFM is not a horizon forecaster; ``predict`` reports the current
        positioning (the trading output), which is what the model produces.
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        ds = data if data is not None else self._dataset
        if ds is None:
            raise DataValidationError("no data to predict on")
        returns, characteristics = self._as_batch(ds)
        with torch.no_grad():
            out = self.forward(returns, characteristics)
        return {"residuals": out["residuals"].cpu().numpy(),
                "weights": out["weights"].cpu().numpy(),
                "book_returns": out["book_returns"].cpu().numpy()}

    def _as_batch(self, ds: Any):
        returns, characteristics = ds.get_tensors(self.device)
        if returns.dim() == 2:
            returns = returns.unsqueeze(0)
        if characteristics is not None and characteristics.dim() == 3:
            characteristics = characteristics.unsqueeze(0)
        return returns, characteristics

    # ----------------------------------------------------------------- update
    def update(self, data: Any, *args: Any, retrain: bool = False, **kwargs: Any) -> None:
        """Attach new data; optionally retrain from it."""
        if data is not None:
            self._dataset = data
            self._infer_dims(data, getattr(self, "n_char", None))
        if retrain:
            self.fit()

    # ------------------------------------------------------------- get_result
    def get_result(self) -> AFMResult:
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        if self._result is not None:
            return self._result
        st = self.training_state
        self._result = AFMResult(
            Z=st.residuals, x_sm=st.residuals, converged=st.converged,
            num_iter=st.num_iter, objective=st.sharpe,
            residual_portfolios=st.residuals, book_returns=st.book_returns,
            sharpe=st.sharpe, explained_variance=st.explained_variance,
            factor_model=self.factor_model, trading=self.trading,
            num_factors=self.n_factors)
        return self._result

    def _ensure_result(self) -> AFMResult:
        """Ensure result exists, computing it if necessary."""
        if self._result is None:
            if self.training_state is None:
                raise ModelNotTrainedError("Model has not been trained yet")
            self._result = self.get_result()
        return self._result

    @property
    def result(self) -> AFMResult:
        return self._ensure_result()
