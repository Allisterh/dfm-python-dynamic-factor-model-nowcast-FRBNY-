"""Optuna hyperparameter search for the iVDFM estimator (needs the ``tune`` extra).

Trials train on a subset of the latest windows of the data. The subset size is
controlled by train_window_ratio (proportion of available windows to use;
1.0 = full data). The last `horizon` steps are held out for the forecasting
objective (sMSE, sMAE or MSE), computed, as before, on the targets scaled by the
estimator's scaler fitted on the full series (data units without a scaler).

optuna is imported lazily, when a search function runs.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from ..dataset.panel import PanelData
from ..logger import get_logger

if TYPE_CHECKING:
    import optuna

    from ..models.ivdfm.estimator import iVDFM

_logger = get_logger(__name__)


def _require_optuna():
    """Import optuna, or raise an ImportError naming the extra that provides it."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for iVDFM hyperparameter search (dfm_python.tuning). "
            "Install it with: pip install 'dfm-python[tune]'"
        ) from e
    return optuna


# -----------------------------------------------------------------------------
# Subset sampling (latest windows)
# -----------------------------------------------------------------------------

def get_optuna_subset(
    data: Union[np.ndarray, pd.DataFrame],
    max_window: int,
    min_stride: int,
    horizon: int,
    *,
    train_window_ratio: float = 1.0,
    subset_ratio: Optional[float] = None,
    subset_max_steps: Optional[int] = None,
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """Take the latest chunk of data and split into train / holdout for forecast metric.

    Uses train_window_ratio (proportion of available windows) to decide how many samples:
    num_windows = (T_total - max_window) // min_stride + 1; use last floor(num_windows * train_window_ratio)
    windows. Default 1.0 = use full windows (same as real training). Pure Optuna param, not searchable.

    Parameters
    ----------
    data : np.ndarray (T, N) or pd.DataFrame
        Full time series.
    max_window : int
        Upper bound on window size (used to size the subset).
    min_stride : int
        Minimum stride (used to size the subset).
    horizon : int
        Forecast horizon for holdout (last horizon steps = test).
    train_window_ratio : float, default 1.0
        Proportion of available windows to use (0 < ratio <= 1). 1.0 = use all (same as real training).
    subset_ratio : float, optional
        If set in (0, 1], use at least this fraction of T_total for the subset (larger subset, slower trials).
    subset_max_steps : int, optional
        If set, use at least this many time steps for the subset (capped by T_total).

    Returns
    -------
    train_data : same type as data, shape (T_subset - horizon, N)
        Training portion (all but last horizon steps of the subset).
    test_true : np.ndarray, shape (horizon, N)
        Last horizon steps (ground truth for forecast metric).
    """
    if isinstance(data, pd.DataFrame):
        # Use only numeric columns (same as iVDFMDataset: exclude time index from variable set)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = list(data.columns)
        arr = data[numeric_cols].values
        is_df = True
        columns = numeric_cols
    else:
        arr = np.asarray(data)
        is_df = False
        columns = None

    T_total, N = arr.shape[0], arr.shape[1]
    min_stride = max(10, min_stride)
    horizon = max(1, horizon)
    ratio = max(0.01, min(1.0, float(train_window_ratio)))

    # Available windows with max_window and min_stride
    num_windows = (T_total - max_window) // min_stride + 1
    num_windows = max(1, num_windows)
    n_use = max(1, int(num_windows * ratio))
    T_subset = (n_use - 1) * min_stride + max_window + horizon
    if subset_ratio is not None and 0 < subset_ratio <= 1:
        T_subset = max(T_subset, min(T_total, int(T_total * subset_ratio)))
    if subset_max_steps is not None and subset_max_steps > 0:
        T_subset = max(T_subset, min(T_total, subset_max_steps))
    T_subset = min(T_subset, T_total)
    if T_subset < horizon + 2:
        raise ValueError(
            f"Data too short: T_total={T_total}, T_subset={T_subset}; need at least horizon+2."
        )

    # Last T_subset time steps
    subset_arr = arr[-T_subset:, :]
    train_end = T_subset - horizon
    train_arr = subset_arr[:train_end, :]
    test_true = np.asarray(subset_arr[train_end:, :], dtype=np.float64)

    if is_df and columns is not None:
        train_data = pd.DataFrame(train_arr, columns=columns)
    else:
        train_data = train_arr

    return train_data, test_true


# -----------------------------------------------------------------------------
# Forecast metrics (sMSE / sMAE)
# -----------------------------------------------------------------------------

def compute_forecast_metric(
    preds: np.ndarray,
    trues: np.ndarray,
    metric: str = "sMSE",
) -> float:
    """Compute multivariate forecast metric: MSE, sMSE, or sMAE.

    MSE: mean squared error over all (pred, true) pairs (one scalar).
    sMSE/sMAE: per-variable normalized then averaged (scale-invariant).
    preds, trues: (horizon, N) or (n_origins, horizon, N); flattened to (n_points, N).
    """
    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    n = preds_flat.shape[1]
    if n == 0:
        return float("nan")

    metric_upper = metric.upper()
    if metric_upper == "MSE":
        return float(np.mean((preds_flat - trues_flat) ** 2))

    out_per_var: List[float] = []
    for j in range(n):
        p_j = preds_flat[:, j]
        t_j = trues_flat[:, j]
        var_j = float(np.var(t_j))
        std_j = float(np.std(t_j)) if var_j >= 1e-20 else float("nan")
        if metric_upper == "SMSE":
            if np.isfinite(var_j) and var_j >= 1e-10:
                mse_j = float(np.mean((p_j - t_j) ** 2))
                out_per_var.append(mse_j / var_j)
            else:
                out_per_var.append(float("nan"))
        elif metric_upper == "SMAE":
            if np.isfinite(std_j) and std_j >= 1e-10:
                mae_j = float(np.mean(np.abs(p_j - t_j)))
                out_per_var.append(mae_j / std_j)
            else:
                out_per_var.append(float("nan"))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'MSE', 'sMSE', or 'sMAE'.")

    return float(np.nanmean(out_per_var)) if out_per_var else float("nan")


# -----------------------------------------------------------------------------
# Optuna suggest (hyperparameters)
# -----------------------------------------------------------------------------

# Structural / causal parameters that MUST NOT be tuned (model selection, not hyperparameters).
# Fix to ground truth in synthetic experiments; change only via ablation elsewhere.
def _aligned_high(low: int, high: int, step: int) -> int:
    """The largest ``low + k * step`` <= ``high``: the upper bound optuna falls back to, with a warning,
    when ``high - low`` is not a multiple of ``step``. Passing it keeps the same search space, silently."""
    return low + (high - low) // step * step


_STRUCTURAL_PARAMS = frozenset({
    "n_factors",
    "factor_order",
    "n_regimes",
    "mixing",
    "innovation_distribution",
    "context",
    "time_context",
    "window",
    "stride",
})


def suggest_ivdfm_hyperparameters_causal(
    trial: optuna.Trial,
    fixed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Suggest only non-structural iVDFM hyperparameters for causal / IRF experiments.

    Tunes optimization and network capacity only. Does NOT suggest:
    n_factors, factor_order, n_regimes, mixing, innovation_distribution,
    context, time_context, window, stride (these define the causal model class).

    Caller must merge the returned dict with fixed structural config (e.g. from base model).
    """
    optuna = _require_optuna()

    fixed = dict(fixed or {})

    params: Dict[str, Any] = {}

    # Optimization (safe)
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["max_epochs"] = trial.suggest_int("max_epochs", 30, 80, step=10)
    params["optimizer_weight_decay"] = trial.suggest_float(
        "optimizer_weight_decay", 1e-6, 1e-3, log=True
    )
    params["patience"] = trial.suggest_int("patience", 10, 20)

    # Encoder / decoder / prior capacity (safe)
    params["encoder_hidden_dim"] = trial.suggest_categorical(
        "encoder_hidden_dim", [64, 128, 256]
    )
    params["encoder_n_hidden_layers"] = trial.suggest_int("encoder_n_hidden_layers", 1, 2)
    params["decoder_hidden_dim"] = trial.suggest_categorical(
        "decoder_hidden_dim", [64, 128, 256]
    )
    params["decoder_n_hidden_layers"] = trial.suggest_int("decoder_n_hidden_layers", 1, 2)
    params["prior_hidden_dim"] = trial.suggest_categorical(
        "prior_hidden_dim", [64, 128, 256]
    )
    params["prior_n_hidden_layers"] = trial.suggest_int("prior_n_hidden_layers", 1, 2)

    # Regularization and KL (safe; beta_kl constrained)
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
    params["use_layer_norm"] = trial.suggest_categorical("use_layer_norm", [True, False])
    params["beta_kl"] = trial.suggest_float("beta_kl", 0.5, 1.2)

    # Pass through any fixed non-structural keys not already suggested
    for k, v in fixed.items():
        if k not in params and k not in _STRUCTURAL_PARAMS:
            params[k] = v

    return params


def suggest_ivdfm_hyperparameters(
    trial: optuna.Trial,
    max_window: int,
    min_window: int = 500,
    max_regimes: int = 7,
    metric: str = "sMSE",
    fixed: Optional[Dict[str, Any]] = None,
    max_hidden_dim: Optional[int] = None,
    search_scope: Optional[str] = None,
    min_stride: int = 10,
) -> Dict[str, Any]:
    """Suggest iVDFM hyperparameters for a trial, under the iVDFM estimator's parameter names.

    ``dropout`` is suggested too (it keeps the sampler's draw sequence) but is
    not an iVDFM parameter; tune_hyperparameters drops it.

    If a key is in fixed, that value is used (structural design); otherwise the param is suggested (study-able).
    max_hidden_dim: if <= 200 use small-dataset space (encoder/decoder max 200); if None or > 200 use large (max 512).
    search_scope: "focused" = narrower lr/epochs for already-optimized datasets (ETT); None/"full" = wider for others.
    For causal/IRF experiments use suggest_ivdfm_hyperparameters_causal instead.
    """
    optuna = _require_optuna()

    fixed = dict(fixed or {})
    _exclude = {"max_regimes", "max_window", "min_window", "train_window_ratio", "min_stride"}

    params: Dict[str, Any] = {}
    window_high = max(min_window, max_window)

    # Structural: use fixed value when provided, else suggest
    if "n_factors" in fixed:
        params["n_factors"] = fixed["n_factors"]
    else:
        params["n_factors"] = trial.suggest_int("n_factors", 2, min(8, max(2, window_high // 50)), step=1)
    if "window" in fixed:
        params["window"] = fixed["window"]
    else:
        # window and stride step by 10; their upper bounds are aligned down to the step (see _aligned_high).
        params["window"] = trial.suggest_int("window", min_window, _aligned_high(min_window, window_high, 10), step=10)
    if "stride" in fixed:
        params["stride"] = fixed["stride"]
    else:
        # Min stride at least 10; when window fixed, at least half window
        stride_low = max(min_stride, (fixed.get("window") or 0) // 2) if fixed.get("window") else min_stride
        stride_low = max(10, stride_low)
        params["stride"] = trial.suggest_int("stride", stride_low, _aligned_high(stride_low, 1000, 10), step=10)
    if "n_regimes" in fixed:
        params["n_regimes"] = fixed["n_regimes"]
    else:
        params["n_regimes"] = trial.suggest_int("n_regimes", 1, max(1, max_regimes), step=1)
    if "regime_temperature" in fixed:
        params["regime_temperature"] = fixed["regime_temperature"]
    else:
        params["regime_temperature"] = trial.suggest_float("regime_temperature", 0.1, 1.0)
    if "mixing" in fixed:
        params["mixing"] = fixed["mixing"]
    else:
        params["mixing"] = trial.suggest_categorical("mixing", [True, False])
    if "factor_order" in fixed:
        params["factor_order"] = fixed["factor_order"]
    else:
        params["factor_order"] = trial.suggest_int("factor_order", 1, 5, step=1)

    # Encoder / decoder / prior: dataset-specific. Small (max 200) = scrutinize; large = up to 512
    small = max_hidden_dim is not None and max_hidden_dim <= 200
    if small:
        encoder_decoder_cats = [64, 96, 128, 160, 200]
        prior_cats = [32, 64, 96, 128]
    else:
        encoder_decoder_cats = [64, 128, 200, 256, 384, 512]
        prior_cats = [32, 64, 96, 128, 160, 192]

    if "encoder_hidden_dim" in fixed:
        params["encoder_hidden_dim"] = fixed["encoder_hidden_dim"]
    else:
        params["encoder_hidden_dim"] = trial.suggest_categorical("encoder_hidden_dim", encoder_decoder_cats)
    if "encoder_n_hidden_layers" in fixed:
        params["encoder_n_hidden_layers"] = fixed["encoder_n_hidden_layers"]
    else:
        params["encoder_n_hidden_layers"] = trial.suggest_int("encoder_n_hidden_layers", 1, 3)
    if "decoder_hidden_dim" in fixed:
        params["decoder_hidden_dim"] = fixed["decoder_hidden_dim"]
    else:
        params["decoder_hidden_dim"] = trial.suggest_categorical("decoder_hidden_dim", encoder_decoder_cats)
    if "decoder_n_hidden_layers" in fixed:
        params["decoder_n_hidden_layers"] = fixed["decoder_n_hidden_layers"]
    else:
        params["decoder_n_hidden_layers"] = trial.suggest_int("decoder_n_hidden_layers", 1, 3)
    if "prior_hidden_dim" in fixed:
        params["prior_hidden_dim"] = fixed["prior_hidden_dim"]
    else:
        params["prior_hidden_dim"] = trial.suggest_categorical("prior_hidden_dim", prior_cats)
    if "prior_n_hidden_layers" in fixed:
        params["prior_n_hidden_layers"] = fixed["prior_n_hidden_layers"]
    else:
        params["prior_n_hidden_layers"] = trial.suggest_int("prior_n_hidden_layers", 1, 3)

    # Training & regularization: focused (ETT, already optimized) vs full (others, need more search)
    focused = search_scope == "focused"
    if "learning_rate" in fixed:
        params["learning_rate"] = fixed["learning_rate"]
    else:
        if focused:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        else:
            params["learning_rate"] = trial.suggest_float("learning_rate", 3e-5, 2e-2, log=True)
    if "max_epochs" in fixed:
        params["max_epochs"] = fixed["max_epochs"]
    else:
        if focused:
            params["max_epochs"] = trial.suggest_int("max_epochs", 50, 95, step=5)
        else:
            params["max_epochs"] = trial.suggest_int("max_epochs", 60, 130, step=5)
    if "decoder_var" in fixed:
        params["decoder_var"] = fixed["decoder_var"]
    else:
        if focused:
            params["decoder_var"] = trial.suggest_float("decoder_var", 0.01, 0.08)
        else:
            params["decoder_var"] = trial.suggest_float("decoder_var", 0.003, 0.15)
    if "beta_kl" in fixed:
        params["beta_kl"] = fixed["beta_kl"]
    else:
        params["beta_kl"] = trial.suggest_float("beta_kl", 0.5, 1.4)
    if "dropout" in fixed:
        params["dropout"] = fixed["dropout"]
    else:
        if focused:
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
        else:
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.35)

    # Any other fixed keys (e.g. time_context, decoder_type) not in suggest set
    for k, v in fixed.items():
        if k not in params and k not in _exclude:
            params[k] = v

    return params


# -----------------------------------------------------------------------------
# Objective and study
# -----------------------------------------------------------------------------

# Meta keys: used for Optuna/subsets only, not passed to iVDFM
_META_KEYS = {"max_regimes", "max_window", "min_window", "train_window_ratio", "min_stride"}
# Suggested (it keeps the sampler's draw sequence) but not an iVDFM parameter.
_NOT_PARAMS = {"dropout"}
_METRICS = ("MSE", "SMSE", "SMAE")


def _imputed_arrays(estimator: iVDFM, panel: PanelData) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Targets (T, N) and exog (T, K) or None, gap-filled as ``fit`` fills them.

    Built through the estimator's dataset adapter: linear interpolation, then
    forward/backward fill, then zeros. The targets are scaled by (a clone of)
    the estimator's scaler fitted on the whole series, as the old
    ``hyperparameter_optimization`` took them from its dataset; the exog is
    never scaled.
    """
    dataset = clone(estimator)._adapter(panel, torch.device("cpu"))
    targets = np.asarray(dataset.target, dtype=np.float64)
    exog = dataset.aux_context_data
    return targets, (None if exog is None else np.asarray(exog, dtype=np.float64))


def _make_objective(
    estimator: iVDFM,
    train_panel: PanelData,
    test_true: np.ndarray,
    futr_exog: Optional[np.ndarray],
    *,
    max_window: int,
    min_window: int,
    min_stride: int,
    metric: str,
    max_regimes: int,
    max_epochs_per_trial: Optional[int],
    fixed: Dict[str, Any],
    max_hidden_dim: Optional[int],
    search_scope: Optional[str],
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective: suggest params, fit a clone of ``estimator``, forecast the holdout, score it."""
    T_train = train_panel.n_timesteps
    horizon = test_true.shape[0]

    def objective(trial: optuna.Trial) -> float:
        optuna = _require_optuna()

        try:
            params = suggest_ivdfm_hyperparameters(
                trial,
                max_window=max_window,
                min_window=min_window,
                max_regimes=max_regimes,
                metric=metric,
                fixed=fixed,
                max_hidden_dim=max_hidden_dim,
                search_scope=search_scope,
                min_stride=min_stride,
            )
        except Exception as e:
            _logger.warning(f"Suggest failed: {e}")
            raise optuna.TrialPruned()

        window = min(int(params.pop("window")), T_train - 1)
        window = max(2, window)
        stride = max(10, int(params.pop("stride")))
        stride = min(stride, window)
        max_epochs = (
            max_epochs_per_trial if max_epochs_per_trial is not None else params.get("max_epochs", estimator.max_epochs)
        )

        # Drop meta keys and dropout so the estimator does not receive them
        model_params = {k: v for k, v in params.items() if k not in _META_KEYS and k not in _NOT_PARAMS}
        # The targets are already scaled (see _imputed_arrays), so the trials fit no scaler of their own.
        model_params.update(window=window, stride=stride, max_epochs=max_epochs, scaler=None, verbose=0)
        try:
            model = clone(estimator).set_params(**model_params)
        except Exception as e:
            _logger.warning(f"Model build failed: {e}")
            raise optuna.TrialPruned()

        try:
            model.fit(train_panel)
        except Exception as e:
            _logger.warning(f"Fit failed: {e}")
            raise optuna.TrialPruned()

        try:
            preds = model.predict(horizon, futr_exog=futr_exog).to_numpy(dtype=np.float64)
        except Exception as e:
            _logger.warning(f"Predict failed: {e}")
            raise optuna.TrialPruned()

        value = compute_forecast_metric(preds, test_true, metric=metric)
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        return float(value)

    return objective


def tune_hyperparameters(
    estimator: iVDFM,
    data: Any,
    *,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    max_window: Optional[int] = None,
    min_window: int = 500,
    max_regimes: int = 7,
    train_window_ratio: float = 1.0,
    min_stride: int = 10,
    horizon: int = 96,
    metric: str = "sMSE",
    max_epochs_per_trial: Optional[int] = None,
    fixed: Optional[Dict[str, Any]] = None,
    subset_ratio: Optional[float] = None,
    subset_max_steps: Optional[int] = None,
    min_batch_size: Optional[int] = None,
    initial_params: Optional[Dict[str, Any]] = None,
    max_hidden_dim: Optional[int] = None,
    search_scope: Optional[str] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = False,
    pruner: Optional[Any] = None,
    return_study: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], optuna.Study]]:
    """Search iVDFM hyperparameters with optuna, minimizing a holdout forecast metric.

    The latest windows of ``data`` (see :func:`get_optuna_subset`) are split
    into training rows and a holdout of the last ``horizon`` steps. Each
    trial fits ``clone(estimator)`` with the suggested hyperparameters
    (:func:`suggest_ivdfm_hyperparameters`) on the training rows, forecasts
    the holdout with ``predict(horizon, futr_exog=<holdout exog>)`` and
    scores it with :func:`compute_forecast_metric`. Trials that fail or score
    NaN are pruned. ``estimator`` and ``data`` are not changed and nothing is
    fitted in place: refit with ``estimator.set_params(**best).fit(data)``.

    Missing values are filled once, as ``fit`` fills them (linear
    interpolation, then forward/backward fill, then zeros), in targets and
    exog. As in the old ``hyperparameter_optimization``, the targets are then
    scaled once by the estimator's ``scaler`` (or ``fixed["scaler"]``) fitted
    on the whole series; the trials train on these scaled targets without a
    scaler of their own, and the metric compares forecasts and holdout in
    these scaled units (data units when the scaler is None). The trial
    estimators inherit every other hyperparameter of ``estimator`` that is
    not suggested or fixed, including ``batch_size``, ``random_state`` and
    ``device``; ``verbose`` is 0.

    The search is reproducible: the trials' ``random_state``
    (``fixed["random_state"]`` if given, else ``estimator.random_state``)
    also seeds optuna's default sampler, ``TPESampler(seed=random_state)``,
    so two calls with the same seed, data and arguments draw the same
    trials and, on CPU, return the same ``best``; different seeds draw
    different trials. ``random_state=None`` leaves the sampler unseeded, as
    before. A ``timeout`` can stop runs after different numbers of trials.
    A study resumed from ``storage`` (``load_if_exists=True``) that already
    has ``n`` trials is sampled with seed ``random_state + n`` (modulo
    2**32), so the resumed run draws new trials rather than replaying the
    first run's.

    ``window`` and ``stride`` are suggested in steps of 10. When the range
    is not a multiple of 10 (``max_window - min_window``, or ``1000`` minus
    the lowest stride), its upper bound is lowered to the last step (e.g.
    ``min_window=10, max_window=25`` searches 10 and 20), the range optuna
    itself fell back to with a warning; ranges that are multiples of 10 are
    unchanged.

    Parameters
    ----------
    estimator : iVDFM
        The estimator to tune (fitted or not).
    data : PanelData, DataFrame or 2-D array
        Targets in ``y`` and, optionally, context variables in ``exog``.
    n_trials : int, default 30
        Number of trials.
    timeout : float, optional
        Total time budget in seconds.
    max_window : int, optional
        Upper bound on the window (and used to size the subset). None or <= 0
        uses the series length.
    min_window : int, default 500
        Lower bound on the suggested window (step 10; the upper bound is
        lowered to ``min_window`` plus a multiple of 10).
    max_regimes : int, default 7
        Upper bound on the suggested ``n_regimes``.
    train_window_ratio : float, default 1.0
        Proportion of available windows to use (0 < ratio <= 1). 1.0 = full
        data; e.g. 0.5 = the latest 50%.
    min_stride : int, default 10
        Minimum stride (at least 10), for subset sizing.
    horizon : int, default 96
        Forecast horizon of the holdout.
    metric : {"sMSE", "sMAE", "MSE"}, default "sMSE"
        Holdout metric to minimize.
    max_epochs_per_trial : int, optional
        Epochs of every trial; None uses the suggested ``max_epochs``.
    fixed : dict, optional
        Hyperparameters that are not suggested, under the estimator's names
        (e.g. ``{"n_factors": 4, "n_regimes": 1}``). ``time_context``
        defaults to ``estimator.time_context``.
    subset_ratio : float, optional
        If in (0, 1], use at least this fraction of the series for the subset.
    subset_max_steps : int, optional
        If set, use at least this many time steps for the subset.
    min_batch_size : int, optional
        Batch size of every trial, returned in ``best``; not suggested. The
        same as ``fixed={"batch_size": ...}`` (passing both with different
        values raises ValueError). None trains the trials with
        ``fixed["batch_size"]`` if given, else ``estimator.batch_size``.
    initial_params : dict, optional
        Enqueued as the first trial; keys of the suggest space (n_factors,
        window, stride, n_regimes, regime_temperature, mixing, factor_order,
        encoder/decoder/prior hidden dims and layers, learning_rate,
        max_epochs, decoder_var, beta_kl, dropout).
    max_hidden_dim : int, optional
        <= 200 selects the small search space for hidden widths (up to 200);
        None or larger the large one (up to 512).
    search_scope : str, optional
        "focused" narrows learning rate, epochs and decoder variance; None or
        "full" searches the wider ranges.
    study_name, storage, load_if_exists, pruner
        Passed to ``optuna.create_study`` (the study name defaults to
        ``"ivdfm_optuna"``).
    return_study : bool, default False
        Also return the optuna study.

    Returns
    -------
    best : dict
        The best trial's hyperparameters plus ``fixed``, under the estimator's
        names (``estimator.set_params(**best)`` accepts it). Empty, with a
        warning, when no trial completed.
    study : optuna.Study
        Only with ``return_study=True``.

    Raises
    ------
    TypeError
        If ``estimator`` is not an iVDFM.
    ImportError
        If optuna is not installed (``pip install 'dfm-python[tune]'``).
    ValueError
        For an unknown metric, a ``fixed`` key that is not an iVDFM
        hyperparameter, ``min_batch_size`` different from
        ``fixed["batch_size"]``, or data too short for ``horizon``.
    """
    from ..models.ivdfm.estimator import iVDFM

    if not isinstance(estimator, iVDFM):
        raise TypeError(
            f"tune_hyperparameters supports the iVDFM estimator only, got {type(estimator).__name__}."
        )
    optuna = _require_optuna()
    if str(metric).upper() not in _METRICS:
        raise ValueError(f"Unknown metric: {metric}. Use 'MSE', 'sMSE', or 'sMAE'.")

    fixed = dict(fixed or {})
    known = set(estimator.get_params(deep=False)) | _META_KEYS | _NOT_PARAMS
    unknown = sorted(str(k) for k in fixed if k not in known)
    if unknown:
        raise ValueError(f"fixed has keys that are not iVDFM hyperparameters: {unknown}.")
    fixed.setdefault("time_context", estimator.time_context)
    if min_batch_size is not None:
        if "batch_size" in fixed and fixed["batch_size"] != min_batch_size:
            raise ValueError(
                f"min_batch_size={min_batch_size} conflicts with fixed['batch_size']={fixed['batch_size']}; "
                "pass one of them."
            )
        fixed["batch_size"] = min_batch_size

    panel = estimator._validate_data(data)
    scaled_by = clone(estimator).set_params(scaler=fixed["scaler"]) if "scaler" in fixed else estimator
    targets, exog = _imputed_arrays(scaled_by, panel)
    T_total = targets.shape[0]
    effective_max_window = T_total if (max_window is None or max_window <= 0) else max_window

    train_y, test_true = get_optuna_subset(
        targets,
        max_window=effective_max_window,
        min_stride=min_stride,
        horizon=horizon,
        train_window_ratio=train_window_ratio,
        subset_ratio=subset_ratio,
        subset_max_steps=subset_max_steps,
    )
    # The subset is the last len(train_y) + len(test_true) rows; the holdout is its end.
    test_start = T_total - test_true.shape[0]
    train_rows = slice(test_start - train_y.shape[0], test_start)
    train_index = panel.index[train_rows]
    train_panel = PanelData(
        y=pd.DataFrame(train_y, index=train_index, columns=list(panel.series_names)),
        exog=None if exog is None else pd.DataFrame(exog[train_rows], index=train_index, columns=list(panel.exog_names)),
    )
    futr_exog = None if exog is None else exog[test_start:]

    objective = _make_objective(
        estimator,
        train_panel,
        test_true,
        futr_exog,
        max_window=effective_max_window,
        min_window=min_window,
        min_stride=min_stride,
        metric=metric,
        max_regimes=max_regimes,
        max_epochs_per_trial=max_epochs_per_trial,
        fixed=fixed,
        max_hidden_dim=max_hidden_dim,
        search_scope=search_scope,
    )

    # The trials' random_state (fixed["random_state"], else the estimator's) also seeds optuna's
    # default TPE sampler, so a search is reproducible; None leaves it unseeded.
    seed = fixed.get("random_state", estimator.random_state)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name or "ivdfm_optuna",
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=None if seed is None else optuna.samplers.TPESampler(seed=int(seed)),
        pruner=pruner,
    )
    n_existing = len(study.trials)
    if seed is not None and n_existing:
        # A study resumed from storage (load_if_exists) already holds trials; reseeding with the same
        # seed would replay the earlier run's draws, so the seed is offset by the number of trials.
        study.sampler = optuna.samplers.TPESampler(seed=(int(seed) + n_existing) % 2**32)
    if initial_params:
        study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=bool(estimator.verbose))

    try:
        best_params = {k: v for k, v in study.best_params.items() if k not in _NOT_PARAMS}
    except ValueError:
        # No completed trials (all pruned or failed)
        warnings.warn(
            "tune_hyperparameters: no trial completed (all were pruned or failed); returning empty best params.",
            UserWarning,
            stacklevel=2,
        )
        best_params = {}
    else:
        for k, v in fixed.items():
            if k not in _META_KEYS and k not in _NOT_PARAMS:
                best_params.setdefault(k, v)
    if return_study:
        return best_params, study
    return best_params
