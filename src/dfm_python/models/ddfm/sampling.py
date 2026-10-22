"""One MCMC-denoising iteration of DDFM training: denoise, sample, train, predict."""

import numpy as np
import torch
import pandas as pd
from typing import Tuple, Optional, List, Any
from ...logger import get_logger
from ...numeric.estimator import get_idio
from ...numeric.statistic import (
    compute_variance_mean, compute_tensor_stats, diagnose_variance_collapse
)
from ...config.constants import (
    DEFAULT_MCMC_EPOCHS,
    DEFAULT_VARIANCE_COLLAPSE_THRESHOLD,
)
from ...config.types import to_numpy
from ...utils.helper import interpolate_dataframe

_logger = get_logger(__name__)


def denoise_targets(
    eps: np.ndarray,
    data_imputed: pd.DataFrame,
    data_denoised: pd.DataFrame,
    dataset: Any,
    lags_input: int,
    interpolation_method: str,
    interpolation_limit: Optional[int],
    interpolation_limit_direction: str,
    min_obs: int = 1
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the conditional AR-idio mean from the targets of ``data_denoised``, in place.

    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals ``(T, num_target_series)``.
    data_imputed : pd.DataFrame
        The data with the missing targets imputed.
    data_denoised : pd.DataFrame
        Updated in place on the target columns, rows ``lags_input + 1`` on.
    dataset : DDFMDataset
        Provides ``observed_y`` and ``target_indices``.
    lags_input : int
    interpolation_method, interpolation_limit, interpolation_limit_direction
        ``interpolate_dataframe`` settings for the returned frame.
    min_obs : int, default 1
        Minimum observations for the AR-idio estimate.

    Returns
    -------
    data_denoised_interpolated : pd.DataFrame
    Phi : np.ndarray
        AR-idio transition matrix.
    mu_eps, std_eps : np.ndarray
        AR-idio mean and std, ``(num_target_series,)``.
    """
    Phi, mu_eps, std_eps = get_idio(eps, dataset.observed_y, min_obs=min_obs)

    eps_denoise = eps[:-1, :] @ Phi  # (T-1, num_target_series)
    # Write through the frame (positional .iloc), not through ``.values``: under
    # pandas copy-on-write (the default from pandas 3) ``.values`` is read-only.
    data_denoised.iloc[lags_input+1:, dataset.target_indices] = \
        data_imputed.to_numpy()[lags_input+1:, dataset.target_indices] - eps_denoise

    data_denoised_interpolated = interpolate_dataframe(
        data_denoised,
        method=interpolation_method,
        limit=interpolation_limit,
        limit_direction=interpolation_limit_direction
    )

    return data_denoised_interpolated, Phi, mu_eps, std_eps


def train_on_mc_samples(
    autoencoder: Any,
    autoencoder_datasets: List[Any],
    window_size: int,
    learning_rate: float,
    optimizer_type: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    target_indices: Optional[torch.Tensor],
    device: torch.device
) -> None:
    """``autoencoder.fit`` on each MC sample in turn, ``DEFAULT_MCMC_EPOCHS`` epochs of ``window_size`` batches.

    ``target_indices`` and ``device`` are unused.
    """
    for ae_dataset in autoencoder_datasets:
        autoencoder.fit(
            dataset=ae_dataset,
            epochs=DEFAULT_MCMC_EPOCHS,
            batch_size=window_size,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            optimizer=optimizer,
            scheduler=scheduler,
            target_indices=target_indices
        )


def extract_predictions_from_mc_samples(
    encoder: Any,
    decoder: Any,
    autoencoder_datasets: List[Any],
    n_mc_samples: int,
    extract_target_predictions: callable,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode and decode each MC sample; return the factors and the prediction statistics.

    Parameters
    ----------
    encoder, decoder : nn.Module
    autoencoder_datasets : list of AutoencoderDataset
        One per MC sample.
    n_mc_samples : int
        Expected number of samples (validated).
    extract_target_predictions : callable
        Maps the mean prediction tensor to the target predictions.
    device : torch.device
        Unused.

    Returns
    -------
    factors : np.ndarray
        ``(n_mc_samples, T, num_factors)``.
    y_pred, y_pred_full : np.ndarray
        Target predictions and the mean prediction, ``(T, num_target_series)``.
    y_pred_std : np.ndarray
        Prediction std over the samples, ``(T, num_target_series)``.
    factor_std, factors_mean : np.ndarray
        Factor std and mean over the samples, ``(T, num_factors)``.
    """
    with torch.no_grad():
        factors_list = [encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
        factors_tensor = torch.stack(factors_list, dim=0)
        y_pred_samples_tensor = torch.stack([decoder(f) for f in factors_list], dim=0)

        if factors_tensor.shape[0] != n_mc_samples:
            raise ValueError(
                f"MC samples dimension mismatch: factors_tensor.shape[0]={factors_tensor.shape[0]} != n_mc_samples={n_mc_samples}"
            )

        y_pred_full_tensor, y_pred_std_tensor = compute_tensor_stats(y_pred_samples_tensor)
        y_pred_tensor = extract_target_predictions(y_pred_full_tensor)
        y_pred_full = to_numpy(y_pred_full_tensor)
        y_pred = to_numpy(y_pred_tensor)
        y_pred_std = to_numpy(y_pred_std_tensor)
        factors = to_numpy(factors_tensor)

        factors_mean_tensor, factors_std_tensor = compute_tensor_stats(factors_tensor)
        factor_std = to_numpy(factors_std_tensor)
        factors_mean = to_numpy(factors_mean_tensor)

    return factors, y_pred, y_pred_full, y_pred_std, factor_std, factors_mean


def check_variance_collapse(
    y_pred_std: np.ndarray,
    y_pred_full: np.ndarray,
    factors_mean: np.ndarray,
    y_actual: np.ndarray,
    target_scaler: Any,
    encoder: Any,
    decoder: Any,
    factors_std: np.ndarray,
    num_iter: int,
    disp: int
) -> Optional[dict]:
    """Run ``diagnose_variance_collapse`` every ``disp`` iterations, or when the mean prediction std is small.

    Logs a warning when a collapse is detected. Returns the diagnostics
    dict when the check ran, else None.
    """
    y_pred_std_mean_check = compute_variance_mean(y_pred_std)
    should_check_variance = (
        (y_pred_std_mean_check is not None and y_pred_std_mean_check < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD) or
        (num_iter % disp == 0)
    )

    if should_check_variance:
        variance_diagnostics = diagnose_variance_collapse(
            prediction_std=y_pred_std,
            prediction_mean=y_pred_full,
            factors_mean=factors_mean,
            y_actual=y_actual,
            target_scaler=target_scaler,
            encoder=encoder,
            decoder=decoder,
            factors_std=factors_std
        )
        if variance_diagnostics['variance_collapse_detected']:
            _logger.warning(f"Variance collapse detected at iteration {num_iter}: {', '.join(variance_diagnostics['warnings'])}")
        return variance_diagnostics

    return None


def run_mcmc_iteration(
    eps: np.ndarray,
    data_imputed: pd.DataFrame,
    data_denoised: pd.DataFrame,
    dataset: Any,
    encoder: Any,
    decoder: Any,
    autoencoder: Any,
    y_actual: np.ndarray,
    lags_input: int,
    n_mc_samples: int,
    window_size: int,
    learning_rate: float,
    optimizer_type: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    extract_target_predictions: callable,
    interpolation_method: str,
    interpolation_limit: Optional[int],
    interpolation_limit_direction: str,
    target_scaler: Any,
    num_iter: int,
    disp: int,
    device: torch.device,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[Any]]:
    """One MCMC iteration: denoise, draw the MC samples, train on them, predict, check for collapse.

    ``data_denoised`` is updated in place (:func:`denoise_targets`); the MC
    noise is drawn from ``rng``. See the called functions for the arguments.

    Returns
    -------
    factors : np.ndarray
        ``(n_mc_samples, T, num_factors)``.
    y_pred, y_pred_full, y_pred_std : np.ndarray
        ``(T, num_target_series)``.
    factor_std : np.ndarray
        ``(T, num_factors)``.
    data_denoised_interpolated : pd.DataFrame
    autoencoder_datasets : list of AutoencoderDataset
        The MC samples (the MLP decoder's ``last_neurons`` are read off them).
    """
    data_denoised_interpolated, Phi, mu_eps, std_eps = denoise_targets(
        eps=eps,
        data_imputed=data_imputed,
        data_denoised=data_denoised,
        dataset=dataset,
        lags_input=lags_input,
        interpolation_method=interpolation_method,
        interpolation_limit=interpolation_limit,
        interpolation_limit_direction=interpolation_limit_direction
    )

    X_features_df, y_tmp = dataset.split_features_and_targets(data_denoised_interpolated)
    X_features = X_features_df if X_features_df is not None else pd.DataFrame()

    autoencoder_datasets = dataset.create_autoencoder_datasets_list(
        n_mc_samples=n_mc_samples,
        mu_eps=mu_eps,
        std_eps=std_eps,
        X=X_features,
        y_tmp=y_tmp,
        y_actual=y_actual,
        rng=rng,
        device=device
    )

    train_on_mc_samples(
        autoencoder=autoencoder,
        autoencoder_datasets=autoencoder_datasets,
        window_size=window_size,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        optimizer=optimizer,
        scheduler=scheduler,
        target_indices=None,
        device=device
    )

    factors, y_pred, y_pred_full, y_pred_std, factor_std, factors_mean = extract_predictions_from_mc_samples(
        encoder=encoder,
        decoder=decoder,
        autoencoder_datasets=autoencoder_datasets,
        n_mc_samples=n_mc_samples,
        extract_target_predictions=extract_target_predictions,
        device=device
    )

    check_variance_collapse(
        y_pred_std=y_pred_std,
        y_pred_full=y_pred_full,
        factors_mean=factors_mean,
        y_actual=y_actual,
        target_scaler=target_scaler,
        encoder=encoder,
        decoder=decoder,
        factors_std=factor_std,
        num_iter=num_iter,
        disp=disp
    )

    return factors, y_pred, y_pred_full, y_pred_std, factor_std, data_denoised_interpolated, autoencoder_datasets
