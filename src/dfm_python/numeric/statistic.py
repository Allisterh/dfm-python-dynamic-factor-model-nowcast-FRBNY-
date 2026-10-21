"""Summary statistics and variance-collapse diagnostics for the DDFM MC sampler."""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from torch import Tensor
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..config.types import to_numpy
from ..config.constants import (
    DEFAULT_VARIANCE_COLLAPSE_THRESHOLD,
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
    DEFAULT_TARGET_PREDICTION_STD,
)


def compute_variance_mean(variance_array: Optional[np.ndarray]) -> Optional[float]:
    """``float(mean(variance_array))``, or None for None."""
    if variance_array is None:
        return None
    return float(np.mean(variance_array))


def compute_array_stats(array: np.ndarray, use_nan: bool = False) -> Tuple[float, float, float, float]:
    """Return ``(mean, std, min, max)`` as floats, NaN-aware when ``use_nan``."""
    if use_nan:
        return (
            float(np.nanmean(array)),
            float(np.nanstd(array)),
            float(np.nanmin(array)),
            float(np.nanmax(array))
        )
    return (
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array))
    )


def compute_tensor_stats(tensor: Tensor, dim: int = 0, unbiased: bool = False) -> Tuple[Tensor, Tensor]:
    """Return ``(mean, std)`` of ``tensor`` along ``dim``.

    The default is the population std (``unbiased=False``), like TensorFlow's
    ``tf.reduce_std`` in the reference implementation, not torch's default
    Bessel-corrected std; with ``n_mc_samples=10`` the difference is material.
    """
    return tensor.mean(dim=dim), tensor.std(dim=dim, unbiased=unbiased)


def average_3d_array(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean over ``axis`` (the MC-sample axis) when ``array`` is 3-D, else ``array`` unchanged."""
    if array.ndim == 3:
        return np.mean(array, axis=axis)
    return array


def extract_batchnorm_statistics(
    encoder: nn.Module,
    decoder: nn.Module,
    compute_variance_mean_fn: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Running statistics of every BatchNorm1d/2d layer in ``encoder`` and ``decoder``.

    Returns one dict per layer with keys ``module`` ('encoder' or 'decoder'),
    ``layer``, ``running_mean_abs`` (mean |running_mean|) and
    ``running_var_mean`` (``compute_variance_mean_fn(running_var)``, default
    :func:`compute_variance_mean`).
    """
    if compute_variance_mean_fn is None:
        compute_variance_mean_fn = compute_variance_mean

    batchnorm_stats = []
    for module_name, module in [('encoder', encoder), ('decoder', decoder)]:
        for name, submodule in module.named_modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
                running_mean = to_numpy(submodule.running_mean) if submodule.running_mean.numel() > 0 else None
                running_var = to_numpy(submodule.running_var) if submodule.running_var.numel() > 0 else None
                if running_mean is not None and running_var is not None:
                    mean_abs = float(np.mean(np.abs(running_mean)))
                    var_mean = compute_variance_mean_fn(running_var)
                    batchnorm_stats.append({
                        'module': module_name,
                        'layer': name,
                        'running_mean_abs': mean_abs,
                        'running_var_mean': var_mean
                    })
    return batchnorm_stats


def diagnose_variance_collapse(
    prediction_std: np.ndarray,
    prediction_mean: np.ndarray,
    factors_mean: np.ndarray,
    y_actual: np.ndarray,
    target_scaler: Optional[Any],
    encoder: nn.Module,
    decoder: nn.Module,
    factors_std: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Diagnose why the MC prediction std (T x N) is too low.

    Compares prediction scale with data scale, factor magnitudes, BatchNorm
    running variances and per-time-step std. ``prediction_mean`` is unused.
    Shapes: ``prediction_std``, ``prediction_mean``, ``y_actual`` (T x N);
    ``factors_mean``, ``factors_std`` (T x m). ``target_scaler`` must be a
    ``StandardScaler`` for the standardization check to pass.

    Returns a dict with ``prediction_std_mean``, ``variance_collapse_detected``
    and ``warnings`` (list of str); when ``prediction_std`` is a valid 2-D
    array also ``data_std_mean``, ``data_mean_abs``, ``is_standardized``,
    ``scale_ratio``, ``factors_mean_abs``, ``batchnorm_stats``,
    ``factors_std_mean`` (if ``factors_std`` given) and, for T > 1,
    ``timestep_collapse_count`` / ``timestep_collapse_ratio``.
    """
    diagnostics = {
        'prediction_std_mean': None,
        'variance_collapse_detected': False,
        'warnings': []
    }

    if not isinstance(prediction_std, np.ndarray):
        diagnostics['warnings'].append(f"Invalid prediction_std type: {type(prediction_std).__name__}, expected np.ndarray")
        return diagnostics
    if prediction_std.ndim == 0 or prediction_std.size == 0:
        diagnostics['warnings'].append(f"Invalid prediction_std shape: {prediction_std.shape}, expected 2D array (T, N)")
        return diagnostics
    if prediction_std.ndim == 1:
        diagnostics['warnings'].append(f"1D prediction_std array (shape: {prediction_std.shape}), per-time-step analysis skipped")
        return diagnostics
    if prediction_std.ndim != 2:
        diagnostics['warnings'].append(f"Invalid prediction_std dimensions: {prediction_std.ndim}, expected 2D array (T, N)")
        return diagnostics

    diagnostics['prediction_std_mean'] = compute_variance_mean(prediction_std)

    data_mean, data_std, _, _ = compute_array_stats(y_actual)
    diagnostics['data_std_mean'] = data_std
    diagnostics['data_mean_abs'] = abs(data_mean)

    is_standardized = (
        isinstance(target_scaler, StandardScaler) and
        abs(data_mean) < DEFAULT_STANDARDIZATION_MEAN_THRESHOLD and
        DEFAULT_STANDARDIZATION_STD_MIN <= data_std <= DEFAULT_STANDARDIZATION_STD_MAX
    )
    diagnostics['is_standardized'] = is_standardized
    
    if not is_standardized:
        diagnostics['warnings'].append(
            f"Data standardization assumption not verified: data_mean={data_mean:.6f}, data_std={data_std:.6f} "
            f"(expected: mean≈0, std≈{DEFAULT_TARGET_PREDICTION_STD} for StandardScaler). Diagnostics may be inaccurate."
        )
        target_std = data_std
    else:
        target_std = DEFAULT_TARGET_PREDICTION_STD
    
    diagnostics['scale_ratio'] = diagnostics['prediction_std_mean'] / data_std if data_std > 0 else float('inf')
    
    if diagnostics['prediction_std_mean'] < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD:
        diagnostics['variance_collapse_detected'] = True
        diagnostics['warnings'].append(
            f"Variance collapse detected: prediction_std={diagnostics['prediction_std_mean']:.6f} << target ~{target_std:.6f}"
        )
    
    if diagnostics['scale_ratio'] < DEFAULT_SCALE_RATIO_MIN:
        diagnostics['warnings'].append(f"Scale mismatch: prediction_std/data_std={diagnostics['scale_ratio']:.6f} << target ~{target_std:.6f}")
    
    factors_mean_abs = float(np.mean(np.abs(factors_mean)))
    diagnostics['factors_mean_abs'] = factors_mean_abs
    if factors_mean_abs < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
        diagnostics['warnings'].append(f"Factor collapse: |factors_mean|={factors_mean_abs:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
    
    if factors_std is not None:
        factors_std_mean = compute_variance_mean(factors_std)
        diagnostics['factors_std_mean'] = factors_std_mean
        if factors_std_mean < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
            diagnostics['warnings'].append(f"Factor variance collapse: factors_std={factors_std_mean:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
    
    batchnorm_stats = extract_batchnorm_statistics(encoder, decoder)
    for stat in batchnorm_stats:
        if stat['running_var_mean'] < DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD:
            diagnostics['warnings'].append(
                f"BatchNorm signal suppression in {stat['module']}.{stat['layer']}: running_var={stat['running_var_mean']:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}"
            )
    diagnostics['batchnorm_stats'] = batchnorm_stats
    
    if len(prediction_std) > 1:
        per_timestep_std = np.mean(prediction_std, axis=1)
        timestep_collapse_count = int(np.sum(per_timestep_std < DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD))
        timestep_collapse_ratio = timestep_collapse_count / len(per_timestep_std)
        diagnostics['timestep_collapse_count'] = timestep_collapse_count
        diagnostics['timestep_collapse_ratio'] = timestep_collapse_ratio
        if timestep_collapse_ratio > DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD:
            diagnostics['warnings'].append(
                f"Localized variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD} "
                f"(ratio={timestep_collapse_ratio:.2%})"
            )
        elif timestep_collapse_count > 0:
            diagnostics['warnings'].append(
                f"Partial variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD}"
            )
    
    return diagnostics


__all__ = [
    'compute_variance_mean',
    'compute_array_stats',
    'compute_tensor_stats',
    'average_3d_array',
    'extract_batchnorm_statistics',
    'diagnose_variance_collapse',
]

