"""Parameter resolution, config access, target-series and scaler lookup helpers."""

from typing import Optional, Any, List, Tuple, Dict, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..config.schema.model import _DFMEngineConfig

from ..logger import get_logger
from ..config.constants import DEFAULT_CLOCK_FREQUENCY, MAX_WARNING_ITEMS, MAX_ERROR_ITEMS
from .errors import ConfigValidationError, DataError

_logger = get_logger(__name__)


def resolve_named_param(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Any:
    """First of ``kwargs[name]`` (popped), ``getattr(config, name)``, ``defaults[name]``; else None."""
    if kwargs is not None and name in kwargs:
        return kwargs.pop(name) if isinstance(kwargs, dict) else kwargs.get(name)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    if defaults is not None and name in defaults:
        return defaults[name]
    return None


def resolve_param(
    override: Optional[Any] = None,
    default: Any = None,
    *,
    name: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Any:
    """``override`` if not None else ``default``; with ``name`` defer to :func:`resolve_named_param`."""
    if name is not None:
        return resolve_named_param(name, kwargs=kwargs, config=config, defaults=defaults)
    if override is not None:
        return override
    return default


def get_config_attr(
    config: Optional[Any],
    attr_name: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """``getattr(config, attr_name)`` when set and not None, else ``default``.

    Raises ``ConfigValidationError`` instead of returning ``default`` when ``required``.
    """
    if config is None:
        if required:
            raise ConfigValidationError(f"Config is None, cannot access required attribute '{attr_name}'")
        return default

    if hasattr(config, attr_name):
        value = getattr(config, attr_name)
        if value is not None:
            return value

    if required:
        raise ConfigValidationError(f"Config missing required attribute '{attr_name}'")
    return default


def get_clock_frequency(config: Optional["_DFMEngineConfig"], default: Optional[str] = None) -> str:
    """``config.clock``, else ``default``, else ``DEFAULT_CLOCK_FREQUENCY``."""
    return get_config_attr(config, 'clock', default or DEFAULT_CLOCK_FREQUENCY)


def resolve_target_series(
    dataset: Optional[Any],
    series_ids: Optional[List[str]] = None,
    result: Optional[Any] = None,
    model_name: str = "model"
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Target series = all series minus ``dataset.covariates``.

    Without covariates (or without a dataset) every entry of ``series_ids``
    is a target. ``result`` is unused. Returns ``(target_ids, target_indices)``
    with indices into ``series_ids``; both are None when they cannot be
    resolved. Targets missing from ``series_ids`` are skipped with a warning.

    Raises
    ------
    DataError
        If none of the targets is in ``series_ids``.
    """
    target_series = None
    if dataset is not None:
        covariates = getattr(dataset, 'covariates', None)
        if covariates is not None and len(covariates) > 0:
            if series_ids is not None:
                valid_covariates = [c for c in covariates if c in series_ids]
                target_series = [s for s in series_ids if s not in valid_covariates]
            else:
                processed_columns = getattr(dataset, '_processed_columns', None)
                if processed_columns is not None:
                    valid_covariates = [c for c in covariates if c in processed_columns]
                    target_series = [s for s in processed_columns if s not in valid_covariates]

    if (target_series is None or len(target_series) == 0) and series_ids is not None:
        target_series = list(series_ids)
        _logger.debug(f"{model_name}: No target_series specified, using all {len(series_ids)} series as targets")

    target_indices = None
    if target_series is not None and series_ids is not None:
        target_indices = []
        for tgt_id in target_series:
            if tgt_id in series_ids:
                target_indices.append(series_ids.index(tgt_id))
            else:
                _logger.warning(
                    f"{model_name} prediction: target series '{tgt_id}' not found in series_ids. "
                    f"Available: {series_ids[:MAX_WARNING_ITEMS]}{'...' if len(series_ids) > MAX_WARNING_ITEMS else ''}. "
                    f"Skipping this target series."
                )

        if len(target_indices) == 0:
            raise DataError(
                f"{model_name} prediction failed: none of the specified target series found",
                details=f"Target: {target_series}, Available: {series_ids[:MAX_ERROR_ITEMS]}{'...' if len(series_ids) > MAX_ERROR_ITEMS else ''}"
            )

    return target_series, target_indices


def _check_sklearn() -> bool:
    """Always True: scikit-learn is a required dependency. Kept for existing callers."""
    return True


def select_columns_by_prefix(df: pd.DataFrame, prefixes: List[str], count_per_prefix: int = 2) -> List[str]:
    """Up to ``count_per_prefix`` column names per prefix, in column order."""
    selected = []
    for prefix in prefixes:
        matching = [col for col in df.columns if col.startswith(prefix)]
        selected.extend(matching[:count_per_prefix])
    return selected


def get_target_scaler(dataset: Optional[Any] = None, model: Optional[Any] = None) -> Optional[Any]:
    """``scaler`` (else the older ``target_scaler``) attribute of ``dataset`` if given, else of ``model``."""
    if dataset is not None:
        return getattr(dataset, 'scaler', None) or getattr(dataset, 'target_scaler', None)
    if model is not None:
        return getattr(model, 'scaler', None) or getattr(model, 'target_scaler', None)
    return None


# Re-exported so existing ``from dfm_python.utils.misc import calculate_*`` keeps working.
from .metric import (  # noqa: E402, F401
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
)
