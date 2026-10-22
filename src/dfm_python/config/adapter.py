"""Private parsing helpers of the DFM engine config (``_DFMEngineConfig``).

``_DFMEngineConfig.from_dict`` and ``BaseModelConfig`` use these helpers to
normalize the frequency mapping (per-series ``{"gdp": "q"}`` or grouped
``{"q": ["gdp"]}``) and the blocks mapping, to raise ``ConfigurationError``,
and (:func:`detect_config_type`) to reject DDFM parameters passed to the DFM
engine config. Estimator parameters are read from YAML with
:func:`dfm_python.config.load_yaml` or ``<Model>Config.from_yaml``.
"""

from typing import Optional, Dict, Any, NoReturn

from .constants import DEFAULT_CLOCK_FREQUENCY, VALID_FREQUENCIES


def _raise_config_error(message: str, details: Optional[str] = None) -> NoReturn:
    """Raise ``ConfigurationError``; ``details`` go on a second line."""
    # Lazy import: dfm_python.utils imports dfm_python.config.
    from dfm_python.utils.errors import ConfigurationError
    if details:
        raise ConfigurationError(f"{message}\nDetails: {details}")
    raise ConfigurationError(message)


def _is_dict_like(obj: Any) -> bool:
    return isinstance(obj, dict) or hasattr(obj, 'keys') and hasattr(obj, '__getitem__')


def _convert_series_to_frequency_dict(
    series_data: Any,
    default_clock: str = DEFAULT_CLOCK_FREQUENCY
) -> Dict[str, str]:
    """Frequency dict from the legacy ``series`` entry.

    ``series_data`` is ``{series_id: {"frequency": ...}}``,
    ``{series_id: "q"}`` or ``[{"series_id": ..., "frequency": ...}, ...]``;
    a missing frequency is ``default_clock``.
    """
    result = {}
    if _is_dict_like(series_data):
        for series_id, item in series_data.items():
            if _is_dict_like(item):
                freq = item.get('frequency', default_clock)
            elif isinstance(item, str):
                freq = item
            else:
                freq = default_clock
            result[series_id] = freq
    elif isinstance(series_data, list):
        for item in series_data:
            if _is_dict_like(item):
                series_id = item.get('series_id', f"series_{len(result)}")
                result[series_id] = item.get('frequency', default_clock)
    return result


def _extract_frequency_dict(
    data: Dict[str, Any],
    clock: str = DEFAULT_CLOCK_FREQUENCY
) -> Optional[Dict[str, str]]:
    """Per-series frequency dict from ``data["frequency"]``, or None when absent.

    Accepts the grouped form ``{'w': [series...], 'm': [...]}`` (every key a
    frequency code and every value a list), the per-series form
    ``{'series1': 'w', ...}`` and the legacy ``data["series"]`` entry (see
    :func:`_convert_series_to_frequency_dict`).
    """
    if 'frequency' not in data:
        if 'series' in data:
            return _convert_series_to_frequency_dict(data['series'], clock)
        return None

    freq_data = data['frequency']
    if not _is_dict_like(freq_data):
        return None

    grouped = all(isinstance(k, str) and k in VALID_FREQUENCIES for k in freq_data.keys()) and all(
        isinstance(v, (list, tuple)) for v in freq_data.values()
    )
    result = {}
    if grouped:
        for freq, series_list in freq_data.items():
            for series_name in series_list:
                if not isinstance(series_name, str):
                    _raise_config_error(
                        f"Series names must be strings, got {type(series_name)} in frequency '{freq}'"
                    )
                if series_name in result:
                    _raise_config_error(f"Series '{series_name}' appears in multiple frequency groups")
                result[series_name] = freq
        return result
    for series_name, freq in freq_data.items():
        if not isinstance(series_name, str):
            _raise_config_error(f"Frequency dict keys must be strings (series names), got {type(series_name)}")
        if not isinstance(freq, str):
            _raise_config_error(
                f"Frequency dict values must be strings (frequency codes), got {type(freq)} for series '{series_name}'"
            )
        result[series_name] = freq
    return result


def _normalize_blocks_dict(blocks_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize to ``{"block_name": {"num_factors": int, "series": [str]}}``.

    The legacy ``factors`` key is read as ``num_factors``; a ``series`` entry
    that is not a list becomes ``[]``. Raises ``ConfigurationError`` for a
    block that is not a dict.
    """
    normalized = {}
    for block_name, block_cfg in blocks_dict.items():
        if not isinstance(block_cfg, dict):
            _raise_config_error(f"Block '{block_name}' must be a dict")
        normalized[block_name] = {
            'num_factors': block_cfg.get('num_factors', block_cfg.get('factors', 1)),
            'series': block_cfg.get('series', []) if isinstance(block_cfg.get('series'), list) else []
        }
    return normalized


MODEL_TYPE_DDFM = 'ddfm'
MODEL_TYPE_DFM = 'dfm'

_DDFM_TYPE_ALIASES = {'ddfm', 'deep'}
_DDFM_PARAMS = {'encoder_layers', 'epochs', 'learning_rate', 'batch_size'}


def detect_config_type(data: Dict[str, Any]) -> str:
    """Return ``'ddfm'`` or ``'dfm'`` for a config dict.

    An explicit ``model_type`` wins; otherwise ``blocks`` means DFM (the only
    model with blocks), and a ``ddfm_*`` key or one of ``encoder_layers``,
    ``epochs``, ``learning_rate``, ``batch_size`` means DDFM. Raises
    ``ConfigurationError`` if ``data`` is not a dict.
    """
    if not _is_dict_like(data):
        _raise_config_error(f"data must be a dictionary, got {type(data).__name__}")

    model_type = data.get('model_type', '').lower()
    if model_type in _DDFM_TYPE_ALIASES:
        return MODEL_TYPE_DDFM
    if model_type == MODEL_TYPE_DFM:
        return MODEL_TYPE_DFM
    if data.get('blocks'):
        return MODEL_TYPE_DFM
    keys = set(data.keys())
    if any(k.startswith('ddfm_') for k in keys) or (keys & _DDFM_PARAMS):
        return MODEL_TYPE_DDFM
    return MODEL_TYPE_DFM
