"""Private parsing helpers of the DFM engine config (``_DFMEngineConfig``).

``_DFMEngineConfig.from_dict`` and ``BaseModelConfig`` use these helpers to
normalize the frequency mapping (per-series ``{"gdp": "q"}`` or grouped
``{"q": ["gdp"]}``) and the blocks mapping, to raise ``ConfigurationError``,
and (:func:`detect_config_type`) to reject DDFM parameters passed to the DFM
engine config.

The old configuration sources (``ConfigSource``, ``YamlSource``,
``DictSource`` and ``make_config_source``) were removed. Configs are now the
estimators' constructor parameters: read a YAML file with
:func:`dfm_python.config.load_yaml` or ``<Model>Config.from_yaml`` (for
example ``dfm_python.DFMConfig.from_yaml(path)``), then
``Model.from_config(config)``.
"""

from typing import Optional, Dict, Any, NoReturn

from .constants import DEFAULT_CLOCK_FREQUENCY

# Error helper to reduce repetitive imports
def _get_config_error():
    """Get ConfigurationError (lazy import to avoid circular dependencies)."""
    from dfm_python.utils.errors import ConfigurationError
    return ConfigurationError


def _raise_config_error(message: str, details: Optional[str] = None) -> NoReturn:
    """Raise ConfigurationError with message and optional details.
    
    Parameters
    ----------
    message : str
        Error message
    details : str, optional
        Additional error details
        
    Raises
    ------
    ConfigurationError
        Always raises (NoReturn)
    """
    ConfigurationError = _get_config_error()
    if details:
        raise ConfigurationError(f"{message}\nDetails: {details}")
    else:
        raise ConfigurationError(message)


# ============================================================================
# Helper Functions for Config Parsing
# ============================================================================

def _is_dict_like(obj: Any) -> bool:
    """Check if object is dict-like (dict, OrderedDict, etc.).
    
    Parameters
    ----------
    obj : Any
        Object to check
        
    Returns
    -------
    bool
        True if object is dict-like, False otherwise
    """
    return isinstance(obj, dict) or hasattr(obj, 'keys') and hasattr(obj, '__getitem__')


def _convert_series_to_frequency_dict(
    series_data: Any,
    default_clock: str = DEFAULT_CLOCK_FREQUENCY
) -> Dict[str, str]:
    """Convert series data (dict or list) to frequency dict.
    
    Parameters
    ----------
    series_data : dict, list, or None
        Series data in various formats:
        - dict: {series_id: {frequency: ...}} or {series_id: frequency_string}
        - list: [{'series_id': ..., 'frequency': ...}, ...]
    default_clock : str
        Default clock frequency if not specified
        
    Returns
    -------
    Dict[str, str]
        Frequency dict mapping series_id to frequency
    """
    result = {}
    
    if _is_dict_like(series_data):
        for series_id, item in series_data.items():
            if _is_dict_like(item):
                freq = item.get('frequency', default_clock)
            elif isinstance(item, str):
                freq = item  # frequency as string
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
    """Extract frequency dict from data, handling multiple formats.
    
    Supports:
    1. Grouped format: {'w': [series1, series2, ...], 'm': [series3, ...]}
    2. Individual format: {'series1': 'w', 'series2': 'm', ...}
    3. Legacy format: {'series': [{'series_id': ..., 'frequency': ...}, ...]}
    
    Parameters
    ----------
    data : Dict[str, Any]
        Configuration data dictionary
    clock : str, default DEFAULT_CLOCK_FREQUENCY
        Default clock frequency if not specified
        
    Returns
    -------
    Optional[Dict[str, str]]
        Frequency dict mapping column names to frequencies, or None if not found
    """
    if 'frequency' not in data:
        # Legacy: check for 'series' key
        if 'series' in data:
            return _convert_series_to_frequency_dict(data['series'], clock)
        return None
    
    freq_data = data['frequency']
    
    # Check if it's in grouped format: {'w': [...], 'm': [...]}
    if _is_dict_like(freq_data):
        from .constants import VALID_FREQUENCIES
        
        # Check if all keys are valid frequency codes (grouped format)
        keys_are_frequencies = all(
            isinstance(k, str) and k in VALID_FREQUENCIES 
            for k in freq_data.keys()
        )
        
        # Check if values are lists (grouped format)
        values_are_lists = all(
            isinstance(v, (list, tuple)) for v in freq_data.values()
        )
        
        if keys_are_frequencies and values_are_lists:
            # Grouped format: convert to individual format
            result = {}
            for freq, series_list in freq_data.items():
                if not isinstance(series_list, (list, tuple)):
                    _raise_config_error(
                        f"Frequency '{freq}' must map to a list of series names, got {type(series_list)}"
                    )
                for series_name in series_list:
                    if not isinstance(series_name, str):
                        _raise_config_error(
                            f"Series names must be strings, got {type(series_name)} in frequency '{freq}'"
                        )
                    if series_name in result:
                        _raise_config_error(
                            f"Series '{series_name}' appears in multiple frequency groups"
                        )
                    result[series_name] = freq
            return result
        else:
            # Individual format: already in correct format, validate structure
            result = {}
            for series_name, freq in freq_data.items():
                if not isinstance(series_name, str):
                    _raise_config_error(
                        f"Frequency dict keys must be strings (series names), got {type(series_name)}"
                    )
                if not isinstance(freq, str):
                    _raise_config_error(
                        f"Frequency dict values must be strings (frequency codes), got {type(freq)} for series '{series_name}'"
                    )
                result[series_name] = freq
            return result
    
    return None


def _normalize_blocks_dict(blocks_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize blocks dict to format: {"block_name": {"num_factors": int, "series": [str]}}.
    
    Parameters
    ----------
    blocks_dict : Dict[str, Any]
        Blocks dict (may contain legacy 'factors' key)
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Normalized blocks dict with 'num_factors' and 'series' keys
        
    Raises
    ------
    ConfigurationError
        If block config is invalid
    """
    ConfigurationError = _get_config_error()
    
    normalized = {}
    for block_name, block_cfg in blocks_dict.items():
        if not isinstance(block_cfg, dict):
            _raise_config_error(f"Block '{block_name}' must be a dict")
        normalized[block_name] = {
            'num_factors': block_cfg.get('num_factors', block_cfg.get('factors', 1)),
            'series': block_cfg.get('series', []) if isinstance(block_cfg.get('series'), list) else []
        }
    return normalized


# ============================================================================
# Configuration Parsing Utilities
# ============================================================================

# Model type constants (exported for use in schema/model.py)
MODEL_TYPE_DDFM = 'ddfm'
MODEL_TYPE_DFM = 'dfm'

# Model type detection patterns
_DDFM_TYPE_ALIASES = {'ddfm', 'deep'}
_DDFM_PARAMS = {'encoder_layers', 'epochs', 'learning_rate', 'batch_size'}

def detect_config_type(data: Dict[str, Any]) -> str:
    """Detect config type (DFM or DDFM) from data dictionary.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Configuration data dictionary
        
    Returns
    -------
    str
        'ddfm' or 'dfm'
        
    Raises
    ------
    ConfigurationError
        If data is not a dictionary
    """
    ConfigurationError = _get_config_error()
    
    if not _is_dict_like(data):
        _raise_config_error(f"data must be a dictionary, got {type(data).__name__}")
    
    model_type = data.get('model_type', '').lower()
    
    # Check explicit model type
    if model_type in _DDFM_TYPE_ALIASES:
        return MODEL_TYPE_DDFM
    if model_type == MODEL_TYPE_DFM:
        return MODEL_TYPE_DFM
    
    # Priority check: blocks parameter indicates DFM (highest priority)
    # DFM is the only model type that uses blocks structure
    if 'blocks' in data and data.get('blocks'):
        return MODEL_TYPE_DFM
    
    # Check for model-specific parameters
    keys = set(data.keys())
    if any(k.startswith('ddfm_') for k in keys) or (keys & _DDFM_PARAMS):
        return MODEL_TYPE_DDFM
    
    return MODEL_TYPE_DFM

