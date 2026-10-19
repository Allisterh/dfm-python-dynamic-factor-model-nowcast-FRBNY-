"""Checkpoint loading utilities for factor models.

This module provides utilities for loading and parsing PyTorch model checkpoints,
inferring model parameters from state dictionaries, and extracting input dimensions.

The state-dict helper defaults to the DDFM autoencoder's key names and can be
parameterized for other autoencoder-based models. Saved DDFM runs are
directories (see ``DDFM.save``/``DDFM.load``); the single-file checkpoint
helpers ``infer_ddfm_input_dim`` and ``infer_ddfm_params_from_state_dict``
were removed with the old ``torch.save`` checkpoints.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None

from ..logger import get_logger
from ..utils.errors import DataError, DataValidationError, ConfigurationError
from ..utils.validation import has_shape_with_min_dims
from ..config.constants import MAX_WARNING_ITEMS

_logger = get_logger(__name__)


def parse_checkpoint(checkpoint: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract state_dict and hyperparameters from checkpoint.
    
    Parameters
    ----------
    checkpoint : Any
        Checkpoint object (dict with 'state_dict' key or raw state_dict)
        
    Returns
    -------
    state_dict : Dict[str, Any]
        Model state dictionary
    hparams : Dict[str, Any]
        Hyperparameters dictionary (empty if not available)
    """
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint.get('hyper_parameters', {})
    return checkpoint, {}


def infer_input_dim_from_data(
    data: Union[pd.DataFrame, np.ndarray, "torch.Tensor"],
    date_id_col: str = "date_id"
) -> int:
    """Infer input dimension from data.
    
    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or torch.Tensor
        Input data
    date_id_col : str, default "date_id"
        Column name to exclude if data is DataFrame
        
    Returns
    -------
    int
        Input dimension (number of features)
        
    Raises
    ------
    DataError
        If data type is unsupported or shape is invalid
    """
    if isinstance(data, pd.DataFrame):
        # Exclude date column if present
        feature_cols = [c for c in data.columns if c != date_id_col]
        return len(feature_cols)
    elif isinstance(data, np.ndarray):
        if not has_shape_with_min_dims(data, min_dims=2):
            raise DataError(
                "Data must be at least 2D to infer input dimension",
                details=f"Got shape: {data.shape}"
            )
        return data.shape[1]
    elif _has_torch and isinstance(data, torch.Tensor):
        if not has_shape_with_min_dims(data, min_dims=2):
            raise DataError(
                "Data must be at least 2D to infer input dimension",
                details=f"Got shape: {data.shape}"
            )
        return data.shape[1]
    else:
        raise DataError(
            f"Unsupported data type: {type(data)}",
            details="Data must be pandas.DataFrame, numpy.ndarray, or torch.Tensor"
        )


def infer_input_dim_from_state_dict(
    state_dict: Dict[str, Any],
    encoder_prefix: str = "encoder.layers.0.weight",
    decoder_prefixes: Optional[List[str]] = None,
    output_dim_equals_input_dim: bool = True
) -> Optional[int]:
    """Infer input_dim from checkpoint state_dict (generalized for autoencoder models).
    
    This function can be used for DDFM or other autoencoder-based models by
    specifying the appropriate key prefixes.
    
    Parameters
    ----------
    state_dict : Dict[str, Any]
        Model state dictionary
    encoder_prefix : str, default "encoder.layers.0.weight"
        Key pattern for encoder first layer weight (shape: (hidden_dim, input_dim))
    decoder_prefixes : List[str], optional
        Key patterns for decoder layers. If None, uses default DDFM patterns.
        For DDFM: ["decoder.decoder.weight", "decoder.layers.0.weight"]
    output_dim_equals_input_dim : bool, default True
        If True, assumes decoder output_dim equals input_dim (DDFM convention).
        If False, decoder shape interpretation may differ.
        
    Returns
    -------
    Optional[int]
        Inferred input dimension, or None if cannot be determined
        
    Raises
    ------
    DataValidationError
        If state_dict is invalid
    """
    if not isinstance(state_dict, dict):
        raise DataValidationError(
            "Cannot infer input_dim: state_dict must be a dictionary",
            details=f"Received type: {type(state_dict).__name__}"
        )
    
    # Check encoder first layer: (hidden_dim, input_dim)
    first_layer_keys = [k for k in state_dict.keys() if encoder_prefix in k]
    if first_layer_keys:
        weight = state_dict[first_layer_keys[0]]
        if _has_torch and isinstance(weight, torch.Tensor):
            return weight.shape[1]  # input_dim is second dimension
    
    # Check decoder layers: (output_dim, num_factors) or (output_dim, input_dim)
    if decoder_prefixes is None:
        decoder_prefixes = ["decoder.decoder.weight", "decoder.layers.0.weight"]
    
    for prefix in decoder_prefixes:
        decoder_keys = [k for k in state_dict.keys() if prefix in k]
        for key in decoder_keys:
            weight = state_dict[key]
            if _has_torch and isinstance(weight, torch.Tensor):
                if output_dim_equals_input_dim:
                    return weight.shape[0]  # output_dim = input_dim (DDFM)
                else:
                    return weight.shape[1]  # input_dim is second dimension
    
    return None
