"""Inferring input dimensions from data or a checkpoint state dict.

The state-dict helper defaults to the DDFM autoencoder's key names. Saved
DDFM runs are directories (see ``DDFM.save`` / ``DDFM.load``), so these
helpers only serve callers that still hold a raw ``state_dict``.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ..utils.errors import DataError, DataValidationError
from ..utils.validation import has_shape_with_min_dims


def parse_checkpoint(checkpoint: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a checkpoint into ``(state_dict, hyper_parameters)``.

    A dict with a ``'state_dict'`` key is unpacked (``hyper_parameters``
    defaults to ``{}``); anything else is taken to be the state dict itself.
    """
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint.get('hyper_parameters', {})
    return checkpoint, {}


def infer_input_dim_from_data(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    date_id_col: str = "date_id"
) -> int:
    """Number of features: DataFrame columns other than ``date_id_col``, or ``shape[1]``.

    Raises
    ------
    DataError
        If an array or tensor has fewer than two dimensions, or the type is unsupported.
    """
    if isinstance(data, pd.DataFrame):
        return len([c for c in data.columns if c != date_id_col])
    if isinstance(data, (np.ndarray, torch.Tensor)):
        if not has_shape_with_min_dims(data, min_dims=2):
            raise DataError(
                "Data must be at least 2D to infer input dimension",
                details=f"Got shape: {data.shape}"
            )
        return data.shape[1]
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
    """Infer the autoencoder input dimension from weight shapes in ``state_dict``.

    The first key containing ``encoder_prefix`` gives ``weight.shape[1]``.
    Otherwise the first key containing one of ``decoder_prefixes`` (default
    ``["decoder.decoder.weight", "decoder.layers.0.weight"]``) gives
    ``weight.shape[0]`` when ``output_dim_equals_input_dim`` else
    ``weight.shape[1]``. Non-tensor values are skipped; returns None when
    nothing matches.

    Raises
    ------
    DataValidationError
        If ``state_dict`` is not a dict.
    """
    if not isinstance(state_dict, dict):
        raise DataValidationError(
            "Cannot infer input_dim: state_dict must be a dictionary",
            details=f"Received type: {type(state_dict).__name__}"
        )

    first_layer_keys = [k for k in state_dict.keys() if encoder_prefix in k]
    if first_layer_keys:
        weight = state_dict[first_layer_keys[0]]
        if isinstance(weight, torch.Tensor):
            return weight.shape[1]

    if decoder_prefixes is None:
        decoder_prefixes = ["decoder.decoder.weight", "decoder.layers.0.weight"]

    for prefix in decoder_prefixes:
        for key in [k for k in state_dict.keys() if prefix in k]:
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor):
                if output_dim_equals_input_dim:
                    return weight.shape[0]
                return weight.shape[1]

    return None
