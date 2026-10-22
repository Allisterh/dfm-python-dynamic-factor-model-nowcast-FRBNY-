"""Linear and MLP decoders for DDFM."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from ..base import BaseDecoder
from ...utils.errors import ConfigurationError
from ...utils.seed import apply_seed
from ...config.constants import (
    DEFAULT_XAVIER_GAIN,
    DEFAULT_OUTPUT_LAYER_GAIN,
    DEFAULT_ZERO_VALUE,
    DEFAULT_CLEAN_NAN,
)
from ...logger import get_logger

_logger = get_logger(__name__)


def _linear_params(layer: nn.Linear, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(weight, bias)`` of ``layer`` as numpy, shapes ``(out, in)`` and ``(out,)``.

    A missing bias is zeros; NaN weights are replaced by zeros with a warning
    logged under ``name``.
    """
    weight = layer.weight.data.cpu().numpy()
    bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])
    if np.any(np.isnan(weight)):
        _logger.warning(f"{name}: C matrix contains NaN values. Replacing with zeros.")
        weight = np.nan_to_num(weight, nan=DEFAULT_CLEAN_NAN)
    return weight, bias


class DDFMLinearDecoder(BaseDecoder):
    """Linear decoder: one ``nn.Linear`` from the factors to the targets.

    Parameters
    ----------
    input_dim : int
        Number of factors.
    output_dim : int
        Number of targets.
    use_bias : bool, default True
    seed : int, optional
        Seeds the global RNGs before the Xavier-normal initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)

        if seed is not None:
            apply_seed(seed)

        # The same gain as the encoder's initialization (matches GlorotNormal).
        nn.init.xavier_normal_(self.decoder.weight, gain=DEFAULT_XAVIER_GAIN)
        if self.decoder.bias is not None:
            nn.init.constant_(self.decoder.bias, DEFAULT_ZERO_VALUE)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """``(..., num_factors)`` factors to ``(..., output_dim)`` observations."""
        return self.decoder(f)

    def extract_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(weight, bias)`` of the decoder, shapes ``(output_dim, input_dim)`` and ``(output_dim,)``."""
        return _linear_params(self.decoder, "LinearDecoder")

    def get_last_linear_layer(self) -> nn.Linear:
        """The output layer (the loadings of the state space)."""
        return self.decoder

    def get_intermediate(self) -> Optional[nn.Module]:
        """None: a linear decoder has no hidden layers."""
        return None


class DDFMMLPDecoder(BaseDecoder):
    """MLP decoder: hidden layers with activations, then a linear output layer.

    Parameters
    ----------
    input_dim : int
        Number of factors.
    output_dim : int
        Number of targets.
    hidden_dims : list of int, optional
        Hidden layer widths; ``[output_dim]`` when None.
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
    use_bias : bool, default True
    seed : int, optional
        Seeds the global RNGs before the initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        use_bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [output_dim]

        self.layers = nn.ModuleList()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ConfigurationError(f"Unknown activation: {activation}")

        if seed is not None:
            apply_seed(seed)

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim, bias=use_bias)
            if activation == 'relu':
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
            self.layers.append(layer)
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=DEFAULT_OUTPUT_LAYER_GAIN)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, DEFAULT_ZERO_VALUE)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """``(..., num_factors)`` factors to ``(..., output_dim)`` observations."""
        x = f
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    def extract_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(weight, bias)`` of the output layer, shapes ``(output_dim, hidden_dims[-1])`` and ``(output_dim,)``."""
        return _linear_params(self.output_layer, "DDFMMLPDecoder")

    def get_last_linear_layer(self) -> nn.Linear:
        """The output layer (the loadings of the state space)."""
        return self.output_layer

    def get_intermediate(self) -> nn.Sequential:
        """The hidden layers with their activations, whose output is ``last_neurons``.

        Raises
        ------
        ConfigurationError
            If the decoder has no hidden layer.
        """
        if len(self.layers) == 0:
            raise ConfigurationError(
                "MLPDecoder has no intermediate layers for last_neurons extraction. "
                "Decoder must have at least one hidden layer."
            )
        intermediate_modules = []
        for layer in self.layers:
            intermediate_modules.append(layer)
            intermediate_modules.append(self.activation)
        return nn.Sequential(*intermediate_modules)
