"""Multi-layer perceptron shared by the iVDFM encoder, prior and regime networks."""

from numbers import Number
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.constants import (
    DEFAULT_XAVIER_GAIN,
    DEFAULT_ZERO_VALUE,
)
from ..utils.seed import apply_seed


class MLP(nn.Module):
    """``n_hidden_layers`` hidden layers (activation, then optional batch/layer norm and dropout) and a linear output.

    Every linear layer is Xavier-normal initialized with zero bias.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dim : int or list of int, default 200
        One width for every hidden layer, or one per layer (``n_hidden_layers`` entries).
    n_hidden_layers : int, default 2
        0 (with ``hidden_dim=[]``) gives a single linear layer.
    activation : str or list of str, default "relu"
        'relu', 'lrelu', 'tanh', 'sigmoid', 'swish', 'silu' or 'none'; one per hidden layer when a list.
    slope : float, default 0.1
        Negative slope of 'lrelu'.
    use_bias : bool, default True
    dropout : float, default 0.0
        Dropout probability after each hidden layer; 0 disables it.
    use_batch_norm : bool, default False
    use_layer_norm : bool, default False
    device : str or torch.device, optional
    seed : int, optional
        Seeds the global RNGs before the initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, List[int]] = 200,
        n_hidden_layers: int = 2,
        activation: Union[str, List[str]] = 'relu',
        slope: float = 0.1,
        use_bias: bool = True,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers

        if isinstance(hidden_dim, Number):
            if self.n_hidden_layers < 1:
                raise ValueError(f"n_hidden_layers must be >= 1, got {self.n_hidden_layers}")
            self.hidden_dim = [int(hidden_dim)] * self.n_hidden_layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != self.n_hidden_layers:
                raise ValueError(
                    f"hidden_dim list length ({len(hidden_dim)}) must equal n_hidden_layers ({self.n_hidden_layers})"
                )
            self.hidden_dim = [int(d) for d in hidden_dim]
        else:
            raise ValueError(f'Wrong argument type for hidden_dim: {type(hidden_dim)}')

        if isinstance(activation, str):
            self.activation = [activation] * self.n_hidden_layers
        elif isinstance(activation, list):
            if len(activation) != self.n_hidden_layers:
                raise ValueError(
                    f"activation list length ({len(activation)}) must equal n_hidden_layers ({self.n_hidden_layers})"
                )
            self.activation = activation
        else:
            raise ValueError(f'Wrong argument type for activation: {type(activation)}')

        if seed is not None:
            apply_seed(seed)

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x, s=slope: F.leaky_relu(x, negative_slope=s))
            elif act == 'relu':
                self._act_f.append(F.relu)
            elif act == 'tanh':
                self._act_f.append(torch.tanh)
            elif act == 'sigmoid':
                self._act_f.append(torch.sigmoid)
            elif act == 'swish' or act == 'silu':
                self._act_f.append(F.silu)
            elif act == 'none' or act is None:
                self._act_f.append(lambda x: x)
            else:
                raise ValueError(f'Incorrect activation: {act}. Options: relu, lrelu, tanh, sigmoid, swish, silu, none')

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        if self.n_hidden_layers == 0:
            layer = nn.Linear(input_dim, output_dim, bias=use_bias)
            self._init_linear(layer)
            self.layers.append(layer)
        else:
            layer = nn.Linear(input_dim, self.hidden_dim[0], bias=use_bias)
            self._init_linear(layer)
            self.layers.append(layer)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.hidden_dim[0]))

            for i in range(1, len(self.hidden_dim)):
                layer = nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i], bias=use_bias)
                self._init_linear(layer)
                self.layers.append(layer)

                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i]))
                if use_layer_norm:
                    self.layer_norms.append(nn.LayerNorm(self.hidden_dim[i]))

            layer = nn.Linear(self.hidden_dim[-1], output_dim, bias=use_bias)
            self._init_linear(layer)
            self.layers.append(layer)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None

        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)

    @staticmethod
    def _init_linear(layer: nn.Linear) -> None:
        nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., input_dim)`` to ``(..., output_dim)``; leading dimensions are flattened and restored."""
        original_shape = x.shape
        needs_reshape = x.dim() > 2
        if needs_reshape:
            # reshape, not view: the input may be non-contiguous.
            x = x.reshape(-1, x.shape[-1])

        h = x
        n_hidden = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < n_hidden:
                h = self._act_f[i](h)
                if self.batch_norms is not None:
                    h = self.batch_norms[i](h)
                if self.layer_norms is not None:
                    h = self.layer_norms[i](h)
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)

        if needs_reshape:
            h = h.reshape(*(list(original_shape[:-1]) + [self.output_dim]))
        return h
