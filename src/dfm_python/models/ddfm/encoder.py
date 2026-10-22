"""Encoder and autoencoder for DDFM."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...dataset.ddfm_dataset import AutoencoderDataset

from ..base import BaseEncoder, BaseDecoder
from .decoder import DDFMLinearDecoder, DDFMMLPDecoder, _linear_params
from ...utils.errors import ConfigurationError, DataValidationError
from ...utils.seed import apply_seed
from ...config.constants import (
    DEFAULT_XAVIER_GAIN,
    DEFAULT_ZERO_VALUE,
    DEFAULT_BATCH_NORM_MOMENTUM,
    DEFAULT_BATCH_NORM_EPS,
    DEFAULT_AUTOENCODER_FIT_EPOCHS,
    DEFAULT_DDFM_BATCH_SIZE,
    DEFAULT_DDFM_LEARNING_RATE,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_N_MC_SAMPLES,
)


def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    """``(weight, bias)`` of a decoder with ``extract_params`` or of an ``nn.Linear``.

    Shapes ``(output_dim, input_dim)`` and ``(output_dim,)``; NaN weights are
    replaced by zeros.

    Raises
    ------
    DataValidationError
        For any other decoder.
    """
    if hasattr(decoder, 'extract_params'):
        return decoder.extract_params()
    if isinstance(decoder, nn.Linear):
        return _linear_params(decoder, "extract_decoder_params")
    raise DataValidationError(
        f"decoder must have 'extract_params' method or be a Linear layer. Got: {type(decoder)}"
    )


class DDFMEncoder(BaseEncoder):
    """Encoder: linear layers with an activation, batch-normalized between layers.

    The last width is the number of factors. The first layer has no batch
    norm before it; each later layer is preceded by one.

    Parameters
    ----------
    input_dim : int
    encoder_dims : list of int
        Layer widths, at least one.
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        activation: str = 'relu',
    ):
        super().__init__()

        if len(encoder_dims) == 0:
            raise ValueError("encoder_dims must have at least one element")

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if activation not in activations:
            raise ConfigurationError(f"Unknown activation: {activation}")
        self.activation = activations[activation]

        first_layer = nn.Linear(input_dim, encoder_dims[0])
        self._init_linear(first_layer)
        self.layers.append(first_layer)
        prev_dim = encoder_dims[0]

        for dim in encoder_dims[1:]:
            self.batch_norms.append(
                nn.BatchNorm1d(prev_dim, momentum=DEFAULT_BATCH_NORM_MOMENTUM, eps=DEFAULT_BATCH_NORM_EPS)
            )
            layer = nn.Linear(prev_dim, dim)
            self._init_linear(layer)
            self.layers.append(layer)
            prev_dim = dim

    @staticmethod
    def _init_linear(layer: nn.Linear) -> None:
        nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
        nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., input_dim)`` to ``(..., encoder_dims[-1])`` factors."""
        x = self.activation(self.layers[0](x))
        for i in range(1, len(self.layers)):
            x = self.batch_norms[i - 1](x)
            x = self.activation(self.layers[i](x))
        return x


class SimpleAutoencoder(nn.Module):
    """DDFM autoencoder: a ``DDFMEncoder`` and a linear or MLP decoder.

    Parameters
    ----------
    encoder : BaseEncoder
    decoder : BaseDecoder
    """

    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build(
        cls,
        input_dim: int,
        encoder_size: Tuple[int, ...],
        decoder_size: Optional[Tuple[int, ...]] = None,
        decoder_type: str = "linear",
        output_dim: Optional[int] = None,
        activation: str = 'relu',
        seed: Optional[int] = None
    ) -> "SimpleAutoencoder":
        """Build the encoder and decoder.

        Parameters
        ----------
        input_dim : int
        encoder_size : tuple of int
            Encoder layer widths; the last is the number of factors.
        decoder_size : tuple of int, optional
            MLP decoder hidden widths; required (non-empty) with ``decoder_type="mlp"``.
        decoder_type : {"linear", "mlp"}, default "linear"
        output_dim : int, optional
            Defaults to ``input_dim``.
        activation : str, default "relu"
        seed : int, optional
            Seeds the global RNGs before building (and again before the decoder).
        """
        if output_dim is None:
            output_dim = input_dim

        if seed is not None:
            apply_seed(seed)

        if len(encoder_size) == 0:
            raise ValueError("encoder_size must have at least one element")

        encoder = DDFMEncoder(input_dim, list(encoder_size), activation)

        latent_dim = encoder_size[-1]
        if decoder_type == "mlp":
            if decoder_size is None or len(decoder_size) == 0:
                raise ValueError("decoder_size must be provided when decoder_type='mlp'")
            decoder = DDFMMLPDecoder(latent_dim, output_dim, list(decoder_size), activation, seed=seed)
        else:
            decoder = DDFMLinearDecoder(latent_dim, output_dim, seed=seed)

        return cls(encoder=encoder, decoder=decoder)

    @classmethod
    def from_dataset(
        cls,
        dataset: 'AutoencoderDataset',
        encoder_size: Tuple[int, ...],
        decoder_size: Optional[Tuple[int, ...]] = None,
        decoder_type: str = "linear",
        activation: str = 'relu',
        seed: Optional[int] = None
    ) -> "SimpleAutoencoder":
        """:meth:`build` with ``input_dim`` and ``output_dim`` read from ``dataset``."""
        return cls.build(
            input_dim=dataset.full_input.shape[1],
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            decoder_type=decoder_type,
            output_dim=dataset.y_clean.shape[1],
            activation=activation,
            seed=seed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` in eval mode without gradients (the TensorFlow ``predict``)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def pretrain(
        self,
        full_input: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        use_mse_loss: bool = True
    ) -> List[float]:
        """Train on uncorrupted data before the MCMC loop; returns each epoch's mean batch loss.

        Parameters
        ----------
        full_input : torch.Tensor
            ``(T, input_dim)`` encoder input (covariates and targets).
        y : torch.Tensor
            ``(T, output_dim)`` targets.
        epochs : int
        batch_size : int
        optimizer : torch.optim.Optimizer
        use_mse_loss : bool, default True
            Plain MSE; False masks the NaN targets (zeroing them and the
            matching predictions) before the MSE.
        """
        self.train()
        T = len(full_input)
        final_epoch_losses = []

        for epoch in range(epochs):
            epoch_losses = []
            for i in range(0, T, batch_size):
                batch_full_input = full_input[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                optimizer.zero_grad()
                y_pred = self.forward(batch_full_input)

                if use_mse_loss:
                    loss = torch.nn.functional.mse_loss(y_pred, batch_y, reduction='mean')
                else:
                    mask = ~torch.isnan(batch_y)
                    y_actual = torch.where(torch.isnan(batch_y), torch.zeros_like(batch_y), batch_y)
                    y_pred_masked = y_pred * mask.float()
                    loss = torch.nn.functional.mse_loss(y_pred_masked, y_actual, reduction='mean')

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if epoch_losses:
                final_epoch_losses.append(np.mean(epoch_losses))

        return final_epoch_losses

    def fit(
        self,
        dataset: 'AutoencoderDataset',
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        optimizer_type: str = 'Adam',
        decay_learning_rate: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        target_indices: Optional[torch.Tensor] = None
    ) -> None:
        """Train on ``dataset`` (corrupted inputs, clean targets) with the masked MSE loss.

        The scheduler steps after every batch.

        Parameters
        ----------
        dataset : AutoencoderDataset
        epochs : int, optional
            Default ``DEFAULT_AUTOENCODER_FIT_EPOCHS``.
        batch_size : int, optional
            Default ``DEFAULT_DDFM_BATCH_SIZE``.
        learning_rate : float, optional
            Default ``DEFAULT_DDFM_LEARNING_RATE``; only for an optimizer built here.
        optimizer_type : {"Adam", "SGD"}, default "Adam"
            Only for an optimizer built here; anything else builds Adam.
        decay_learning_rate : bool, default True
            Build a ``StepLR`` scheduler when none is given.
        optimizer : torch.optim.Optimizer, optional
            Used as given; None builds one.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        target_indices : torch.Tensor, optional
            Unused: the output layer already has one unit per target.
        """
        if epochs is None:
            epochs = DEFAULT_AUTOENCODER_FIT_EPOCHS
        if batch_size is None:
            batch_size = DEFAULT_DDFM_BATCH_SIZE
        if learning_rate is None:
            learning_rate = DEFAULT_DDFM_LEARNING_RATE

        if optimizer is None:
            optimizers = {
                'Adam': lambda: torch.optim.Adam(self.parameters(), lr=learning_rate),
                'SGD': lambda: torch.optim.SGD(self.parameters(), lr=learning_rate)
            }
            optimizer = optimizers.get(optimizer_type, optimizers['Adam'])()

        if scheduler is None and decay_learning_rate:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=DEFAULT_N_MC_SAMPLES, gamma=DEFAULT_LR_DECAY_RATE
            )

        self.train()
        full_input = dataset.full_input
        y_clean = dataset.y_clean
        T = len(dataset)

        for epoch in range(epochs):
            for i in range(0, T, batch_size):
                batch_input = full_input[i:i+batch_size]
                batch_target = y_clean[i:i+batch_size]

                optimizer.zero_grad()
                pred = self.forward(batch_input)
                mask = ~torch.isnan(batch_target)
                y_actual_ = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                y_predicted_ = pred * mask.float()
                loss = torch.nn.functional.mse_loss(y_predicted_, y_actual_, reduction='mean')
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
