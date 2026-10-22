"""Abstract bases of the DDFM and iVDFM network components.

The estimator base class is :class:`dfm_python.base.BaseFactorModel`; the
engines share no base class.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Encoder: observations ``x`` to latent factors."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the factors of ``x``."""


class BaseDecoder(nn.Module, ABC):
    """Decoder: latent factors ``f`` to reconstructed observations."""

    @abstractmethod
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction of ``f``."""
