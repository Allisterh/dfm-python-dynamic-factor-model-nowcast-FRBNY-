"""Abstract base classes for the neural network components of DDFM and iVDFM.

The estimator base class is :class:`dfm_python.base.BaseFactorModel`. The old
abstract engine base that lived here was removed; the engines set its
attributes themselves (``_init_engine_state`` in each ``_engine.py``).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for model encoders.
    
    Model-specific encoder base class. Encoders extract latent factors
    from observed data. This is a PyTorch module for neural encoders.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        
        Returns
        -------
        torch.Tensor
            Encoded representation (factors)
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for model decoders.
    
    Model-specific decoder base class. Decoders map latent factors
    to observed data. This is a PyTorch module for neural decoders.
    """
    
    @abstractmethod
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Parameters
        ----------
        f : torch.Tensor
            Latent factors
        
        Returns
        -------
        torch.Tensor
            Reconstructed observations
        """
        pass
