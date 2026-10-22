"""Regime probability network of iVDFM: π_t = softmax(h(u_t) / τ) from the observed context only.

τ < 1 sharpens π toward one-hot; τ = 1 is the plain softmax. No latent
inputs, so no gradient reaches η through π.
"""

from typing import Optional
import torch
import torch.nn as nn

from .mlp import MLP


class RegimeNet(nn.Module):
    """Context ``u`` to probabilities over ``num_regimes`` regimes: an MLP (or a linear map) and a softmax."""

    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        hidden_dim: int = 32,
        n_hidden_layers: int = 1,
        activation: str = "lrelu",
        slope: float = 0.1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        if n_hidden_layers < 1:
            self.net = nn.Linear(input_dim, num_regimes)
        else:
            self.net = MLP(
                input_dim=input_dim,
                output_dim=num_regimes,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation=activation,
                slope=slope,
                device=device,
                seed=seed,
            )
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Regime probabilities ``(..., num_regimes)``; the logits are divided by ``temperature``."""
        logits = self.net(x)
        if temperature != 1.0:
            logits = logits / temperature
        return torch.softmax(logits, dim=-1)
