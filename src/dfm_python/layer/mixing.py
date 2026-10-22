"""Post-factor mixing layer of iVDFM: f -> M f (r -> r) before the decoder when ``mixing=True``."""

from typing import Optional
import torch
import torch.nn as nn


class FactorMixer(nn.Module):
    """Bias-free linear map ``M`` of the factors; ``init="identity"`` starts it at the identity."""

    def __init__(self, r: int, init: str = "identity", device: Optional[torch.device] = None):
        super().__init__()
        self.r = r
        self.M = nn.Linear(r, r, bias=False)
        if init == "identity":
            nn.init.eye_(self.M.weight)
        if device is not None:
            self.to(device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """``(..., r)`` to ``(..., r)``."""
        return self.M(f)


def build_factor_mixing(
    r: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> FactorMixer:
    """An identity-initialized ``FactorMixer`` on ``device``, cast to ``dtype`` when given."""
    module = FactorMixer(r=r, init="identity", device=device)
    if dtype is not None:
        module = module.to(dtype=dtype)
    return module
