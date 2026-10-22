"""Parametric Avellaneda--Lee trading policy (the paper's PCA+OU benchmark).

Avellaneda and Lee (2010) cumulate the residual returns into a level, fit an
Ornstein--Uhlenbeck / AR(1) mean-reversion model on the trailing window and
trade an s-score threshold rule: a non-learned mapping from residual histories
to portfolio weights that drops in where the LongConv filter would sit.
"""

from __future__ import annotations

import torch


def ou_sscore(residual_hist: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """s-score of each asset's cumulated residual under a trailing AR(1) fit.

    Parameters
    ----------
    residual_hist : torch.Tensor
        Residual returns ``eps_{i,(t-s,t-1)}``, shape ``(..., s)`` (oldest first).
    eps : float
        Numerical floor.

    Returns
    -------
    torch.Tensor
        s-score per leading-axis element, shape ``(...)``.
    """
    level = torch.cumsum(residual_hist, dim=-1)                     # (..., s)
    x_lag = level[..., :-1]
    dx = level[..., 1:] - level[..., :-1]
    mx = x_lag.mean(dim=-1, keepdim=True)
    md = dx.mean(dim=-1, keepdim=True)
    cov = ((x_lag - mx) * (dx - md)).sum(dim=-1)
    var = ((x_lag - mx) ** 2).sum(dim=-1) + eps
    b = cov / var
    a = md.squeeze(-1) - b * mx.squeeze(-1)
    kappa = torch.clamp(-b, min=1e-4)                              # mean-reversion speed
    m = a / kappa                                                  # equilibrium
    resid = dx - (a.unsqueeze(-1) + b.unsqueeze(-1) * x_lag)
    sigma = resid.std(dim=-1) + eps
    sigma_eq = sigma / torch.sqrt(2.0 * kappa)
    return (level[..., -1] - m) / (sigma_eq + eps)


def avellaneda_lee_weights(residual_hist: torch.Tensor, entry: float = 1.25,
                           hl_max: float = 60.0) -> torch.Tensor:
    """Fade-the-stretch weights ``-sign(s) * (|s| >= entry)`` from the OU s-score.

    Parameters
    ----------
    residual_hist : torch.Tensor
        Residual returns, shape ``(..., s)``.
    entry : float
        s-score entry threshold.
    hl_max : float
        Unused; the half-life eligibility screen is the caller's.

    Returns
    -------
    torch.Tensor
        Portfolio weight per asset, shape ``(...)``.
    """
    s = ou_sscore(residual_hist)
    active = (s.abs() >= entry).to(residual_hist.dtype)
    return -torch.sign(s) * active
