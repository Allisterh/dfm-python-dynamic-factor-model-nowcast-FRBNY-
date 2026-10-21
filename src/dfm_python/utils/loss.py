"""Loss functions for the torch factor models: masked reconstruction and the iVDFM ELBO."""

import torch
import numpy as np
from typing import Literal, Dict, Tuple, Optional
from ..config.constants import DEFAULT_EPSILON, HUBER_QUADRATIC_COEFF


def compute_masked_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_function: Literal['mse', 'huber'] = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """Reconstruction loss over the observed entries (``mask`` True), as a scalar tensor.

    The sum of per-entry losses is divided by ``target.numel()`` (all
    entries, observed or not) plus ``DEFAULT_EPSILON``, matching the
    reference TensorFlow ``MeanSquaredError``. Huber uses
    ``0.5 * d^2`` for ``|d| <= huber_delta`` and
    ``huber_delta * (|d| - 0.5 * huber_delta)`` beyond. ``mask`` is cast to
    bool; all three tensors share one shape.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Same masking order as the reference implementation: zero the target,
    # multiply the reconstruction by the mask, then difference.
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    reconstructed_masked = reconstructed * mask
    diff = target_clean - reconstructed_masked

    if loss_function == 'huber':
        abs_diff = torch.abs(diff)
        loss_values = torch.where(
            abs_diff <= huber_delta,
            HUBER_QUADRATIC_COEFF * diff ** 2,
            huber_delta * (abs_diff - HUBER_QUADRATIC_COEFF * huber_delta)
        )
    else:
        loss_values = diff ** 2

    return torch.sum(loss_values * mask) / (target.numel() + DEFAULT_EPSILON)


# The KL divergences live in models.ivdfm.prior, which (through the ivdfm
# package) imports this module, so they are resolved lazily at call time.
def _prior():
    from ..models.ivdfm import prior
    return prior


def compute_kl_gaussian_laplace(*args, **kwargs):
    return _prior().compute_kl_gaussian_laplace(*args, **kwargs)


def compute_kl_gaussian_gaussian(*args, **kwargs):
    return _prior().compute_kl_gaussian_gaussian(*args, **kwargs)


def compute_kl_gaussian_gamma(*args, **kwargs):
    return _prior().compute_kl_gaussian_gamma(*args, **kwargs)


def compute_kl_gaussian_beta(*args, **kwargs):
    return _prior().compute_kl_gaussian_beta(*args, **kwargs)


def compute_kl_gaussian_exponential(*args, **kwargs):
    return _prior().compute_kl_gaussian_exponential(*args, **kwargs)


# KL(q || p) per innovation distribution: the function and the prior-param
# keys it takes after (mu_q, logvar_q). The Student-t prior reuses the
# Laplace bound.
_KL_BY_DISTRIBUTION = {
    'laplace': (compute_kl_gaussian_laplace, ('location', 'log_scale')),
    'student_t': (compute_kl_gaussian_laplace, ('location', 'log_scale')),
    'gaussian': (compute_kl_gaussian_gaussian, ('mu', 'logvar')),
    'gamma': (compute_kl_gaussian_gamma, ('shape', 'log_rate')),
    'beta': (compute_kl_gaussian_beta, ('log_alpha', 'log_beta')),
    'exponential': (compute_kl_gaussian_exponential, ('log_rate',)),
}


def compute_reconstruction_loss_gaussian(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: float = 1.0
) -> torch.Tensor:
    """Gaussian negative log-likelihood ``0.5 * (log(2 pi s2) + (y - g(f))^2 / s2)``, summed over the last dim and averaged over the rest."""
    recon_loss = 0.5 * (
        np.log(2 * np.pi * variance) +
        ((y_true - y_pred) ** 2) / variance
    )
    return recon_loss.sum(dim=-1).mean()


def compute_elbo_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    encoder_params: list,
    prior_params: list,
    innovation_distribution: str = 'laplace',
    decoder_variance: float = 1.0,
    beta_kl: float = 1.0,
    reduction: str = 'mean',
    regime_weights: Optional[torch.Tensor] = None,
    prior_params_per_regime: Optional[list] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Negative ELBO ``recon + beta_kl * KL`` for the iVDFM.

    ``KL = sum_t KL(q(eta_t | y, u_t) || p(eta_t | u_t))``, or with a regime
    prior ``KL_t = sum_k pi_{t,k} KL(q || p_k)``.

    Parameters
    ----------
    y_true, y_pred : torch.Tensor
        (batch, T, N).
    encoder_params : list
        T dicts with ``mu`` and ``logvar`` of shape (batch, dim).
    prior_params : list
        T dicts of prior parameters (keys depend on ``innovation_distribution``).
    innovation_distribution : str, default 'laplace'
        'laplace', 'student_t', 'gaussian', 'gamma', 'beta' or 'exponential'.
    decoder_variance : float, default 1.0
    beta_kl : float, default 1.0
    reduction : str, default 'mean'
        'mean' over (batch, T) for the KL term; anything else sums.
    regime_weights : torch.Tensor, optional
        (batch, T, K) regime probabilities; with ``prior_params_per_regime``
        (K lists of T dicts) the KL is regime-weighted.

    Returns
    -------
    elbo : torch.Tensor
        Scalar loss.
    loss_dict : dict
        ``{'elbo', 'reconstruction', 'kl'}``.
    """
    batch_size, T, _ = y_true.shape

    recon_loss = compute_reconstruction_loss_gaussian(
        y_true, y_pred, variance=decoder_variance
    )

    mu_q_all = torch.stack([encoder_params[t]['mu'] for t in range(T)], dim=1)  # (batch, T, dim)
    logvar_q_all = torch.stack([encoder_params[t]['logvar'] for t in range(T)], dim=1)  # (batch, T, dim)

    if regime_weights is not None and prior_params_per_regime is not None:
        kl_per_k = torch.stack([
            _compute_kl_bt(mu_q_all, logvar_q_all, prior_params_k, T, batch_size, innovation_distribution)
            for prior_params_k in prior_params_per_regime
        ], dim=2)  # (batch, T, K)
        kl_loss_bt = (regime_weights * kl_per_k).sum(dim=2)  # (batch, T)
    else:
        kl_loss_bt = _compute_kl_bt(
            mu_q_all, logvar_q_all, prior_params, T, batch_size, innovation_distribution
        )

    if reduction == 'mean':
        kl_loss = kl_loss_bt.mean()
    else:
        kl_loss = kl_loss_bt.sum()

    elbo = recon_loss + beta_kl * kl_loss
    loss_dict = {'elbo': elbo, 'reconstruction': recon_loss, 'kl': kl_loss}
    return elbo, loss_dict


def _compute_kl_bt(
    mu_q_all: torch.Tensor,
    logvar_q_all: torch.Tensor,
    prior_params: list,
    T: int,
    batch_size: int,
    innovation_distribution: str,
) -> torch.Tensor:
    """KL(q || p) per (batch, time) from (batch, T, dim) posterior moments; returns (batch, T)."""
    if innovation_distribution not in _KL_BY_DISTRIBUTION:
        raise NotImplementedError(
            f"KL divergence for {innovation_distribution} not implemented"
        )
    kl_fn, keys = _KL_BY_DISTRIBUTION[innovation_distribution]

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, x.shape[-1])  # (batch*T, dim)

    prior_flat = [
        flat(torch.stack([prior_params[t][key] for t in range(T)], dim=1)) for key in keys
    ]
    kl_all = kl_fn(flat(mu_q_all), flat(logvar_q_all), *prior_flat)
    return kl_all.view(batch_size, T)
