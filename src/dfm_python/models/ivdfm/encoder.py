"""Innovation encoder for iVDFM: q(η_t | y_t, u_t), conditioned on the current time step only."""

from typing import Tuple, Union, List, Optional
import torch

from ..base import BaseEncoder
from ...layer.mlp import MLP


class iVDFMInnovationEncoder(BaseEncoder):
    """Innovation encoder q(η_t | y_t, u_t): two MLPs give the posterior mean and log-variance.

    Parameters
    ----------
    data_dim : int
        Dimension N of y_t.
    latent_dim : int
        Dimension r of the innovations.
    aux_dim : int
        Dimension of u_t.
    hidden_dim : int or list of int, default 200
    n_hidden_layers : int, default 2
    activation : str, default 'lrelu'
        One of 'lrelu', 'relu', 'tanh', 'sigmoid'.
    slope : float, default 0.1
        Leaky ReLU slope.
    use_layer_norm : bool, default False
    device : str or torch.device, optional
    seed : int, optional
        Seeds the mean network; the log-variance network uses ``seed + 1``.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        aux_dim: int,
        hidden_dim: Union[int, List[int]] = 200,
        n_hidden_layers: int = 2,
        activation: str = 'lrelu',
        slope: float = 0.1,
        use_layer_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim

        encoder_input_dim = data_dim + aux_dim
        self.mu_network = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            slope=slope,
            use_layer_norm=use_layer_norm,
            device=device,
            seed=seed,
        )
        # A different seed so the two networks do not start identical.
        logvar_seed = seed + 1 if seed is not None else None
        self.logvar_network = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            slope=slope,
            use_layer_norm=use_layer_norm,
            device=device,
            seed=logvar_seed,
        )

        if device is not None:
            self.to(device)

    def forward(
        self,
        y_t: torch.Tensor,
        u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and log-variance (clamped to [-10, 10]) of the innovations.

        ``y_t`` is (batch, N) or (batch, T, N) and ``u_t`` (batch, aux_dim) or
        (batch, T, aux_dim) to match; the outputs are (batch, r) or (batch, T, r).
        """
        if y_t.dim() == 3:
            batch_size, T, N = y_t.shape
            y_flat = y_t.reshape(-1, N)
            u_flat = u_t.reshape(-1, self.aux_dim)
            needs_reshape = True
        elif y_t.dim() == 2:
            y_flat = y_t
            u_flat = u_t
            needs_reshape = False
        else:
            raise ValueError(f"y_t must be 2D (batch, N) or 3D (batch, T, N), got {y_t.dim()}D")

        xu = torch.cat([y_flat, u_flat], dim=1)
        mu = self.mu_network(xu)
        logvar = self.logvar_network(xu)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if needs_reshape:
            mu = mu.reshape(batch_size, T, -1)
            logvar = logvar.reshape(batch_size, T, -1)

        return mu, logvar

    def sample(
        self,
        y_t: torch.Tensor,
        u_t: torch.Tensor,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Reparameterized sample η = mu + eps * std; with ``return_params`` also ``(mu, logvar)``."""
        mu, logvar = self.forward(y_t, u_t)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        eta = mu + eps * std
        if return_params:
            return eta, mu, logvar
        return eta
