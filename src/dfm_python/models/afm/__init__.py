"""Attention Factor Model (AFM) for statistical arbitrage.

Epstein et al. (2025), "Attention Factors for Statistical Arbitrage": learned
conditional factors via cross-sectional attention, a LongConv residual trading
filter, and an after-cost Sharpe + explained-variance objective, with PCA and
Avellaneda--Lee benchmarks. ``AFM`` is the public estimator; the training code
is the private engine ``_engine.AFMEngine``.
"""

from .estimator import AFM
from .factors import AttentionFactors, PCAFactors, residuals, ridge_loadings
from .longconv import LongConv1d
from .baselines import avellaneda_lee_weights, ou_sscore

__all__ = [
    "AFM",
    "AttentionFactors",
    "PCAFactors",
    "LongConv1d",
    "residuals",
    "ridge_loadings",
    "avellaneda_lee_weights",
    "ou_sscore",
]
