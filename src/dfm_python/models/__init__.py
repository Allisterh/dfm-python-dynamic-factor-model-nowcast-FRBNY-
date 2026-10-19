"""Factor model estimators.

- ``DFM``: linear, mixed-frequency dynamic factor model estimated by EM.
- ``DDFM``: deep dynamic factor model with a nonlinear (PyTorch) encoder.
- ``iVDFM``: identifiable variational dynamic factor model.
- ``AFM``: attention factor model for statistical arbitrage.

Each model's engine is private (``models/<model>/_engine.py``); the estimators
share the base class :class:`dfm_python.base.BaseFactorModel`.
"""

from .dfm import DFM
from .ddfm.estimator import DDFM
from .ivdfm.estimator import iVDFM
from .afm.estimator import AFM
from ..config import BaseResult, DFMResult, DDFMResult

__all__ = [
    'DFM', 'DDFM', 'iVDFM', 'AFM',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult',
]
