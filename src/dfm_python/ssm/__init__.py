"""State-space models.

- ``DFMKalmanFilter``: Kalman filter and Durbin-Koopman smoother of the DFM
  E-step (per-element missing data, exact log-likelihood).
- ``BaseSSM``: base class of the iVDFM SSMs (innovations to factors).
- ``iVDFMCompanionSSM``: block-diagonal companion-form SSM of iVDFM.

``CompanionSSM``, ``MACompanionSSM`` and ``CompanionSSMBase`` are None: no such
classes exist, and the names stay importable.
"""

from .kalman import DFMKalmanFilter
from .base import BaseSSM
from .companion import iVDFMCompanionSSM

CompanionSSM = None
MACompanionSSM = None
CompanionSSMBase = None

__all__ = [
    'DFMKalmanFilter',
    'BaseSSM',
    'iVDFMCompanionSSM',
    'CompanionSSM',
    'MACompanionSSM',
    'CompanionSSMBase',
]
