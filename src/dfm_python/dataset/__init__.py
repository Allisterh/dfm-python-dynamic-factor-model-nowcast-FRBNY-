"""Data container and time utilities.

This package exports:
- ``PanelData``: the one data container every estimator's ``fit`` takes.
- ``TimeIndex``: time-index utilities.

The per-model datasets are the estimators' private adapters
(``dataset.dfm_dataset``, ``dataset.ddfm_dataset``, ``dataset.ivdfm_dataset``,
``dataset.afm_dataset``) and are not exported; ``fit`` builds them from
``PanelData`` and the hyperparameters.
"""

from .panel import PanelData
from .time import TimeIndex

__all__ = [
    # Data container
    'PanelData',
    # Time utilities
    'TimeIndex',
]
