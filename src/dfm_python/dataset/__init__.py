"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM) that use gradient descent training.

This module provides:
- Data container: PanelData
- Dataset classes: AFMDataset

The DFM, DDFM and iVDFM datasets are their estimators' private adapters
(``dataset.dfm_dataset``, ``dataset.ddfm_dataset``, ``dataset.ivdfm_dataset``)
and are not exported.
"""

from .afm_dataset import AFMDataset
from .panel import PanelData
from .time import TimeIndex

__all__ = [
    # Data container
    'PanelData',
    # Datasets
    'AFMDataset',
    # Time utilities
    'TimeIndex',
]
