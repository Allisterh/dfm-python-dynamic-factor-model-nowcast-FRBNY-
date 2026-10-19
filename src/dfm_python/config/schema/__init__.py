"""Configuration schema package for DFM models.

This package contains:
- model.py: the estimator config mirrors (DFMConfig, DDFMConfig, iVDFMConfig,
  AFMConfig) and the private engine configs (BaseModelConfig and its
  subclasses, not exported)
- results.py: BaseResult, DFMResult, DDFMResult - result structures

Note: Series are specified via frequency dict mapping column names to frequencies.
"""

from .model import DFMConfig, DDFMConfig, iVDFMConfig, AFMConfig
from .results import BaseResult, DFMResult, DDFMResult, iVDFMResult, AFMResult

__all__ = [
    'DFMConfig', 'DDFMConfig', 'iVDFMConfig', 'AFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult', 'iVDFMResult', 'AFMResult',
]

