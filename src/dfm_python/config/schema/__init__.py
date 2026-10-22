"""Configuration schema package.

- ``model``: the estimator config mirrors (``DFMConfig``, ``DDFMConfig``,
  ``iVDFMConfig``, ``AFMConfig``) and the private engine configs
  (``BaseModelConfig`` and its subclasses, not exported).
- ``results``: ``BaseResult`` and the per-model result dataclasses.
"""

from .model import DFMConfig, DDFMConfig, iVDFMConfig, AFMConfig
from .results import BaseResult, DFMResult, DDFMResult, iVDFMResult, AFMResult

__all__ = [
    'DFMConfig', 'DDFMConfig', 'iVDFMConfig', 'AFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult', 'iVDFMResult', 'AFMResult',
]
