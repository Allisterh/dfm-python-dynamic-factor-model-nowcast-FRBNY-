"""The linear DFM: the public ``DFM`` estimator over the private ``DFMEngine``.

Submodules: ``estimator`` (DFM), ``_engine`` (DFMEngine), ``initialization``,
``mixed_freq`` and ``tent``. Only ``DFM`` is exported; the engine helpers are
importable from their submodules.
"""

from .estimator import DFM

__all__ = ['DFM']
