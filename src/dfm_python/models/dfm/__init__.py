"""DFM model package.

This package contains the DFM estimator and its submodules:
- estimator: DFM, the public estimator (PanelData input, sklearn conventions)
- _engine: DFMEngine, the private EM engine the estimator runs
- initialization: Parameter initialization functions
- mixed_freq: Mixed-frequency parameter setup
- tent: Tent kernels of slower-frequency series

Only ``DFM`` is exported; the engine helpers are importable from their
submodules (for example ``dfm_python.models.dfm.mixed_freq.find_slower_frequency``).
"""

from .estimator import DFM

__all__ = ['DFM']
