"""Dynamic factor models in Python.

Models
------
- ``DFM``: linear, mixed-frequency dynamic factor model estimated by EM with
  Kalman filtering and smoothing. Latent factors evolve at a common clock
  frequency; lower-frequency series load on them through tent kernels.
- ``DDFM``: deep dynamic factor model with a nonlinear (PyTorch) encoder.
- ``iVDFM``: identifiable variational dynamic factor model.
- ``AFM``: attention factor model for cross-sectional returns.

Each model is built from a dataset and a config, then fitted::

    >>> from dfm_python import DFM, DFMDataset, DFMConfig
    >>> config = DFMConfig(
    ...     blocks={"global": {"num_factors": 1, "series": cols}},
    ...     frequency={c: "m" for c in cols},
    ...     clock="m",
    ... )
    >>> dataset = DFMDataset(config=config, data=df, time_index="date")
    >>> model = DFM(dataset=dataset, config=config)
    >>> model.fit()
    >>> X_forecast, Z_forecast = model.predict(horizon=6)

See README.md for complete examples of every model.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("dfm-python")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

del PackageNotFoundError, _version

# Configuration
from .config import (
    ConfigSource, YamlSource,
    make_config_source,
)
from .config import DFMConfig, DDFMConfig, iVDFMConfig, AFMConfig

# Results
from .config import DFMResult, DDFMResult, BaseResult

# Utilities
from .utils.metric import calculate_rmse

# Datasets
from .dataset.dfm_dataset import DFMDataset
from .dataset.ddfm_dataset import DDFMDataset
from .dataset.ivdfm_dataset import iVDFMDataset
from .dataset.afm_dataset import AFMDataset

# Models
from .models.base import BaseFactorModel
from .models.dfm import DFM
from .models.ddfm.ddfm import DDFM
from .models.ivdfm.ivdfm import iVDFM
from .models.afm.afm import AFM

__all__ = [
    # Models
    'BaseFactorModel',
    'DFM', 'DDFM', 'iVDFM', 'AFM',
    # Configs
    'DFMConfig', 'DDFMConfig', 'iVDFMConfig', 'AFMConfig',
    # Config sources
    'ConfigSource', 'YamlSource', 'make_config_source',
    # Datasets
    'DFMDataset', 'DDFMDataset', 'iVDFMDataset', 'AFMDataset',
    # Results and metrics
    'BaseResult', 'DFMResult', 'DDFMResult', 'calculate_rmse',
]
