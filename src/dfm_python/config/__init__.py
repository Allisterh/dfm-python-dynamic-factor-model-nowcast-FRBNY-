"""Configuration subpackage.

- The constructor mirrors ``DFMConfig``, ``DDFMConfig``, ``iVDFMConfig`` and
  ``AFMConfig`` (subclasses of :class:`EstimatorConfig`) and :func:`load_yaml`,
  which reads the ``model:``/``data:`` YAML layout.
- The result classes (``DFMResult``, ``DDFMResult``, ``AFMResult``).
- Constants, frequency utilities and type aliases.

The engines' own config classes (``schema.model``) and the parsing helpers in
``adapter`` are private.
"""

from .schema import (
    DFMConfig, DDFMConfig, iVDFMConfig, AFMConfig,
    BaseResult, DFMResult, DDFMResult, AFMResult,
)
from .constants import (
    DEFAULT_CLOCK_FREQUENCY,
    FREQUENCY_HIERARCHY,
    PERIODS_PER_YEAR,
)
from .schema.model import validate_frequency
from .estimator import EstimatorConfig, load_yaml
from .types import (
    ArrayLike,
    FloatArray,
    IntArray,
    BoolArray,
    OptionalArray,
    OptionalTensor,
    OptionalArrayLike,
    FactorState,
    ObservationState,
    ForecastResult,
    CoefficientMatrix,
    CovarianceMatrix,
    SeriesID,
    Frequency,
    DatasetName,
    ModelName,
    Batch,
    Loss,
    Optimizer,
    ResultDict,
    ForecastDict,
    MetricsDict,
    CheckpointDict,
    ConfigDict,
    Device,
    Shape,
    Shape2D,
    Shape3D,
    Shape4D,
    ForecastHorizon,
    IRFHorizon,
    LagOrder,
    NumFactors,
    NumVars,
    PathLike,
    ValidationIssue,
    ValidationResult,
    Tensor,
    is_numpy_array,
    is_torch_tensor,
    is_array_like,
    get_array_shape,
    to_numpy,
    to_tensor,
)
# Circular: models.dfm.tent imports this package while it is initializing.
try:
    from ..models.dfm.tent import get_agg_structure, group_by_freq
except ImportError:
    get_agg_structure = None
    group_by_freq = None


def get_periods_per_year(frequency: str) -> int:
    """Periods per year of a frequency code (the clock frequency's for an unknown code)."""
    return PERIODS_PER_YEAR.get(frequency, PERIODS_PER_YEAR.get(DEFAULT_CLOCK_FREQUENCY, 12))


__all__ = [
    'BaseResult',
    'EstimatorConfig', 'load_yaml',
    'DFMConfig', 'DDFMConfig', 'iVDFMConfig', 'AFMConfig',
    'DFMResult', 'DDFMResult', 'AFMResult',
    'validate_frequency',
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'get_periods_per_year',
    'get_agg_structure',
    'group_by_freq',
    'ArrayLike',
    'FloatArray',
    'IntArray',
    'BoolArray',
    'OptionalArray',
    'OptionalTensor',
    'OptionalArrayLike',
    'FactorState',
    'ObservationState',
    'ForecastResult',
    'CoefficientMatrix',
    'CovarianceMatrix',
    'SeriesID',
    'Frequency',
    'DatasetName',
    'ModelName',
    'Batch',
    'Loss',
    'Optimizer',
    'ResultDict',
    'ForecastDict',
    'MetricsDict',
    'CheckpointDict',
    'ConfigDict',
    'Device',
    'Shape',
    'Shape2D',
    'Shape3D',
    'Shape4D',
    'ForecastHorizon',
    'IRFHorizon',
    'LagOrder',
    'NumFactors',
    'NumVars',
    'PathLike',
    'ValidationIssue',
    'ValidationResult',
    'Tensor',
    'is_numpy_array',
    'is_torch_tensor',
    'is_array_like',
    'get_array_shape',
    'to_numpy',
    'to_tensor',
]
