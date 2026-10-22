"""Type aliases and the numpy/torch conversion helpers used across the package."""

from typing import Union, Tuple, Optional, Dict, Any, Literal
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

# Array types
ArrayLike = Union[np.ndarray, Tensor]
FloatArray = Union[np.ndarray, Tensor]
IntArray = Union[np.ndarray, Tensor]
BoolArray = Union[np.ndarray, Tensor]
OptionalArray = Optional[ArrayLike]
OptionalTensor = Optional[Tensor]
OptionalArrayLike = Optional[ArrayLike]

# Model types
FactorState = np.ndarray
"""Factor state array (m,) or (T, m)."""

ObservationState = np.ndarray
"""Observation array (N,) or (T, N)."""

ForecastResult = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
"""Series only, or (series, factors)."""

CoefficientMatrix = np.ndarray
"""Coefficient matrix (K, K) or (p, K, K)."""

CovarianceMatrix = np.ndarray
"""Covariance matrix (K, K) or (m, m)."""

# Configuration types
SeriesID = str
Frequency = Literal['d', 'w', 'm', 'q', 'sa', 'a']
"""Frequency code; matches ``constants.VALID_FREQUENCIES``."""
DatasetName = str
ModelName = Literal['var', 'vecm', 'dfm', 'ddfm']

# Training types
Batch = Union[Tensor, Tuple[Tensor, Tensor]]
"""Data tensor or (data, target) tuple."""
Loss = Tensor
Optimizer = torch.optim.Optimizer

# Result types
ResultDict = Dict[str, Any]
ForecastDict = Dict[str, Union[float, np.ndarray, Dict[str, Any]]]
MetricsDict = Dict[str, float]
CheckpointDict = Dict[str, Any]
ConfigDict = Dict[str, Any]

Device = Union[torch.device, str]

# Shape types
Shape = Tuple[int, ...]
Shape2D = Tuple[int, int]
"""(T, N): time steps, variables."""
Shape3D = Tuple[int, int, int]
"""(T, N, N) or (H, N, N)."""
Shape4D = Tuple[int, int, int, int]
"""(B, T, N, N)."""

# Forecast and analysis types
ForecastHorizon = int
IRFHorizon = int
LagOrder = int
NumFactors = int
NumVars = int

PathLike = Union[str, Path]

# Validation types
ValidationIssue = Dict[str, Any]
ValidationResult = Dict[str, Any]


def is_numpy_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray)


def is_torch_tensor(obj: Any) -> bool:
    return isinstance(obj, Tensor)


def is_array_like(obj: Any) -> bool:
    return is_numpy_array(obj) or is_torch_tensor(obj)


def get_array_shape(arr: ArrayLike) -> Shape:
    """Shape of a numpy array or torch tensor as a tuple; raises ``DataValidationError`` otherwise."""
    if is_numpy_array(arr):
        return arr.shape
    elif is_torch_tensor(arr):
        return tuple(arr.shape)
    else:
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Expected numpy array or torch tensor, got {type(arr)}",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


def to_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert to a numpy array (tensors are detached and moved to CPU), cast to ``dtype`` if given."""
    if is_numpy_array(arr):
        result = arr
    elif is_torch_tensor(arr):
        if arr.requires_grad:
            arr = arr.detach()
        result = arr.cpu().numpy()
    else:
        result = np.asarray(arr)

    if dtype is not None:
        result = result.astype(dtype)
    return result


def to_tensor(arr: ArrayLike, device: Optional[Device] = None, dtype: Optional['torch.dtype'] = None) -> Tensor:
    """Convert a numpy array or tensor to a tensor on ``device`` with ``dtype``; raises ``DataValidationError`` otherwise."""
    if is_torch_tensor(arr):
        result = arr
        if device is not None:
            result = result.to(device)
        if dtype is not None:
            result = result.to(dtype)
        return result
    elif is_numpy_array(arr):
        tensor = torch.from_numpy(arr)
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    else:
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Cannot convert {type(arr)} to torch tensor",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


__all__ = [
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
    'is_numpy_array',
    'is_torch_tensor',
    'is_array_like',
    'get_array_shape',
    'to_numpy',
    'to_tensor',
    'Tensor',
]
