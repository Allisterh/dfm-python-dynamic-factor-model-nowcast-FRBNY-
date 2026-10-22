"""Generic condition checks that raise a caller-chosen exception."""

from typing import Any, Optional, Type


def check_condition(
    condition: bool,
    error_class: Type[Exception],
    message: str,
    details: Optional[str] = None
) -> None:
    """Raise ``error_class`` when ``condition`` is false; ``details`` is appended as ``"\\nDetails: ..."``."""
    if not condition:
        if details:
            raise error_class(f"{message}\nDetails: {details}")
        raise error_class(message)


def check_not_none(
    value: Any,
    name: str,
    error_class: Type[Exception] = ValueError
) -> None:
    """Raise ``error_class`` when ``value`` is None."""
    if value is None:
        raise error_class(f"{name} must not be None")


def check_has_attr(
    obj: Any,
    attr: str,
    name: str,
    error_class: Type[Exception] = AttributeError
) -> None:
    """Raise ``error_class`` when ``obj`` lacks attribute ``attr``."""
    if not hasattr(obj, attr):
        raise error_class(f"{name} must have attribute '{attr}'")


def has_shape_with_min_dims(
    obj: Any,
    min_dims: int = 2
) -> bool:
    """True when ``obj`` is not None and has a ``shape`` of at least ``min_dims`` dimensions."""
    return obj is not None and hasattr(obj, 'shape') and len(obj.shape) >= min_dims
