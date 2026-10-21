"""Exception classes for dfm-python.

Every library error derives from :class:`DFMError`. The concrete classes also
derive from the matching built-in (or scikit-learn) exception, so callers can
catch them the standard way:

==========================  ============================================
Exception                   Bases
==========================  ============================================
``ConfigurationError``      ``DFMError``, ``ValueError``
``ConfigValidationError``   ``ConfigurationError``
``DataValidationError``     ``DFMError``, ``ValueError``
``UnsupportedInputError``   ``DataValidationError``
``NotFittedError``          ``DFMError``, ``sklearn.exceptions.NotFittedError``
``ModelNotTrainedError``    alias of ``NotFittedError``
``NumericalError``          ``DFMError``, ``ArithmeticError``
``NumericalStabilityError`` ``NumericalError``
==========================  ============================================
"""

from typing import Optional

from sklearn.exceptions import NotFittedError as _SklearnNotFittedError


class DFMError(Exception):
    """Base of all dfm-python errors; ``details`` is appended to ``str(err)`` as ``"\\nDetails: ..."``."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        if details:
            super().__init__(f"{message}\nDetails: {details}")
        else:
            super().__init__(message)


class ModelNotInitializedError(DFMError):
    """Raised when model operations are attempted before initialization."""


class NotFittedError(DFMError, _SklearnNotFittedError):
    """Raised when a fitted-only operation is attempted before ``fit``.

    Subclasses :class:`sklearn.exceptions.NotFittedError`, so it is also a
    ``ValueError`` and an ``AttributeError``.
    """


# Engines raise ``ModelNotTrainedError``; the alias keeps their raise sites unchanged.
ModelNotTrainedError = NotFittedError


class ConfigurationError(DFMError, ValueError):
    """Raised when configuration is invalid or missing."""


class DataError(DFMError):
    """Raised when data validation fails."""


class NumericalError(DFMError, ArithmeticError):
    """Raised when numerical stability issues are detected."""


class PredictionError(DFMError):
    """Raised when prediction fails."""


class DataValidationError(DFMError, ValueError):
    """Raised when data validation fails (e.g., scale mismatches)."""


class UnsupportedInputError(DataValidationError):
    """Raised when an input slot, frequency layout or option is not supported by the model."""


class NumericalStabilityError(NumericalError):
    """Raised when numerical stability issues are detected (e.g., near-singular matrices)."""


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
