"""Custom exception classes for dfm-python.

This module provides specialized exception classes for different error types,
enabling better error handling and more informative error messages.

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
    """Base exception class for all dfm-python errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize error with message and optional details.
        
        Parameters
        ----------
        message : str
            Main error message
        details : str, optional
            Additional details about the error
        """
        self.message = message
        self.details = details
        if details:
            super().__init__(f"{message}\nDetails: {details}")
        else:
            super().__init__(message)


class ModelNotInitializedError(DFMError):
    """Raised when model operations are attempted before initialization."""
    pass


class NotFittedError(DFMError, _SklearnNotFittedError):
    """Raised when a fitted-only operation is attempted before ``fit``.

    Subclasses :class:`sklearn.exceptions.NotFittedError`, so it is also a
    ``ValueError`` and an ``AttributeError``.
    """
    pass


# Engines raise ``ModelNotTrainedError``; the alias keeps their raise sites unchanged.
ModelNotTrainedError = NotFittedError


class ConfigurationError(DFMError, ValueError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(DFMError):
    """Raised when data validation fails."""
    pass


class NumericalError(DFMError, ArithmeticError):
    """Raised when numerical stability issues are detected."""
    pass


class PredictionError(DFMError):
    """Raised when prediction fails."""
    pass


class DataValidationError(DFMError, ValueError):
    """Raised when data validation fails (e.g., scale mismatches)."""
    pass


class UnsupportedInputError(DataValidationError):
    """Raised when an input slot, frequency layout or option is not supported by the model."""
    pass


class NumericalStabilityError(NumericalError):
    """Raised when numerical stability issues are detected (e.g., near-singular matrices)."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass
