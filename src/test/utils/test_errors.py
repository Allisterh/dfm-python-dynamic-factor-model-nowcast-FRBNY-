"""Tests for utils.errors module."""

import pytest
from dfm_python.utils.errors import (
    NumericalError,
    ModelNotTrainedError,
    ConfigurationError
)


class TestNumericalError:
    """Test suite for NumericalError."""
    
    def test_numerical_error_initialization(self):
        """Test NumericalError can be raised."""
        with pytest.raises(NumericalError):
            raise NumericalError("Test error")


class TestModelNotTrainedError:
    """Test suite for ModelNotTrainedError."""
    
    def test_model_not_trained_error(self):
        """Test ModelNotTrainedError can be raised."""
        with pytest.raises(ModelNotTrainedError):
            raise ModelNotTrainedError("Model not trained")


class TestConfigurationError:
    """Test suite for ConfigurationError."""
    
    def test_configuration_error(self):
        """Test ConfigurationError can be raised."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Configuration error")
    
    def test_configuration_error_with_details(self):
        """Test ConfigurationError with details parameter."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Main error", details="Additional details")
        assert "Main error" in str(exc_info.value)
        assert exc_info.value.details == "Additional details"



class TestExceptionBases:
    """The exception hierarchy of the error-handling conventions (spec section 9)."""

    def test_configuration_errors_are_value_errors(self):
        from dfm_python.utils.errors import ConfigValidationError, DFMError
        assert issubclass(ConfigurationError, DFMError)
        assert issubclass(ConfigurationError, ValueError)
        assert issubclass(ConfigValidationError, ConfigurationError)
        with pytest.raises(ValueError):
            raise ConfigValidationError("bad key")

    def test_data_validation_errors(self):
        from dfm_python.utils.errors import (DataValidationError, DFMError,
                                             UnsupportedInputError)
        assert issubclass(DataValidationError, DFMError)
        assert issubclass(DataValidationError, ValueError)
        assert issubclass(UnsupportedInputError, DataValidationError)
        with pytest.raises(ValueError):
            raise UnsupportedInputError("exog is not supported")

    def test_not_fitted_error_is_sklearn_not_fitted(self):
        from sklearn.exceptions import NotFittedError as SkNotFittedError
        from dfm_python.utils.errors import DFMError, NotFittedError
        assert issubclass(NotFittedError, DFMError)
        assert issubclass(NotFittedError, SkNotFittedError)
        assert issubclass(NotFittedError, ValueError)
        assert issubclass(NotFittedError, AttributeError)
        err = NotFittedError("call fit first", details="predict before fit")
        assert err.details == "predict before fit"
        assert "call fit first" in str(err)

    def test_model_not_trained_error_is_alias(self):
        from sklearn.exceptions import NotFittedError as SkNotFittedError
        from dfm_python.utils.errors import NotFittedError
        assert ModelNotTrainedError is NotFittedError
        with pytest.raises(SkNotFittedError):
            raise ModelNotTrainedError("Model has not been trained yet")

    def test_numerical_errors_are_arithmetic_errors(self):
        from dfm_python.utils.errors import DFMError, NumericalStabilityError
        assert issubclass(NumericalError, DFMError)
        assert issubclass(NumericalError, ArithmeticError)
        assert issubclass(NumericalStabilityError, NumericalError)

    def test_unchanged_errors(self):
        from dfm_python.utils.errors import (DataError, DFMError, ModelNotInitializedError,
                                             PredictionError)
        for cls in (ModelNotInitializedError, DataError, PredictionError):
            assert cls.__bases__ == (DFMError,)

    def test_exported(self):
        import dfm_python
        import dfm_python.utils as utils
        from dfm_python.utils.errors import NotFittedError, UnsupportedInputError
        assert dfm_python.NotFittedError is NotFittedError
        assert dfm_python.UnsupportedInputError is UnsupportedInputError
        assert utils.NotFittedError is NotFittedError
        assert utils.UnsupportedInputError is UnsupportedInputError
        assert {"NotFittedError", "UnsupportedInputError"} <= set(dfm_python.__all__)
