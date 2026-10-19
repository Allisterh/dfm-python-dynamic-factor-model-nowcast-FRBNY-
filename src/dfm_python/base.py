"""``BaseFactorModel``: the public estimator base class.

Every migrated model subclasses :class:`BaseFactorModel`, which builds on
:class:`sklearn.base.BaseEstimator`:

- ``__init__`` takes keyword-only hyperparameters and only stores them, so
  ``get_params``, ``set_params``, ``clone`` and ``repr`` work unchanged.
- ``fit(data)`` coerces ``data`` to :class:`~dfm_python.PanelData`, rejects
  inputs the tags exclude, records the data attributes, runs the subclass's
  ``_fit`` inside :func:`~dfm_python.logger.fit_verbosity` and returns ``self``.
- Fitted state lives in attributes ending in ``_``; before ``fit`` they raise
  :class:`~dfm_python.utils.errors.NotFittedError` (an ``AttributeError``), so
  ``hasattr(model, "result_")`` is False.
- Tags (``get_tags()``) describe the task, backend and capabilities; the
  default ``predict``, ``predict_factors``, ``update`` and ``allocate`` raise
  ``TypeError`` or ``UnsupportedInputError`` when the tags exclude them.
- ``get_config()`` / ``from_config()`` convert to and from the model's
  :class:`~dfm_python.config.EstimatorConfig` mirror.

This is not the engine base class ``dfm_python.models.base.BaseFactorModel``,
which the unmigrated models still inherit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .config.estimator import EstimatorConfig, _unknown_keys_message
from .dataset.panel import PanelData
from .logger.progress import fit_verbosity
from .utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError

__all__ = ["BaseFactorModel", "DEFAULT_TAGS"]

#: Tag defaults. Subclasses override any subset in their own ``_tags``.
DEFAULT_TAGS: dict = {
    "task": "forecast",  # "forecast" or "allocation"
    "backend": "numpy",  # "numpy" or "torch"
    "capability:exog": False,
    "capability:series_exog": False,
    "capability:mixed_frequency": False,
    "capability:missing_values": False,
    "capability:update": False,
    "capability:factor_forecast": False,
    "capability:device": False,
    "update_kind": None,  # "filter", "encoder_forward_rescale", "encoder_forward" or "reattach"
    "predict_data": None,  # "new_rows", "full_panel" or "window"
    "non_deterministic": False,
}


def _is_fitted_name(name: str) -> bool:
    """Public fitted-attribute names: a trailing underscore and no leading one."""
    return name.endswith("_") and not name.startswith("_")


class BaseFactorModel(BaseEstimator):
    """Base class of the public estimators (sklearn conventions, PanelData input).

    Subclasses set ``_tags`` (overrides of :data:`DEFAULT_TAGS`) and
    ``_config_class``, declare keyword-only hyperparameters in ``__init__``
    that are stored unchanged, and implement ``_fit(data, **fit_kwargs)`` plus
    the methods their tags enable.
    """

    _tags: ClassVar[dict] = dict(DEFAULT_TAGS)
    _config_class: ClassVar[Optional[type]] = None

    # -- tags -------------------------------------------------------------

    @classmethod
    def get_tags(cls) -> dict:
        """Return the tags, merging ``_tags`` along the MRO (base classes first).

        Raises
        ------
        ConfigurationError
            If a class in the MRO sets a tag name that is not in :data:`DEFAULT_TAGS`.
        """
        tags: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_tags")
            if own is None:
                continue
            unknown = [k for k in own if k not in DEFAULT_TAGS]
            if unknown:
                raise ConfigurationError(f"{klass.__name__}._tags: " + _unknown_keys_message(unknown, list(DEFAULT_TAGS), "tag"))
            tags.update(own)
        return tags

    # -- parameters and config ---------------------------------------------

    def set_params(self, **params: Any) -> BaseFactorModel:
        """Set hyperparameters; an unknown name raises ``ConfigurationError`` naming the closest one."""
        valid = self._get_param_names()
        unknown = [k for k in params if k.split("__", 1)[0] not in valid]
        if unknown:
            raise ConfigurationError(f"{type(self).__name__}.set_params: " + _unknown_keys_message(unknown, valid, "parameter"))
        return super().set_params(**params)

    @classmethod
    def _require_config_class(cls) -> type:
        cfg_cls = cls._config_class
        if cfg_cls is None:
            raise NotImplementedError(f"{cls.__name__} defines no _config_class.")
        return cfg_cls

    def get_config(self) -> EstimatorConfig:
        """Return the config mirror whose fields equal ``get_params(deep=False)``."""
        return self._require_config_class()(**self.get_params(deep=False))

    @classmethod
    def from_config(cls, cfg: EstimatorConfig | Mapping) -> BaseFactorModel:
        """Build the estimator from its config: ``cls(**cfg.to_dict())``.

        A mapping is first validated with ``_config_class.from_dict``, so unknown
        keys raise ``ConfigurationError``.
        """
        cfg_cls = cls._require_config_class()
        if isinstance(cfg, Mapping):
            cfg = cfg_cls.from_dict(cfg)
        elif not isinstance(cfg, cfg_cls):
            raise ConfigurationError(
                f"{cls.__name__}.from_config expects a {cfg_cls.__name__} (or a mapping), got {type(cfg).__name__}."
            )
        return cls(**cfg.to_dict())

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Parameter sets for :func:`dfm_python.testing.check_estimator` (small and fast on CPU)."""
        return [{}]

    # -- fitted state ------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. A missing public fitted attribute
        # of an unfitted model raises NotFittedError, which is an AttributeError,
        # so hasattr(model, "result_") is False before fit.
        if _is_fitted_name(name) and not self._is_fitted():
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet, so it has no attribute {name!r}. "
                "Call fit(data) first."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _is_fitted(self) -> bool:
        return any(_is_fitted_name(k) for k in vars(self))

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted()

    def _check_is_fitted(self, method: str) -> None:
        """Raise ``NotFittedError`` naming ``method`` if the model is not fitted."""
        if not self._is_fitted():
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call fit(data) before {method}()."
            )

    def _clear_fitted(self) -> None:
        for name in [k for k in vars(self) if _is_fitted_name(k)]:
            delattr(self, name)

    # -- data --------------------------------------------------------------

    def _validate_data(self, data: Any) -> PanelData:
        """Coerce ``data`` to ``PanelData`` and reject what the tags exclude.

        Raises
        ------
        UnsupportedInputError
            For ``exog``, ``series_exog``, a mixed-frequency layout or missing
            values when the matching ``capability:*`` tag is False; the
            message names the tag.
        """
        panel = PanelData._coerce(data)
        tags = self.get_tags()
        name = type(self).__name__
        if panel.exog_names and not tags["capability:exog"]:
            raise UnsupportedInputError(
                f"{name} does not accept exog (tag capability:exog is False); got exog columns "
                f"{list(panel.exog_names)}. Pass the data without exog."
            )
        if panel.series_exog_names and not tags["capability:series_exog"]:
            raise UnsupportedInputError(
                f"{name} does not accept series_exog (tag capability:series_exog is False); got features "
                f"{list(panel.series_exog_names)}. Pass the data without series_exog."
            )
        if panel.is_mixed_frequency and not tags["capability:mixed_frequency"]:
            raise UnsupportedInputError(
                f"{name} does not accept mixed-frequency data (tag capability:mixed_frequency is False); got "
                f"freq={panel.freq}. Pass series sampled at one frequency."
            )
        if not tags["capability:missing_values"]:
            for slot, frame in (("y", panel.y), ("exog", panel.exog), ("series_exog", panel.series_exog)):
                if frame is not None and frame.isna().to_numpy().any():
                    raise UnsupportedInputError(
                        f"{name} does not accept missing values (tag capability:missing_values is False); "
                        f"{slot} contains NaN. Impute or drop them before fitting."
                    )
        return panel

    def _set_data_attributes(self, data: PanelData) -> None:
        """Record the fitted data attributes shared by every model."""
        self.n_features_in_ = len(data.series_names)
        self.feature_names_in_ = np.asarray(data.series_names, dtype=object)
        self.exog_names_in_ = np.asarray(data.exog_names, dtype=object)
        if self.get_tags()["capability:series_exog"]:
            self.series_exog_names_in_ = np.asarray(data.series_exog_names, dtype=object)
        self.index_ = data.index

    @staticmethod
    def _validate_horizon(h: Any) -> int:
        """Return ``h`` as an int; a non-integer raises ``TypeError`` and ``h < 1`` raises ``ValueError``."""
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)):
            raise TypeError(f"h must be a positive integer (the forecast horizon), got {type(h).__name__} {h!r}.")
        h = int(h)
        if h < 1:
            raise ValueError(f"h must be a positive integer (the forecast horizon), got {h}.")
        return h

    # -- fit ---------------------------------------------------------------

    def fit(self, data: Any, **fit_kwargs: Any) -> BaseFactorModel:
        """Fit on ``data`` (``PanelData``, a DataFrame or a 2-D array) and return ``self``.

        Refitting starts from scratch: earlier fitted attributes are removed
        first, and also when fitting fails. Progress output follows the
        ``verbose`` hyperparameter (0 silent, 1 progress, 2 debug).
        """
        panel = self._validate_data(data)
        verbose = getattr(self, "verbose", 0)
        self._clear_fitted()
        try:
            with fit_verbosity(verbose):
                self._set_data_attributes(panel)
                self._fit(panel, **fit_kwargs)
        except BaseException:
            self._clear_fitted()
            raise
        return self

    def _fit(self, data: PanelData, **fit_kwargs: Any) -> None:
        """Train on validated data and set the fitted attributes (``result_`` and the rest)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _fit().")

    # -- methods gated by tags ---------------------------------------------

    def predict(self, h: int, data: Any = None, **predict_kwargs: Any) -> pd.DataFrame:
        """Forecast ``h`` steps as a time-indexed DataFrame (``task == "forecast"``)."""
        if self.get_tags()["task"] != "forecast":
            raise TypeError(
                f"{type(self).__name__} is an allocation model (tag task='allocation') and has no predict(h); "
                "use allocate(data=None) for portfolio weights."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement predict().")

    def predict_factors(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (``capability:factor_forecast``)."""
        if not self.get_tags()["capability:factor_forecast"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not forecast factors (tag capability:factor_forecast is False)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement predict_factors().")

    def update(self, data: Any, *, refit: bool = False) -> BaseFactorModel:
        """Condition later predictions on ``data``; parameters change only with ``refit=True`` (``capability:update``)."""
        if not self.get_tags()["capability:update"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not support update (tag capability:update is False); "
                "call fit(data) to refit on the new data."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement update().")

    def allocate(self, data: Any = None) -> pd.DataFrame:
        """Return portfolio weights as a time-indexed DataFrame (``task == "allocation"``)."""
        if self.get_tags()["task"] != "allocation":
            raise TypeError(
                f"{type(self).__name__} is a forecasting model (tag task={self.get_tags()['task']!r}) and has no "
                "allocate(); use predict(h)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement allocate().")
