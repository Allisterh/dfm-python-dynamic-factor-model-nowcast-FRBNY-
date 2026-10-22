"""``DFMDataset``: the DFM engine's view of a preprocessed pandas frame.

DFM trains on NumPy arrays, so this is not a torch ``Dataset``; it holds the
numeric columns of the frame with the time column as index, ordered
``[clock-frequency series | slower-frequency series]`` as the mixed-frequency
engine requires.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from ..logger import get_logger
from ..config.schema.model import _DFMEngineConfig
from ..config.constants import FREQUENCY_HIERARCHY, DEFAULT_HIERARCHY_VALUE
from ..dataset.time import TimeIndex
from ..utils.errors import DataValidationError

_logger = get_logger(__name__)


class DFMDataset:
    """Numeric columns of ``data`` indexed by its ``time_index`` column.

    Parameters
    ----------
    config : _DFMEngineConfig
        Engine configuration; its ``frequency`` and ``clock`` order the columns.
    data : pd.DataFrame
        Preprocessed data (imputed, scaled) with a time column.
    time_index : str
        Name of the time column, set as the index and dropped from the columns.
    """

    def __init__(self, config: _DFMEngineConfig, data: pd.DataFrame, time_index: str):
        self.config = config
        self.data = data
        self._time_index: Optional[TimeIndex] = None
        self.data_df: Optional[pd.DataFrame] = None
        self._columns: Optional[List[str]] = None
        self._setup(time_index)

    def _setup(self, time_index: str) -> None:
        X_df = self.data.copy()
        try:
            self._time_index = TimeIndex(pd.to_datetime(X_df[time_index]).tolist())
            X_df = X_df.set_index(time_index)
        except KeyError:
            raise DataValidationError(
                f"time_index column '{time_index}' not found in DataFrame. "
                f"Available columns: {list(X_df.columns)}",
                details=f"Requested column: {time_index}. DataFrame has {len(X_df.columns)} columns."
            )
        numeric_columns = list(X_df.select_dtypes(include=[np.number]).columns)
        if not numeric_columns:
            raise DataValidationError(
                "DFMDataset: columns not available. "
                "Please ensure dataset was initialized with valid data.",
                details="Dataset validation failed: _columns is None or empty."
            )
        self._columns = self._reorder_columns_for_mixed_freq(numeric_columns)
        self.data_df = X_df[self._columns]

    @property
    def variables(self) -> pd.DataFrame:
        return self.data_df

    @property
    def time_index(self) -> Optional[TimeIndex]:
        return self._time_index

    def _reorder_columns_for_mixed_freq(self, columns: List[str]) -> List[str]:
        """``columns`` with the clock-frequency series first, then the slower ones (stable within each group).

        ``config.frequency`` may be grouped (``{'m': [series, ...]}``) or
        per-series (``{series: 'm'}``); an unlisted series counts as clock
        frequency. Without a frequency mapping the order is unchanged.
        """
        if not self.config.frequency:
            return columns

        frequency_dict = {}
        if isinstance(self.config.frequency, dict):
            for key, value in self.config.frequency.items():
                if isinstance(value, list):
                    for series in value:
                        if series in columns:
                            frequency_dict[series] = key
                else:
                    if key in columns:
                        frequency_dict[key] = value
        if not frequency_dict:
            return columns

        clock_hierarchy = FREQUENCY_HIERARCHY.get(self.config.clock, DEFAULT_HIERARCHY_VALUE)
        clock_cols = []
        slower_cols = []
        for col in columns:
            freq = frequency_dict.get(col)
            if freq is None:
                clock_cols.append(col)
            else:
                freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
                if freq_hierarchy <= clock_hierarchy:
                    clock_cols.append(col)
                else:
                    slower_cols.append(col)

        reordered = clock_cols + slower_cols
        if reordered != columns:
            _logger.info(
                f"Reordered columns for mixed-frequency model: "
                f"{len(clock_cols)} clock-frequency, {len(slower_cols)} slower-frequency"
            )
        return reordered
