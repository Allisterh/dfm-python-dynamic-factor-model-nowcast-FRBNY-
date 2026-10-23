"""Dataset of the DDFM engine: the scaled frame ``[covariates | targets]`` and its MC samples."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any

from ..config.constants import DEFAULT_TORCH_DTYPE


class DDFMDataset(Dataset):
    """Dataset for DDFM training.

    Every column but ``time_idx`` is a target unless listed in
    ``covariates``. The targets are scaled with ``scaler`` (fitted here) and
    the covariates with a fresh instance of its class, kept as
    ``covariate_scaler`` so that new rows are scaled with the same statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Sorted by index. ``time_idx`` is dropped from the variables when present.
    time_idx : str
        Time column name.
    covariates : list of str, optional
        Columns fed to the encoder but not reconstructed or forecast.
    scaler : sklearn transformer, optional
        None leaves the data as given.

    Attributes
    ----------
    data : pd.DataFrame
        The scaled frame ``[covariates | targets]`` (NaN kept).
    X, y : np.ndarray
        Its covariate and target columns, ``(T, 0)`` without covariates.
    missing_y, observed_y : np.ndarray
        Boolean masks of ``y``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        covariates: Optional[List[str]] = None,
        scaler: Optional[Any] = None,
    ):
        data = data.copy()
        data.sort_index(inplace=True)

        self.time_idx = time_idx
        variables = [col for col in data.columns if col != time_idx] if time_idx in data.columns else list(data.columns)
        covariates = [c for c in (covariates or []) if c in variables]
        self.covariates = covariates
        self.target_series = [col for col in variables if col not in covariates]

        data_for_split = data.drop(columns=[time_idx]) if time_idx in data.columns else data
        y = data_for_split[self.target_series]
        X = data_for_split.drop(columns=self.target_series) if self.target_series else pd.DataFrame()

        self.covariate_scaler = None
        if scaler is not None and not X.empty:
            self.covariate_scaler = type(scaler)()
            X = pd.DataFrame(self.covariate_scaler.fit_transform(X.values), index=X.index, columns=X.columns)

        if scaler is not None:
            y = pd.DataFrame(scaler.fit_transform(y.values), index=y.index, columns=y.columns)

        self.scaler = scaler

        self.data = pd.concat([X, y], axis=1) if not X.empty else y
        self.X = X.values if not X.empty else np.empty((len(y), 0))
        self.y = y.values
        self.missing_y = y.isna().values
        self.observed_y = ~self.missing_y

    @property
    def target_shape(self) -> Tuple[int, int]:
        """``(T, num_target_series)``."""
        return self.y.shape

    @property
    def data_shape(self) -> Tuple[int, int]:
        """``(T, num_covariates + num_target_series)``."""
        return self.X.shape[0], self.X.shape[1] + self.y.shape[1]

    @property
    def all_columns_are_targets(self) -> bool:
        return len(self.covariates) == 0

    def split_features_and_targets(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """``(covariates, targets)`` of a frame with ``self.data``'s columns; covariates None when there are none."""
        if self.all_columns_are_targets:
            return None, data
        X = data.drop(columns=self.target_series)
        y = data[self.target_series]
        return X, y

    @property
    def target_indices(self) -> np.ndarray:
        """Column positions of the targets in ``self.data``."""
        return np.array([self.data.columns.get_loc(col) for col in self.target_series])

    @classmethod
    def from_dataset(cls, new_data: pd.DataFrame, dataset: 'DDFMDataset') -> 'DDFMDataset':
        """A dataset of ``new_data`` with ``dataset``'s ``time_idx``, ``covariates`` and ``scaler`` (refitted)."""
        return cls(
            data=new_data,
            time_idx=dataset.time_idx,
            covariates=dataset.covariates,
            scaler=dataset.scaler
        )

    def create_autoencoder_dataset(
        self,
        X: Optional[torch.Tensor],
        y_tmp: torch.Tensor,
        y_actual: torch.Tensor,
        eps_draw: torch.Tensor
    ) -> 'AutoencoderDataset':
        """An ``AutoencoderDataset`` whose corrupted targets are ``y_tmp - eps_draw``."""
        return AutoencoderDataset(X=X, y_corrupted=y_tmp - eps_draw, y_clean=y_actual)

    def create_pretrain_dataset(
        self,
        data: pd.DataFrame,
        device: Optional[torch.device] = None
    ) -> 'AutoencoderDataset':
        """An ``AutoencoderDataset`` of ``data`` without corruption (``y_corrupted`` is ``y_clean``)."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_df, y_df = self.split_features_and_targets(data)
        X = None if X_df is None else torch.tensor(X_df.values, dtype=DEFAULT_TORCH_DTYPE, device=device)
        y = torch.tensor(y_df.values, dtype=DEFAULT_TORCH_DTYPE, device=device)

        return AutoencoderDataset(X=X, y_corrupted=y, y_clean=y)

    def create_autoencoder_datasets_list(
        self,
        n_mc_samples: int,
        mu_eps: np.ndarray,
        std_eps: np.ndarray,
        X: Any,
        y_tmp: Any,
        y_actual: np.ndarray,
        rng: np.random.RandomState,
        device: Optional[torch.device] = None
    ) -> List['AutoencoderDataset']:
        """``n_mc_samples`` corrupted copies of ``y_tmp``, noise drawn from ``rng`` at once.

        ``X`` and ``y_tmp`` may be arrays or frames; the noise is
        ``N(mu_eps, diag(std_eps))`` per row, ``(n_mc_samples, T)`` draws.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_tmp_array = y_tmp.values if isinstance(y_tmp, pd.DataFrame) else y_tmp
        T = y_tmp_array.shape[0]

        eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (n_mc_samples, T))

        X_tensor = torch.tensor(X_array, dtype=DEFAULT_TORCH_DTYPE, device=device) if X_array.size > 0 else None
        y_tmp_tensor = torch.tensor(y_tmp_array, dtype=DEFAULT_TORCH_DTYPE, device=device)
        y_actual_tensor = torch.tensor(y_actual, dtype=DEFAULT_TORCH_DTYPE, device=device)
        eps_draws_tensor = torch.tensor(eps_draws, dtype=DEFAULT_TORCH_DTYPE, device=device)

        return [
            self.create_autoencoder_dataset(X_tensor, y_tmp_tensor, y_actual_tensor, eps_draws_tensor[i])
            for i in range(n_mc_samples)
        ]


class AutoencoderDataset:
    """Tensors of one training sample: clean covariates, corrupted targets and clean targets.

    Slice the tensors directly (``dataset.full_input[i:j]``); ``full_input``
    is ``[X | y_corrupted]``, computed once.

    Parameters
    ----------
    X : torch.Tensor, optional
        ``(T, num_covariates)``, not corrupted.
    y_corrupted : torch.Tensor
        ``(T, num_target_series)``.
    y_clean : torch.Tensor
        ``(T, num_target_series)``, the reconstruction target.
    """

    def __init__(
        self,
        X: Optional[torch.Tensor],
        y_corrupted: torch.Tensor,
        y_clean: torch.Tensor
    ):
        self.X = X
        self.y_corrupted = y_corrupted
        self.y_clean = y_clean
        if self.X is not None:
            self._full_input = torch.cat([self.X, self.y_corrupted], dim=1)
        else:
            self._full_input = self.y_corrupted

    @property
    def full_input(self) -> torch.Tensor:
        return self._full_input

    def __len__(self) -> int:
        return self.y_corrupted.shape[0]
