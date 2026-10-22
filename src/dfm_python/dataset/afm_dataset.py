"""Dataset for the Attention Factor Model.

AFM consumes a cross-sectional panel: at each date the asset returns ``R_t``
(N,) and, for the attention factors, the firm characteristics (N, M) the
factor model conditions on at ``t``. Row ``t`` pairs the returns at ``t`` with
the characteristics of the same row; the dataset does not lag them, so pass
the characteristics known before ``R_t`` realizes (typically ``X_{t-1}``).
The trading step applies the weights of date ``t-1`` to the returns of ``t``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AFMDataset(Dataset):
    """Returns panel (T, N) with optional characteristics (T, N, M).

    Parameters
    ----------
    returns : pandas.DataFrame or numpy.ndarray
        Asset returns, shape (T, N).
    characteristics : numpy.ndarray, optional
        Firm characteristics, shape (T, N, M); required for attention factors.
    window : int, optional
        Sliding-window length for batched training; None makes the full panel
        one sequence.
    stride : int
        Step between windows.
    """

    def __init__(self, returns: Union[pd.DataFrame, np.ndarray],
                 characteristics: Optional[np.ndarray] = None,
                 window: Optional[int] = None, stride: int = 1) -> None:
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.to_numpy(dtype=np.float64)
        else:
            returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, N), got {returns_arr.shape}")
        self.returns = returns_arr                                  # (T, N)
        self.T, self.N = returns_arr.shape

        if characteristics is not None:
            characteristics = np.asarray(characteristics, dtype=np.float64)
            if characteristics.shape[:2] != (self.T, self.N):
                raise ValueError(
                    "characteristics must be (T, N, M) aligned with returns, "
                    f"got {characteristics.shape} vs returns {returns_arr.shape}")
        self.characteristics = characteristics                     # (T, N, M) or None
        self.n_char: Optional[int] = None if characteristics is None else characteristics.shape[2]
        self.n_factors: Optional[int] = None
        self.window = window
        self.stride = max(1, int(stride))

        if window is not None:
            self._starts = list(range(0, self.T - window + 1, self.stride))
        else:
            self._starts = [0]

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        length = self.window if self.window is not None else self.T
        r = torch.tensor(self.returns[start:start + length], dtype=torch.float32)
        if self.characteristics is None:
            return r, None
        c = torch.tensor(self.characteristics[start:start + length], dtype=torch.float32)
        return r, c

    def get_tensors(self, device: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full panel as float32 tensors ``(T, N)`` and ``(T, N, M)`` (or None)."""
        r = torch.tensor(self.returns, dtype=torch.float32)
        c = (None if self.characteristics is None
             else torch.tensor(self.characteristics, dtype=torch.float32))
        if device is not None:
            r = r.to(device)
            c = None if c is None else c.to(device)
        return r, c

    def get_all_tensors_on_device(self, device: Optional[Any] = None
                                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Alias of :meth:`get_tensors` (the ``iVDFMDataset`` name)."""
        return self.get_tensors(device)

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True,
                       num_workers: int = 0):
        """DataLoader over the sliding windows.

        Each batch is ``(returns, characteristics)`` of shape ``(B, W, N)`` and
        ``(B, W, N, M)`` (characteristics is ``None`` when absent).
        """
        from torch.utils.data import DataLoader

        def collate(batch):
            rs = torch.stack([b[0] for b in batch])
            cs = (None if batch[0][1] is None
                  else torch.stack([b[1] for b in batch]))
            return rs, cs

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collate)
