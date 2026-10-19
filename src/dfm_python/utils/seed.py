"""Global RNG seeding.

Every global seeding call in the package goes through :func:`apply_seed`, so
there is one place that touches the torch and numpy global generators. Call
sites stay where they always were: moving one would change the order in
which random numbers are consumed and therefore the numbers a model produces.

Local generators (``np.random.RandomState(seed)``, seeded optuna samplers)
are not global seeding and do not use this helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["apply_seed"]


def apply_seed(
    seed: Any,
    *,
    torch_global: bool = True,
    numpy_global: bool = False,
) -> None:
    """Seed the global torch and/or numpy generators.

    Parameters
    ----------
    seed : int or None
        Seed passed unchanged to ``torch.manual_seed`` and
        ``np.random.seed``. Any cast (for example ``int(...)``) is the
        caller's job. ``None`` does nothing.
    torch_global : bool, default True
        Call ``torch.manual_seed(seed)``.
    numpy_global : bool, default False
        Call ``np.random.seed(seed)``, after torch when both are set.
    """
    if seed is None:
        return
    if torch_global:
        torch.manual_seed(seed)
    if numpy_global:
        np.random.seed(seed)
