"""Tests for utils.checkpoint.

The generic dimension helpers moved here from the DDFM tests. The DDFM
single-file checkpoint helpers (``infer_ddfm_input_dim``,
``infer_ddfm_params_from_state_dict``) were removed with the ``torch.save``
checkpoints; their state-dict cases are kept against
``infer_input_dim_from_state_dict``, whose defaults are the DDFM key names.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from dfm_python.utils import checkpoint
from dfm_python.utils.checkpoint import infer_input_dim_from_data, infer_input_dim_from_state_dict
from dfm_python.utils.errors import DataError, DataValidationError


@pytest.mark.parametrize("make", [lambda: np.array([[1, 2, 3], [4, 5, 6]]),
                                  lambda: torch.tensor([[1, 2, 3], [4, 5, 6]])])
def test_infer_input_dim_from_2d_data(make):
    assert infer_input_dim_from_data(make()) == 3


def test_infer_input_dim_from_a_frame_skips_the_date_id_column():
    frame = pd.DataFrame(np.zeros((2, 4)), columns=["date_id", "a", "b", "c"])
    assert infer_input_dim_from_data(frame) == 3


@pytest.mark.parametrize("make", [lambda: np.array([1, 2, 3]), lambda: torch.tensor([1, 2, 3])])
def test_infer_input_dim_from_1d_data_raises(make):
    with pytest.raises(DataError, match="Data must be at least 2D"):
        infer_input_dim_from_data(make())


@pytest.mark.parametrize("invalid_input", ["not a dict", [1, 2, 3], 42, None])
def test_infer_input_dim_from_state_dict_rejects_non_dicts(invalid_input):
    with pytest.raises(DataValidationError, match="state_dict must be a dictionary"):
        infer_input_dim_from_state_dict(invalid_input)


def test_infer_input_dim_from_state_dict_of_an_empty_dict_is_none():
    assert infer_input_dim_from_state_dict({}) is None


def test_infer_input_dim_from_state_dict_without_matching_keys_is_none():
    assert infer_input_dim_from_state_dict({"some.other.key": torch.zeros(10, 5)}) is None


def test_infer_input_dim_from_the_first_encoder_layer():
    assert infer_input_dim_from_state_dict({"autoencoder.encoder.layers.0.weight": torch.zeros(32, 64)}) == 64


def test_infer_input_dim_from_the_decoder_weight():
    assert infer_input_dim_from_state_dict({"autoencoder.decoder.decoder.weight": torch.zeros(10, 5)}) == 10


@pytest.mark.parametrize("name", ["infer_ddfm_input_dim", "infer_ddfm_params_from_state_dict"])
def test_ddfm_single_file_checkpoint_helpers_are_removed(name):
    assert not hasattr(checkpoint, name)
