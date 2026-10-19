"""Unit tests for the golden-suite helpers (pinning, fingerprint, comparison)."""

import json

import numpy as np
import pytest
import threadpoolctl
import torch

from . import _golden
from ._golden import GoldenToleranceWarning, assert_golden, fingerprint, load_golden, pinned


def _write_fixture(directory, name, arrays, fp):
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    (directory / f"{name}.json").write_text(json.dumps({"name": name, "fingerprint": fp}))


@pytest.fixture
def fixture_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_golden, "FIXTURES_DIR", tmp_path)
    return tmp_path


ARRAYS = {
    "f64": np.array([1.0, 2.0, 3.0]),
    "f32": np.array([1.0, 2.0], dtype=np.float32),
    "i": np.int64(7),
}


class TestPinned:
    def test_sets_and_restores(self):
        torch_threads = torch.get_num_threads()
        det = torch.are_deterministic_algorithms_enabled()
        with pinned():
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled()
            assert all(lib["num_threads"] == 1 for lib in threadpoolctl.threadpool_info())
        assert torch.get_num_threads() == torch_threads
        assert torch.are_deterministic_algorithms_enabled() == det

    def test_restores_on_error(self):
        torch_threads = torch.get_num_threads()
        det = torch.are_deterministic_algorithms_enabled()
        with pytest.raises(RuntimeError), pinned():
            raise RuntimeError("boom")
        assert torch.get_num_threads() == torch_threads
        assert torch.are_deterministic_algorithms_enabled() == det

    def test_fixture(self, pinned):
        assert torch.get_num_threads() == 1
        assert torch.are_deterministic_algorithms_enabled()


class TestFingerprint:
    def test_keys(self):
        fp = fingerprint()
        assert set(fp) == {"machine", "processor", "numpy", "torch", "blas"}
        assert fp["numpy"] == np.__version__
        assert fp["torch"] == torch.__version__
        assert all(e["internal_api"] for e in fp["blas"])

    def test_independent_of_thread_count(self):
        outside = fingerprint()
        with pinned():
            assert fingerprint() == outside


class TestAssertGolden:
    def test_load_round_trip(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        manifest, arrays = load_golden("case")
        assert manifest["name"] == "case"
        assert set(arrays) == set(ARRAYS)

    def test_bitwise_pass(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        assert_golden(dict(ARRAYS), "case")

    def test_bitwise_detects_last_bit(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        actual = dict(ARRAYS, f64=np.nextafter(ARRAYS["f64"], np.inf))
        with pytest.raises(AssertionError):
            assert_golden(actual, "case")

    def test_fallback_warns_and_uses_tolerance(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        actual = dict(ARRAYS, f64=ARRAYS["f64"] * (1 + 1e-12), f32=ARRAYS["f32"] * np.float32(1 + 1e-6))
        with pytest.warns(GoldenToleranceWarning, match="assert_allclose"):
            assert_golden(actual, "case")

    def test_fallback_rejects_beyond_tolerance(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        actual = dict(ARRAYS, f64=ARRAYS["f64"] * (1 + 1e-8))
        with pytest.warns(GoldenToleranceWarning), pytest.raises(AssertionError):
            assert_golden(actual, "case")

    def test_fallback_keeps_integers_exact(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        with pytest.warns(GoldenToleranceWarning), pytest.raises(AssertionError):
            assert_golden(dict(ARRAYS, i=np.int64(8)), "case")

    @pytest.mark.parametrize("actual", [
        {k: v for k, v in ARRAYS.items() if k != "i"},               # missing key
        dict(ARRAYS, extra=np.zeros(1)),                              # unexpected key
        dict(ARRAYS, f64=ARRAYS["f64"].astype(np.float32)),           # dtype
        dict(ARRAYS, f64=ARRAYS["f64"][None, :]),                     # shape
    ], ids=["missing", "extra", "dtype", "shape"])
    def test_structure_mismatch(self, fixture_dir, actual):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        with pytest.raises(AssertionError):
            assert_golden(actual, "case")

    def test_exclude_leaves_fixture_outputs_out(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        partial = {k: v for k, v in ARRAYS.items() if k != "i"}
        assert_golden(partial, "case", exclude=["i"])
        with pytest.raises(AssertionError):  # the remaining outputs are still compared
            assert_golden(dict(partial, f64=np.nextafter(ARRAYS["f64"], np.inf)), "case", exclude=["i"])

    @pytest.mark.parametrize("exclude", [["nope"], ["i"]], ids=["not-a-fixture-key", "also-given"])
    def test_exclude_must_name_absent_fixture_outputs(self, fixture_dir, exclude):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        with pytest.raises(AssertionError, match="excluded"):
            assert_golden(dict(ARRAYS), "case", exclude=exclude)
