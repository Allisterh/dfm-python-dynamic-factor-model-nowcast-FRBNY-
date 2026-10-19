"""``atomic_save_dir``: temporary sibling, verify, rename; no debris on failure."""

import os

import pytest

from dfm_python.io import directory as directory_module
from dfm_python.io.directory import atomic_save_dir


def _writer(content: str):
    def write(tmp):
        assert tmp.is_dir() and not any(tmp.iterdir())
        assert tmp.name.startswith(".run.tmp-")
        (tmp / "a.txt").write_text(content)
        (tmp / "sub").mkdir()
        (tmp / "sub" / "b.txt").write_text(content)

    return write


def _entries(root):
    return sorted(p.name for p in root.iterdir())


def test_writes_the_directory(tmp_path):
    seen = []
    out = atomic_save_dir(tmp_path / "run", _writer("new"), lambda tmp: seen.append(tmp))
    assert out == tmp_path / "run"
    assert (tmp_path / "run" / "a.txt").read_text() == "new"
    assert (tmp_path / "run" / "sub" / "b.txt").read_text() == "new"
    assert len(seen) == 1 and seen[0].parent == tmp_path and seen[0] != tmp_path / "run"
    assert _entries(tmp_path) == ["run"]


def test_creates_missing_parents(tmp_path):
    atomic_save_dir(tmp_path / "runs" / "deep" / "run", _writer("x"))
    assert (tmp_path / "runs" / "deep" / "run" / "a.txt").is_file()
    assert _entries(tmp_path / "runs" / "deep") == ["run"]


def test_refuses_an_existing_directory(tmp_path):
    atomic_save_dir(tmp_path / "run", _writer("old"))
    calls = []
    with pytest.raises(FileExistsError, match="overwrite=True"):
        atomic_save_dir(tmp_path / "run", lambda tmp: calls.append(tmp))
    assert not calls
    assert (tmp_path / "run" / "a.txt").read_text() == "old"
    assert _entries(tmp_path) == ["run"]


def test_refuses_an_existing_file(tmp_path):
    (tmp_path / "run").write_text("checkpoint")
    with pytest.raises(FileExistsError):
        atomic_save_dir(tmp_path / "run", _writer("new"))
    assert (tmp_path / "run").read_text() == "checkpoint"


def test_overwrite_replaces_and_leaves_no_backup(tmp_path):
    atomic_save_dir(tmp_path / "run", _writer("old"))
    (tmp_path / "run" / "stale.txt").write_text("stale")
    atomic_save_dir(tmp_path / "run", _writer("new"), overwrite=True)
    assert (tmp_path / "run" / "a.txt").read_text() == "new"
    assert not (tmp_path / "run" / "stale.txt").exists()
    assert _entries(tmp_path) == ["run"]


def test_overwrite_replaces_a_file(tmp_path):
    (tmp_path / "run").write_text("checkpoint")
    atomic_save_dir(tmp_path / "run", _writer("new"), overwrite=True)
    assert (tmp_path / "run" / "a.txt").read_text() == "new"
    assert _entries(tmp_path) == ["run"]


def _failing_verify(tmp):
    assert (tmp / "a.txt").is_file()
    raise ValueError("cannot reload")


@pytest.mark.parametrize("overwrite", [False, True])
def test_failing_verify_leaves_no_debris(tmp_path, overwrite):
    with pytest.raises(ValueError, match="cannot reload"):
        atomic_save_dir(tmp_path / "run", _writer("new"), _failing_verify, overwrite=overwrite)
    assert _entries(tmp_path) == []


def test_failing_verify_keeps_the_old_run(tmp_path):
    atomic_save_dir(tmp_path / "run", _writer("old"))
    with pytest.raises(ValueError, match="cannot reload"):
        atomic_save_dir(tmp_path / "run", _writer("new"), _failing_verify, overwrite=True)
    assert (tmp_path / "run" / "a.txt").read_text() == "old"
    assert _entries(tmp_path) == ["run"]


def test_failing_write_leaves_no_debris(tmp_path):
    def write(tmp):
        (tmp / "partial.txt").write_text("x")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        atomic_save_dir(tmp_path / "run", write)
    assert _entries(tmp_path) == []


def test_interrupt_leaves_no_debris(tmp_path):
    def write(tmp):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        atomic_save_dir(tmp_path / "run", write)
    assert _entries(tmp_path) == []


def test_path_created_while_writing_is_not_clobbered(tmp_path):
    def write(tmp):
        (tmp / "a.txt").write_text("new")
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "other.txt").write_text("other")

    with pytest.raises(FileExistsError, match="created while saving"):
        atomic_save_dir(tmp_path / "run", write)
    assert _entries(tmp_path / "run") == ["other.txt"]
    assert _entries(tmp_path) == ["run"]


def test_failed_rename_restores_the_backup(tmp_path, monkeypatch):
    atomic_save_dir(tmp_path / "run", _writer("old"))
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append((str(src), str(dst)))
        if len(calls) == 2:  # the rename of the new directory into place
            raise OSError("rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(directory_module.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="rename failed"):
        atomic_save_dir(tmp_path / "run", _writer("new"), overwrite=True)
    assert (tmp_path / "run" / "a.txt").read_text() == "old"
    assert _entries(tmp_path) == ["run"]


def test_rejects_empty_names(tmp_path):
    with pytest.raises(ValueError, match="directory name"):
        atomic_save_dir(tmp_path / "..", _writer("x"))
