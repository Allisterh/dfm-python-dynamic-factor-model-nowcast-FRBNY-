"""Atomic, verified writes of a save directory.

:func:`atomic_save_dir` writes a run into a sibling temporary directory,
verifies it, and only then renames it into place, so a failed write or a
failed verification never leaves a half-written run at ``path`` and never
destroys the run that was there before.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

__all__ = ["atomic_save_dir"]

PathLike = Union[str, "os.PathLike[str]"]


def _remove(path: Path) -> None:
    """Best-effort removal of a file or a directory tree (cleanup never raises)."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def _sibling(path: Path, kind: str) -> Path:
    return path.parent / f".{path.name}.{kind}-{uuid.uuid4().hex}"


def atomic_save_dir(
    path: PathLike,
    write_fn: Callable[[Path], None],
    verify_fn: Optional[Callable[[Path], None]] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a directory atomically: write into a temporary sibling, verify, rename.

    Parameters
    ----------
    path : str or os.PathLike
        The directory to create. Missing parent directories are created.
    write_fn : callable
        Called with the temporary directory (an existing, empty
        :class:`~pathlib.Path`); writes every file of the run into it.
    verify_fn : callable, optional
        Called with the temporary directory after ``write_fn``; raises if the
        written run is not valid (for example, when it cannot be reloaded).
    overwrite : bool, default False
        Replace an existing ``path``. The old content is first moved to a
        sibling backup and removed only after the new directory is in place;
        if the final rename fails, the backup is moved back.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False (checked before writing
        and again before the rename).
    Exception
        Whatever ``write_fn`` or ``verify_fn`` raises. The temporary directory
        is removed first, so a failed save leaves no debris.
    """
    path = Path(path)
    if path.name in ("", ".", ".."):
        raise ValueError(f"Cannot save to {str(path)!r}: give a directory name.")
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"{str(path)!r} already exists. Pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = _sibling(path, "tmp")
    tmp.mkdir()
    try:
        write_fn(tmp)
        if verify_fn is not None:
            verify_fn(tmp)
        if path.exists() or path.is_symlink():
            if not overwrite:
                raise FileExistsError(f"{str(path)!r} was created while saving. Pass overwrite=True to replace it.")
            backup = _sibling(path, "bak")
            os.replace(path, backup)
            try:
                os.replace(tmp, path)
            except BaseException:
                os.replace(backup, path)
                raise
            _remove(backup)
        else:
            os.replace(tmp, path)
    except BaseException:
        _remove(tmp)
        raise
    return path
