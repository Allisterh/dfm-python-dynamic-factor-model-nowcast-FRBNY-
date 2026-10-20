"""Run the Python examples of README.md.

Each ``## `` or ``### `` section's ```` ```python ```` blocks run in order in
one fresh namespace (a ``## `` section ends at its first ``### ``
subsection), in a temporary working directory that links the repository's
``config/`` and ``data/`` (the examples read the shipped YAMLs and CSVs and
save under ``runs/``). Global state an example changes on purpose (torch
threads and determinism, the package logger) is restored afterwards.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

_SECTION = re.compile(r"^#{2,3} (.+)$", re.M)
_PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _sections() -> dict[str, list[str]]:
    """``{section title: [python block, ...]}`` for the sections with Python code."""
    text = README.read_text(encoding="utf-8")
    heads = list(_SECTION.finditer(text))
    out: dict[str, list[str]] = {}
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        blocks = _PYTHON_BLOCK.findall(text, head.end(), end)
        if blocks:
            out[head.group(1).strip()] = blocks
    return out


SECTIONS = _sections()


def _missing_requirement(block: str) -> str | None:
    """The optional package a block needs and that is not installed, if any."""
    if "dfm_python.tuning" in block and importlib.util.find_spec("optuna") is None:
        return "optuna (the tune extra)"
    if ("load_yaml" in block or "from_yaml" in block) and not (
        importlib.util.find_spec("omegaconf") or importlib.util.find_spec("yaml")
    ):
        return "omegaconf or PyYAML (the yaml extra)"
    return None


@pytest.fixture
def readme_cwd(tmp_path, monkeypatch):
    for name in ("config", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    package_logger = logging.getLogger("dfm_python")
    saved_logger = (package_logger.level, list(package_logger.handlers))
    saved_torch = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    yield tmp_path
    package_logger.setLevel(saved_logger[0])
    package_logger.handlers[:] = saved_logger[1]
    torch.set_num_threads(saved_torch[0])
    torch.use_deterministic_algorithms(saved_torch[1])


def test_readme_has_examples_for_every_model():
    expected = {
        "Quick start", "The estimator API", "Data", "DFM", "DDFM", "iVDFM", "AFM",
        "Saving and loading", "YAML configurations", "Reproducibility", "Logging",
    }
    assert expected <= set(SECTIONS), sorted(expected - set(SECTIONS))


@pytest.mark.parametrize("title", list(SECTIONS))
def test_readme_section_runs(title, readme_cwd, capsys):
    skipped = []
    module_name = "readme_" + re.sub(r"\W+", "_", title)
    namespace: dict = {"__name__": module_name}
    for i, block in enumerate(SECTIONS[title]):
        missing = _missing_requirement(block)
        if missing is not None:
            skipped.append(f"block {i} needs {missing}")
            continue
        exec(compile(block, f"README.md [{title}, block {i}]", "exec"), namespace)
    assert os.getcwd() == str(readme_cwd)
    if skipped:
        pytest.skip("; ".join(skipped))
