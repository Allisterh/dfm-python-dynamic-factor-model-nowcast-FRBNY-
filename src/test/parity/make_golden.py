"""Capture the golden fixtures from the current code.

Run from the repository root::

    uv run python src/test/parity/make_golden.py            # every case
    uv run python src/test/parity/make_golden.py dfm_synthetic ivdfm_macro

Each case runs on CPU under ``pinned()`` and writes
``src/test/parity/fixtures/<case>.npz`` (engine outputs) and ``<case>.json``
(git commit, platform fingerprint, data recipe, constructed hyperparameters,
output names). Fixtures are captured before any engine code moves; regenerate
them only when a numerical change is intended.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported anywhere

import json
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
# src/ must come first: the editable install appends it after the stdlib, whose
# own ``test`` package would otherwise shadow src/test.
sys.path.insert(0, str(_SRC))

import numpy as np

from test.parity._cases import CASES, GLOBAL_RNG_PRESEED, normalized
from test.parity._golden import FIXTURES_DIR, fingerprint, pinned

REPO_ROOT = _SRC.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def main(names: list[str]) -> int:
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown case(s): {unknown}; known: {sorted(CASES)}", file=sys.stderr)
        return 2
    selected = [CASES[n] for n in names] if names else list(CASES.values())

    commit = _git("rev-parse", "HEAD")
    src_dirty = bool(_git("status", "--porcelain", "--", "src/dfm_python"))
    if src_dirty:
        print("warning: src/dfm_python has uncommitted changes", file=sys.stderr)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for case in selected:
        start = time.perf_counter()
        with pinned():
            fp = fingerprint()
            run = case.run()
        elapsed = time.perf_counter() - start

        np.savez_compressed(FIXTURES_DIR / f"{case.name}.npz", **run.outputs)
        manifest = {
            "name": case.name,
            "model": case.model,
            "git_commit": commit,
            "src_dirty": src_dirty,
            "fingerprint": fp,
            "global_rng_preseed": GLOBAL_RNG_PRESEED,
            "data": normalized(run.data),
            "hyperparameters": normalized(run.hyperparameters),
            "raised": normalized(run.raised),
            "outputs": {k: {"dtype": str(np.asarray(v).dtype), "shape": list(np.asarray(v).shape)}
                        for k, v in sorted(run.outputs.items())},
        }
        (FIXTURES_DIR / f"{case.name}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        size = (FIXTURES_DIR / f"{case.name}.npz").stat().st_size
        status = f"raised {run.raised['type']}" if run.raised else f"{len(run.outputs)} outputs"
        print(f"{case.name:40s} {elapsed:6.2f}s  {size / 1024:8.1f} KiB  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
