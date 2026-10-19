# Changelog

## Unreleased

- **Breaking (installation):** a bare `pip install dfm-python` no longer
  installs optuna, sktime, omegaconf, hydra-core, pyarrow, matplotlib or
  pytorch-lightning. Code that used them through dfm-python now raises an
  `ImportError` naming the extra to install: `calculate_rmse`/`calculate_mae`
  need `[metrics]`, `dfm_python.tuning` needs `[tune]`,
  `YamlSource.load()` needs `[yaml]`, polars input needs `[polars]` (polars
  and pyarrow), and the `tutorial/` scripts need `[examples]`.
  **Migration:** to keep the 0.7.0 behaviour, depend on `dfm-python[all]`
  (for example in a `requirements.txt` that pinned bare `dfm-python`).
- Runtime dependencies reduced to numpy, pandas, scipy, scikit-learn, torch,
  pykalman and einops. pytorch-lightning and matplotlib were never imported
  and are dropped; optuna, omegaconf, sktime, polars (with pyarrow) and
  hydra-core moved to the `tune`, `yaml`, `metrics`, `polars` and `examples`
  extras (`all` installs every extra).
- **Breaking (extras):** the `db`, `deep`, `transform`, `forecasting` and
  `hydra` extras are removed. pip only warns about an unknown extra, so
  `dfm-python[hydra]` now silently installs neither omegaconf nor hydra-core;
  use `[yaml]` or `[examples]` instead (`[metrics]` replaces `[transform]` and
  `[forecasting]`).
- The `dev` dependency group lists its packages directly (it works with
  `pip install --group dev` as well as uv) and includes the optional packages
  the test suite imports.
- `iVDFM` and the four config classes (`DFMConfig`, `DDFMConfig`,
  `iVDFMConfig`, `AFMConfig`) are exported from `dfm_python`. `iVDFMDataset`
  is no longer exported (from `dfm_python` or `dfm_python.dataset`); it is the
  iVDFM estimator's private window adapter in `dfm_python.dataset.ivdfm_dataset`.
- `__version__` is read from the installed package metadata.
- The package ships a `py.typed` marker.
- Fixed: `compute_cov_safe` with listwise deletion raised on every
  non-square input; `run_em_algorithm` returned non-finite parameters when an
  iteration diverged instead of the last finite ones (runs whose parameters
  stay finite, including those that stop because the log-likelihood is
  non-finite, are unchanged); `summary()` again prints
  the log-likelihood.
- Test suite repaired: tests for removed helpers deleted, the rest updated to
  the current API. It now collects and passes.
- CI (tests on Python 3.10-3.13, ruff, sdist/wheel build check), ruff and
  pre-commit configuration, and pytest `testpaths` added.
- Hygiene: undefined names on error paths fixed; debug `print`/`breakpoint`
  calls removed from `functional/krylov.py`; refactoring-history comment
  prefixes and leftover KDFM references removed; the unused
  `DEFAULT_KDFM_*` and `DEFAULT_INF_VALUE` constants removed.

## 0.7.0

- The Attention Factor Model (`dfm_python.models.afm`: `AFM`, attention and PCA
  factors, LongConv filter, Avellaneda-Lee baseline) now ships in the
  distribution; the 0.6.14 wheel on PyPI did not include it.
- README added and used as the package long description.
- Metadata: project URLs point to `github.com/minkeymouse/dfm-python`.
- `build` and `twine` moved from runtime dependencies to the `dev` extra.
- `__version__` now matches the release; the sdist no longer bundles the example CSVs or the lock file.
