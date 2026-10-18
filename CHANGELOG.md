# Changelog

## Unreleased

- Runtime dependencies reduced to numpy, pandas, scipy, scikit-learn, torch,
  pykalman and einops. pytorch-lightning, pyarrow and matplotlib were never
  imported and are dropped; optuna, omegaconf, sktime, polars and hydra-core
  moved to the `tune`, `yaml`, `metrics`, `polars` and `examples` extras
  (`all` installs every extra). The unused `db`, `deep`, `transform`,
  `forecasting` and `hydra` extras are removed.
- `iVDFM`, `iVDFMDataset` and the four config classes (`DFMConfig`,
  `DDFMConfig`, `iVDFMConfig`, `AFMConfig`) are exported from `dfm_python`.
- `__version__` is read from the installed package metadata.
- The package ships a `py.typed` marker.
- Fixed: `compute_cov_safe` with listwise deletion raised on every
  non-square input; `run_em_algorithm` returned non-finite parameters when an
  iteration diverged instead of the last finite ones; `summary()` again prints
  the log-likelihood.
- Test suite repaired: tests for removed helpers deleted, the rest updated to
  the current API. It now collects and passes.
- CI (tests on Python 3.10-3.13, ruff, sdist/wheel build check), ruff and
  pre-commit configuration, and pytest `testpaths` added.
- Hygiene: undefined names on error paths fixed; debug `print`/`breakpoint`
  calls removed from `functional/krylov.py`; refactoring-history comment
  prefixes and leftover KDFM references removed; the unused
  `DEFAULT_KDFM_*` and `DEFAULT_INF_VALUE` constants removed;
  `get_target_scaler` reads only the `scaler` attribute.

## 0.7.0

- The Attention Factor Model (`dfm_python.models.afm`: `AFM`, attention and PCA
  factors, LongConv filter, Avellaneda-Lee baseline) now ships in the
  distribution; the 0.6.14 wheel on PyPI did not include it.
- README added and used as the package long description.
- Metadata: project URLs point to `github.com/minkeymouse/dfm-python`.
- `build` and `twine` moved from runtime dependencies to the `dev` extra.
- `__version__` now matches the release; the sdist no longer bundles the example CSVs or the lock file.
