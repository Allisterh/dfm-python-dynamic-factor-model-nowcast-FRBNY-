# Changelog

## 0.7.0

- The Attention Factor Model (`dfm_python.models.afm`: `AFM`, attention and PCA
  factors, LongConv filter, Avellaneda-Lee baseline) now ships in the
  distribution; the 0.6.14 wheel on PyPI did not include it.
- README added and used as the package long description.
- Metadata: project URLs point to `github.com/minkeymouse/dfm-python`.
- `build` and `twine` moved from runtime dependencies to the `dev` extra.
- `__version__` now matches the release; the sdist no longer bundles the example CSVs or the lock file.
