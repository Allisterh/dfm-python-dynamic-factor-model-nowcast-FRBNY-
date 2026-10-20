# Security policy

## Supported versions

Security fixes are made in the latest release on PyPI. Older releases do not
receive fixes; upgrade with `pip install -U dfm-python`.

| Version | Supported |
|---------|-----------|
| 0.8.x   | yes       |
| < 0.8   | no        |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it
privately through GitHub:
[Security -> Report a vulnerability](https://github.com/minkeymouse/dfm-python/security/advisories/new).
Include the dfm-python, Python and torch versions, and a minimal example
(for instance a crafted run directory or YAML file) that shows the problem.

The report stays private between you and the maintainer while a fix is
prepared; once the fix is released, the advisory is published with credit to
the reporter unless you prefer otherwise.

## Trust model

dfm-python is a library that runs inside your own Python process. The inputs
it treats as potentially untrusted are files you load:

- **Run directories** (`Model.load(path)`, `dfm_python.load(path)`). With the
  default `allow_pickle=False`, nothing is unpickled: arrays are read with
  `numpy.load(..., allow_pickle=False)`, JSON files rebuild only the public
  estimator classes of `sklearn.preprocessing`, and `weights.pt` is read with
  `torch.load(..., weights_only=True)`. The package requires torch 2.10 or
  newer, because earlier releases have published flaws in that loader
  (CVE-2025-32434, CVE-2026-24747); keep torch up to date.
  `allow_pickle=True` unpickles `scaler.pkl` and imports any estimator class
  named in `config.json`, which can run arbitrary code. Use it only for
  directories you created or trust. A way to execute code, or to read or
  write files outside the directory, from a run directory loaded with
  `allow_pickle=False` is a vulnerability we want to hear about.
- **YAML configurations** (`load_yaml`, `XConfig.from_yaml`). Files are
  parsed with safe loaders, so YAML tags cannot construct Python objects.
  With omegaconf installed, interpolations are resolved, including
  `${oc.env:NAME}`, which reads environment variables; resolved values are
  passed to the estimator and written to `config.json` by `save()`. Do not
  load configuration files you have not read.
- **`save(path, overwrite=True)`** replaces whatever is at `path`, including
  a directory that is not a saved run. Pass paths you control.

Out of scope: bugs that need the attacker to already run code in your
process, numerical accuracy issues (please open a normal issue), and
vulnerabilities in dependencies that dfm-python does not reach (report those
upstream).
