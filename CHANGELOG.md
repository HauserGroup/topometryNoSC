# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Packaging
- Migrated the build backend to **Hatchling** with dynamic versioning sourced
  from `src/topo/version.py`.
- Adopted a **src-layout** (`src/topo/`) so tests run against the installed
  package.
- Completed project metadata: real description, authors/maintainers, SPDX
  `license`, classifiers, keywords and project URLs.
- **Slimmed the core dependencies** to numpy/scipy/scikit-learn/numba/tqdm.
  Moved plotting, dataframe and third-party layout/ANN dependencies into
  optional extras: `plot`, `pandas`, `ann`, `layouts`, `notebooks`, and `all`.

### Changed
- Centralised optional-dependency and ANN-backend detection in
  `topo._optional` (single source of truth replacing duplicated `_have_*`
  flags). Missing optional dependencies now raise a clear
  `pip install topometry[<extra>]` hint instead of a raw traceback.
- Replaced `print()` calls in library code with the `topo` logger hierarchy;
  verbosity now maps onto log levels via `topo.configure_logging`.
- `IntrinsicDim.transform` is now a clean no-op for scikit-learn compatibility
  (previously printed a placeholder message); invalid `random_state` values now
  raise `TypeError`.

### Tooling
- CI rewritten to use `uv` with a Python 3.10–3.13 matrix, plus ruff lint,
  ruff format check and a minimal-core import smoke test.
- Added a tag-triggered release workflow (builds sdist + wheel; PyPI publishing
  stubbed and ready to enable).
- Added `Makefile`, `CONTRIBUTING.md`, ruff/mypy/pytest/coverage configuration.

## [2.0.0]

### Removed
- Single-cell / scanpy / AnnData wrappers — TopoMetry is now a standalone
  geometry toolkit.

### Notes
- Core API unchanged: `TopOGraph`, spectral scaffolds, graph operators,
  layouts, metrics and plotting.
