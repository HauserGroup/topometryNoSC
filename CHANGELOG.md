# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-10

First PyPI release of the fork.

### Fixed
- Repaired the breakage left behind by the spectral/ann/dists simplification:
  restored imports and call sites (`pairwise_distances` now delegated to
  scikit-learn, spectral initialization via `EigenDecomposition`/`LE`), broke
  the `kernels <-> spectral.eigen` circular import, removed the dead
  `graph_kernel` state, and made the UoM path populate `P_Z_`/`P_msZ_`.
- kNN graphs no longer silently drop genuine zero-distance neighbors
  (duplicate points); off-diagonal zeros are clamped to a tiny positive float
  so every row keeps exactly `k` edges (both sklearn and HNSWlib backends).

### Changed
- Deleted `topo.base.dists`; the Euclidean gradient is delegated to
  `umap.distances.euclidean_grad`.
- Runtime dependencies slimmed: Jupyter packages moved to the dev group;
  `hnswlib` and `pacmap` are optional extras only (`[ann]`, `[layouts]`),
  with corrected install hints.

### Documentation
- API reference now covers the standalone building blocks (`compute_kernel`,
  `eigendecompose`, `LE`, `graph_laplacian`, `diffusion_operator`,
  `automated_scaffold_sizing`, projection diagnostics).
- `cknn_graph` gained a full NumPy-style docstring with the Berry & Sauer
  reference. Example notebooks updated to the current API and re-synced with
  their jupytext sources.

### Tests
- Added integration tests composing the standalone estimators outside
  `TopOGraph`, plus a duplicate-point kNN regression test. Suite: 213 tests,
  ruff/mypy/pyright clean.

### Packaging
- **Renamed the distribution to `topometry-nosc`** (independently maintained
  fork) to avoid colliding with the upstream `topometry` project on PyPI. The
  import package is still `topo`. Reset the version to `0.1.0`.
- Added the **`py.typed`** marker so the declared `Typing :: Typed` support is
  honoured by downstream type checkers (PEP 561).
- Set fork authorship/maintainership, repointed project URLs to the fork (with
  an "Original project" link to upstream), and fork-ified `CITATION.cff` to
  cite this fork while referencing the original TopOMetry software and paper.
- Migrated the build backend to **Hatchling** with dynamic versioning sourced
  from `src/topo/version.py`.
- Adopted a **src-layout** (`src/topo/`) so tests run against the installed
  package.
- Completed project metadata: real description, authors/maintainers, SPDX
  `license`, classifiers, keywords and project URLs.
- **Slimmed the core dependencies** to numpy/scipy/scikit-learn/numba/tqdm.
  Moved plotting, dataframe and third-party layout/ANN dependencies into
  optional extras: `plot`, `pandas`, `ann`, `layouts`, `notebooks`, and `all`.

### Documentation
- Documented the entire public API to a single NumPy-convention standard, added
  type hints across public signatures, and enabled ruff `D` (pydocstyle) lint
  enforcement to keep docstrings consistent.

### Changed
- Split the monolithic `TopOGraph` (≈1370 → ≈750 lines) into focused mixins
  under `topo._pipeline` (`GraphBuildMixin`, `EigenBuildMixin`,
  `LayoutBuildMixin`), mirroring the existing `UoMMixin` pattern. Public API,
  attributes and method-resolution order are unchanged.
- Centralised optional-dependency and ANN-backend detection in
  `topo._optional` (single source of truth replacing duplicated `_have_*`
  flags). Missing optional dependencies now raise a clear
  `pip install topometry-nosc[<extra>]` hint instead of a raw traceback.
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
