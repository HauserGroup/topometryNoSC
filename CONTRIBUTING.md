# Contributing to TopoMetry

First off, thanks for taking the time to contribute! 🎉

I appreciate your help and this document provides a guide to help you get started quickly, hopefully.

## 🛠️ Quick Start

We use [uv](https://docs.astral.sh/uv/) to make environment and dependency management fast and painless.

1. **Fork and clone** the repository.
2. **Install `uv`** if you haven't already (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).
3. **Set up the environment**:

```bash
uv sync --all-extras   # Creates a .venv and installs everything you need
```

Or use the Makefile shortcuts:

```bash
make install   # uv sync --all-extras
make test      # run tests
make lint      # ruff check
make format    # ruff format + autofix
make typecheck # mypy
make build     # build sdist + wheel
```

## Project layout

- `src/topo/` — the package (src-layout; tests run against the installed package).
- `tests/` — pytest suite.
- Optional dependencies are grouped as extras: `plot`, `pandas`, `ann`,
  `layouts`, `notebooks`, and `all`. The core install depends only on
  numpy/scipy/scikit-learn/numba/tqdm. New optional dependencies must be guarded
  through `topo._optional` so a missing extra raises a clear, actionable error.

## Coding standards

- Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/); run
  `make format` before committing. CI enforces `ruff check` and
  `ruff format --check`.
- Library code logs through the `topo` logger hierarchy
  (`logging.getLogger(__name__)`); do not `print()` from library code. The one
  exception is numba-jitted functions, where `print` is the only option.
- Public APIs should carry NumPy-style docstrings and type hints.
- `pre-commit` hooks are configured; install them with `uv run pre-commit install`.

## Tests

- Add tests under `tests/` for new behavior.
- Mark tests that exercise optional-dependency guard rails with
  `@pytest.mark.optional_deps`, and slow tests with `@pytest.mark.slow`.
- Ensure `uv run pytest -q` is green across supported Python versions
  (3.10–3.13) before opening a PR.

## Pull requests

1. Create a feature branch.
2. Keep the public API (`topo.TopOGraph().fit(X)` and its result attributes)
   backwards-compatible unless a change is explicitly discussed.
3. Make sure CI passes (lint, format, tests).
