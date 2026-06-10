.PHONY: help install test lint format typecheck build publish clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the dev environment with all extras
	uv sync --all-extras

test:  ## Run the test suite
	uv run pytest -q

lint:  ## Run ruff lint checks
	uv run ruff check src/topo

format:  ## Auto-format the codebase
	uv run ruff format src/topo tests
	uv run ruff check --fix src/topo

typecheck:  ## Run mypy
	uv run mypy src/topo

build:  ## Build sdist + wheel
	uv build

publish: build  ## Publish to PyPI (reads API token from .pypi-token)
	@test -f .pypi-token || { echo "ERROR: .pypi-token not found. Create it with your PyPI API token."; exit 1; }
	@if grep -q "REPLACE_ME" .pypi-token; then \
		echo "ERROR: .pypi-token still contains the placeholder."; \
		echo "Paste your real PyPI API token (starts with 'pypi-') into .pypi-token first."; \
		exit 1; \
	fi
	UV_PUBLISH_TOKEN="$$(cat .pypi-token)" uv publish

clean:  ## Remove build artifacts and caches
	rm -rf dist build *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
