# Installation

`topometry-nosc` is implemented in Python, with models that inherit from
[scikit-learn](https://github.com/scikit-learn/scikit-learn) `BaseEstimator` and
`TransformerMixin`. This makes the building-block classes compatible with
scikit-learn pipelines and easy to combine with other workflows.

The **core** install depends only on numpy, scipy, scikit-learn, numba and
joblib:

```bash
pip install topometry-nosc            # core
pip install "topometry-nosc[all]"     # core + plotting, dataframes, ANN backends, extra layouts
```

!!! danger "Do not install alongside upstream `topometry`"
    This fork ships the **same import package name** (`topo`) as the original
    [`topometry`](https://pypi.org/project/topometry/). Installing both
    distributions in one environment makes them silently overwrite each other's
    `topo/` files.

    To prevent broken installs, `import topo` **fails fast** with an
    `ImportError` if it detects the upstream `topometry` distribution present in
    the same environment:

    ```text
    Conflicting installation detected: the upstream 'topometry' distribution is
    installed in the same environment as this fork ('topometry-nosc')...
    ```

    Use a fresh virtual environment, or run `pip uninstall topometry` before
    installing this fork.

## Optional dependencies (extras)

Optional features are grouped into extras — install only what you need:

| Extra        | Adds                                                    |
|--------------|---------------------------------------------------------|
| `plot`       | matplotlib (plotting)                                   |
| `pandas`     | pandas (DataFrame I/O)                                  |
| `ann`        | hnswlib (fast approximate nearest neighbors)           |
| `amg`        | pyamg (algebraic-multigrid `eigensolver='amg'`)        |
| `layouts`    | pacmap, pymde, trimap, umap-learn (extra projections)  |
| `notebooks`  | jupyterlab / ipywidgets                                |
| `all`        | everything above                                       |

```bash
pip install "topometry-nosc[plot]"        # one extra
pip install "topometry-nosc[ann,layouts]" # several
```

Missing an optional dependency raises a clear message telling you which extra to
install (e.g. `pip install topometry-nosc[plot]`).

### Approximate Nearest Neighbors

The kNN-graph builder (`topo.base.ann.kNN`) wraps several ANN backends. If none
of the optional libraries is installed, it falls back to `scikit-learn`
neighborhood search (slower on large datasets). Supported backends include
[HNSWlib](https://github.com/nmslib/hnswlib) (default, via the `ann` extra) and
[NMSlib](https://github.com/nmslib/nmslib). If your CPU supports advanced
instructions, NMSlib built from source can be faster:

```bash
pip install --no-binary :all: nmslib
```

### Additional layout methods

Fast implementations of [Isomap](https://doi.org/10.1126/science.290.5500.2319)
and a cross-entropy–minimization layout (MAP) are built in. Other layout
algorithms are available via the `layouts` extra:
[UMAP](https://umap-learn.readthedocs.io/en/latest/), `umap-learn`;
[PaCMAP](http://jmlr.org/papers/v22/20-1061.html), `pacmap`;
[TriMAP](https://github.com/eamid/trimap), `trimap`;
IsomorphicMDE / IsometricMDE, `pymde`. These are handled by
`topo.layouts.Projector`.

## Development install

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras   # package + all extras + dev tooling
uv run pytest -q       # run the tests
```

To build the documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve
```
