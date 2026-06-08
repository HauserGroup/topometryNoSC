# Quick-start

## Install

```bash
pip install topometry-nosc
```

See [Installation](installation.md) for extras and the development setup.

## Fit a TopOGraph

TopoMetry revolves around the `TopOGraph` class. Given a data matrix `X` (NumPy array or SciPy sparse matrix), a single call to `fit()` runs the full pipeline:

1. Build a kNN graph in input space
2. Compute a density-corrected diffusion kernel
3. Estimate intrinsic dimensionality and extract DM/msDM spectral scaffolds
4. Build refined kNN graphs and kernels in scaffold space
5. Compute 2-D layouts (MAP and PaCMAP by default)

```python
import topo as tp
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=2000, noise=0.5, random_state=42)

tg = tp.TopOGraph(base_knn=15, graph_knn=15, verbosity=1)
tg.fit(X)

print(tg)
```

## Access results

```python
# 2-D embeddings
tg.msTopoMAP       # MAP layout on msDM scaffold  (n, 2)
tg.TopoMAP         # MAP layout on DM scaffold    (n, 2)
tg.msTopoPaCMAP    # PaCMAP on msDM               (n, 2)
tg.TopoPaCMAP      # PaCMAP on DM                 (n, 2)

# Spectral scaffolds (high-dimensional)
tg.spectral_scaffold(multiscale=True)   # msDM scaffold (n, n_eigs)
tg.spectral_scaffold(multiscale=False)  # DM scaffold   (n, n_eigs)

# Operators
tg.P_of_X          # diffusion operator on input space
tg.P_of_msZ        # diffusion operator on msDM scaffold
tg.P_of_Z          # diffusion operator on DM scaffold

# Intrinsic dimensionality
tg.global_id        # global ID estimate
tg.intrinsic_dim    # dict with method, global, local, details

# Eigenvalues
tg.eigenvalues      # eigenvalues of the active eigenbasis
tg.eigenspectrum()  # scree plot
```

The fitted operator properties above are read-only views of the fitted pipeline
state. Re-fit the model, or build a new `TopOGraph`, to change them.

## Choosing kernel versions

The `base_kernel_version` and `graph_kernel_version` parameters control the graph construction. Available options:

| Version | Description |
|---------|-------------|
| `bw_adaptive` (default) | Adaptive bandwidth + α=1 density correction |
| `bw_adaptive_alpha_decaying` | Adaptive bandwidth with exponential decay |
| `bw_adaptive_nbr_expansion` | Adaptive bandwidth with neighbor expansion |
| `bw_adaptive_alpha_decaying_nbr_expansion` | Both α-decay and neighbor expansion |
| `cknn` | Binary Continuous k-NN graph (Berry & Sauer 2019) |
| `fuzzy` | Fuzzy simplicial set (UMAP-style) |
| `gaussian` | Fixed-bandwidth Gaussian |

```python
tg = tp.TopOGraph(
    base_kernel_version='cknn',
    graph_kernel_version='bw_adaptive',
)
tg.fit(X)
```

## Compute additional projections

```python
# Compute a specific projection
Y = tg.project(projection_method='MAP', multiscale=True, num_iters=600)

# Available methods: 'MAP', 'PaCMAP', 'Isomap', 'IsomorphicMDE', 'IsometricMDE',
#                    't-SNE', 'UMAP', 'TriMAP', 'NCVis'
# (some require optional dependencies)
```

## Analysis utilities

```python
from topo import analysis

# Spectral selectivity — per-sample geometry diagnostics
sel = tg.spectral_selectivity(multiscale=True, k_neighbors=10)

# Diffusion-filter a signal
smoothed = tg.filter_signal(color, t=3, which='msZ')

# Diffusion-based imputation
X_imputed = tg.impute(X, t=8, which='msZ')

# Riemannian distortion diagnostics
diag = tg.riemann_diagnostics()
```

For target-aware checks, such as asking whether a 2-D embedding explains a
continuous or binary per-sample variable, see the
[Practical FAQ](faq.md#how-should-i-evaluate-an-embedding-against-a-target-variable).

## Save and load

```python
tg.save("my_topograph.pkl")
tg2 = tp.load_topograph("my_topograph.pkl")
```

## Union of Manifolds (UoM)

For datasets with disconnected components, enable block-diagonal scaffolding:

```python
tg = tp.TopOGraph(uom=True)
tg.fit(X)
```

This detects disconnected components in the base graph and builds per-component scaffolds that are assembled into block-diagonal operators and a global layout.
