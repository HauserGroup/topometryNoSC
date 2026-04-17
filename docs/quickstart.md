# Quick-start

## Install

```bash
pip install topometry
```

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

## Choosing kernel versions

The `base_kernel_version` and `graph_kernel_version` parameters control the graph construction. Available options:

| Version | Description |
|---------|-------------|
| `bw_adaptive` (default) | Adaptive bandwidth + α=1 density correction |
| `bw_adaptive_alpha_decaying` | Adaptive bandwidth with exponential decay |
| `bw_adaptive_nbr_expansion` | Adaptive bandwidth with neighbor expansion |
| `bw_adaptive_alpha_decaying_nbr_expansion` | Both α-decay and neighbor expansion |
| `cknn` | Continuous k-NN (Berry & Sauer 2019) |
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
# Quick-start cheat-sheet

## Fitting a TopOGraph
Now, let's go through a quick start!

TopoMetry functions around the `TopOGraph` class. It contains dictionaries, attributes and functions to analyse your data.
From a  data matrix ``data`` (np.ndarray, pd.DataFrame or sp.csr_matrix), you can set up a ``TopoGraph``
with default parameters:

```
import topo as tp

# Learn topological metrics and basis from data. The default is to use diffusion harmonics.
tg = tp.TopOGraph()
tg.fit(data)
```

After learning a topological basis, we can access topological metrics and basis in the ``TopOGraph`` object, and build different
topological graphs.

```
# Learn a topological graph. Again, the default is to use diffusion harmonics.
tgraph = tg.transform(data)
```

Then, it is possible to optimize the topological graph layout. TopoMetry has 5 different layout options: tSNE, MAP,
TriMAP, PaCMAP and MDE.

```
# Graph layout optimization
map_emb = tg.MAP()
mde_emb = tg.MDE()
pacmap_emb = tg.PaCMAP()
trimap_emb = tg.TriMAP()
tsne_emb = tg.tSNE()
```

We can also plot the embeddings:

```
tp.plot.scatter(map_emb)
```

## Computing several models at once

The `run_layouts()` attribute of the TopOGraph object runs all possible combinations of algorithms to perform DR
in the TopoMetry framework.

```
# These settings run all models and layouts
tg.run_layouts(X, n_components=2,
                    bases=['diffusion', 'fuzzy', 'continuous'],
                    graphs=['diff', 'cknn', 'fuzzy'],
                    layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])
```

If no parameters are passed to the `run_layouts()` function, by default it will perform the following steps:

Similarity learning and building a topological orthogonal basis with:
* Multiscale diffusion maps (`'diffusion'`)
* Fuzzy simplicial sets Laplacian Eigenmaps ('fuzzy');

Learn the topological graphs with:
* Diffusion harmonics (`'diff'`)
* Fuzzy simplicial sets

Next, it will use all layout optimization methods:
* MAP - a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions;
    - MAP and MDE use information both from the orthogonal basis and the topological graph
* [MDE](https://web.stanford.edu/~boyd/papers/min_dist_emb.html) - a general framework for graph layout optimization, with the pyMDE implementation.
* [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbclid=IwA) - using [MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
* [PaCMAP](https://arxiv.org/abs/2012.04456)
* [TriMAP](https://arxiv.org/abs/1910.00204)
* [NCVis](https://dl.acm.org/doi/abs/10.1145/3366423.3380061)

MAP, MDE, PaCMAP and NCVis use an spectral initialization from the learned topological graph. TriMAP uses PCA internally
as a initialization. NCVis uses a custom initialization procedure.

So if you want to compute the diffusion basis, its diffusion and fuzzy topological graphs, and the associated MAP
and PaCMAP layouts, you can simply run:

```
tg.run_layouts(X, n_components=2,
                    bases=['diffusion'],
                    graphs=['diff', 'fuzzy'],
                    layouts=['MAP','PaCMAP'])
```

This diversity of options is useful for comparisons and scoring, instead of selecting a single layout algorithm
_a priori_.
