# Practical FAQ

Short answers to common workflow questions that are not tied to one API call.

## Should I use UMAP or TopoMetry?

Use UMAP when you mainly need a fast, familiar standalone 2-D visualization and
its assumptions are working well for your data. UMAP is a layout method; it
learns a neighbor graph and optimizes a low-dimensional embedding from that
graph.

Use TopoMetry when you want to inspect and control the full geometry pipeline:
nearest-neighbor graph, kernel, diffusion or Laplacian operator, spectral
scaffold, refined graph, layout, and diagnostics. This is useful when you care
about geometry-preserving representations, method comparison, nonuniform
sampling density, disconnected or weakly connected structure, or quantitative
operator-native checks rather than a single visualization.

The two are not mutually exclusive. TopoMetry can use UMAP-style ideas through
the `fuzzy` kernel and can compute UMAP as one projection option when the
optional layout dependencies are installed. In that setting, UMAP is one layout
inside a broader graph/operator workflow.

If you need streaming updates or an inverse transform for new points, TopoMetry
is usually not the right first tool because embeddings are recomputed from the
fitted graph/operator state. UMAP, parametric UMAP, or an autoencoder-style model
may fit that requirement better.

Relevant background:

- [Layouts and projections](background.md#6-layouts-and-projections)
- [Kernel graph](concepts.md#kernel-graph)
- [Spectral scaffold](concepts.md#spectral-scaffold)
- [Topology-preservation metrics](concepts.md#topology-preservation-metrics)

## How should I evaluate an embedding against a target variable?

TopoMetry's built-in scores mainly ask whether an embedding preserves geometry
or graph/operator structure. They do not directly answer whether a 2-D embedding
explains a binary label, clinical score, gene expression value, or other
per-sample target.

For a continuous or binary target, treat target-aware evaluation as a separate
question:

1. First inspect the target visually by coloring the embedding. This often
   catches artifacts, class imbalance, outliers, and batch effects that a scalar
   score hides.
2. Compare geometry-preservation metrics across candidate embeddings so a
   target score does not reward a layout that simply tears the manifold apart.
3. Use a simple cross-validated predictor from the embedding to the target when
   you want a supervised summary. Prefer simple models such as kNN or logistic
   regression before using highly tuned models, otherwise the score may mostly
   reflect the predictor's hyperparameters.
4. Compare against baselines: the same predictor on PCA coordinates, the
   spectral scaffold, the original features if feasible, and a shuffled target.
5. For continuous signals, inspect graph smoothness on the fitted operator. A
   signal that is smooth on the graph is aligned with the learned geometry; a
   signal that is not smooth may still be biologically meaningful, but it is not
   captured as a simple geometric trend.

Example: cross-validated target predictability from a 2-D embedding.

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

Y = tg.msTopoMAP

# Continuous target, e.g. expression of one gene.
reg_cv = KFold(n_splits=5, shuffle=True, random_state=0)
r2 = cross_val_score(
    KNeighborsRegressor(n_neighbors=15),
    Y,
    continuous_target,
    cv=reg_cv,
    scoring="r2",
)

# Binary target.
clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc = cross_val_score(
    KNeighborsClassifier(n_neighbors=15),
    Y,
    binary_target,
    cv=clf_cv,
    scoring="roc_auc",
)
```

Example: a simple graph-smoothness check for a continuous signal.

```python
import numpy as np

signal = np.asarray(continuous_target, dtype=float)
smoothed = tg.filter_signal(signal, t=3, which="msZ")

residual = signal - smoothed
smoothness = 1.0 - np.var(residual) / (np.var(signal) + 1e-12)
```

The smoothness value above is a practical diagnostic, not a universal benchmark.
It depends on the chosen operator, diffusion time, preprocessing, and target
scale. Use it to compare nearby choices under the same setup, not as an
absolute measure of biological or scientific importance.

Relevant background:

- [Evaluation and geometry diagnostics](background.md#7-evaluation-and-geometry-diagnostics)
- [Diffusion operator](concepts.md#diffusion-operator)
- [Riemannian deformation](math_details.md#9-measuring-distortion-the-riemannian-metric)

## Why can different layouts of the same data look different?

Layouts optimize different objectives and emphasize different scales. MAP/UMAP
style objectives emphasize neighbor preservation through a fuzzy-graph loss;
PaCMAP explicitly balances nearby, mid-near, and farther pairs; Isomap tries to
preserve graph geodesic distances; t-SNE is often strongest for local cluster
separation but can distort global distances.

Differences between layouts are not automatically errors. They are a sign to
compare geometry metrics, inspect the spectral scaffold, check sensitivity to
neighbors/kernel choices, and verify that scientific conclusions do not depend
on one visually convenient picture.

## PaCMAP reports an ndarray initialization error. What should I do?

Older PaCMAP/TopOMetry combinations could fail when a NumPy array was passed as
the initial layout because PaCMAP compared that array to the string `"pca"`.
This fork includes a compatibility workaround in its `Projector` implementation.

If you still see an error such as "The truth value of an array with more than
one element is ambiguous", check that you are importing this package rather than
the upstream `topometry` package, and upgrade the optional layout dependency:

```bash
pip install -U "topometry-nosc[layouts]"
```

Remember that the import package is still named `topo`, so `topometry` and
`topometry-nosc` should not be installed in the same environment.
