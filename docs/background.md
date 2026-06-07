# Background and further reading

A light, readable tour of the ideas behind `topometry-nosc`, with pointers to the
papers and package docs that introduce each one. You do **not** need any of this
to use the package — it is here to explain *why* each piece exists.

The sections follow the pipeline the package runs:

> **neighbor graph → graph operator (Laplace–Beltrami) → spectral scaffold →
> refined graph → 2-D layout → evaluation**

!!! tip "Start here"
    If you read only three things, read the **UMAP documentation** (intuition for
    neighbor graphs), **von Luxburg's spectral-clustering tutorial** (graph
    Laplacians), and **Coifman & Lafon on diffusion maps** (the spectral
    backbone). Everything else specializes from there.

---

## 1. Nearest-neighbor graphs

The first step turns a table of points into a graph that connects each point to
its closest neighbors. *(Package: `topo.base.ann`.)*

- **Approximate nearest neighbors (HNSW).** Malkov & Yashunin's Hierarchical
  Navigable Small World graphs give fast, scalable neighbor search — the default
  backend for large data. [IEEE TPAMI 2020 / arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- **Continuous k-nearest neighbors (CkNN).** Berry & Sauer show that a
  *continuous* kNN graph yields a Laplacian that converges to the
  Laplace–Beltrami operator, capturing topology in a single unweighted graph.
  [*Foundations of Data Science* 2019 / arXiv:1606.02353](https://arxiv.org/abs/1606.02353)

## 2. Kernels and the Laplace–Beltrami operator

Edge weights (a *kernel*) turn the neighbor graph into a discrete approximation
of a continuous operator on the data's underlying shape. *(Package:
`topo.tpgraph.kernels`.)*

- **Diffusion maps.** Coifman & Lafon's foundational paper: a diffusion operator
  built from a Gaussian kernel whose eigenfunctions parameterize the manifold and
  approximate the Laplace–Beltrami operator. [*Appl. Comput. Harmon. Anal.* 2006](https://doi.org/10.1016/j.acha.2006.04.006)
- **Variable-bandwidth kernels.** Berry & Harlim generalize diffusion maps:
  scaling bandwidth by local density improves convergence on unevenly sampled
  data — the basis for the package's adaptive-bandwidth kernels.
  [*Appl. Comput. Harmon. Anal.* 2016 / arXiv:1406.5064](https://arxiv.org/abs/1406.5064)
- **Laplacian eigenmaps.** Belkin & Niyogi: embed data using eigenvectors of the
  graph Laplacian — the simplest member of this family and a useful baseline.
  [*Neural Computation* 2003](https://doi.org/10.1162/089976603321780317)
- **Fuzzy simplicial sets (UMAP's graph).** UMAP builds its graph from *fuzzy*
  neighborhoods combined by a probabilistic union. For `fuzzy` kernels,
  `topometry-nosc` delegates this UMAP-specific graph construction to
  `umap-learn`; TopoMetry then continues with its own spectral scaffolds,
  diffusion operators and graph-refinement pipeline. The docs are the gentlest
  introduction.
  [UMAP docs](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) ·
  [McInnes, Healy & Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426)

## 3. Spectral scaffolds (the eigenbasis)

Eigenvectors of the graph operator form an orthonormal basis — the "spectral
scaffold" — that captures structure across scales. *(Package:
`topo.spectral.eigen`.)*

- **Spectral-clustering tutorial.** Von Luxburg's self-contained tutorial:
  similarity graphs, the different graph Laplacians and their normalizations, and
  why their eigenvectors reveal structure. The best single reference for this
  layer. [*Statistics and Computing* 2007 / arXiv:0711.0189](https://arxiv.org/abs/0711.0189)
- **Multiscale diffusion (msDM).** Diffusion at multiple time scales (the package's
  `msDM` scaffold) follows directly from the diffusion-map eigenstructure in
  Coifman & Lafon (above); the diffusion *time* controls the scale of structure
  emphasized.

## 4. Union of Manifolds (UoM)

When data splits into disconnected or weakly connected pieces, the package builds
per-component scaffolds and assembles them block-diagonally. *(Package:
`topo.uom`.)*

- **Louvain community detection.** Blondel et al.'s greedy modularity method —
  used to partition the graph into coherent communities before per-component
  scaffolding. [*J. Stat. Mech.* 2008 / arXiv:0803.0476](https://arxiv.org/abs/0803.0476)
- **Leiden (modern refinement).** Traag et al. fix Louvain's badly-connected-community
  issue; useful background even though the package's own splitter is Louvain-style.
  [*Scientific Reports* 2019](https://doi.org/10.1038/s41598-019-41695-z)
- **Cuts and conductance.** Shi & Malik's normalized cuts motivate the
  conductance-based merging of fragile components; von Luxburg's tutorial (§3)
  also covers cut measures. [*IEEE TPAMI* 2000](https://doi.org/10.1109/34.868688)

## 5. Intrinsic dimensionality

How many underlying factors really shape the data — used to size the scaffold.
*(Package: `topo.tpgraph.intrinsic_dim`, methods `fsa` and `mle`.)*

- **FSA estimator.** Farahmand, Szepesvári & Audibert's manifold-adaptive
  dimension estimator from nearest-neighbor distance ratios (the `fsa` method).
  [*ICML* 2007](https://doi.org/10.1145/1273496.1273530)
- **MLE estimator.** Levina & Bickel's maximum-likelihood intrinsic-dimension
  estimator (the `mle` method). [*NeurIPS* 2004](https://papers.nips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html)
- **Practical overview.** `scikit-dimension` collects and benchmarks many such
  estimators — handy for context. [Bac et al., *Entropy* 2021](https://doi.org/10.3390/e23101368)

## 6. Layouts and projections

Turning the scaffold and refined graph into a 2-D picture. *(Package:
`topo.layouts`.)* Different methods trade off local detail against the global
arrangement; trying a few is normal.

- **Isomap.** Tenenbaum, de Silva & Langford: geodesic distances + classical MDS;
  the foundational geodesic-preserving embedding. [*Science* 2000](https://doi.org/10.1126/science.290.5500.2319)
- **t-SNE.** Van der Maaten & Hinton: heavy-tailed neighbor embedding, excellent
  for local cluster structure. [*JMLR* 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- **UMAP / MAP.** `projection_method="UMAP"` uses `umap-learn`'s UMAP estimator.
  The package's `MAP` remains a local checkpoint-aware graph-layout optimizer
  for TopoMetry refined graphs. [McInnes et al., arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- **PaCMAP.** Wang, Huang, Rudin & Shaposhnik: balances local and global structure
  by design, with a clear analysis of what makes a good DR loss.
  [*JMLR* 2021](https://www.jmlr.org/papers/v22/20-1061.html)
- **TriMap.** Amid & Warmuth: triplet-constraint embedding that preserves global
  layout and scales to large data. [arXiv:1910.00204](https://arxiv.org/abs/1910.00204)
- **Minimum-Distortion Embedding (MDE).** Agrawal, Ali & Boyd: a general framework
  (generalizing PCA, MDS, spectral embedding, UMAP…) implemented in PyMDE.
  [*Found. Trends ML* 2021 / arXiv:2103.02559](https://arxiv.org/abs/2103.02559) ·
  [PyMDE docs](https://pymde.org)

## 7. Evaluation and geometry diagnostics

How faithful is the embedding? The package emphasizes geometry and operator
preservation rather than trusting any single layout. *(Packages: `topo.eval`,
`topo.eval.topo_metrics`, `topo.eval.rmetric`.)*

- **Geodesic rank correlation.** The package compares geodesic distances in the
  original vs. embedded space with Spearman/Kendall correlation (built on Isomap's
  geodesic idea, §6). [Tenenbaum et al., *Science* 2000](https://doi.org/10.1126/science.290.5500.2319)
- **Operator-native topology metrics.** The package also compares transition
  neighborhoods, transition probabilities, and spectral coordinates between
  graph operators. These scores are meant to compare representations and
  parameter choices in the same graph/diffusion language as the pipeline.
- **Trustworthiness & continuity.** These are standard DR diagnostics in the
  broader literature: do embedded neighbors reflect real neighbors, and vice
  versa? They are useful context, but they are not the main built-in evaluation
  API in this fork.
  [Venna & Kaski, *Neural Networks* 2006](https://doi.org/10.1016/j.neunet.2006.05.014)
- **Co-ranking framework.** Lee & Verleysen unify these rank-based metrics in one
  co-ranking matrix — the cleanest way to think about DR quality.
  [*Neurocomputing* 2009](https://doi.org/10.1016/j.neucom.2008.12.017)
- **Riemannian metric estimation.** Perrault-Joncas & Meila estimate the distortion
  an embedding introduces from its Laplacian — the basis for the package's
  `rmetric` diagnostics (after the megaman library). [arXiv:1305.7255](https://arxiv.org/abs/1305.7255) ·
  [megaman, *JMLR* 2016](https://www.jmlr.org/papers/v17/16-109.html)
- **Target-aware evaluation.** Predicting or smoothing a per-sample target
  answers a different question from geometry preservation. See the
  [Practical FAQ](faq.md#how-should-i-evaluate-an-embedding-against-a-target-variable)
  for recommended checks.

## 8. Surveys and starting points

- **Comparative review of nonlinear DR.** Van der Maaten, Postma & van den Herik
  compare many techniques on common datasets — a map of the whole field.
  [Report, 2009 (PDF)](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf)
- **UMAP documentation.** Still the friendliest on-ramp to neighbor graphs and
  fuzzy topology. [umap-learn docs](https://umap-learn.readthedocs.io/en/latest/)
