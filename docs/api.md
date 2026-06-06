# API Reference

This reference is organized by user workflows: from the high-level orchestrator down to the individual building blocks and evaluation metrics.

## Pipeline Orchestrator

The easiest way to use TopoMetry is through the `TopOGraph` orchestrator, which runs the full geometry-learning and layout pipeline and stores the results as accessible properties.

::: topo.topograph.TopOGraph
    options:
      show_root_heading: true
      inherited_members: false
      members:
        - fit
        - fit_transform
        - transform
        - spectral_scaffold
        - TopoMAP
        - msTopoMAP
        - TopoPaCMAP
        - msTopoPaCMAP
        - eigenvalues
        - global_id
        - intrinsic_dim
        - save
      filters:
        - "!^__"

### Persistence

::: topo.topograph.save_topograph
::: topo.topograph.load_topograph

## Core Estimators (Custom Pipelines)

These scikit-learn compatible transformers can be used individually to build custom manifold learning pipelines.

::: topo.base.ann.kNN
::: topo.tpgraph.kernels.Kernel
    options:
      inherited_members: false
      filters:
        - "!^__"
::: topo.spectral.eigen.EigenDecomposition
    options:
      inherited_members: false
      filters:
        - "!^__"
::: topo.layouts.projector.Projector
    options:
      inherited_members: false
      filters:
        - "!^__"
::: topo.tpgraph.intrinsic_dim.IntrinsicDim
    options:
      inherited_members: false
      filters:
        - "!^__"

## Evaluation & Diagnostics

### Topology Preservation Metrics
Operator-native metrics that quantify how well geometry is preserved between different representations.

::: topo.eval.topo_metrics.topo_preserve_score
::: topo.eval.topo_metrics.multiscale_diffusion_emd
::: topo.eval.topo_metrics.spectral_procrustes
::: topo.eval.topo_metrics.diffusion_rank_biased_overlap
::: topo.eval.topo_metrics.rowwise_js_similarity
::: topo.eval.topo_metrics.sparse_neighborhood_f1
::: topo.eval.topo_metrics.spectral_similarity
::: topo.eval.topo_metrics.commute_time_trace_gap
::: topo.eval.topo_metrics.rank_diffusion_correlation

### Riemannian Distortion Diagnostics
Metrics that compute local expansion, contraction, and anisotropy fields over the manifold.

::: topo.eval.rmetric.RiemannMetric
::: topo.eval.rmetric.riemann_metric

## Global and Local Euclidean Scores
Classic manifold-learning scores for benchmarking.

::: topo.eval.global_scores.global_score_pca
::: topo.eval.global_scores.global_score_laplacian
::: topo.eval.local_scores.geodesic_correlation
::: topo.eval.local_scores.knn_spearman_r
::: topo.eval.local_scores.knn_kendall_tau

## Plotting

::: topo.plot.scatter
::: topo.plot.scatter3d
::: topo.plot.decay_plot
::: topo.plot.plot_riemann_metric
::: topo.plot.plot_eigenvectors
::: topo.plot.plot_dimensionality_histograms
::: topo.plot.visualize_optimization

## Standalone Analysis Functions
Advanced functional tools for signal processing on learned graphs.
*(Note: If you are using the `TopOGraph` orchestrator, these are already available directly as convenient object methods, e.g., `tg.filter_signal()`.)*

::: topo.analysis.spectral_selectivity
::: topo.analysis.filter_signal
::: topo.analysis.impute
::: topo.analysis.riemann_diagnostics
