# API Reference

The high-level entry point is [`TopOGraph`](#topograph). The remaining sections
expose the main public building blocks it orchestrates. Lower-level helpers are
included only when they are useful for building custom workflows directly.

## High-level workflow

### TopOGraph

The fitted-result properties on `TopOGraph` are read-only views into the
pipeline state created by `fit()`: input-space graphs/operators (`X`),
fixed-time diffusion-map scaffold results (`Z`), multiscale scaffold results
(`msZ`), and stored 2-D layouts. They are useful for inspection, plotting,
diagnostics, and custom workflows; they are not constructor parameters.

::: topo.topograph.TopOGraph
    options:
      inherited_members: false
      members:
        - fit
        - spectral_scaffold
        - eigenvalues
        - global_id
        - intrinsic_dim
        - knn_X
        - P_of_X
        - knn_Z
        - P_of_Z
        - knn_msZ
        - P_of_msZ
        - TopoMAP
        - msTopoMAP
        - TopoPaCMAP
        - msTopoPaCMAP
        - project
        - eigenspectrum
        - spectral_selectivity
        - filter_signal
        - impute
        - riemann_diagnostics
        - save

### Persistence

::: topo.topograph.save_topograph

::: topo.topograph.load_topograph

### Logging

::: topo._logging.configure

## Neighbor search and distances

### kNN graph construction

::: topo.base.ann.kNN

::: topo.base.ann.grid_search

### ANN transformers

::: topo.base.ann.NMSlibTransformer

::: topo.base.ann.HNSWlibTransformer

## Graph kernels and intrinsic dimension

### Kernel

::: topo.tpgraph.kernels.Kernel
    options:
      inherited_members: false
      members:
        - fit
        - transform
        - fit_transform
        - knn
        - K
        - A
        - degree
        - weighted_degree
        - laplacian
        - L
        - diff_op
        - P
        - shortest_paths
        - SP
        - get_indices_distances
        - impute
        - filter
        - is_connected

### Kernel construction

::: topo.tpgraph.kernels.compute_kernel

::: topo.tpgraph.cknn.cknn_graph

::: topo.tpgraph.fuzzy.fuzzy_simplicial_set

### Intrinsic dimension

::: topo.tpgraph.intrinsic_dim.IntrinsicDim
    options:
      inherited_members: false

::: topo.tpgraph.intrinsic_dim.automated_scaffold_sizing

::: topo.tpgraph.intrinsic_dim.fsa_local

::: topo.tpgraph.intrinsic_dim.fsa_global

::: topo.tpgraph.intrinsic_dim.mle_local

::: topo.tpgraph.intrinsic_dim.mle_global

## Spectral operators and scaffolds

### EigenDecomposition

::: topo.spectral.eigen.EigenDecomposition
    options:
      inherited_members: false
      members:
        - fit
        - transform
        - fit_transform
        - results
        - rescale
        - spectral_layout
        - plot_eigenspectrum

### Eigendecomposition helpers

::: topo.spectral.eigen.eigendecompose

::: topo.spectral.eigen.spectral_layout

### Graph operators

::: topo.spectral._spectral.graph_laplacian

::: topo.spectral._spectral.diffusion_operator

::: topo.spectral._spectral.LE

::: topo.spectral._spectral.degree

## Layouts and projections

### Projector

::: topo.layouts.projector.Projector
    options:
      inherited_members: false
      members:
        - fit
        - transform
        - fit_transform

### Layout functions

::: topo.layouts.map.fuzzy_embedding

::: topo.layouts.isomap.Isomap

## Evaluation metrics

### Global and local scores

::: topo.eval.global_scores.global_score_pca

::: topo.eval.global_scores.global_score_laplacian

::: topo.eval.local_scores.geodesic_distance

::: topo.eval.local_scores.knn_spearman_r

::: topo.eval.local_scores.knn_kendall_tau

::: topo.eval.local_scores.geodesic_correlation

### Operator-native topology metrics

::: topo.eval.topo_metrics.get_P

::: topo.eval.topo_metrics.diffusion_coordinates

::: topo.eval.topo_metrics.diffusion_distance_from_eigs

::: topo.eval.topo_metrics.rank_diffusion_correlation

::: topo.eval.topo_metrics.multiscale_diffusion_emd

::: topo.eval.topo_metrics.spectral_procrustes

::: topo.eval.topo_metrics.diffusion_rank_biased_overlap

::: topo.eval.topo_metrics.rowwise_js_similarity

::: topo.eval.topo_metrics.sparse_neighborhood_f1

::: topo.eval.topo_metrics.spectral_similarity

::: topo.eval.topo_metrics.commute_time_trace_gap

::: topo.eval.topo_metrics.topo_preserve_score

### Riemannian distortion diagnostics

::: topo.eval.rmetric.riemann_metric

::: topo.eval.rmetric.RiemannMetric

::: topo.eval.rmetric.calculate_deformation

::: topo.eval.rmetric.get_eccentricity

## Analysis utilities

::: topo.analysis.spectral_selectivity

::: topo.analysis.filter_signal

::: topo.analysis.impute

::: topo.analysis.riemann_diagnostics

## Plotting

### Embedding plots

::: topo.plot.scatter

::: topo.plot.scatter3d

### Diagnostics and scores

::: topo.plot.decay_plot

::: topo.plot.plot_riemann_metric

::: topo.plot.plot_eigenvectors

::: topo.plot.plot_dimensionality_histograms

::: topo.plot.visualize_optimization

## Low-level utilities

These functions are useful when building custom workflows around the core
objects. They are lower-level than `TopOGraph`, `Kernel`, `EigenDecomposition`,
and `Projector`.

::: topo.utils.get_landmark_indices

::: topo.utils.get_sparse_matrix_from_indices_distances

::: topo.utils.get_indices_distances_from_sparse_matrix
