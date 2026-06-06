# Evaluation Metrics

## Metrics for Comparing Two Embeddings

::: topo.eval.topo_metrics.spectral_procrustes
::: topo.eval.global_scores.global_score_pca
::: topo.eval.global_scores.global_score_laplacian
::: topo.eval.local_scores.geodesic_correlation
::: topo.eval.local_scores.knn_spearman_r
::: topo.eval.local_scores.knn_kendall_tau

## Metrics for Comparing Two Sparse Graphs/Operators

::: topo.eval.topo_metrics.rowwise_js_similarity
::: topo.eval.topo_metrics.sparse_neighborhood_f1
::: topo.eval.topo_metrics.diffusion_rank_biased_overlap

## Metrics for Evaluating Laplacian Spectra

::: topo.eval.topo_metrics.spectral_similarity
::: topo.eval.topo_metrics.commute_time_trace_gap

## Metric for End-to-End Layout Quality

::: topo.eval.topo_metrics.topo_preserve_score

## Riemannian Distortion Diagnostics

::: topo.eval.rmetric.RiemannMetric
    options:
      filters:
        - "!^__"
        - "!^get_mdimG$"
        - "!^get_detG$"
        - "!^get_dual_rmetric$"
        - "!^get_rmetric$"
::: topo.eval.rmetric.riemann_metric
