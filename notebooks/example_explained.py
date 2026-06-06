# ---  # noqa: D100
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# # TopOMetry explained example
#
# This notebook is the documented version of the basic and extended examples.
# It runs the same helper-driven workflow, but explains each reported object,
# plot, and metric next to the cell that produces it.
#
# The workflow is:
#
# 1. configure data, graph, kernel, spectral scaffold, and final layout;
# 2. load or generate data;
# 3. run the TopOMetry pipeline;
# 4. inspect intermediate objects;
# 5. visualize the pipeline;
# 6. compute and interpret geometry-preservation metrics;
# 7. optionally compare method variants.
#
# The implementation details live in `_example_utils.py`. This notebook is
# intentionally descriptive: it should help you understand the output, not hide
# the method choices.

# %%
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _example_utils import (
    DemoConfig,
    compute_metrics,
    load_demo_data,
    plot_deep_dive,
    plot_metric_overview,
    plot_pipeline_overview,
    print_metric_summary,
    print_options,
    run_pipeline,
)

# %%
# ## 1. Configure the run
#
# This is the main tuning cell. Edit it first, then rerun the cells below.
#
# **Data parameters**
#
# - `use_custom_data=False` generates a Swiss roll. Set it to `True` to load a
#   `.npy` feature matrix from `data_path`.
# - `color_path` is optional. It should point to a one-dimensional `.npy` array
#   with one label or continuous value per sample. If omitted, points are colored
#   by sample index or Swiss-roll position.
# - `scale_data=True` applies `StandardScaler`. This is useful when feature
#   columns have different physical units or very different variances.
#
# **Graph and kernel parameters**
#
# - `n_neighbors` controls locality. Smaller values emphasize local structure;
#   larger values make the graph more connected and more global.
# - `metric` is the distance used for nearest-neighbor search.
# - `backend` chooses the neighbor-search implementation.
# - `kernel_version` controls how distances are converted into affinities.
# - `sigma` is used by the fixed-bandwidth Gaussian kernel.
# - `anisotropy` controls diffusion normalization. Values near 1 reduce the
#   influence of sampling density; values near 0 preserve more density signal.
#
# **Spectral scaffold parameters**
#
# - `n_components_dm` is the number of spectral coordinates computed before the
#   final 2-D layout.
# - `dm_method` chooses diffusion maps (`DM`), multiscale diffusion maps
#   (`msDM`), or Laplacian Eigenmaps (`LE`).
# - `diffusion_time=0` gives the multiscale diffusion representation when using
#   diffusion maps. Positive values give a fixed diffusion time.
#
# **Layout parameters**
#
# - `projection_method` chooses the final 2-D optimizer or projection method.
# - `num_iters` affects iterative methods such as MAP, UMAP, and PaCMAP.

config = DemoConfig(
    # -- Data source ---------------------------------------------------------
    # False generates the built-in Swiss roll. True loads data_path instead.
    # Custom data should be a 2-D .npy array with shape (n_samples, n_features).
    use_custom_data=False,
    data_path=Path("../data/my_data.npy"),
    # Optional .npy vector with shape (n_samples,). Used only for plot colors.
    # This can be a class label, time point, pseudotime value, batch id, etc.
    color_path=None,
    # Recommended when feature columns have very different units or scales.
    # Leave False if distances in the original units are scientifically useful.
    scale_data=False,
    # Swiss-roll settings, used only when use_custom_data=False.
    # Increase n_samples for smoother plots; decrease it for faster iteration.
    n_samples=2000,
    # Swiss-roll noise controls how hard the manifold-learning problem is.
    noise=0.5,
    # Reused by data generation, landmark sampling, and stochastic layouts.
    random_state=42,
    # -- kNN graph -----------------------------------------------------------
    # Main locality knob. Smaller values emphasize fine local neighborhoods;
    # larger values make the graph more connected and more globally stable.
    n_neighbors=15,
    # Distance used by neighbor search. Must be supported by the chosen backend.
    metric="euclidean",
    # "hnswlib" is fast and approximate; "sklearn" is exact and dependency-light;
    # "nmslib" can be useful on large data if installed.
    backend="hnswlib",
    # -- Kernel --------------------------------------------------------------
    # Converts neighbor distances into affinities. Try "cknn" and "fuzzy" first.
    # Other options: "bw_adaptive", "bw_adaptive_alpha_decaying",
    # "bw_adaptive_nbr_expansion", "bw_adaptive_alpha_decaying_nbr_expansion",
    # and "gaussian".
    kernel_version="cknn",
    # Bandwidth for kernel_version="gaussian" only. Smaller values sharpen
    # neighborhoods; larger values smooth affinities over longer distances.
    sigma=1.0,
    # Diffusion alpha in [0, 1]. Near 1 reduces sampling-density effects; near 0
    # preserves density differences more strongly.
    anisotropy=1.0,
    # -- Spectral scaffold ---------------------------------------------------
    # Number of eigenvectors/scaffold coordinates to compute before the final
    # 2-D layout. More components retain more structure but cost more.
    n_components_dm=64,
    # "DM" = diffusion maps, "msDM" = multiscale diffusion maps,
    # "LE" = Laplacian Eigenmaps.
    dm_method="msDM",
    # "arpack" is a robust default. "lobpcg" may help for some large sparse
    # problems. "amg" requires the optional pyamg dependency.
    eigensolver="arpack",
    # For diffusion maps: 0 means multiscale; positive values use fixed-time
    # diffusion and increasingly emphasize coarser structure as t grows.
    diffusion_time=0,
    # -- Final 2-D layout ----------------------------------------------------
    # Final projection/optimizer. Common choices include "MAP", "PaCMAP",
    # "UMAP", "Isomap", "t-SNE", "TriMAP", "IsomorphicMDE", "IsometricMDE".
    projection_method="PaCMAP",
    # Keep this at 2 for the plotting cells in this notebook.
    n_components_2d=2,
    # Optimization iterations for iterative layout methods. More iterations can
    # improve convergence but increase runtime.
    num_iters=500,
    # -- Output --------------------------------------------------------------
    # If True, helper plotting functions save PNGs into output_dir.
    save_figures=False,
    output_dir=Path("../figures"),
)

# %%
# ## 2. Available options
#
# The printed reference below lists the common option values accepted by the
# helper pipeline.
#
# **Kernel versions**
#
# - `bw_adaptive`: adaptive-bandwidth Gaussian affinities.
# - `bw_adaptive_alpha_decaying`: adaptive bandwidth with alpha-decaying
#   exponent behavior.
# - `bw_adaptive_nbr_expansion`: adaptive bandwidth with expanded neighbor
#   search.
# - `bw_adaptive_alpha_decaying_nbr_expansion`: combines both extensions.
# - `fuzzy`: UMAP-style fuzzy simplicial set affinities.
# - `cknn`: continuous k-nearest-neighbor affinities.
# - `gaussian`: fixed-bandwidth Gaussian affinities; tune `sigma`.
#
# **Spectral scaffold methods**
#
# - `DM`: diffusion maps at a chosen diffusion time.
# - `msDM`: multiscale diffusion maps, often a strong default.
# - `LE`: Laplacian Eigenmaps.
#
# **Projection methods**
#
# `MAP`, `PaCMAP`, `Isomap`, `UMAP`, `t-SNE`, `TriMAP`, `IsomorphicMDE`, and
# `IsometricMDE` are common choices. Some require optional layout dependencies.

print_options()

# %%
# ## 3. Load or generate data
#
# The output reports the shape of the feature matrix.
#
# - `data.X` has shape `(n_samples, n_features)`.
# - `data.color` has one value per sample and is used only for coloring plots.
#
# To use your own data:
#
# ```python
# config = replace(
#     config,
#     use_custom_data=True,
#     data_path=Path("../data/my_data.npy"),
#     color_path=Path("../data/my_labels.npy"),  # optional
# )
# ```

data = load_demo_data(config)

# %%
# ## 4. Run the pipeline
#
# `run_pipeline` performs the six main computational steps:
#
# 1. build a k-nearest-neighbor graph in input space;
# 2. fit a kernel in input space;
# 3. compute a spectral eigenbasis;
# 4. represent the data in spectral scaffold coordinates `Z`;
# 5. fit a second graph/kernel in scaffold space;
# 6. compute a spectral initialization and optimize the final 2-D layout `Y`.
#
# The printed lines summarize the kernel matrix, scaffold, and final projection
# shapes. Sparse matrix `nnz` values tell you how many nonzero graph or kernel
# entries were retained.

result = run_pipeline(data, config)

# %%
# ## 5. Inspect intermediate objects
#
# These objects are useful when a parameter change gives an unexpected result.
#
# - `knn_X`: sparse nearest-neighbor distance graph in input space.
# - `K_X`: sparse input-space kernel or affinity matrix.
# - `P_X`: row-normalized diffusion operator from `K_X`.
# - `L_X`: graph Laplacian from the input-space kernel.
# - `evals`: eigenvalues from the spectral decomposition.
# - `Z`: high-dimensional spectral scaffold used by the final layout method.
# - `knn_Z`, `K_Z`, `L_Z`: graph, kernel, and Laplacian in scaffold space.
# - `init_Y`: 2-D spectral initialization.
# - `Y`: final 2-D embedding.
#
# What to watch:
#
# - Very low `nnz` can mean the graph is too sparse.
# - Very flat leading eigenvalues can indicate weak spectral separation.
# - If `Z` has fewer useful dimensions than expected, increase data quality,
#   adjust `n_neighbors`, or try a different kernel.

print("Input")
print(f"  X                  {data.X.shape}")
print(f"  color              {data.color.shape}")

print("\nGraph and kernels")
print(f"  knn_X              {result.knn_X.shape}, nnz={result.knn_X.nnz}")
print(f"  K_X                {result.K_X.shape}, nnz={result.K_X.nnz}")
print(f"  P_X                {result.P_X.shape}, nnz={result.P_X.nnz}")
print(f"  L_X                {result.L_X.shape}, nnz={result.L_X.nnz}")
print(f"  knn_Z              {result.knn_Z.shape}, nnz={result.knn_Z.nnz}")
print(f"  K_Z                {result.K_Z.shape}, nnz={result.K_Z.nnz}")

print("\nEmbeddings")
print(f"  eigenvalues        {result.evals.shape}")
print(f"  Z scaffold         {result.Z.shape}")
print(f"  spectral init_Y    {result.init_Y.shape}")
print(f"  final Y            {result.Y.shape}")

print("\nLeading eigenvalues")
print(np.array2string(result.evals[:10], precision=4))

# %%
# ### Quick visual inspection
#
# This compact figure is a first sanity check before looking at the full
# overview.
#
# - **Eigenvalue spectrum**: shows the strength of the first spectral
#   components. Large gaps suggest natural low-dimensional structure.
# - **Spectral scaffold**: plots the first two coordinates of `Z`, colored by
#   `data.color`. This is not the final embedding; it is the geometry-aware
#   coordinate system used before optimization.
# - **Final layout**: plots `Y`, the 2-D output that most users inspect or use
#   downstream.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

n_show = min(30, result.evals.shape[0])
axes[0].plot(np.arange(1, n_show + 1), result.evals[:n_show], marker="o", lw=1)
axes[0].set_title("Eigenvalue spectrum")
axes[0].set_xlabel("rank")
axes[0].set_ylabel("eigenvalue")

axes[1].scatter(
    result.Z[:, 0],
    result.Z[:, 1],
    c=data.color,
    cmap="Spectral",
    s=4,
    linewidths=0,
)
axes[1].set_title("Spectral scaffold")
axes[1].set_xlabel("Z1")
axes[1].set_ylabel("Z2")

axes[2].scatter(
    result.Y[:, 0],
    result.Y[:, 1],
    c=data.color,
    cmap="Spectral",
    s=4,
    linewidths=0,
)
axes[2].set_title(f"{config.projection_method} layout")
axes[2].set_xlabel("Y1")
axes[2].set_ylabel("Y2")

plt.tight_layout()
plt.show()

# %%
# ## 6. Full pipeline overview
#
# The 3x3 overview explains how the data move through the pipeline.
#
# Row 1: input space
#
# - **(a) Input data**: raw input coordinates. For the default Swiss roll this
#   is a 3-D plot. For custom data it shows the first two dimensions.
# - **(b) Input kNN distances**: distribution of neighbor edge distances. A
#   long tail can indicate uneven sampling density, outliers, or too many
#   neighbors.
# - **(c) Distance to affinity**: shows how graph distances become kernel
#   weights. Nearby points should generally receive stronger affinity.
#
# Row 2: spectral scaffold
#
# - **(d) Spectrum**: leading eigenvalues. A visible gap can suggest an
#   intrinsic scale or dimensionality.
# - **(e) First scaffold dimensions**: first two spectral coordinates.
# - **(f) Higher harmonics**: later spectral coordinates, useful for seeing
#   structure that is not captured by the first pair.
#
# Row 3: layout
#
# - **(g) Spectral initialization**: the 2-D starting point for optimization.
# - **(h) Final layout**: the optimized 2-D result.
# - **(i) Layout density**: point positions colored by scaffold-kernel weighted
#   degree. High values mark locally dense or strongly connected regions.

plot_pipeline_overview(data, result, config)

# %%
# ## 7. Compute quality metrics
#
# These metrics compare the final 2-D layout `Y` against the input geometry.
# They are complementary: no single score completely describes embedding
# quality.
#
# **Global scores**
#
# - `global_score_pca(X, Y)` returns a value in `[0, 1]`. Higher means `Y`
#   preserves global variance structure better relative to a PCA baseline.
# - `global_score_laplacian(X, Y)` also returns a value in `[0, 1]`. Higher
#   means `Y` preserves graph-aware global structure better relative to a
#   Laplacian Eigenmaps baseline.
#
# **Local and geodesic score**
#
# - `geodesic_correlation(X, Y)` computes Spearman rank correlation between
#   graph geodesic distances in input space and Euclidean distances in `Y`.
#   Values near 1 are better; values below about 0.5 often indicate substantial
#   distortion.
#
# **Diffusion-operator topology**
#
# These metrics compare transition operators built from the input and embedding:
#
# - `TopoPreserve composite`: combined topology-preservation score in `[0, 1]`.
# - `PF1`: neighborhood F1, measuring overlap of local neighbor sets.
# - `PJS`: Jensen-Shannon similarity, measuring whether transition weights are
#   preserved, not just neighbor identities.
# - `SP`: spectral Procrustes alignment, measuring global alignment of
#   diffusion coordinates.
# - `Rank diffusion corr.`: rank correlation between diffusion similarities.
# - `Spectral similarity`: eigen-spectrum similarity between operators.
#
# **Riemannian deformation**
#
# The deformation values come from a local Riemannian metric estimated on the
# embedding with the scaffold graph Laplacian.
#
# - Negative values indicate local contraction.
# - Positive values indicate local expansion.
# - A narrow distribution around 0 indicates more uniform local area
#   preservation.
# - Large absolute values identify regions to inspect in the deep-dive plots.

metrics = compute_metrics(data, result, config)
print_metric_summary(metrics)

# %%
# ## 8. Metric overview figure
#
# This 2x2 diagnostic figure is a compact dashboard for the metric output.
#
# - **(a) Local metric ellipses**: each ellipse represents local stretching
#   estimated by the Riemannian metric. Nearly circular ellipses mean isotropic
#   local scaling; elongated ellipses mean direction-dependent distortion.
# - **(b) Local area change**: red/positive regions are local expansion;
#   blue/negative regions are local contraction.
# - **(c) Deformation distribution**: shows whether distortion is concentrated
#   in a few regions or spread across the layout. Narrow and centered near zero
#   is preferable.
# - **(d) Metrics at a glance**: bar chart of the printed scores. Higher is
#   better for all displayed scores except deformation is summarized separately
#   in panel (c).

plot_metric_overview(data, result, metrics, config)

# %%
# ## 9. Optional deep diagnostics
#
# This figure is heavier than the overview because it computes per-point and
# pairwise distortion summaries. Run it when you need to understand where and
# how an embedding is distorted.
#
# Row 1: Euclidean and rank distortion
#
# - **(a) Per-point rank distortion**: for landmark distances, compares the
#   rank order of each point's distances in input space and layout space.
#   Lower values are better. Hot regions are points whose relative position
#   changed substantially.
# - **(b) Shepard diagram**: compares input pairwise distances with layout
#   pairwise distances after rescaling. Points close to the diagonal preserve
#   pairwise distance well.
# - **(c) Pairwise distance residuals**: histogram of rescaled
#   `layout distance - input distance`. Positive values indicate expansion;
#   negative values indicate contraction.
#
# Row 2: diffusion-operator preservation
#
# - **(d) Per-point JS similarity**: how well local transition weights are
#   preserved for each point. Higher is better.
# - **(e) Per-point neighborhood F1**: how well neighbor sets are preserved for
#   each point. Higher is better.
# - **(f) JS similarity vs F1**: separates weight preservation from set
#   preservation. Points above the diagonal preserve weights better than hard
#   neighbor identity; points below preserve neighbor identity better than
#   transition weights.
#
# Row 3: deformation structure
#
# - **(g) Absolute deformation vs kernel density**: checks whether low-density
#   or weakly connected regions are distorting more.
# - **(h) Absolute deformation vs rank distortion**: checks whether metric
#   deformation and rank-order distortion occur in the same points.
# - **(i) Cumulative deformation magnitude**: shows whether a small fraction of
#   points accounts for most total distortion.

plot_deep_dive(data, result, metrics, config)

# %%
# ## 10. Tune one variant manually
#
# This section keeps method exploration explicit. Change a few values in
# `variant_config`, set `RUN_VARIANT = True`, and rerun the cells in this
# section.
#
# Useful experiments:
#
# - increase or decrease `n_neighbors` to change the local/global tradeoff;
# - try `kernel_version="fuzzy"` for UMAP-style affinities;
# - try `kernel_version="gaussian"` and tune `sigma`;
# - compare `dm_method="DM"`, `"msDM"`, and `"LE"`;
# - compare final layouts such as `"MAP"`, `"PaCMAP"`, `"UMAP"`, and `"Isomap"`.

RUN_VARIANT = False

variant_config = replace(
    config,
    # Common graph/kernel experiments
    n_neighbors=25,
    kernel_version="fuzzy",
    # Try "DM", "msDM", or "LE"
    dm_method="msDM",
    n_components_dm=64,
    diffusion_time=0,
    # Try "MAP", "PaCMAP", "UMAP", "Isomap", "t-SNE", "TriMAP", ...
    projection_method="PaCMAP",
    num_iters=500,
)

variant_result = None
variant_metrics = None

if RUN_VARIANT:
    variant_result = run_pipeline(data, variant_config)
    variant_metrics = compute_metrics(data, variant_result, variant_config)
    print_metric_summary(variant_metrics)
else:
    print("Set RUN_VARIANT = True to run this comparison.")

# %%
# ### Variant visual comparison
#
# This plot compares only the final layouts. Use the metric table in the next
# cell to check whether an apparently cleaner layout also preserves geometry
# better.

if variant_result is None:
    print("Run the previous cell with RUN_VARIANT = True first.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(
        result.Y[:, 0],
        result.Y[:, 1],
        c=data.color,
        cmap="Spectral",
        s=4,
        linewidths=0,
    )
    axes[0].set_title(
        f"Baseline: {config.kernel_version} + "
        f"{config.dm_method} + {config.projection_method}"
    )
    axes[0].set_xlabel("Y1")
    axes[0].set_ylabel("Y2")

    axes[1].scatter(
        variant_result.Y[:, 0],
        variant_result.Y[:, 1],
        c=data.color,
        cmap="Spectral",
        s=4,
        linewidths=0,
    )
    axes[1].set_title(
        f"Variant: {variant_config.kernel_version} + "
        f"{variant_config.dm_method} + {variant_config.projection_method}"
    )
    axes[1].set_xlabel("Y1")
    axes[1].set_ylabel("Y2")

    plt.tight_layout()
    plt.show()

# %%
# ### Variant metric comparison
#
# The `delta` column is `variant - baseline`. Positive values are better for
# all scores shown here. A useful variant usually improves several metrics
# without causing an obvious visual artifact or a large deformation increase.

if variant_metrics is None:
    print("Run the variant cell with RUN_VARIANT = True first.")
else:
    comparison_rows = [
        ("PCA", metrics.gs_pca, variant_metrics.gs_pca),
        ("Laplacian", metrics.gs_lap, variant_metrics.gs_lap),
        ("Geodesic rho", metrics.geo_r, variant_metrics.geo_r),
        ("TopoPreserve", metrics.topo_score, variant_metrics.topo_score),
        ("PF1", metrics.neighborhood_f1, variant_metrics.neighborhood_f1),
        ("PJS", metrics.js_similarity, variant_metrics.js_similarity),
        ("SP", metrics.spectral_procrustes_r2, variant_metrics.spectral_procrustes_r2),
        ("RDC", metrics.rdc, variant_metrics.rdc),
        (
            "Spectral sim.",
            metrics.spectral_similarity,
            variant_metrics.spectral_similarity,
        ),
    ]

    name_width = max(len(name) for name, _, _ in comparison_rows)
    print(f"{'metric':<{name_width}}  baseline  variant   delta")
    for name, baseline, variant in comparison_rows:
        print(
            f"{name:<{name_width}}  "
            f"{baseline:8.4f}  {variant:7.4f}  {variant - baseline:+7.4f}"
        )

# %%
# ## 11. Optional parameter sweep
#
# Set `RUN_PARAMETER_SWEEP = True` after editing `sweep_configs`.
#
# Keep this list small for interactive work. Each row runs the complete
# pipeline and the metric suite. The printed table is intentionally narrow:
# it is meant to identify promising variants, not replace the full plots.

RUN_PARAMETER_SWEEP = False

sweep_configs = [
    (
        "baseline",
        config,
    ),
    (
        "fuzzy kernel",
        replace(config, kernel_version="fuzzy"),
    ),
    (
        "more neighbors",
        replace(config, n_neighbors=30),
    ),
    (
        "MAP layout",
        replace(config, projection_method="MAP", num_iters=800),
    ),
    (
        "diffusion maps",
        replace(config, dm_method="DM", diffusion_time=1),
    ),
]

sweep_rows = []

if RUN_PARAMETER_SWEEP:
    for label, sweep_config in sweep_configs:
        print(f"\nRunning {label}: {sweep_config}")
        sweep_result = run_pipeline(data, sweep_config)
        sweep_metrics = compute_metrics(data, sweep_result, sweep_config)
        sweep_rows.append(
            {
                "label": label,
                "kernel": sweep_config.kernel_version,
                "neighbors": sweep_config.n_neighbors,
                "scaffold": sweep_config.dm_method,
                "layout": sweep_config.projection_method,
                "global_lap": sweep_metrics.gs_lap,
                "geo_r": sweep_metrics.geo_r,
                "topo": sweep_metrics.topo_score,
                "pf1": sweep_metrics.neighborhood_f1,
                "pjs": sweep_metrics.js_similarity,
                "rdc": sweep_metrics.rdc,
                "deform_std": float(np.std(sweep_metrics.deform)),
            }
        )

if sweep_rows:
    header = (
        "label                 kernel     k   scaffold  layout     "
        "lap     geo     topo    deform_std"
    )
    print(header)
    print("-" * len(header))
    for row in sweep_rows:
        print(
            f"{row['label']:<21} "
            f"{row['kernel']:<10} "
            f"{row['neighbors']:>3} "
            f"{row['scaffold']:<9} "
            f"{row['layout']:<9} "
            f"{row['global_lap']:>6.3f} "
            f"{row['geo_r']:>7.3f} "
            f"{row['topo']:>7.3f} "
            f"{row['deform_std']:>10.3f}"
        )

# %%
# ## 12. Final interpretation checklist
#
# Use this checklist when deciding whether a layout is trustworthy.
#
# 1. The input kNN distance histogram should not be dominated by a long outlier
#    tail unless the data really contain isolated points.
# 2. The distance-to-affinity plot should show stronger weights for shorter
#    distances, with behavior that matches the selected kernel.
# 3. The spectral scaffold should reveal structure before the final optimizer
#    does any work.
# 4. The final layout should preserve visible structure without producing
#    unexplained tears, isolated islands, or dense knots.
# 5. Global, geodesic, and diffusion-topology scores should generally agree.
#    If they disagree, inspect the deep-dive plots to see what kind of
#    structure is being preserved or lost.
# 6. Riemannian deformation should be centered near zero and not concentrated in
#    a tiny subset of points unless the data contain true irregular regions.
