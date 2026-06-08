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
# # TopOMetry extended exploration
#
# **Purpose:** hands-on parameter exploration and method comparison.
#
# This notebook keeps the same high-level architecture as `example.py`, but adds
# tuning cells for comparing graph, kernel, spectral, and layout choices.
#
# Most mechanics live in `_example_utils.py`; this notebook stays focused on:
#
# 1. choosing parameters;
# 2. running the pipeline;
# 3. inspecting intermediate objects;
# 4. comparing method variants;
# 5. evaluating geometry preservation.

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
# ## 1. Configure the baseline
#
# Edit this cell first. The rest of the notebook reads from `config`, so every
# method choice is visible in one place.

config = DemoConfig(
    # Data
    use_custom_data=False,  # True loads data_path instead of the Swiss roll
    data_path=Path("../data/my_data.npy"),  # (n_samples, n_features) float array
    color_path=None,  # optional: (n_samples,) color values
    scale_data=False,  # recommended when features have very different units
    n_samples=2000,
    noise=0.5,
    random_state=42,
    # Graph/kernel
    n_neighbors=15,
    metric="euclidean",  # any metric accepted by the chosen backend
    backend="hnswlib",  # "hnswlib"  | "sklearn"
    kernel_version="cknn",  # "bw_adaptive" | "fuzzy" | "cknn" | "gaussian"
    sigma=1.0,  # bandwidth for "gaussian" only
    anisotropy=1.0,  # alpha for the diffusion operator (0-1)
    # Spectral scaffold
    n_components_dm=64,  # eigenvectors to compute
    dm_method="msDM",  # "DM" | "msDM" | "LE"
    eigensolver="arpack",  # "arpack" | "lobpcg" | "amg"
    diffusion_time=0,  # t=0 means multiscale; t>0 means fixed-time DM
    # Final 2-D layout
    projection_method="PaCMAP",  # "MAP" | "PaCMAP" | "UMAP" | "Isomap" | ...
    n_components_2d=2,
    num_iters=500,  # optimization iterations for iterative layouts
    # Output
    save_figures=False,
    output_dir=Path("../figures"),
)

# %%
# ## 2. Option reference
#
# Use these names in the configuration cells below.

print_options()

# %%
# ## 3. Load or generate data
#
# To use your own data, set:
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
# ## 4. Run the baseline pipeline
#
# `run_pipeline` returns every intermediate object so you can inspect the graph,
# kernel, eigenbasis, scaffold, initialization, and final layout.

result = run_pipeline(data, config)

# %%
# ## 5. Inspect intermediate objects
#
# These are the main objects to look at when method changes behave unexpectedly.

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

plot_pipeline_overview(data, result, config)

# %%
# ## 7. Compute baseline metrics

metrics = compute_metrics(data, result, config)
print_metric_summary(metrics)

# %%
# ## 8. Metric overview

plot_metric_overview(data, result, metrics, config)

# %%
# ## 9. Tune one variant manually
#
# Change a few values here, then run this section to compare against the
# baseline. This is the most useful cell for hands-on method tuning.

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
# ## 10. Optional parameter sweep
#
# Set `RUN_PARAMETER_SWEEP = True` after editing `sweep_configs`. Keep this
# list small for interactive work because each entry runs the full pipeline and
# metric suite.

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
# ## 11. Optional deep diagnostics
#
# This figure is heavier than the overview. Run it after choosing one result
# that deserves closer inspection.

plot_deep_dive(data, result, metrics, config)

# %%
# ## 12. Interpreting the main metrics
#
# **Global PCA / Laplacian scores**
#
# Higher is better. PCA is a linear baseline; the Laplacian score is usually
# more relevant for graph-aware embeddings.
#
# **Geodesic correlation**
#
# Spearman correlation between graph geodesic distances in the input space and
# Euclidean distances in the final layout.
#
# **TopoPreserve score**
#
# Composite score based on:
#
# - `PF1`: overlap of neighborhood sets;
# - `PJS`: Jensen-Shannon similarity of transition weights;
# - `SP`: spectral Procrustes alignment.
#
# **Riemannian deformation**
#
# Local expansion/contraction estimated from the embedding and graph Laplacian.
# A narrow distribution around zero means more uniform local area preservation.
