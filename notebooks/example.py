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
# # TopOMetry demo
#
# **Purpose:** a compact first-run path through the full TopOMetry workflow.
#
# This notebook gives a compact tour of the TopOMetry workflow:
#
# 1. load or generate data;
# 2. build a k-nearest-neighbor graph;
# 3. convert the graph to a kernel;
# 4. compute a spectral scaffold;
# 5. optimize a 2-D layout;
# 6. evaluate geometry preservation.
#
# Most implementation details live in `_example_utils.py` so this notebook stays
# readable.

# %%
# ## Colab Setup
# This cell automatically sets up the environment if running in Google Colab.
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Colab detected: installing topometry-nosc and downloading utils...")
    subprocess.run(["pip", "install", "-q", "topometry-nosc[all]"], check=True)
    subprocess.run(
        [
            "wget",
            "-q",
            "https://raw.githubusercontent.com/HauserGroup/topometryNoSC/master/notebooks/_example_utils.py",
        ],
        check=True,
    )
    subprocess.run(["mkdir", "-p", "../data", "../figures"], check=True)
    print("Setup complete.")

# %%
from pathlib import Path

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
# ## 1. Configure the demo
#
# The defaults generate a Swiss roll and run:
#
# - cKNN kernel;
# - multiscale diffusion maps scaffold;
# - PaCMAP final projection.
#
# Edit this cell and rerun the notebook to try other options.

config = DemoConfig(
    # Data
    use_custom_data=False,
    data_path=Path("../data/my_data.npy"),
    color_path=None,
    scale_data=False,
    n_samples=2000,
    noise=0.5,
    random_state=42,
    # Graph/kernel
    n_neighbors=15,
    metric="euclidean",
    backend="hnswlib",
    kernel_version="cknn",
    sigma=1.0,
    anisotropy=1.0,
    # Spectral scaffold
    n_components_dm=64,
    dm_method="msDM",
    eigensolver="arpack",
    diffusion_time=0,
    # Final 2-D layout
    projection_method="PaCMAP",
    n_components_2d=2,
    num_iters=500,
    # Output
    save_figures=False,
    output_dir=Path("../figures"),
)

# %%
# ## 2. Available options

print_options()

# %%
# ## 3. Load or generate data
#
# To use your own data:
#
# ```python
# config = DemoConfig(
#     use_custom_data=True,
#     data_path=Path("../data/my_data.npy"),
#     color_path=Path("../data/my_labels.npy"),  # optional
# )
# ```
#
# `data_path` should point to a `.npy` array with shape
# `(n_samples, n_features)`.

data = load_demo_data(config)

# %%
# ## 4. Run the pipeline
#
# The pipeline returns all intermediate objects:
#
# - `kernel_X`: kernel in input space;
# - `Z`: spectral scaffold;
# - `kernel_Z`: kernel in scaffold space;
# - `init_Y`: spectral initialization;
# - `Y`: final 2-D layout.

result = run_pipeline(data, config)

# %%
# ## 5. Visual overview
#
# This figure shows the main stages of the workflow:
#
# - input data;
# - input graph distances;
# - distance-to-affinity transformation;
# - spectral scaffold;
# - spectral initialization;
# - final layout.

plot_pipeline_overview(data, result, config)

# %%
# ## 6. Compute quality metrics
#
# These metrics compare the final layout against the input geometry using:
#
# - global baseline scores;
# - geodesic rank correlation;
# - diffusion-operator topology preservation;
# - local Riemannian deformation.

metrics = compute_metrics(data, result, config)
print_metric_summary(metrics)

# %%
# ## 7. Metric overview
#
# The compact diagnostic figure gives a first-pass view of whether the layout
# preserves local and global structure.

plot_metric_overview(data, result, metrics, config)

# %%
# ## 8. Optional deeper diagnostics
#
# This cell is heavier and more detailed. Use it when you want to inspect where
# distortion occurs in the layout.

plot_deep_dive(data, result, metrics, config)

# %%
# ## 9. Interpreting the main metrics
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
