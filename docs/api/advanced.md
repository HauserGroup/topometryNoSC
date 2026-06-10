# Advanced Estimators (Custom Pipelines)

These scikit-learn compatible transformers can be used individually to build custom manifold learning pipelines.

The `Kernel(..., fuzzy=True)` path delegates UMAP-specific fuzzy simplicial-set
construction to `umap-learn`; TopoMetry uses the resulting graph inside its own
spectral-scaffold and graph-refinement pipeline.

The `Kernel(..., cknn=True)` path builds the paper-defined binary CkNN graph:
samples are adjacent when `d(i, j) < delta * sqrt(rho_i * rho_j)`. This is an
unweighted graph construction, not a weighted adaptive kernel; use
`cknn_ratio_matrix` separately when normalized CkNN distance ratios are needed.

::: topo.base.ann.kNN
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      filters:
        - "!^__"
        - "!^_"

::: topo.tpgraph.cknn.cknn_graph
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.tpgraph.cknn.cknn_ratio_matrix
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.tpgraph.kernels.compute_kernel
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.tpgraph.kernels.Kernel
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      inherited_members: false
      filters:
        - "!^__"
        - "!^_"
        - "!^resistance_distance$"
        - "!^sparsify$"
        - "!^interpolate$"

::: topo.spectral.eigen.EigenDecomposition
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      inherited_members: false
      filters:
        - "!^__"
        - "!^_"

::: topo.spectral.eigen.eigendecompose
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.spectral.LE
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.spectral.graph_laplacian
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.spectral.diffusion_operator
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.layouts.projector.Projector
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      inherited_members: false
      filters:
        - "!^__"
        - "!^_"

::: topo.tpgraph.intrinsic_dim.IntrinsicDim
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      inherited_members: false
      filters:
        - "!^__"
        - "!^_"

::: topo.tpgraph.intrinsic_dim.automated_scaffold_sizing
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.layouts.diagnostics.find_ideal_projection
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true

::: topo.layouts.diagnostics.run_best_projection
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
