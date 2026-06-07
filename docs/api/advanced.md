# Advanced Estimators (Custom Pipelines)

These scikit-learn compatible transformers can be used individually to build custom manifold learning pipelines.

The `Kernel(..., fuzzy=True)` path delegates UMAP-specific fuzzy simplicial-set
construction to `umap-learn`; TopoMetry uses the resulting graph inside its own
spectral-scaffold and graph-refinement pipeline.

::: topo.base.ann.kNN
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      filters:
        - "!^__"
        - "!^_"

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
