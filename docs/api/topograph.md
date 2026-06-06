# Main workflow: TopOGraph

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

## Advanced graph state

These properties are fitted attributes that cache the intermediate states of the graphs.

::: topo.topograph.TopOGraph
    options:
      show_root_heading: false
      inherited_members: false
      members:
        - knn_X
        - P_of_X
        - knn_Z
        - P_of_Z
        - knn_msZ
        - P_of_msZ
      filters:
        - "!^__"

## Persistence

::: topo.topograph.save_topograph
::: topo.topograph.load_topograph
