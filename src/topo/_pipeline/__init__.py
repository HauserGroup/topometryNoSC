"""Internal pipeline mixins for :class:`topo.topograph.TopOGraph`.

The orchestrator's behaviour is split across three cooperating mixins, each
owning one phase of the fit pipeline and operating on shared ``self`` state:

- :class:`~topo._pipeline.graph.GraphBuildMixin` — base neighborhood graph and
  kernel construction.
- :class:`~topo._pipeline.eigen.EigenBuildMixin` — intrinsic-dimension sizing,
  the dual (DM / msDM) eigenbasis and refined scaffold graphs.
- :class:`~topo._pipeline.layout.LayoutBuildMixin` — spectral initialization,
  projections and visualization.

These are private implementation details; they are not part of the public API
and should be used only through ``TopOGraph``.
"""
