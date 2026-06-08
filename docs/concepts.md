# Concepts

A short tour of the core objects. See [Math details](math_details.md) for the
theory and [API Reference](api/index.md) for signatures.

## Kernel graph

The package builds a sparse affinity graph from nearest-neighbor distances. The
[`Kernel`](api/advanced.md#topo.tpgraph.kernels.Kernel) class wraps kNN construction
(`topo.base.ann.kNN`) and several bandwidth/density-correction schemes
(adaptive bandwidth, continuous k-NN, fuzzy simplicial sets, Gaussian).

## Diffusion operator

A Markov-type graph operator derived from the kernel, used for diffusion maps and
related spectral decompositions. Powers of the operator describe diffusion of a
signal across the graph over time `t`.

## Laplacian Eigenmaps

Eigenvectors of a graph Laplacian represent the geometry of the graph in a
low-dimensional, orthonormal basis. The
[`EigenDecomposition`](api/advanced.md#topo.spectral.eigen.EigenDecomposition) class computes these spectral
scaffolds (diffusion-map `DM`, multiscale `msDM`, and Laplacian-eigenmap variants).

## Spectral scaffold

The collection of eigenfunctions forms the **spectral scaffold** — an orthonormal
basis capturing intrinsic geometry across scales. Refined kNN graphs and kernels
are then built in scaffold space before computing 2-D layouts.

## Topology-preservation metrics

`topo.eval` provides metrics comparing local and global geometric structure
between the original data representation and learned embeddings, plus Riemannian
distortion diagnostics. These scores evaluate geometry/topology preservation,
not supervised target predictiveness; for target-aware checks see the
[Practical FAQ](faq.md#how-should-i-evaluate-an-embedding-against-a-target-variable).

## High-level orchestrator

[`TopOGraph`](api/topograph.md#topo.topograph.TopOGraph) ties the pipeline together: kNN → kernel →
eigenbasis → scaffold → refined graph → 2-D layouts, with a scikit-learn-style
`fit` / `transform` interface.
