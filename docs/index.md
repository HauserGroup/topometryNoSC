# topometry-nosc

!!! note "Provenance"
    `topometry-nosc` is an independently maintained, heavily modified fork of
    [TopOMetry](https://github.com/davisidarta/topometry) by David S Oliveira
    (original copyright and MIT license preserved). Its API, internals and
    behaviour may differ substantially from upstream. It is **not** an official
    release of, affiliated with, or endorsed by the original project. The import
    package is still named `topo`, so it cannot be installed alongside the
    upstream `topometry` distribution in the same environment. Please cite both
    this fork and the original work — see [Citation](citation.md).

**topometry-nosc** is a geometry-aware Python toolkit for exploring
high-dimensional data via diffusion/Laplacian operators. It learns
**neighborhood graphs → Laplace–Beltrami–type operators → spectral scaffolds →
refined graphs** and then builds low-dimensional layouts for analysis and
visualization.

- **scikit-learn–style transformers** with a high-level `TopOGraph` orchestrator
- **Fixed-time & multiscale spectral scaffolds**
- **Operator-native metrics** to quantify geometry preservation and
  **Riemannian diagnostics** to evaluate distortion in visualizations
- Designed for **large, diverse datasets**

For background, see the preprint: <https://doi.org/10.1101/2022.03.14.484134>

## Status

Under active development. Interfaces may still change.

## Geometry-first rationale

We approximate the **Laplace–Beltrami operator (LBO)** by learning well-weighted
similarity graphs and their Laplacian/diffusion operators. The **eigenfunctions**
of these operators form an orthonormal basis — the **spectral scaffold** — that
captures the dataset's intrinsic geometry across scales. This view connects to
**Diffusion Maps**, **Laplacian Eigenmaps**, and related kernel eigenmaps, and
enables downstream tasks such as clustering and graph-layout optimization with
geometry preserved.

## When to use it

- Geometry-faithful representations beyond variance maximization (e.g., PCA)
- Robust low-dimensional views and clustering from operator-grounded features
- Quantitative **operator-native** metrics to compare methods and parameter choices
- Reproducible, **non-destructive** pipelines

### When not to use it

- **Very small sample sizes** where the manifold hypothesis is weak
- Workflows needing **streaming/online** updates or **inverse transforms**
  (embedding new points without recomputing operators is not currently supported)

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md) — fit a `TopOGraph`, inspect scaffolds, make 2-D layouts
- [Practical FAQ](faq.md) — when to use TopoMetry, target-aware checks, troubleshooting
- [Concepts](concepts.md) and [Math details](math_details.md)
- [API Reference](api.md)
