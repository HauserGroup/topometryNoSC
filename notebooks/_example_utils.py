from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from topo.base.ann import kNN
from topo.eval.global_scores import global_score_laplacian, global_score_pca
from topo.eval.local_scores import geodesic_correlation
from topo.eval.rmetric import (
    RiemannMetric,
    calculate_deformation,
    plot_metric_contraction_expansion,
    plot_riemann_metric_localized,
)
from topo.eval.topo_metrics import (
    get_P,
    rank_diffusion_correlation,
    rowwise_js_similarity,
    sparse_neighborhood_f1,
    spectral_procrustes,
    spectral_similarity,
    topo_preserve_score,
)
from topo.layouts.projector import Projector
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.topograph import _KERNEL_CONFIGS
from topo.tpgraph.kernels import Kernel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the TopOMetry demo pipeline."""

    # Data
    use_custom_data: bool = False
    data_path: Path = Path("../data/my_data.npy")
    color_path: Path | None = None
    scale_data: bool = False
    n_samples: int = 2000
    noise: float = 0.5
    random_state: int = 42

    # Graph/kernel
    n_neighbors: int = 15
    metric: str = "euclidean"
    backend: str = "hnswlib"
    kernel_version: str = "cknn"
    sigma: float = 1.0
    anisotropy: float = 1.0

    # Spectral scaffold
    n_components_dm: int = 64
    dm_method: str = "msDM"
    eigensolver: str = "arpack"
    diffusion_time: int = 0

    # Layout
    projection_method: str = "PaCMAP"
    n_components_2d: int = 2
    num_iters: int = 500

    # Output
    save_figures: bool = False
    output_dir: Path = Path("../figures")


@dataclass
class DemoData:
    X: FloatArray
    color: NDArray[Any]


@dataclass
class PipelineResult:
    knn_X: csr_matrix
    kernel_X: Kernel
    P_X: csr_matrix
    L_X: csr_matrix
    K_X: csr_matrix
    eigen: EigenDecomposition
    evals: FloatArray
    Z: FloatArray
    knn_Z: csr_matrix
    kernel_Z: Kernel
    K_Z: csr_matrix
    L_Z: csr_matrix
    init_Y: FloatArray
    Y: FloatArray


@dataclass
class MetricResult:
    gs_pca: float
    gs_lap: float
    geo_r: float
    topo_score: float
    parts: dict[str, float]
    rdc: float
    spectral_procrustes_r2: float
    js_similarity: float
    neighborhood_f1: float
    spectral_similarity: float
    G: FloatArray
    deform: FloatArray
    deformation_range: tuple[float, float]
    P_input: csr_matrix
    P_emb: csr_matrix


def print_options() -> None:
    """Print a compact overview of common user-facing options."""
    print(
        """
Available options

Kernel versions
  bw_adaptive
  bw_adaptive_alpha_decaying
  bw_adaptive_nbr_expansion
  bw_adaptive_alpha_decaying_nbr_expansion
  fuzzy
  cknn
  gaussian

Spectral scaffold
  DM      fixed-time diffusion maps
  msDM    multiscale diffusion maps
  LE      Laplacian eigenmaps

Projection methods
  MAP
  UMAP
  PaCMAP
  Isomap
  t-SNE
  TriMAP
  NCVis
  IsomorphicMDE
  IsometricMDE

Backends
  hnswlib
  nmslib
  sklearn
"""
    )


def as_2d_float_array(value: Any, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; got shape {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature/component.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


def as_1d_array(value: Any, name: str) -> NDArray[Any]:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {arr.shape}.")
    return arr


def as_1d_float_array(value: Any, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64).ravel()

    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


def as_csr_matrix(value: Any, name: str) -> csr_matrix:
    if sparse.issparse(value):
        mat = value.tocsr()
    else:
        mat = csr_matrix(np.asarray(value, dtype=np.float64))

    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")

    return mat


def as_scalar_float(value: Any, name: str) -> float:
    """Normalize metric returns that may be scalar or tuple-like."""
    if isinstance(value, tuple):
        if len(value) == 0:
            raise ValueError(f"{name} returned an empty tuple.")
        value = value[0]

    if isinstance(value, Mapping):
        raise TypeError(f"{name} returned a mapping, not a scalar value.")

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must return a scalar float-like value.") from exc

    if not np.isfinite(out):
        raise ValueError(f"{name} returned a non-finite value: {out}.")

    return out


def checked_color(value: Any, n_samples: int) -> NDArray[Any]:
    color = as_1d_array(value, "color")
    if color.shape[0] != n_samples:
        raise ValueError(f"color must have length {n_samples}; got {color.shape[0]}.")
    return color


def checked_embedding(
    value: Any,
    name: str,
    *,
    n_samples: int,
    min_columns: int = 2,
) -> FloatArray:
    arr = as_2d_float_array(value, name)

    if arr.shape[0] != n_samples:
        raise ValueError(f"{name} must have {n_samples} rows; got {arr.shape}.")
    if arr.shape[1] < min_columns:
        raise ValueError(
            f"{name} must have at least {min_columns} columns; got {arr.shape}."
        )

    return arr


def load_demo_data(config: DemoConfig) -> DemoData:
    """Load custom data or generate a Swiss roll demo dataset."""
    if config.use_custom_data:
        X = as_2d_float_array(np.load(config.data_path), "X")

        if config.color_path is None:
            color = np.arange(X.shape[0])
        else:
            color = checked_color(np.load(config.color_path), X.shape[0])

        print(f"Loaded custom data: {X.shape}")
    else:
        X_raw, color_raw = make_swiss_roll(
            n_samples=config.n_samples,
            noise=config.noise,
            random_state=config.random_state,
        )
        X = as_2d_float_array(X_raw, "X")
        color = checked_color(color_raw, X.shape[0])
        print(f"Generated Swiss roll: {X.shape}")

    if config.scale_data:
        X = as_2d_float_array(StandardScaler().fit_transform(X), "scaled X")
        print("Applied StandardScaler to X")

    return DemoData(X=X, color=color)


def kernel_kwargs(config: DemoConfig) -> dict[str, Any]:
    """Resolve a named TopOMetry kernel preset."""
    kernel_configs = cast(Mapping[str, Mapping[str, Any]], _KERNEL_CONFIGS)

    if config.kernel_version not in kernel_configs:
        valid = ", ".join(sorted(kernel_configs))
        raise ValueError(
            f"Unknown kernel_version={config.kernel_version!r}. "
            f"Available versions: {valid}."
        )

    kwargs = dict(kernel_configs[config.kernel_version])

    if config.kernel_version == "gaussian":
        kwargs["sigma"] = config.sigma

    return kwargs


def eigen_embedding(eigen: EigenDecomposition, n_samples: int) -> FloatArray:
    result = eigen.transform()
    if isinstance(result, tuple):
        result = result[0]
    return checked_embedding(
        result,
        "eigen.transform()",
        n_samples=n_samples,
        min_columns=1,
    )


def eigenvalues_array(eigen: EigenDecomposition) -> FloatArray:
    if eigen.eigenvalues is None:
        raise ValueError("EigenDecomposition did not populate eigenvalues.")
    return as_1d_float_array(eigen.eigenvalues, "eigen.eigenvalues")


def graph_layout_array(value: Any, n_samples: int) -> FloatArray:
    if isinstance(value, tuple):
        value = value[0]
    return checked_embedding(
        value,
        "spectral_layout()",
        n_samples=n_samples,
        min_columns=2,
    )


def projector_embedding(projector: Projector, n_samples: int) -> FloatArray:
    if not hasattr(projector, "Y_"):
        raise ValueError("Projector did not expose Y_ after fit().")
    return checked_embedding(
        projector.Y_,
        "projector.Y_",
        n_samples=n_samples,
        min_columns=2,
    )


def run_pipeline(data: DemoData, config: DemoConfig) -> PipelineResult:
    """Run the full TopOMetry demo pipeline."""
    X = data.X
    kwargs = kernel_kwargs(config)

    knn_X = as_csr_matrix(
        kNN(
            X,
            n_neighbors=config.n_neighbors,
            metric=config.metric,
            backend=config.backend,
        ),
        "knn_X",
    )

    kernel_X = Kernel(
        n_neighbors=config.n_neighbors,
        metric=config.metric,
        backend=config.backend,
        anisotropy=config.anisotropy,
        **kwargs,
    )
    kernel_X.fit(X)

    P_X = as_csr_matrix(kernel_X.P, "kernel_X.P")
    L_X = as_csr_matrix(kernel_X.L, "kernel_X.L")
    K_X = as_csr_matrix(kernel_X.K, "kernel_X.K")

    eigen = EigenDecomposition(
        n_components=config.n_components_dm,
        method=config.dm_method,
        eigensolver=config.eigensolver,
        drop_first=True,
        weight=True,
        t=config.diffusion_time,
    )
    eigen.fit(kernel_X)

    Z = eigen_embedding(eigen, n_samples=X.shape[0])
    evals = eigenvalues_array(eigen)

    knn_Z = as_csr_matrix(
        kNN(
            Z,
            n_neighbors=config.n_neighbors,
            metric=config.metric,
            backend=config.backend,
        ),
        "knn_Z",
    )

    kernel_Z = Kernel(
        n_neighbors=config.n_neighbors,
        metric=config.metric,
        backend=config.backend,
        anisotropy=config.anisotropy,
        **kwargs,
    )
    kernel_Z.fit(Z)

    K_Z = as_csr_matrix(kernel_Z.K, "kernel_Z.K")
    L_Z = as_csr_matrix(kernel_Z.L, "kernel_Z.L")

    init_raw = spectral_layout(
        graph=K_Z,
        dim=config.n_components_2d,
        random_state=config.random_state,
    )
    init_Y = graph_layout_array(init_raw, n_samples=X.shape[0])

    projector = Projector(
        projection_method=config.projection_method,
        n_components=config.n_components_2d,
        n_neighbors=config.n_neighbors,
        num_iters=config.num_iters,
        init="spectral",
    )

    if config.projection_method == "MAP":
        projector.fit(K_Z)
    else:
        projector.fit(Z)

    Y = projector_embedding(projector, n_samples=X.shape[0])

    print(f"Kernel      : {config.kernel_version}  {K_X.shape}, nnz={K_X.nnz}")
    print(f"Scaffold    : {config.dm_method}  {Z.shape}")
    print(f"Projection  : {config.projection_method}  {Y.shape}")

    return PipelineResult(
        knn_X=knn_X,
        kernel_X=kernel_X,
        P_X=P_X,
        L_X=L_X,
        K_X=K_X,
        eigen=eigen,
        evals=evals,
        Z=Z,
        knn_Z=knn_Z,
        kernel_Z=kernel_Z,
        K_Z=K_Z,
        L_Z=L_Z,
        init_Y=init_Y,
        Y=Y,
    )


def compute_metrics(
    data: DemoData,
    result: PipelineResult,
    config: DemoConfig,
) -> MetricResult:
    """Compute a compact set of geometry-preservation diagnostics."""
    X = data.X
    Y = result.Y

    gs_pca = as_scalar_float(global_score_pca(X, Y), "global_score_pca")
    gs_lap = as_scalar_float(
        global_score_laplacian(
            X,
            Y,
            k=config.n_neighbors,
            random_state=config.random_state,
        ),
        "global_score_laplacian",
    )

    landmarks = min(500, X.shape[0])
    geo_r = as_scalar_float(
        geodesic_correlation(
            X,
            Y,
            n_neighbors=config.n_neighbors,
            metric=config.metric,
            cor_method="spearman",
            landmarks=landmarks,
            random_state=config.random_state,
        ),
        "geodesic_correlation",
    )

    P_input = result.P_X.tocsr()
    P_emb = as_csr_matrix(
        get_P(Y, metric="euclidean", n_neighbors=config.n_neighbors),
        "P_emb",
    )
    r_eigs = min(64, X.shape[0] - 2)

    topo_result = topo_preserve_score(P_input, P_emb, r=r_eigs)
    if not isinstance(topo_result, tuple) or len(topo_result) != 2:
        raise TypeError("topo_preserve_score must return (score, parts).")

    topo_score = as_scalar_float(topo_result[0], "topo_preserve_score score")
    if not isinstance(topo_result[1], Mapping):
        raise TypeError("topo_preserve_score parts must be a mapping.")

    parts = {
        str(key): as_scalar_float(value, f"topo_preserve_score[{key!r}]")
        for key, value in topo_result[1].items()
    }

    rdc = as_scalar_float(
        rank_diffusion_correlation(P_input, P_emb, r=r_eigs),
        "rank_diffusion_correlation",
    )
    sp_r2 = as_scalar_float(
        spectral_procrustes(P_input, P_emb, r=r_eigs),
        "spectral_procrustes",
    )
    js_sim = as_scalar_float(
        rowwise_js_similarity(P_input, P_emb),
        "rowwise_js_similarity",
    )
    pf1 = as_scalar_float(
        sparse_neighborhood_f1(P_input, P_emb),
        "sparse_neighborhood_f1",
    )
    spec_sim = as_scalar_float(
        spectral_similarity(P_input, P_emb, r=r_eigs),
        "spectral_similarity",
    )

    rm = RiemannMetric(Y, result.L_Z)
    G = as_2d_or_3d_float_array(rm.get_rmetric(), "Riemann metric tensor")

    deformation_raw, deformation_range_raw = calculate_deformation(
        Y,
        result.L_Z,
        G_emb=G,
    )
    deform = as_1d_float_array(deformation_raw, "deformation")
    deformation_range = (
        float(deformation_range_raw[0]),
        float(deformation_range_raw[1]),
    )

    return MetricResult(
        gs_pca=gs_pca,
        gs_lap=gs_lap,
        geo_r=geo_r,
        topo_score=topo_score,
        parts=parts,
        rdc=rdc,
        spectral_procrustes_r2=sp_r2,
        js_similarity=js_sim,
        neighborhood_f1=pf1,
        spectral_similarity=spec_sim,
        G=G,
        deform=deform,
        deformation_range=deformation_range,
        P_input=P_input,
        P_emb=P_emb,
    )


def as_2d_or_3d_float_array(value: Any, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2-D or 3-D; got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


def print_metric_summary(metrics: MetricResult) -> None:
    """Print a compact metric summary."""
    print(
        f"""
Geometry-preservation summary

Global
  PCA baseline score          {metrics.gs_pca: .4f}
  Laplacian baseline score    {metrics.gs_lap: .4f}

Local / geodesic
  Geodesic Spearman rho       {metrics.geo_r: .4f}

Diffusion topology
  TopoPreserve composite      {metrics.topo_score: .4f}
  PF1 neighborhood overlap    {metrics.parts["PF1"]: .4f}
  PJS transition similarity   {metrics.parts["PJS"]: .4f}
  SP spectral Procrustes      {metrics.parts["SP"]: .4f}
  Rank diffusion corr.        {metrics.rdc: .4f}
  Spectral similarity         {metrics.spectral_similarity: .4f}

Riemannian deformation
  mean                        {np.mean(metrics.deform):+.4f}
  std                         {np.std(metrics.deform): .4f}
  range                       [{metrics.deform.min():.3f}, {metrics.deform.max():.3f}]
"""
    )


def save_figure(fig: Any, config: DemoConfig, filename: str) -> None:
    if not config.save_figures:
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


def plot_pipeline_overview(
    data: DemoData,
    result: PipelineResult,
    config: DemoConfig,
) -> None:
    """Plot a readable 3x3 overview of the pipeline."""
    X = data.X
    color = data.color
    Y = result.Y
    Z = result.Z

    use_3d = X.shape[1] >= 3 and not config.use_custom_data

    fig = plt.figure(figsize=(17, 15))
    fig.suptitle(
        f"TopOMetry pipeline · kernel={config.kernel_version} · "
        f"scaffold={config.dm_method} · layout={config.projection_method}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    if use_3d:
        ax1 = cast(Any, fig.add_subplot(3, 3, 1, projection="3d"))
        ax1.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            c=color,
            cmap="Spectral",
            s=4,
            linewidths=0,
        )
        ax1.set_title("(a) Input data", fontsize=10)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.tick_params(labelsize=6)
    else:
        ax1 = fig.add_subplot(3, 3, 1)
        sc1 = ax1.scatter(
            X[:, 0],
            X[:, 1],
            c=color,
            cmap="Spectral",
            s=4,
            linewidths=0,
        )
        ax1.set_title("(a) Input data", fontsize=10)
        ax1.set_xlabel("dim 1")
        ax1.set_ylabel("dim 2")
        plt.colorbar(sc1, ax=ax1, pad=0.02)

    ax2 = fig.add_subplot(3, 3, 2)
    edge_dists = as_1d_float_array(result.knn_X.data, "knn_X.data")
    ax2.hist(edge_dists, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    med = float(np.median(edge_dists))
    ax2.axvline(med, color="tomato", lw=1.2, ls="--", label=f"median = {med:.2f}")
    ax2.set_title("(b) Input kNN distances", fontsize=10)
    ax2.set_xlabel(f"{config.metric} distance")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(3, 3, 3)
    rows, cols = result.K_X.nonzero()
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if rows.size > 10_000:
        rng = np.random.default_rng(config.random_state)
        sel = rng.choice(rows.size, 10_000, replace=False)
        rows = rows[sel]
        cols = cols[sel]

    d_vals = np.asarray(result.knn_X[rows, cols]).ravel()
    k_vals = np.asarray(result.K_X[rows, cols]).ravel()
    mask = d_vals > 0

    ax3.scatter(
        d_vals[mask],
        k_vals[mask],
        s=1,
        alpha=0.3,
        color="darkcyan",
        rasterized=True,
    )
    ax3.set_title("(c) Distance → affinity", fontsize=10)
    ax3.set_xlabel("kNN distance")
    ax3.set_ylabel("kernel weight")
    ax3.tick_params(labelsize=7)

    ax4 = fig.add_subplot(3, 3, 4)
    n_show = min(20, result.evals.shape[0])
    idx = np.arange(1, n_show + 1)
    ax4.bar(idx, result.evals[:n_show], color="darkorange", width=0.7)

    if n_show >= 2:
        gaps = np.abs(np.diff(result.evals[:n_show]))
        gap_idx = int(np.argmax(gaps))
        ax4.axvline(
            gap_idx + 1.5,
            color="navy",
            lw=1.2,
            ls="--",
            label=f"largest gap after λ{gap_idx + 1}",
        )
        ax4.legend(fontsize=7)

    ax4.set_title(f"(d) {config.dm_method} spectrum", fontsize=10)
    ax4.set_xlabel("rank")
    ax4.set_ylabel("λ")
    ax4.set_xticks(idx)
    ax4.tick_params(labelsize=7)

    ax5 = fig.add_subplot(3, 3, 5)
    sc5 = ax5.scatter(Z[:, 0], Z[:, 1], c=color, cmap="Spectral", s=4, linewidths=0)
    ax5.set_title("(e) Scaffold: first two dimensions", fontsize=10)
    ax5.set_xlabel(f"{config.dm_method} 1")
    ax5.set_ylabel(f"{config.dm_method} 2")
    plt.colorbar(sc5, ax=ax5, pad=0.02)

    ax6 = fig.add_subplot(3, 3, 6)
    if Z.shape[1] >= 3:
        sc6 = ax6.scatter(
            Z[:, 1],
            Z[:, 2],
            c=color,
            cmap="Spectral",
            s=4,
            linewidths=0,
        )
        ax6.set_xlabel(f"{config.dm_method} 2")
        ax6.set_ylabel(f"{config.dm_method} 3")
    else:
        sc6 = ax6.scatter(
            Z[:, 0],
            Z[:, 1],
            c=color,
            cmap="Spectral",
            s=4,
            linewidths=0,
        )
        ax6.set_xlabel(f"{config.dm_method} 1")
        ax6.set_ylabel(f"{config.dm_method} 2")

    ax6.set_title("(f) Scaffold: higher harmonics", fontsize=10)
    plt.colorbar(sc6, ax=ax6, pad=0.02)

    ax7 = fig.add_subplot(3, 3, 7)
    sc7 = ax7.scatter(
        result.init_Y[:, 0],
        result.init_Y[:, 1],
        c=color,
        cmap="Spectral",
        s=4,
        linewidths=0,
    )
    ax7.set_title("(g) Spectral initialization", fontsize=10)
    ax7.set_xlabel("init 1")
    ax7.set_ylabel("init 2")
    plt.colorbar(sc7, ax=ax7, pad=0.02)

    ax8 = fig.add_subplot(3, 3, 8)
    sc8 = ax8.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=4, linewidths=0)
    ax8.set_title(f"(h) Final {config.projection_method} layout", fontsize=10)
    ax8.set_xlabel("dim 1")
    ax8.set_ylabel("dim 2")
    plt.colorbar(sc8, ax=ax8, pad=0.02)

    ax9 = fig.add_subplot(3, 3, 9)
    log_degree = np.log1p(np.asarray(result.K_Z.sum(axis=1)).ravel())
    sc9 = ax9.scatter(Y[:, 0], Y[:, 1], c=log_degree, cmap="magma", s=4, linewidths=0)
    ax9.set_title("(i) Layout density", fontsize=10)
    ax9.set_xlabel("dim 1")
    ax9.set_ylabel("dim 2")
    cb = plt.colorbar(sc9, ax=ax9, pad=0.02)
    cb.set_label("log(1 + weighted degree)", fontsize=7)

    plt.tight_layout()
    save_figure(
        fig,
        config,
        f"pipeline_{config.kernel_version}_{config.dm_method}_{config.projection_method}.png",
    )
    plt.show()


def plot_metric_overview(
    data: DemoData,
    result: PipelineResult,
    metrics: MetricResult,
    config: DemoConfig,
) -> None:
    """Plot a compact 2x2 metric summary."""
    Y = result.Y

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Geometry diagnostics · kernel={config.kernel_version} · "
        f"layout={config.projection_method}",
        fontsize=13,
        fontweight="bold",
        y=1.005,
    )

    ax_a = axes[0, 0]
    plot_riemann_metric_localized(
        Y,
        result.L_Z,
        G_emb=metrics.G,
        n_plot=min(800, Y.shape[0]),
        colors=data.color,
        cmap="Spectral",
        ax=ax_a,
        scale_mode="anisotropy",
        scale_gain=1.2,
    )
    ax_a.set_title(
        "(a) Local metric ellipses\ncircles = isotropic, elongated = anisotropic",
        fontsize=9,
    )

    ax_b = axes[0, 1]
    plot_metric_contraction_expansion(
        Y,
        result.L_Z,
        G_emb=metrics.G,
        ax=ax_b,
        cmap="coolwarm",
        title="(b) Local area change",
        title_fontsize=9,
    )

    ax_c = axes[1, 0]
    ax_c.hist(
        metrics.deform,
        bins=60,
        color="mediumpurple",
        edgecolor="none",
        alpha=0.85,
    )
    ax_c.axvline(0, color="k", lw=1, ls="--")
    ax_c.axvline(
        np.median(metrics.deform),
        color="tomato",
        lw=1.2,
        ls="--",
        label=f"median = {np.median(metrics.deform):.3f}",
    )
    ax_c.set_title("(c) Deformation distribution", fontsize=9)
    ax_c.set_xlabel("centered log det(G)")
    ax_c.set_ylabel("count")
    ax_c.legend(fontsize=8)

    ax_d = axes[1, 1]
    metric_names = [
        "Global\nPCA",
        "Global\nLap.",
        "Geo.\nrho",
        "Topo\nscore",
        "PF1",
        "PJS",
        "SP",
        "RDC",
        "Spec.\nsim.",
    ]
    metric_vals = [
        metrics.gs_pca,
        metrics.gs_lap,
        metrics.geo_r,
        metrics.topo_score,
        metrics.neighborhood_f1,
        metrics.js_similarity,
        metrics.spectral_procrustes_r2,
        metrics.rdc,
        metrics.spectral_similarity,
    ]
    bar_colors = ["#3b82f6"] * 3 + ["#f59e0b"] + ["#10b981"] * 5
    bars = ax_d.bar(
        metric_names,
        metric_vals,
        color=bar_colors,
        width=0.65,
        edgecolor="white",
    )
    y_lo = min(0.0, min(metric_vals)) - 0.05
    ax_d.set_ylim(y_lo, 1.10)
    ax_d.axhline(0, color="k", lw=0.8, ls="--")
    ax_d.set_ylabel("score")
    ax_d.set_title("(d) Metrics at a glance", fontsize=9)
    ax_d.tick_params(axis="x", labelsize=7)

    for bar, value in zip(bars, metric_vals):
        va = "bottom" if value >= 0 else "top"
        offset = 0.02 if value >= 0 else -0.02
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    save_figure(
        fig,
        config,
        f"eval_{config.kernel_version}_{config.dm_method}_{config.projection_method}.png",
    )
    plt.show()


def plot_deep_dive(
    data: DemoData,
    result: PipelineResult,
    metrics: MetricResult,
    config: DemoConfig,
) -> None:
    """Plot heavier diagnostic panels. Optional for casual users."""
    X = data.X
    Y = result.Y
    rng = check_random_state(config.random_state)

    P_input = metrics.P_input
    P_emb = metrics.P_emb

    if P_input is None or P_emb is None:
        raise ValueError("MetricResult must contain P_input and P_emb.")

    P_input = P_input.tocsr()
    P_emb = P_emb.tocsr()

    P_input_shape = P_input.shape
    P_emb_shape = P_emb.shape

    if P_input_shape is None or P_emb_shape is None:
        raise ValueError("P_input and P_emb must have valid shapes.")

    if P_input_shape[0] != P_emb_shape[0]:
        raise ValueError(
            f"P_input and P_emb must have the same number of rows; "
            f"got {P_input_shape[0]} and {P_emb_shape[0]}."
        )

    n_rows = int(P_input_shape[0])

    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    fig.suptitle(
        f"Metric deep-dive · kernel={config.kernel_version} · "
        f"scaffold={config.dm_method} · layout={config.projection_method}",
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )

    n_lm = min(300, X.shape[0])
    lm = rng.choice(X.shape[0], n_lm, replace=False)
    D_inp = cdist(X[lm], X, metric="euclidean")
    D_out = cdist(Y[lm], Y, metric="euclidean")

    rank_err = np.mean(
        np.abs(
            np.argsort(np.argsort(D_inp, axis=1), axis=1).astype(float) / X.shape[0]
            - np.argsort(np.argsort(D_out, axis=1), axis=1).astype(float) / X.shape[0]
        ),
        axis=0,
    )

    ax = axes[0, 0]
    sc = ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=rank_err,
        cmap="hot_r",
        s=4,
        linewidths=0,
        vmin=0,
        vmax=np.percentile(rank_err, 95),
    )
    ax.set_title("(a) Per-point rank distortion", fontsize=9)
    ax.set_xlabel("layout dim 1")
    ax.set_ylabel("layout dim 2")
    plt.colorbar(sc, ax=ax, pad=0.02).set_label("mean |delta-rank| / n", fontsize=7)

    n_sh = min(500, X.shape[0])
    sh_idx = rng.choice(X.shape[0], n_sh, replace=False)
    d_in = cdist(X[sh_idx], X[sh_idx], metric="euclidean").ravel()
    d_out = cdist(Y[sh_idx], Y[sh_idx], metric="euclidean").ravel()
    d_out_sc = d_out * float(d_in.max() / (d_out.max() + 1e-12))

    ax = axes[0, 1]
    ax.scatter(d_in, d_out_sc, s=0.5, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot([0, d_in.max()], [0, d_in.max()], "r--", lw=1)
    ax.set_title("(b) Shepard diagram", fontsize=9)
    ax.set_xlabel("input distance")
    ax.set_ylabel("layout distance")

    distortion = d_out_sc - d_in
    ax = axes[0, 2]
    ax.hist(distortion, bins=80, color="darkorange", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(np.median(distortion), color="tomato", lw=1.2, ls="--")
    ax.set_title("(c) Pairwise distance residuals", fontsize=9)
    ax.set_xlabel("layout distance - input distance")
    ax.set_ylabel("count")

    js_per_row = np.asarray(
        [
            as_scalar_float(
                rowwise_js_similarity(
                    P_input[i : i + 1],
                    P_emb[i : i + 1],
                ),
                "rowwise_js_similarity",
            )
            for i in range(n_rows)
        ],
        dtype=np.float64,
    )

    ax = axes[1, 0]
    sc = ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=js_per_row,
        cmap="RdYlGn",
        s=4,
        linewidths=0,
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"(d) Per-point JS similarity, mean={js_per_row.mean():.3f}",
        fontsize=9,
    )
    ax.set_xlabel("layout dim 1")
    ax.set_ylabel("layout dim 2")
    plt.colorbar(sc, ax=ax, pad=0.02).set_label("JS similarity", fontsize=7)

    pf1_per_row = np.asarray(
        [
            as_scalar_float(
                sparse_neighborhood_f1(
                    P_input[i : i + 1],
                    P_emb[i : i + 1],
                ),
                "sparse_neighborhood_f1",
            )
            for i in range(n_rows)
        ],
        dtype=np.float64,
    )

    ax = axes[1, 1]
    sc = ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=pf1_per_row,
        cmap="RdYlGn",
        s=4,
        linewidths=0,
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"(e) Per-point neighborhood F1, mean={pf1_per_row.mean():.3f}",
        fontsize=9,
    )
    ax.set_xlabel("layout dim 1")
    ax.set_ylabel("layout dim 2")
    plt.colorbar(sc, ax=ax, pad=0.02).set_label("F1 score", fontsize=7)

    ax = axes[1, 2]
    ax.scatter(
        pf1_per_row,
        js_per_row,
        s=3,
        alpha=0.3,
        color="mediumpurple",
        rasterized=True,
    )
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_title("(f) JS similarity vs F1", fontsize=9)
    ax.set_xlabel("neighborhood F1")
    ax.set_ylabel("JS similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    log_deg = np.log1p(np.asarray(result.K_Z.sum(axis=1)).ravel())

    ax = axes[2, 0]
    ax.scatter(
        log_deg,
        np.abs(metrics.deform),
        s=3,
        alpha=0.3,
        color="mediumpurple",
        rasterized=True,
    )
    ax.set_title("(g) |Deformation| vs kernel density", fontsize=9)
    ax.set_xlabel("log(1 + weighted degree)")
    ax.set_ylabel("|centered log det G|")

    ax = axes[2, 1]
    ax.scatter(
        rank_err,
        np.abs(metrics.deform),
        s=3,
        alpha=0.3,
        color="darkorange",
        rasterized=True,
    )
    ax.set_title("(h) |Deformation| vs rank distortion", fontsize=9)
    ax.set_xlabel("mean |delta-rank| / n")
    ax.set_ylabel("|centered log det G|")

    sorted_d = np.sort(np.abs(metrics.deform))[::-1]
    denom = sorted_d.sum()
    cumsum_d = np.zeros_like(sorted_d) if denom <= 0 else np.cumsum(sorted_d) / denom
    pct_pts = np.linspace(0, 100, len(sorted_d))

    ax = axes[2, 2]
    ax.plot(pct_pts, cumsum_d, color="navy", lw=1.5)
    ax.axhline(0.5, color="tomato", lw=1, ls="--")

    p50_idx = min(np.searchsorted(cumsum_d, 0.5), len(pct_pts) - 1)
    p50 = float(pct_pts[p50_idx])

    ax.axvline(p50, color="tomato", lw=1, ls="--", label=f"{p50:.1f}% of points")
    ax.set_title("(i) Cumulative deformation magnitude", fontsize=9)
    ax.set_xlabel("% of points")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=7)

    plt.tight_layout()
    save_figure(
        fig,
        config,
        f"deepdive_{config.kernel_version}_{config.dm_method}_{config.projection_method}.png",
    )
    plt.show()
