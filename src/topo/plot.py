"""Plotting helpers for embeddings, spectra and diagnostics.

Scatter plots for 2-D/3-D and manifold embeddings (hyperboloid, Poincaré disk,
sphere, toroid), eigenspectrum decay plots, score bar charts, Riemann-metric
ellipses, heatmaps and a training-animation writer.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.patches import Circle, Ellipse
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state


def decay_plot(
    evals, title=None, figsize=(9, 5), fontsize=14, label_fontsize=14, wspace=0.3
):
    """Plot eigenspectrum decay and its first derivatives.

    Parameters
    ----------
    evals : np.ndarray of shape (n_eigs,)
        Eigenvalues to visualize.
    title : str, optional
        Plot title.
    figsize : tuple of (float, float), default=(9, 5)
        Figure width and height.
    fontsize : int, default=14
        Title font size.
    label_fontsize : int, default=14
        Axis label font size.
    wspace : float, default=0.3
        Width spacing between subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the two subplots.
    ax : np.ndarray of matplotlib.axes.Axes, shape (2,)
        Left axes (spectrum decay) and right axes (first derivatives).
    """
    evals = np.asarray(evals, dtype=float).ravel()
    if evals.size == 0:
        raise ValueError("evals must contain at least one value.")
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.98, wspace=wspace)
    max_eigs = int(np.sum(evals > 0, axis=0))
    first_diff = np.diff(evals)
    eigengap = int(np.argmax(first_diff) + 1) if first_diff.size > 0 else 0
    ax1 = ax[0]
    if title is not None:
        plt.suptitle(title, fontsize=fontsize)
    ax1.plot(range(0, len(evals)), evals, "b")
    ax1.set_ylabel("Eigenvalues", fontsize=label_fontsize)
    ax1.set_xlabel("Eigenvectors", fontsize=label_fontsize)
    if max_eigs == len(evals):
        ax1.vlines(
            eigengap, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="Eigengap"
        )
        plt.suptitle(
            "Spectrum decay and eigengap (%i)" % int(eigengap), fontsize=fontsize
        )
    else:
        ax1.vlines(
            max_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="Eigengap"
        )
        plt.suptitle(
            "Spectrum decay and eigengap (%i)" % int(max_eigs), fontsize=fontsize
        )
    ax1.legend(prop={"size": 12}, fontsize=label_fontsize, loc="best")
    ax2 = ax[1]
    ax2.set_yscale("log")
    ax2.scatter(range(0, len(first_diff)), np.abs(first_diff))
    ax2.set_ylabel("Eigenvalues first derivatives (abs)", fontsize=label_fontsize)
    ax2.set_xlabel("Eigenvalues", fontsize=label_fontsize)
    if max_eigs == len(evals):
        ax2.vlines(
            eigengap, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="Eigengap"
        )
    else:
        ax2.vlines(
            max_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="Eigengap"
        )
    plt.tight_layout()
    return fig, ax


def eigsorted(cov):
    """Return eigenvalues/eigenvectors of a covariance matrix sorted descending."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def _as_2d_array(X, name):
    """Return X as a 2-D ndarray."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    return arr


def _check_n_columns(X, n_cols, name):
    """Require at least n_cols columns."""
    if X.shape[1] < n_cols:
        raise ValueError(f"{name} must have at least {n_cols} columns.")


def scatter(
    res, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral", **kwargs
):
    """Draw a 2-D scatter plot of an embedding.

    Parameters
    ----------
    res : np.ndarray of shape (n_samples, >=2)
        Embedding coordinates; the first two columns are plotted.
    labels : array-like of shape (n_samples,), optional
        Per-point values mapped to colors via ``cmap``.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency (alpha).
    cmap : str, default="Spectral"
        Colormap name.
    **kwargs
        Extra arguments forwarded to ``matplotlib.axes.Axes.scatter``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.
    """
    res = _as_2d_array(res, "res")
    _check_n_columns(res, 2, "res")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    ax.scatter(
        res[:, 0],
        res[:, 1],
        cmap=cmap,
        c=labels,
        s=pt_size,
        marker=marker,
        alpha=opacity,
        **kwargs,
    )
    return fig, ax


def scatter3d(res, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"):
    """Draw a 3-D scatter plot of an embedding (first three columns).

    Parameters
    ----------
    res : np.ndarray of shape (n_samples, >=3)
        Embedding coordinates; the first three columns are plotted.
    labels : array-like of shape (n_samples,), optional
        Per-point values mapped to colors via ``cmap``.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency (alpha).
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The 3-D axes containing the scatter plot.
    """
    res = _as_2d_array(res, "res")
    _check_n_columns(res, 3, "res")
    fig = plt.figure()
    ax = cast(Any, fig.add_subplot(111, projection="3d"))
    ax.scatter(
        res[:, 0],
        res[:, 1],
        res[:, 2],
        cmap=cmap,
        c=labels,
        s=pt_size,
        marker=marker,
        alpha=opacity,
    )
    return fig, ax


def hyperboloid(emb, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"):
    """Plot a 2-D embedding on the hyperboloid (Lorentz) model.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, 2)
        2-D embedding coordinates.
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The 3-D axes.
    """
    x, y, z = two_to_3d_hyperboloid(emb)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, cmap=cmap, c=labels, s=pt_size, marker=marker, alpha=opacity)
    ax.view_init(35, 80)
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def two_to_3d_hyperboloid(emb):
    """Lift 2-D coordinates onto the 3-D hyperboloid surface ``(x, y, z)``."""
    emb = _as_2d_array(emb, "emb")
    _check_n_columns(emb, 2, "emb")
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb**2, axis=1))
    return x, y, z


def poincare(emb, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"):
    """Plot a 2-D embedding on the Poincaré disk model.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, 2)
        2-D embedding coordinates.
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    emb = _as_2d_array(emb, "emb")
    _check_n_columns(emb, 2, "emb")
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb**2, axis=1))
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    boundary = Circle((0, 0), 1, fc="none", ec="k")
    ax.add_artist(boundary)
    ax.scatter(
        disk_x, disk_y, cmap=cmap, c=labels, s=pt_size, marker=marker, alpha=opacity
    )
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    return fig, ax


def sphere(emb, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"):
    """Plot a 2-D (angular) embedding on the surface of a 3-D sphere.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, 2)
        2-D angular embedding (azimuth, elevation).
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The 3-D axes.
    """
    emb = _as_2d_array(emb, "emb")
    _check_n_columns(emb, 2, "emb")
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])
    fig = plt.figure()
    ax = cast(Any, fig.add_subplot(111, projection="3d"))
    ax.scatter(x, y, z, cmap=cmap, c=labels, s=pt_size, marker=marker, alpha=opacity)
    return fig, ax


def sphere_projection(
    emb, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"
):
    """Plot a 2-D spherical embedding as a flat (azimuth/elevation) projection.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, 2)
        2-D angular embedding (azimuth, elevation).
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    emb = _as_2d_array(emb, "emb")
    _check_n_columns(emb, 2, "emb")
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])
    x = np.arctan2(x, y)
    y = -np.arccos(z)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, cmap=cmap, c=labels, s=pt_size, marker=marker, alpha=opacity)
    return fig, ax


def toroid(
    emb, R=3, r=1, labels=None, pt_size=5, marker="o", opacity=1, cmap="Spectral"
):
    """Plot a 2-D (angular) embedding on the surface of a torus.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, 2)
        2-D angular embedding (two angles on the torus).
    R : float, default=3
        Major radius of the torus.
    r : float, default=1
        Minor radius of the torus.
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The 3-D axes.
    """
    emb = _as_2d_array(emb, "emb")
    _check_n_columns(emb, 2, "emb")
    x = (R + r * np.cos(emb[:, 0])) * np.cos(emb[:, 1])
    y = (R + r * np.cos(emb[:, 0])) * np.sin(emb[:, 1])
    z = r * np.sin(emb[:, 0])
    fig = plt.figure()
    ax = cast(Any, fig.add_subplot(111, projection="3d"))
    ax.scatter(x, y, z, cmap=cmap, c=labels, s=pt_size, marker=marker, alpha=opacity)
    ax.set_zlim3d(-3, 3)
    ax.view_init(35, 70)
    return fig, ax


def draw_simple_ellipse(
    position,
    width,
    height,
    angle,
    ax=None,
    from_size=0.1,
    to_size=0.5,
    n_ellipses=3,
    alpha=0.1,
    color=None,
):
    """Draw a nested family of ellipses at ``position`` for a Gaussian potential."""
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    width, height = np.sqrt(width + 10e-4), np.sqrt(height + 10e-4)
    # Draw the Ellipse
    for nsig in np.linspace(from_size, to_size, n_ellipses):
        ax.add_patch(
            Ellipse(
                xy=position,
                width=nsig * width,
                height=nsig * height,
                angle=angle,
                alpha=alpha,
                lw=0,
                color=color,
            )
        )


def gaussian_potential(
    emb,
    dims=(2, 3, 4),
    labels=None,
    pt_size=5,
    marker="o",
    opacity=1,
    cmap="Spectral",
):
    """Scatter a 2-D embedding overlaid with per-point Gaussian-potential ellipses.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, >=max(dims)+1)
        Multi-dimensional embedding; first two columns are plotted.
    dims : tuple of int, default=(2, 3, 4)
        Dimensions of the embedding to use for ellipse width, height, angle.
    labels : array-like of shape (n_samples,), optional
        Per-point labels for coloring.
    pt_size : float, default=5
        Marker size.
    marker : str, default="o"
        Marker style.
    opacity : float, default=1
        Marker transparency.
    cmap : str, default="Spectral"
        Colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    emb = np.asarray(emb)

    if emb.ndim != 2:
        raise ValueError("emb must be a 2-D array.")
    if emb.shape[1] <= max(dims):
        raise ValueError(
            f"emb must have at least {max(dims) + 1} columns for dims={dims}."
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if labels is None:
        label_codes = None
        ellipse_colors = None
    else:
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != emb.shape[0]:
            raise ValueError("labels must have the same length as emb.")
        _, label_codes = np.unique(labels_arr, return_inverse=True)
        ellipse_colors = plt.get_cmap(cmap)(
            np.linspace(0, 1, int(label_codes.max()) + 1)
        )

    for i in range(emb.shape[0]):
        pos = emb[i, :2]
        if ellipse_colors is None or label_codes is None:
            color = None
        else:
            color = ellipse_colors[label_codes[i]]

        draw_simple_ellipse(
            pos,
            emb[i, dims[0]],
            emb[i, dims[1]],
            emb[i, dims[2]],
            ax=ax,
            n_ellipses=1,
            color=color,
            from_size=1.0,
            to_size=1.0,
            alpha=0.01,
        )

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        cmap=cmap,
        c=labels,
        s=pt_size,
        marker=marker,
        alpha=opacity,
    )
    return fig, ax


@numba.njit(fastmath=True)
def eval_gaussian(x, pos=np.array([0, 0]), cov=np.eye(2, dtype=np.float32)):
    """Evaluate a 2-D Gaussian with mean ``pos`` and covariance ``cov`` at ``x``."""
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det > 1e-16:
        cov_inv = (
            np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) * 1.0 / det
        )
        diff = x - pos
        m_dist = (
            cov_inv[0, 0] * diff[0] ** 2
            - (cov_inv[0, 1] + cov_inv[1, 0]) * diff[0] * diff[1]
            + cov_inv[1, 1] * diff[1] ** 2
        )
        return (np.exp(-0.5 * m_dist)) / (2 * np.pi * np.sqrt(np.abs(det)))
    else:
        return 0.0


@numba.njit(fastmath=True)
def eval_density_at_point(x, embedding):
    """Sum per-point Gaussian densities of ``embedding`` evaluated at ``x``."""
    result = 0.0
    for i in range(embedding.shape[0]):
        pos = embedding[i, :2]
        t = embedding[i, 4]
        U = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
        cov = U @ np.diag(embedding[i, 2:4]) @ U
        result += eval_gaussian(x, pos=pos, cov=cov)
    return result


def get_cmap(n, name="hsv"):
    """Return a colormap with ``n`` discrete colors from the named colormap."""
    return plt.cm.get_cmap(name, n)


def create_density_plot(X, Y, embedding):
    """Evaluate a normalized Gaussian-mixture density over a meshgrid ``(X, Y)``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    embedding = _as_2d_array(embedding, "embedding")
    _check_n_columns(embedding, 5, "embedding")
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    Z = np.zeros_like(X, dtype=float)
    tree = KDTree(embedding[:, :2])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            query = np.array([[X[i, j], Y[i, j]]], dtype=float)
            nearby_idx = tree.query_radius(query, r=2)[0]
            nearby_points = embedding[nearby_idx]
            point = np.array([X[i, j], Y[i, j]], dtype=float)
            Z[i, j] = eval_density_at_point(point, nearby_points)
    total = Z.sum()
    if total <= 0 or not np.isfinite(total):
        return Z
    return Z / total


def plot_bases_scores(bases_scores, return_plot=True, figsize=(20, 8), fontsize=20):
    """Bar-plot PCA-loss and geodesic Spearman R for each candidate base."""
    keys = list(bases_scores.keys())
    values = list(bases_scores.values())
    cmap = get_cmap(len(keys), name="tab20")
    k_color = [cmap(k) for k in range(len(keys))]
    pca_vals = [val[0] for val in values]
    r_vals = [val[1] for val in values]
    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Bases scores:", fontsize=fontsize)
    ax1.bar(x, pca_vals, color=k_color)
    ax1.set_title("PCA loss", fontsize=fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, fontsize=fontsize, rotation=90)
    ax2.bar(x, r_vals, color=k_color)
    ax2.set_title("Geodesic Spearman R", fontsize=fontsize)
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, fontsize=fontsize, rotation=90)
    fig.tight_layout()
    return fig


def plot_graphs_scores(graphs_scores, return_plot=True, figsize=(20, 8), fontsize=20):
    """Bar-plot geodesic Spearman R for each candidate graph."""
    keys = list(graphs_scores.keys())
    values = list(graphs_scores.values())
    cmap = get_cmap(len(keys), name="tab20")
    k_color = [cmap(k) for k in range(len(keys))]
    x = np.arange(len(keys))
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("Graphs scores:", fontsize=fontsize)
    ax1.bar(x, values, color=k_color)
    ax1.set_title("Geodesic Spearman R", fontsize=fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, fontsize=fontsize // 2, rotation=90)
    fig.tight_layout()
    return fig


def plot_layouts_scores(layouts_scores, return_plot=True, figsize=(20, 8), fontsize=20):
    """Bar-plot PCA-loss and geodesic Spearman R for each candidate layout."""
    keys = list(layouts_scores.keys())
    values = list(layouts_scores.values())
    cmap = get_cmap(len(keys), name="tab20")
    k_color = [cmap(k) for k in range(len(keys))]
    pca_vals = [val[0] for val in values]
    r_vals = [val[1] for val in values]
    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Layouts scores:", fontsize=fontsize)
    ax1.bar(x, pca_vals, color=k_color)
    ax1.set_title("PCA loss", fontsize=fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, fontsize=fontsize // 2, rotation=90)
    ax2.bar(x, r_vals, color=k_color)
    ax2.set_title("Geodesic Spearman R", fontsize=fontsize)
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, fontsize=fontsize // 2, rotation=90)
    fig.tight_layout()
    return fig


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """Draw the ``nstd``-sigma covariance ellipse of a point cloud."""
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """Draw the ``nstd``-sigma ellipse of covariance ``cov`` centered at ``pos``."""
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(np.absolute(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


def plot_riemann_metric(
    emb,
    L,
    H_emb=None,
    ax=None,
    n_plot=50,
    std=1,
    alpha=0.1,
    title="Riemannian metric",
    title_fontsize=10,
    labels=None,
    pt_size=1,
    cmap="Spectral",
    figsize=(8, 8),
    random_state=None,
    **kwargs,
):
    """Plot Riemannian metric using ellipses. Adapted from Megaman.

    Parameters
    ----------
    emb : np.ndarray of shape (n_samples, n_features)
        Embedding coordinates. Must have at least 2 columns.
    L : scipy.sparse.spmatrix of shape (n_samples, n_samples), optional
        Graph Laplacian defining the reference geometry. Required if H_emb is None.
    H_emb : np.ndarray of shape (n_samples, n_features, n_features), optional
        Inverse (dual) Riemannian metric at each point. If None, computed from L using RiemannMetric.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If None, creates a new figure and axes.
    n_plot : int, default=50
        Number of ellipses to draw (subset of points).
    std : float, default=1
        Standard deviation scaling for ellipse size. Adjust manually for clarity.
    alpha : float, default=0.1
        Transparency of ellipses.
    title : str, default="Riemannian metric"
        Plot title.
    title_fontsize : int, default=10
        Title font size.
    labels : array-like of shape (n_samples,), optional
        Per-point labels mapped to colors via cmap.
    pt_size : float, default=1
        Marker size for points.
    cmap : str, default="Spectral"
        Colormap for labels.
    figsize : tuple of (float, float), default=(8, 8)
        Figure width and height.
    random_state : int or RandomState, optional
        RNG seed for sampling points.
    **kwargs : dict
        Extra arguments forwarded to matplotlib.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    Computing the Riemannian metric may require O(n_samples²) memory if all-pairs distances are
    materialized. For large datasets, use landmark mode or avoid this method.

    References
    ----------
    Perraul-Joncas, D., & Meila, M. (2013).
    Non-linear dimensionality reduction: Riemannian metric estimation and the problem of geometric discovery.
    arXiv preprint arXiv:1305.7255.
    """
    emb = np.asarray(emb)

    if H_emb is None:
        from topo.eval import RiemannMetric

        rmetric = RiemannMetric(emb, L)
        H = np.asarray(rmetric.get_dual_rmetric())
    else:
        H = np.asarray(H_emb)

    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError("emb must be a 2-D array with at least two columns.")
    if H.ndim != 3 or H.shape[0] != emb.shape[0] or H.shape[1] < 2 or H.shape[2] < 2:
        raise ValueError("H_emb must have shape (n_samples, >=2, >=2).")
    N = emb.shape[0]
    n_plot = min(int(n_plot), N)
    rng = check_random_state(random_state)
    sample_points = rng.choice(np.arange(N), n_plot, replace=False)
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(
        1
    )  # if an ellipse is a circle no distortion occured in particular directions
    if labels is not None:
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != emb.shape[0]:
            raise ValueError("labels must have the same length as emb.")
        _, label_codes = np.unique(labels_arr, return_inverse=True)
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, int(label_codes.max()) + 1))
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size, c=labels_arr, cmap=cmap)
    else:
        labels_arr = None
        label_codes = None
        colors = None
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size)

    for i in range(n_plot):
        ii = sample_points[i]
        cov = H[ii, :2, :2]

        if labels_arr is not None and colors is not None and label_codes is not None:
            plot_cov_ellipse(
                cov,
                emb[ii, :],
                nstd=std,
                ax=ax,
                edgecolor="none",
                color=colors[label_codes[ii]],
                alpha=alpha,
            )
        else:
            plot_cov_ellipse(
                cov,
                emb[ii, :],
                nstd=std,
                ax=ax,
                edgecolor="none",
                alpha=alpha,
            )
    return ax


def draw_edges(ax, data, kernel, color="black", **kwargs):
    """Draw graph edges between points, with alpha proportional to kernel affinity."""
    data = _as_2d_array(data, "data")
    _check_n_columns(data, 2, "data")
    kernel = np.asarray(kernel)
    if kernel.shape[0] != data.shape[0] or kernel.shape[1] != data.shape[0]:
        raise ValueError("kernel must have shape (n_samples, n_samples).")
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            affinity = kernel[i, j]
            if affinity > 0:
                ax.plot(
                    data[[i, j], 0],
                    data[[i, j], 1],
                    color=color,
                    alpha=affinity,
                    zorder=0,
                    **kwargs,
                )


def plot_scores(
    scores, return_plot=True, log=True, figsize=(8, 3), fontsize=12, title="Scores"
):
    """Bar-plot a dictionary of named scores, optionally on a log scale."""
    keys = list(scores.keys())
    values = list(scores.values())
    cmap = get_cmap(len(keys), name="tab20")
    k_color = [cmap(k) for k in range(len(keys))]
    x = np.arange(len(keys))
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title, fontsize=round(fontsize * 1.5))
    ax1.bar(x, values, color=k_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, fontsize=fontsize, rotation=90)
    if log:
        ax1.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_all_scores(evaluation_dict, log=False, figsize=(20, 8), fontsize=20):
    """Bar-plot each score group in an evaluation dictionary as a separate figure."""
    for key, value in evaluation_dict.items():
        plot_scores(value, figsize=figsize, log=log, fontsize=fontsize, title=key)


def plot_eigenvectors(
    eigenvectors,
    n_eigenvectors=10,
    labels=None,
    cmap="tab20",
    figsize=(23, 2),
    fontsize=10,
    title="DC",
    orientation="horizontal",  # "horizontal" (1 row) or "vertical" (stacked rows)
    row_height=0.8,  # inches per row when orientation="vertical"
    width=8.0,  # figure width (inches) for vertical layout
    marker_base=6,  # base marker size; auto-scales in vertical mode
    **kwargs,
):
    """Plot the first ``n_eigenvectors`` diffusion components as scatter strips.

    Lays out one panel per component, either horizontally or as vertically
    stacked slender rows (``orientation``).
    """
    X = np.asarray(eigenvectors)
    if X.ndim != 2:
        raise ValueError("eigenvectors must be a 2-D array.")
    n, m = X.shape
    k = int(min(n_eigenvectors, m))
    if k < 1:
        raise ValueError("n_eigenvectors must select at least one eigenvector.")

    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")

    if orientation == "horizontal":
        # original-style strip of columns
        fig, axes = plt.subplots(1, k, figsize=figsize, constrained_layout=False)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.set_title(f"{title} {i + 1}", fontsize=fontsize)
            ax.scatter(
                np.arange(n), X[:, i], c=labels, cmap=cmap, s=marker_base, **kwargs
            )
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.05)
        return fig

    # --- Vertical (stacked) slender composition ---
    # Compute a sane figure size: width × (rows * row_height)
    fig_h = max(2.5, k * float(row_height))
    fig_w = float(width)
    fig, axes = plt.subplots(
        k,
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw=dict(hspace=0.05),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    # Auto marker size so points don’t look gigantic when rows are slender
    # heuristic: smaller points when many rows or many samples
    s = kwargs.pop("s", None)
    if s is None:
        s = max(1.0, marker_base * (row_height / 0.8) * (3000 / max(300.0, n)))

    # common x for all rows
    x = np.arange(n)

    for i, ax in enumerate(axes):
        ax.scatter(x, X[:, i], c=labels, cmap=cmap, s=s, **kwargs)
        # a slim, clean strip
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # put a compact title label at the left of the strip
        ax.text(
            -0.01,
            0.5,
            f"{title} {i + 1}",
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=fontsize,
        )

    axes[-1].set_xticks([])  # keep minimalist look; add ticks if you want
    # slim margins; no tight_layout to avoid warnings
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.04)
    return fig


def plot_dimensionality_histograms(
    local_id_dict,
    global_id_dict,
    bins: int | Sequence[float] | str | None = 50,
    title="FSA",
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    stacked=True,
    density=True,
    log=False,
    title_fontsize=22,
    legend_fontsize=15,
):
    """Overlay per-``k`` local intrinsic-dimension histograms with global estimates."""
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(6)
    fig.set_figheight(8)
    for key in local_id_dict.keys():
        x = local_id_dict[key]
        #
        # Make a multiple-histogram of data-sets with different length.
        label = (
            "k = "
            + str(key)
            + "    ( estim.i.d. = "
            + str(int(global_id_dict[key]))
            + " )"
        )

        _n, _bins, _patches = ax.hist(
            x,
            bins=bins,
            histtype=histtype,
            stacked=stacked,
            density=density,
            log=log,
            label=label,
        )
    ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.legend(prop={"size": 12}, fontsize=legend_fontsize)
    ax.set_xlabel("Estimated intrinsic dimension", fontsize=legend_fontsize)
    ax.set_ylabel("Frequency", fontsize=legend_fontsize)
    ax.legend(prop={"size": 10})
    return fig


def plot_dimensionality_histograms_multiple(
    id_dict,
    bins: int | Sequence[float] | str | None = 50,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    stacked=True,
    density=True,
    log=False,
    title="I.D. estimates",
):
    """Overlay intrinsic-dimension histograms for multiple estimates on shared bins."""
    fig, ax = plt.subplots(1, 1)
    # data
    current_bins: int | Sequence[float] | str | None = bins
    for key in id_dict.keys():
        x = id_dict[key]
        #
        # Make a multiple-histogram of data-sets with different length.
        _n, _bins, _patches = ax.hist(
            x,
            bins=current_bins,
            histtype=histtype,
            stacked=stacked,
            density=density,
            log=log,
            label=key,
        )

        current_bins = cast(Sequence[float], _bins.tolist())
    ax.set_title(title)
    ax.legend(prop={"size": 10})
    fig.tight_layout()
    return fig


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    cbar_fontsize=12,
    shrink=0.6,
    cb_pad=0.3,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    data = _as_2d_array(data, "data")
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=shrink, pad=cb_pad, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=cbar_fontsize)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    an_fontsize=8,
    **textkw,
):
    """Annotate a heatmap with its per-cell values.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **textkw
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if data is None:
        data_arr = np.asarray(im.get_array())
    else:
        data_arr = np.asarray(data)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data_arr.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        from matplotlib import ticker

        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            kw.update(color=textcolors[int(im.norm(data_arr[i, j]) > threshold)])
            text = im.axes.text(
                j, i, valfmt(data_arr[i, j], None), fontsize=an_fontsize, **kw
            )
            texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# MAP training GIF
# ---------------------------------------------------------------------------


def visualize_optimization(
    snapshots,
    *,
    dpi=120,
    color=None,
    filename=None,
    point_size=3.0,
    fps=20,
    tag="msTopoMAP",
    overlay_metrics=False,
):
    """
    Produce an animated GIF from a list of MAP training snapshots.

    Parameters
    ----------
    snapshots : list[dict]
        Each dict must have an ``'embedding'`` key (ndarray, shape (n, 2)) and
        optionally ``'epoch'`` (int) and ``'metrics'`` (dict with PF1/PJS/SP/TP).
    dpi : int
        Figure DPI.
    color : None, array-like, or single color
        Per-point coloring (see TopOGraph.visualize_optimization for details).
    filename : str or None
        Output GIF path.  Auto-generated if None.
    point_size : float
        Scatter marker size.
    fps : int
        Frames per second.
    tag : str
        Label used in title and default filename.
    overlay_metrics : bool
        Draw metric values on each frame if present.

    Returns
    -------
    str
        Path to the generated GIF.
    """
    import time as _time

    import matplotlib.colors as mcolors

    if not snapshots:
        raise RuntimeError("No snapshots provided.")

    snapshots = sorted(snapshots, key=lambda s: int(s.get("epoch", 0)))
    n = snapshots[-1]["embedding"].shape[0]

    def _to_rgba_array(c, n):
        if c is None:
            return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))
        if isinstance(c, (str, tuple)):
            rgba = np.array(mcolors.to_rgba(c), float)
            return np.tile(rgba[None, :], (n, 1))
        c = np.asarray(c)
        if c.ndim == 1:
            if c.shape[0] == n and np.issubdtype(c.dtype, np.number):
                cmap = plt.get_cmap("viridis")
                vmin, vmax = np.nanmin(c), np.nanmax(c)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
                t = (c - vmin) / (vmax - vmin + 1e-12)
                return cmap(np.clip(t, 0, 1))
            elif c.shape[0] == n:
                return np.array([mcolors.to_rgba(ci) for ci in c], float)
            return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))
        if c.ndim == 2 and c.shape[0] == n and c.shape[1] in (3, 4):
            if c.shape[1] == 3:
                return np.concatenate([c, np.ones((n, 1))], axis=1)
            return c.astype(float)
        return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))

    point_colors = _to_rgba_array(color, n)

    all_coords = np.concatenate([s["embedding"] for s in snapshots], axis=0)
    x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    frames = []
    fig_w, fig_h = 6, 5
    for snap in snapshots:
        Y = snap["embedding"]
        epoch = int(snap.get("epoch", 0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=int(dpi))
        ax.scatter(Y[:, 0], Y[:, 1], s=float(point_size), c=point_colors, linewidths=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("TopoMAP_1")
        ax.set_ylabel("TopoMAP_2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{tag} training — epoch {epoch}")

        if overlay_metrics:
            m = snap.get("metrics", None)
            if isinstance(m, dict) and ("PF1" in m or "TP" in m):
                txt = []
                for key in ("PF1", "PJS", "SP", "TP"):
                    if key in m:
                        txt.append(f"{key} {m[key]:.3f}")
                ax.text(
                    0.98,
                    0.02,
                    "\n".join(txt),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                        boxstyle="round,pad=0.25",
                    ),
                )

        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.92)
        fig.canvas.draw()
        canvas = cast(Any, fig.canvas)
        w, h = canvas.get_width_height()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = rgba[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    if filename is None:
        filename = f"{tag}_training_{int(_time.time())}.gif"

    from PIL import Image

    pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / max(1, int(fps))),
        disposal=2,
        optimize=False,
    )
    return filename
