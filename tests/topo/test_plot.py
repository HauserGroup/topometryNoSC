"""Tests for plotting utilities."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use headless backend for CI compatibility
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from topo import plot
from topo.tpgraph.intrinsic_dim import IntrinsicDim


def test_decay_plot(fitted_topograph):
    fig, ax = plot.decay_plot(fitted_topograph.eigenvalues)
    assert fig is not None
    plt.close(fig)


def test_scatter(fitted_topograph):
    fig, ax = plot.scatter(fitted_topograph.msTopoMAP)
    assert fig is not None
    plt.close(fig)


def test_plot_riemann_metric(fitted_topograph):
    emb = fitted_topograph.msTopoMAP
    L = fitted_topograph.P_of_msZ
    ax = plot.plot_riemann_metric(emb, L=L, n_plot=5)
    assert ax is not None
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_dimensionality_histograms(swiss_roll_data):
    X, _ = swiss_roll_data
    id_est = IntrinsicDim(k=[10], plot=False, backend="sklearn")
    id_est.fit(X)
    fig = plot.plot_dimensionality_histograms(
        id_est.local_id["fsa"], id_est.global_id["fsa"]
    )
    assert fig is not None
    plt.close(fig)


def test_heatmap():
    data = np.random.rand(5, 5)
    row_labels = ["a", "b", "c", "d", "e"]
    col_labels = ["1", "2", "3", "4", "5"]
    im, cbar = plot.heatmap(data, row_labels, col_labels)
    assert im is not None
    assert cbar is not None


def test_plot_riemann_metric_accepts_more_than_two_embedding_columns():
    from scipy import sparse

    emb = np.random.default_rng(0).normal(size=(20, 3))
    H = np.tile(np.eye(2)[None, :, :], (20, 1, 1))
    L = sparse.eye(20)

    ax = plot.plot_riemann_metric(emb, L, H_emb=H, n_plot=5, random_state=0)
    assert ax is not None
    plt.close(ax.get_figure())


def test_eval_gaussian_matches_numpy_quadratic_form():
    x = np.array([1.0, 2.0])
    pos = np.array([0.5, -0.25])
    cov = np.array([[2.0, 0.4], [0.4, 1.5]])

    diff = x - pos
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    expected = np.exp(-0.5 * diff @ inv @ diff) / (2 * np.pi * np.sqrt(det))

    actual = plot.eval_gaussian(x, pos, cov)
    np.testing.assert_allclose(actual, expected)


def test_draw_edges_accepts_sparse_kernel():
    from scipy import sparse

    data = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    K = sparse.csr_matrix(
        [
            [0, 1, 0],
            [1, 0, 0.5],
            [0, 0.5, 0],
        ]
    )

    fig, ax = plt.subplots()
    plot.draw_edges(ax, data, K)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_decay_plot_handles_zero_gaps_without_error():
    evals = np.array([1.0, 0.5, 0.5, 0.1])
    fig, axes = plot.decay_plot(evals)
    assert fig is not None
    assert len(axes) == 2
    plt.close(fig)
