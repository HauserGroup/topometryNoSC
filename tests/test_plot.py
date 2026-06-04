"""Tests for plotting utilities."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use headless backend for CI compatibility
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from topo import plot
from topo.tpgraph.intrinsic_dim import IntrinsicDim


def test_decay_plot(fitted_topograph):
    fig = plot.decay_plot(fitted_topograph.eigenvalues)
    assert fig is not None
    plt.close(fig)


def test_scatter(fitted_topograph):
    fig = plot.scatter(fitted_topograph.msTopoMAP)
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
