"""Tests for plotting utilities."""

import matplotlib

matplotlib.use("Agg")  # Use headless backend for CI compatibility
import matplotlib.pyplot as plt

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
    plt.close(ax.figure)


def test_plot_dimensionality_histograms(swiss_roll_data):
    X, _ = swiss_roll_data
    id_est = IntrinsicDim(k=[10], plot=False, backend="sklearn")
    id_est.fit(X)
    fig = plot.plot_dimensionality_histograms(
        id_est.local_id["fsa"], id_est.global_id["fsa"]
    )
    assert fig is not None
    plt.close(fig)
