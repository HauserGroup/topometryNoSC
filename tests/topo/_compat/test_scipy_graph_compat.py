"""Tests for SciPy graph compatibility layer."""

import numpy as np
from scipy.sparse import csr_matrix

from topo._compat.scipy_graph import (
    graph_connected_components,
    graph_laplacian,
    graph_shortest_paths,
)


def test_graph_laplacian_handles_isolated_node():
    """Test that Laplacian handles isolated nodes without inf/nan."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for kind in ["unnormalized", "normalized", "random_walk"]:
        L = graph_laplacian(W, laplacian_type=kind)
        assert np.isfinite(L.toarray()).all()


def test_connected_components_wrapper():
    """Test connected components wrapper."""
    W = csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    n_components, labels = graph_connected_components(W, directed=False)
    assert n_components == 2
    assert labels.shape == (3,)


def test_shortest_paths_wrapper_indices():
    """Test shortest paths with specific indices."""
    W = csr_matrix([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    D = graph_shortest_paths(W, directed=False, indices=[0])
    assert D.shape == (1, 3)
    np.testing.assert_allclose(D[0], [0.0, 2.0, 5.0])


def test_graph_laplacian_return_degree():
    """Test that Laplacian returns degree matrix when requested."""
    W = csr_matrix([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    L, D = graph_laplacian(W, laplacian_type="unnormalized", return_D=True)
    assert L.shape == W.shape
    assert D.shape == W.shape
    # Check degree values
    expected_deg = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(D.diagonal(), expected_deg)
