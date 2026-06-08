"""Tests for SciPy graph operations compatibility layer."""

from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from topo._compat.scipy_graph import (
    graph_connected_components,
    graph_laplacian,
    graph_shortest_paths,
)


def test_graph_laplacian_unnormalized():
    """Test unnormalized Laplacian computation."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L = cast(csr_matrix, graph_laplacian(W, laplacian_type="unnormalized"))

    # Unnormalized Laplacian should be D - W
    assert L.shape == (3, 3)
    assert np.isfinite(L.toarray()).all()
    # Diagonal should be degrees
    assert cast(Any, L[0, 0]) == 1.0
    assert cast(Any, L[1, 1]) == 2.0


def test_graph_laplacian_normalized():
    """Test normalized Laplacian computation."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L = cast(csr_matrix, graph_laplacian(W, laplacian_type="normalized"))

    assert L.shape == (3, 3)
    assert np.isfinite(L.toarray()).all()


def test_graph_laplacian_random_walk():
    """Test random-walk Laplacian computation."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L = cast(csr_matrix, graph_laplacian(W, laplacian_type="random_walk"))

    assert L.shape == (3, 3)
    assert np.isfinite(L.toarray()).all()


def test_graph_laplacian_handles_isolated_node():
    """Test that isolated nodes don't create inf/nan."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for laplacian_type in ["unnormalized", "normalized", "random_walk"]:
        L = cast(csr_matrix, graph_laplacian(W, laplacian_type=laplacian_type))
        assert np.isfinite(L.toarray()).all()


def test_graph_laplacian_return_D():
    """Test that degree matrix is returned when requested."""
    W = csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    result = graph_laplacian(W, laplacian_type="unnormalized", return_D=True)
    L, D = cast(tuple[csr_matrix, csr_matrix], result)

    assert L.shape == (3, 3)
    assert D.shape == (3, 3)
    # D should be diagonal with degrees
    assert cast(Any, D[0, 0]) == 1.0
    assert cast(Any, D[1, 1]) == 2.0


def test_connected_components_wrapper():
    """Test connected components computation."""
    W = csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    n_components, labels = graph_connected_components(W, directed=False)

    assert n_components == 2
    assert labels.shape == (3,)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_shortest_paths_wrapper():
    """Test shortest path distances."""
    W = csr_matrix([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    D = graph_shortest_paths(W, directed=False)

    assert D.shape == (3, 3)


def test_shortest_paths_with_indices():
    """Test shortest paths from specific source nodes."""
    W = csr_matrix([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    D = graph_shortest_paths(W, directed=False, indices=[0])

    assert D.shape == (1, 3)
    np.testing.assert_allclose(D[0], [0.0, 2.0, 5.0])
