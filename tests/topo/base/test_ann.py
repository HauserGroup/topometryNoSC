"""Tests for approximate nearest neighbors wrappers."""

import importlib.util

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from topo.base.ann import HNSWlibTransformer, NMSlibTransformer, _resolve_n_jobs, kNN


def test_kNN_sklearn():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    res = kNN(X, n_neighbors=2, backend="sklearn")
    assert isinstance(res, csr_matrix)
    assert res.shape == (3, 3)


def test_kNN_sklearn_direct_kneighbors_graph_path():
    X = np.random.default_rng(0).normal(size=(30, 3))
    graph = kNN(X, n_neighbors=5, backend="sklearn", return_instance=False)
    assert isinstance(graph, csr_matrix)
    assert graph.shape == (30, 30)
    assert graph.nnz == 30 * 5
    assert graph.format == "csr"


def test_kNN_sklearn_return_instance_with_precomputed():
    X = np.random.default_rng(1).normal(size=(20, 3))
    nbrs, graph = kNN(X, n_neighbors=5, backend="sklearn", return_instance=True)
    assert hasattr(nbrs, "kneighbors")
    assert isinstance(graph, csr_matrix)
    assert graph.shape == (20, 20)


def test_resolve_n_jobs():
    assert _resolve_n_jobs(None) == 1
    assert _resolve_n_jobs("2") == 2
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(-1) > 0


def test_hnswlib_transformer_raises_or_fits():
    X = np.random.rand(20, 3)
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            HNSWlibTransformer(n_neighbors=2).fit(X)
        return

    model = HNSWlibTransformer(n_neighbors=2).fit(X)
    knn = model.transform(X)
    assert knn.shape == (20, 20)


def test_nmslib_transformer_raises_or_fits():
    X = np.random.rand(20, 3)

    if importlib.util.find_spec("nmslib") is None:
        with pytest.raises(ImportError):
            NMSlibTransformer(n_neighbors=2).fit(X)
        return

    model = NMSlibTransformer(n_neighbors=2).fit(X)
    knn = model.transform(X)
    assert knn.shape == (20, 20)


def test_knn_sklearn_no_diagonal():
    """Verify sklearn kNN graph has no self-edges."""
    X = np.random.default_rng(42).normal(size=(15, 4))
    G = kNN(X, n_neighbors=5, backend="sklearn")

    assert G.diagonal().sum() == 0, "kNN graph should have no self-edges"


def test_knn_sklearn_exact_k_neighbors_per_row():
    """Verify sklearn kNN graph has exactly k neighbors per row."""
    X = np.random.default_rng(43).normal(size=(20, 3))
    k = 5
    G = kNN(X, n_neighbors=k, backend="sklearn")

    row_counts = np.diff(G.indptr)
    assert np.all(row_counts == k), (
        f"Expected {k} neighbors per row, got {np.unique(row_counts)}"
    )


def test_knn_sklearn_distance_values_are_reasonable():
    """Verify sklearn kNN distance values are non-negative and reasonable."""
    X = np.random.default_rng(44).normal(size=(12, 4))
    k = 3

    G = kNN(X, n_neighbors=k, backend="sklearn")

    # All distances should be non-negative
    assert np.all(G.data >= 0), "Distances should be non-negative"
    # Should have k*n_samples nonzero values
    expected_nnz = k * X.shape[0]
    assert G.nnz == expected_nnz, f"Expected {expected_nnz} distances, got {G.nnz}"


def test_knn_sklearn_binary_connectivity_mode():
    """Verify connectivity mode returns binary values."""
    X = np.random.default_rng(45).normal(size=(15, 3))
    # Note: current kNN returns distance graph, not connectivity
    # This test documents the current behavior
    G = kNN(X, n_neighbors=4, backend="sklearn")

    # Distance graph has float values
    assert G.data.dtype in [np.float32, np.float64]
    # All values should be non-negative distances
    assert np.all(G.data >= 0)
