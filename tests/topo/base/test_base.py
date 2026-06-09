"""Tests for low-level distance and neighbor graph helpers."""

import numpy as np
import pytest
from scipy import sparse

from topo.base import ann


class TestANNHelpers:
    def test_neighbor_count_validation(self):
        assert ann._validate_n_neighbors(3) == 3
        assert ann._query_k(3, 3) == 3
        with pytest.raises(ValueError, match="n_neighbors"):
            ann._validate_n_neighbors(0)
        with pytest.raises(ValueError, match="empty index"):
            ann._query_k(1, 0)

    def test_build_sparse_knn_graph(self):
        graph = ann._build_sparse_knn_graph(
            indices=np.array([[0, 1], [1, 0]]),
            distances=np.array([[0.0, 2.0], [0.0, 2.0]]),
            n_query_samples=2,
            n_fit_samples=2,
        )

        assert graph.shape == (2, 2)
        np.testing.assert_allclose(graph.toarray(), [[0.0, 2.0], [2.0, 0.0]])
        with pytest.raises(ValueError, match="same shape"):
            ann._build_sparse_knn_graph(np.array([[0]]), np.array([[0, 1]]), 1, 2)

    def test_backend_metric_mapping(self):
        assert ann._hnswlib_space("cosine") == "cosine"
        with pytest.raises(ValueError, match="Unsupported HNSWlib"):
            ann._hnswlib_space("bad")

    def test_knn_sklearn_backend_and_y_fallback(self):
        X = np.array([[0.0], [1.0], [3.0], [10.0]])
        Y = np.array([[2.0], [9.0]])

        graph = ann.kNN(X, n_neighbors=2, backend="sklearn")
        assert sparse.isspmatrix_csr(graph)
        assert graph.shape == (4, 4)
        assert graph.getnnz(axis=1).tolist() == [2, 2, 2, 2]

        with pytest.warns(UserWarning, match="Falling back to sklearn"):
            query_graph = ann.kNN(X, Y=Y, n_neighbors=1, backend="hnswlib")
        assert sparse.isspmatrix_csr(query_graph)
        assert query_graph.shape == (2, 4)
