"""Tests for low-level distance, sparse, and neighbor graph helpers."""

import math

import numpy as np
import pytest
from scipy import sparse

from topo.base import ann, dists
from topo.base import sparse as sparse_dist


class TestDistanceHelpers:
    def test_vector_distances_and_gradients(self):
        x = np.array([1.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)

        assert dists.euclidean(x, y) == pytest.approx(math.sqrt(2.0))
        dist, grad = dists.euclidean_grad(x, y)
        assert dist == pytest.approx(math.sqrt(2.0))
        np.testing.assert_allclose(grad, np.array([1.0, -1.0]) / math.sqrt(2.0))

        assert dists.cosine(x, y) == pytest.approx(1.0)
        cos_dist, cos_grad = dists.cosine_grad(x, y)
        assert cos_dist == pytest.approx(1.0)
        assert cos_grad.shape == x.shape

    def test_poincare_distance_is_symmetric_for_points_inside_ball(self):
        x = np.array([0.1, 0.2], dtype=np.float32)
        y = np.array([0.2, -0.1], dtype=np.float32)

        assert dists.poincare(x, y) == pytest.approx(dists.poincare(y, x))
        dist, grad = dists.poincare_grad(x, y)
        assert dist > 0
        assert grad.shape == x.shape

    def test_pairwise_distance_dispatchers(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        Y = np.array([[0.0, 4.0]], dtype=np.float32)

        np.testing.assert_allclose(dists.pairwise_euclidean(X, Y), [[4.0], [3.0]])
        np.testing.assert_allclose(dists.pairwise_cosine(X, X)[0], [0.0, 1.0])
        np.testing.assert_allclose(
            dists.matrix_pairwise_distance(X, "euclidean"),
            dists.pairwise_distances(X, metric="euclidean"),
        )
        np.testing.assert_allclose(
            dists.matrix_to_matrix_distance(X, Y, "euclidean"),
            dists.pairwise_distances(X, Y, metric="euclidean"),
        )
        with pytest.raises(ValueError, match="Unknown metric"):
            dists.pairwise_distances(X, metric="not-a-metric")

    def test_cosine_vector_to_matrix_matches_pairwise_row(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = dists.cosine_vector_to_matrix(X[0], X)
        np.testing.assert_allclose(out, [0.0, 1.0])


class TestSparseDistanceHelpers:
    def test_index_set_operations(self):
        a = np.array([2, 1, 3], dtype=np.int64)
        b = np.array([3, 4], dtype=np.int64)

        np.testing.assert_array_equal(sparse_dist.arr_unique(a), [1, 2, 3])
        np.testing.assert_array_equal(sparse_dist.arr_union(a, b), [1, 2, 3, 4])
        np.testing.assert_array_equal(sparse_dist.arr_intersect(a, b), [3])

    def test_sparse_arithmetic(self):
        ind1 = np.array([0, 2], dtype=np.int64)
        data1 = np.array([1.0, 2.0], dtype=np.float64)
        ind2 = np.array([1, 2], dtype=np.int64)
        data2 = np.array([3.0, 4.0], dtype=np.float64)

        sum_ind, sum_data = sparse_dist.sparse_sum(ind1, data1, ind2, data2)
        np.testing.assert_array_equal(sum_ind, [0, 1, 2])
        np.testing.assert_allclose(sum_data, [1.0, 3.0, 6.0])

        diff_ind, diff_data = sparse_dist.sparse_diff(ind1, data1, ind2, data2)
        np.testing.assert_array_equal(diff_ind, [0, 1, 2])
        np.testing.assert_allclose(diff_data, [1.0, -3.0, -2.0])

        mul_ind, mul_data = sparse_dist.sparse_mul(ind1, data1, ind2, data2)
        np.testing.assert_array_equal(mul_ind, [2])
        np.testing.assert_allclose(mul_data, [8.0])

    def test_sparse_metric_family(self):
        ind1 = np.array([0, 2], dtype=np.int64)
        data1 = np.array([1.0, 2.0], dtype=np.float64)
        ind2 = np.array([1, 2], dtype=np.int64)
        data2 = np.array([3.0, 4.0], dtype=np.float64)

        assert sparse_dist.sparse_euclidean(ind1, data1, ind2, data2) == pytest.approx(
            math.sqrt(14.0)
        )
        assert sparse_dist.sparse_manhattan(ind1, data1, ind2, data2) == pytest.approx(
            6.0
        )
        assert sparse_dist.sparse_chebyshev(ind1, data1, ind2, data2) == pytest.approx(
            3.0
        )
        assert sparse_dist.sparse_minkowski(
            ind1, data1, ind2, data2, 3.0
        ) == pytest.approx((1.0 + 27.0 + 8.0) ** (1.0 / 3.0))
        assert sparse_dist.sparse_cosine(ind1, data1, ind2, data2) == pytest.approx(
            1.0 - 8.0 / (math.sqrt(5.0) * 5.0)
        )
        assert (
            sparse_dist.sparse_poincare(
                np.array([0], dtype=np.int64),
                np.array([0.1], dtype=np.float64),
                np.array([0], dtype=np.int64),
                np.array([0.2], dtype=np.float64),
            )
            > 0
        )

    def test_sparse_binary_and_probability_metrics_are_bounded(self):
        ind1 = np.array([0, 2], dtype=np.int64)
        data1 = np.array([1.0, 2.0], dtype=np.float64)
        ind2 = np.array([1, 2], dtype=np.int64)
        data2 = np.array([3.0, 4.0], dtype=np.float64)

        metrics = [
            sparse_dist.sparse_hamming(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_canberra(ind1, data1, ind2, data2),
            sparse_dist.sparse_bray_curtis(ind1, data1, ind2, data2),
            sparse_dist.sparse_jaccard(ind1, data1, ind2, data2),
            sparse_dist.sparse_matching(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_dice(ind1, data1, ind2, data2),
            sparse_dist.sparse_kulsinski(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_rogers_tanimoto(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_russellrao(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_sokal_michener(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_sokal_sneath(ind1, data1, ind2, data2),
            sparse_dist.sparse_hellinger(ind1, data1, ind2, data2),
            sparse_dist.sparse_correlation(ind1, data1, ind2, data2, 5),
            sparse_dist.sparse_ll_dirichlet(ind1, data1, ind2, data2),
        ]
        assert np.isfinite(metrics).all()


class TestANNHelpers:
    def test_neighbor_count_validation(self):
        assert ann._validate_n_neighbors(3) == 3
        assert ann._query_k(3, 3) == 3
        with pytest.raises(ValueError, match="n_neighbors"):
            ann._validate_n_neighbors(0)
        with pytest.raises(ValueError, match="empty index"):
            ann._query_k(1, 0)

    def test_data_conversion_and_shape_validation(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        assert ann._as_dense_array(X).shape == (2, 2)
        assert ann._as_csr_matrix(X).format == "csr"
        ann._check_2d_data(np.asarray(X), "X")
        with pytest.raises(ValueError, match="2-D"):
            ann._check_2d_data(np.array([1.0, 2.0]), "bad")

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
        assert ann._nmslib_sparse_space("euclidean") == "l2_sparse"
        assert ann._nmslib_dense_space("inner_product") == "negdotprod"
        assert ann._hnswlib_space("cosine") == "cosine"
        with pytest.raises(ValueError, match="Unsupported NMSlib sparse"):
            ann._nmslib_sparse_space("bad")
        with pytest.raises(ValueError, match="Unsupported NMSlib dense"):
            ann._nmslib_dense_space("bad")
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

    def test_transformers_fail_helpfully_when_not_fitted_or_missing_backend(self):
        X = np.array([[0.0], [1.0], [2.0]])

        hnsw = ann.HNSWlibTransformer(n_neighbors=1)
        with pytest.raises(ValueError, match="not fitted"):
            hnsw.transform(X)
        hnsw.update_search(2)
        assert hnsw.n_neighbors == 2

        nms = ann.NMSlibTransformer(n_neighbors=1)
        with pytest.raises(NotImplementedError, match="return_grad=True"):
            nms.ind_dist_grad(X, return_grad=True)
