"""Tests for utility modules."""

import re

import numpy as np
import pytest
from scipy import sparse

from topo.utils import _utils, umap_utils


class TestLandmarkAndSparseMatrixUtilities:
    def test_random_landmark_indices_are_reproducible(self):
        X = np.arange(20).reshape(10, 2)

        idx1 = _utils.get_landmark_indices(X, n_landmarks=4, random_state=42)
        idx2 = _utils.get_landmark_indices(X, n_landmarks=4, random_state=42)

        np.testing.assert_array_equal(idx1, idx2)
        assert len(np.unique(idx1)) == 4

    def test_kmeans_landmark_indices_are_valid_rows(self):
        X = np.array([[0.0], [0.1], [10.0], [10.1]])
        idx = _utils.get_landmark_indices(
            X, n_landmarks=2, method="kmeans", random_state=0, batch_size=4
        )

        assert idx.shape == (2,)
        assert set(idx).issubset(set(range(X.shape[0])))

    def test_landmark_method_validation(self):
        with pytest.raises(ValueError, match="Unknown landmark"):
            _utils.get_landmark_indices(np.ones((4, 2)), method="bad")

    def test_sparse_knn_matrix_roundtrip_helpers(self):
        indices = np.array([[0, 1], [1, 2], [2, 1]])
        dists = np.array([[0.0, 0.5], [0.0, 0.25], [0.0, 0.25]])

        graph = _utils.get_sparse_matrix_from_indices_distances(
            indices, dists, n_obs=3, n_neighbors=2
        )
        out_idx, out_dist = _utils.get_indices_distances_from_sparse_matrix(
            graph, n_neighbors=1
        )

        assert graph.shape == (3, 3)
        np.testing.assert_array_equal(out_idx.ravel(), [1, 2, 1])
        np.testing.assert_allclose(out_dist.ravel(), [0.5, 0.25, 0.25])

    def test_sparse_knn_matrix_requires_enough_neighbors(self):
        graph = sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="fewer than n_neighbors"):
            _utils.get_indices_distances_from_sparse_matrix(graph, n_neighbors=2)


class TestUmapUtilities:
    def test_gaussian_density_helpers(self):
        cov = np.eye(2, dtype=np.float32)
        assert umap_utils.eval_gaussian(np.array([0.0, 0.0]), cov=cov) == pytest.approx(
            1.0 / (2.0 * np.pi)
        )

        embedding = np.array([[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        density = umap_utils.eval_density_at_point(np.array([0.0, 0.0]), embedding)
        assert density > 0

    def test_create_density_plot_normalizes_grid(self):
        X, Y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
        embedding = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.5, 0.5, 1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        Z = umap_utils.create_density_plot(X, Y, embedding)
        assert Z.shape == X.shape
        assert Z.sum() == pytest.approx(1.0)

    def test_torus_gradient_wraps_across_boundary(self):
        dist, grad = umap_utils.torus_euclidean_grad(
            np.array([0.1, 0.0]),
            np.array([2 * np.pi - 0.1, 0.0]),
            torus_dimensions=(2 * np.pi, 2 * np.pi),
        )

        assert dist == pytest.approx(0.2, rel=1e-5)
        assert grad[0] > 0

    def test_fast_knn_submatrix_and_norm(self):
        dmat = np.array([[0.3, 0.1, 0.2], [0.0, 2.0, 1.0]], dtype=np.float32)
        idx = umap_utils.fast_knn_indices(dmat, 2)
        sub = umap_utils.submatrix(dmat, idx, 2)

        np.testing.assert_array_equal(idx, [[1, 2], [0, 2]])
        np.testing.assert_allclose(sub, [[0.1, 0.2], [0.0, 1.0]])
        assert umap_utils.norm(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_tau_random_updates_state_and_timestamp_is_string(self):
        state = np.array([1, 2, 3], dtype=np.int64)
        first = umap_utils.tau_rand_int(state)
        second = umap_utils.tau_rand(state)

        assert isinstance(int(first), int)
        assert 0.0 <= second <= 1.0
        assert re.match(r"\w{3} ", umap_utils.ts())

    def test_csr_unique_reports_unique_sparse_rows(self):
        mat = sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        index, inverse, counts = umap_utils.csr_unique(mat)

        assert index.ndim == 1
        assert inverse.shape[0] == mat.shape[0]
        assert counts.sum() == inverse.size
