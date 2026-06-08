"""Tests for utility modules."""

import re

import numpy as np
import pytest
from scipy import sparse

from topo.base.graph_matrix import (
    get_indices_distances_from_sparse_matrix,
    get_sparse_matrix_from_indices_distances,
)
from topo.utils import _utils, map_utils


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
        shape = X.shape
        assert shape is not None
        assert set(idx).issubset(set(range(shape[0])))

    def test_kmeans_landmark_indices_accept_sparse_input(self):
        X = sparse.csr_matrix([[0.0], [0.1], [10.0], [10.1]])
        idx = _utils.get_landmark_indices(
            X, n_landmarks=2, method="kmeans", random_state=0, batch_size=4
        )

        assert idx.shape == (2,)
        shape = X.shape
        assert shape is not None
        assert set(idx).issubset(set(range(shape[0])))

    def test_landmark_method_validation(self):
        with pytest.raises(ValueError, match="Unknown landmark"):
            _utils.get_landmark_indices(np.ones((4, 2)), method="bad")

    def test_sparse_knn_matrix_roundtrip_helpers(self):
        indices = np.array([[0, 1], [1, 2], [2, 1]])
        dists = np.array([[0.0, 0.5], [0.0, 0.25], [0.0, 0.25]])

        graph = get_sparse_matrix_from_indices_distances(
            indices, dists, n_obs=3, n_neighbors=2
        )
        out_idx, out_dist = get_indices_distances_from_sparse_matrix(
            graph, n_neighbors=1
        )

        assert graph.shape == (3, 3)
        np.testing.assert_array_equal(out_idx.ravel(), [1, 2, 1])
        np.testing.assert_allclose(out_dist.ravel(), [0.5, 0.25, 0.25])

    def test_sparse_knn_matrix_requires_enough_neighbors(self):
        graph = sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="fewer than n_neighbors"):
            get_indices_distances_from_sparse_matrix(graph, n_neighbors=2)


class TestUmapUtilities:
    def test_gaussian_density_helpers(self):
        x = np.array([0.0, 0.0], dtype=np.float32)
        pos = np.array([0.0, 0.0], dtype=np.float32)
        cov = np.eye(2, dtype=np.float32)

        assert map_utils.eval_gaussian(x, pos, cov) == pytest.approx(
            1.0 / (2.0 * np.pi)
        )

        embedding = np.array([[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        density = map_utils.eval_density_at_point(x, embedding)

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

        Z = map_utils.create_density_plot(X, Y, embedding)
        assert Z.shape == X.shape
        assert Z.sum() == pytest.approx(1.0)

    def test_torus_gradient_wraps_across_boundary(self):
        dist, grad = map_utils.torus_euclidean_grad(
            np.array([0.1, 0.0]),
            np.array([2 * np.pi - 0.1, 0.0]),
            torus_dimensions=(2 * np.pi, 2 * np.pi),
        )

        assert dist == pytest.approx(0.2, rel=1e-5)
        assert grad[0] > 0

    def test_fast_knn_submatrix_and_norm(self):
        dmat = np.array([[0.3, 0.1, 0.2], [0.0, 2.0, 1.0]], dtype=np.float32)
        idx = map_utils.fast_knn_indices(dmat, 2)
        sub = map_utils.submatrix(dmat, idx, 2)

        np.testing.assert_array_equal(idx, [[1, 2], [0, 2]])
        np.testing.assert_allclose(sub, [[0.1, 0.2], [0.0, 1.0]])
        assert map_utils.norm(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_tau_random_updates_state_and_timestamp_is_string(self):
        state = np.array([1, 2, 3], dtype=np.int64)
        first = map_utils.tau_rand_int(state)
        second = map_utils.tau_rand(state)

        assert isinstance(int(first), int)
        assert 0.0 <= second <= 1.0
        assert re.match(r"\w{3} ", map_utils.ts())
