"""Tests for layout utility modules."""

import numpy as np
from scipy import sparse

from topo.layouts import graph_utils
from topo.layouts.isomap import Isomap
from topo.tpgraph import fuzzy


def test_make_epochs_per_sample_handles_zero_weight_edges():
    weights = np.array([1.0, 0.5, 0.0])
    epochs = graph_utils.make_epochs_per_sample(weights, n_epochs=10)

    np.testing.assert_allclose(epochs[:2], [1.0, 2.0])
    assert epochs[2] == -1.0


def test_find_ab_params_returns_positive_curve_parameters():
    a, b = graph_utils.find_ab_params(spread=1.0, min_dist=0.1)

    assert a > 0
    assert b > 0


def test_simplicial_set_embedding_tracks_checkpoints():
    graph = sparse.coo_matrix(
        np.array(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
    )
    init = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)

    emb, aux = graph_utils.simplicial_set_embedding(
        graph,
        n_components=2,
        initial_alpha=0.1,
        a=1.0,
        b=1.0,
        gamma=1.0,
        negative_sample_rate=1,
        n_epochs=2,
        init=init,
        random_state=np.random.RandomState(0),
        metric="euclidean",
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        parallel=False,
        save_every=1,
    )

    assert emb.shape == (3, 2)
    assert "checkpoints" in aux
    assert aux["checkpoints"][0]["epoch"] == 0


def test_isomap_accepts_precomputed_graph():
    graph = sparse.csr_matrix(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0, 0.0],
        ]
    )

    emb = Isomap(graph, n_components=2, metric="precomputed", n_neighbors=2)

    assert emb.shape[0] == 4
    assert emb.shape[1] <= 2
    assert np.isfinite(emb).all()


class TestFuzzyGraphHelpers:
    def test_smooth_knn_dist_returns_positive_sigmas_and_rhos(self):
        dists = np.array(
            [[0.0, 0.5, 1.0], [0.0, 0.25, 0.75], [0.0, 0.3, 0.9]],
            dtype=np.float32,
        )

        sigmas, rhos = fuzzy.smooth_knn_dist(dists, k=3, n_iter=8)

        assert sigmas.shape == (3,)
        assert rhos.shape == (3,)
        assert np.all(sigmas > 0)
        np.testing.assert_allclose(rhos, [0.5, 0.25, 0.3])

    def test_compute_membership_strengths_sets_self_edges_to_zero(self):
        knn_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
        knn_dists = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        sigmas = np.array([1.0, 1.0], dtype=np.float32)
        rhos = np.array([0.1, 0.1], dtype=np.float32)

        rows, cols, vals = fuzzy.compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos
        )

        np.testing.assert_array_equal(rows, [0, 0, 1, 1])
        np.testing.assert_array_equal(cols, [0, 1, 1, 0])
        np.testing.assert_allclose(vals[[0, 2]], [0.0, 0.0])
        assert np.all(vals[[1, 3]] > 0)

    def test_fuzzy_simplicial_set_precomputed_returns_distances(self):
        graph = sparse.csr_matrix(
            [
                [0.0, 0.5, 1.0],
                [0.5, 0.0, 0.75],
                [1.0, 0.75, 0.0],
            ]
        )

        fss, sigmas, rhos, dists = fuzzy.fuzzy_simplicial_set(
            graph,
            n_neighbors=2,
            metric="precomputed",
            return_dists=True,
            apply_set_operations=True,
        )

        assert fss.shape == (3, 3)
        assert sigmas.shape == (3,)
        assert rhos.shape == (3,)
        assert dists.shape == (3, 2)
