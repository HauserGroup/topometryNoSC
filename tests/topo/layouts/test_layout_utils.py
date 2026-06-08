"""Tests for layout utility modules."""

import numpy as np
from scipy import sparse

from topo.layouts import graph_utils
from topo.layouts.isomap import Isomap


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

    emb = Isomap(n_components=2, metric="precomputed", n_neighbors=1).fit_transform(
        graph
    )

    assert emb.shape[0] == 4
    assert emb.shape[1] <= 2
    assert np.isfinite(emb).all()
