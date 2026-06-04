"""Tests for layout projectors and embeddings."""

import numpy as np

from topo.layouts.map import fuzzy_embedding
from topo.layouts.projector import Projector
from topo.tpgraph.kernels import compute_kernel


def test_projector_map(swiss_roll_data):
    X, _ = swiss_roll_data
    proj = Projector(projection_method="MAP", n_components=2, num_iters=10)
    Y = proj.fit_transform(X)
    assert Y.shape == (X.shape[0], 2)
    assert np.isfinite(Y).all()


def test_projector_tsne(swiss_roll_data):
    X, _ = swiss_roll_data
    proj = Projector(
        projection_method="t-SNE",
        n_components=2,
        num_iters=250,
        n_neighbors=100,
    )
    Y = proj.fit_transform(X)
    assert Y.shape == (X.shape[0], 2)


def test_fuzzy_embedding(swiss_roll_data):
    X, _ = swiss_roll_data
    K = compute_kernel(X, n_neighbors=15)
    Y, Y_aux = fuzzy_embedding(K, n_components=2, n_epochs=10)
    assert Y.shape == (X.shape[0], 2)
