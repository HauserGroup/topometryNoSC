"""Tests for distance computations."""

import numpy as np
import pytest

from topo.base import dists


def test_euclidean():
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    assert dists.euclidean(x, y) == pytest.approx(5.0)
    dist, grad = dists.euclidean_grad(x, y)
    assert dist == pytest.approx(5.0)
    assert len(grad) == 2


def test_cosine():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert dists.cosine(x, y) == pytest.approx(1.0)
    dist, grad = dists.cosine_grad(x, y)
    assert dist == pytest.approx(1.0)
    assert len(grad) == 2


def test_poincare():
    x = np.array([0.1, 0.1])
    y = np.array([0.2, -0.1])
    dist = dists.poincare(x, y)
    assert np.isfinite(dist)


def test_pairwise_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    D = dists.pairwise_distances(X, metric="euclidean", n_jobs=1)
    assert D.shape == (2, 2)
    assert D[0, 1] == pytest.approx(5.0)
