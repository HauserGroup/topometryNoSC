"""Tests for scipy/sklearn-delegated metric primitives."""

from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from topo.eval.topo_metrics import rowwise_js_similarity, spectral_procrustes


def test_rowwise_js_similarity_identical_operators():
    """Test JS similarity when operators are identical."""
    P = csr_matrix([[0.5, 0.5], [0.2, 0.8]])
    sim = cast(float, rowwise_js_similarity(P, P))

    assert np.isclose(sim, 1.0)


def test_rowwise_js_similarity_different_operators():
    """Test JS similarity with different operators."""
    P1 = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    P2 = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    sim = cast(float, rowwise_js_similarity(P1, P2))

    assert 0.0 <= sim <= 1.0


def test_rowwise_js_similarity_sparse_matrices():
    """Test JS similarity with sparse matrices."""
    P1 = csr_matrix([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
    P2 = csr_matrix([[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]])
    sim = cast(float, rowwise_js_similarity(P1, P2))

    assert 0.0 <= sim <= 1.0
    assert np.isfinite(sim)


def test_rowwise_js_similarity_return_per_row():
    """Test JS similarity with per-row scores."""
    P1 = csr_matrix([[0.5, 0.5], [0.2, 0.8]])
    P2 = csr_matrix([[0.6, 0.4], [0.3, 0.7]])
    result = rowwise_js_similarity(P1, P2, return_per_row=True)
    sim, per_row = cast(tuple[float, Any], result)

    assert np.isclose(sim, np.mean(per_row))
    assert per_row.shape == (2,)
    assert np.all((per_row >= 0.0) & (per_row <= 1.0))


def test_spectral_procrustes_identical_operators():
    """Test spectral Procrustes alignment of identical operators."""
    from scipy.sparse import identity

    n = 50
    P = 0.5 * identity(n) + 0.5 * csr_matrix(np.eye(n, k=1) + np.eye(n, k=-1)) / 2.0
    r2 = spectral_procrustes(P, P, r=20)

    assert np.isclose(r2, 1.0, atol=1e-4)


def test_spectral_procrustes_returns_valid_r2():
    """Test that Procrustes returns valid R^2."""
    from scipy.sparse import identity

    n = 50
    P1 = identity(n)
    P2 = 0.9 * identity(n) + 0.1 * csr_matrix(np.eye(n, k=1) + np.eye(n, k=-1)) / 2.0
    r2 = spectral_procrustes(P1, P2, r=20)

    assert 0.0 <= r2 <= 1.0
    assert np.isfinite(r2)
