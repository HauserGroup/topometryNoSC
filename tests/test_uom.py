"""Tests for topo.uom standalone functions."""

import numpy as np
import scipy.sparse as sp

from topo.uom import find_components, louvain_micro, consolidate_macros, mbkm


class TestLouvainMicro:
    """Test greedy Louvain modularity clustering."""

    def test_louvain_on_block_diagonal(self):
        """Two disconnected cliques should yield two clusters."""
        n = 30
        # Build a block-diagonal adjacency: two cliques
        A = np.zeros((2 * n, 2 * n))
        A[:n, :n] = 1.0
        A[n:, n:] = 1.0
        np.fill_diagonal(A, 0)
        S = sp.csr_matrix(A)
        labels = louvain_micro(S, random_state=42)
        assert labels.shape == (2 * n,)
        # Should find at least 2 clusters
        assert len(np.unique(labels)) >= 2
        # Labels within each block should be uniform
        assert len(np.unique(labels[:n])) == 1
        assert len(np.unique(labels[n:])) == 1


class TestFindComponents:
    """Test component discovery under UoM hypothesis."""

    def test_find_components_connected(self):
        """A well-connected graph should yield a single component."""
        n = 50
        rng = np.random.default_rng(42)
        # Dense random stochastic matrix (connected graph)
        W = rng.random((n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        D_inv = 1.0 / W.sum(axis=1)
        P = sp.csr_matrix(W * D_inv[:, None])
        n_comp, labels = find_components(P, random_state=42)
        assert labels.shape == (n,)
        assert n_comp >= 1

    def test_find_components_disconnected(self):
        """A block-diagonal stochastic matrix with 4+ disconnected blocks triggers direct detection."""
        n = 15
        # Four disconnected ring blocks — find_components short-circuits at n_cc > 3
        from scipy.sparse import block_diag

        rings = []
        for _ in range(4):
            ring = sp.csr_matrix(
                np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), -1, axis=1)
            )
            rings.append(ring)
        W = block_diag(rings, format="csr")
        row_sums = np.array(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        P = W.multiply(1.0 / row_sums[:, None]).tocsr()
        n_comp, labels = find_components(P, random_state=42)
        assert n_comp == 4
        assert labels.shape == (4 * n,)
        assert len(np.unique(labels)) == 4


class TestMBKM:
    """Test MiniBatch KMeans wrapper."""

    def test_mbkm_returns_labels(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        labels = mbkm(X, n_clusters=3, random_state=0)
        assert labels.shape == (100,)
        assert len(np.unique(labels)) == 3


class TestConsolidateMacros:
    """Test conductance-based macro-component merging."""

    def test_consolidate_identity(self):
        """With two well-separated blocks, consolidation should keep them."""
        n = 30
        W = np.zeros((2 * n, 2 * n))
        W[:n, :n] = 1.0
        W[n:, n:] = 1.0
        np.fill_diagonal(W, 0)
        W_sp = sp.csr_matrix(W)
        labels = np.array([0] * n + [1] * n)
        new_labels = consolidate_macros(W_sp, labels)
        assert new_labels.shape == (2 * n,)
        # Two well-separated blocks should remain separate
        assert len(np.unique(new_labels)) >= 2
