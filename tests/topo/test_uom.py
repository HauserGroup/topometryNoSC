"""Tests for topo.uom standalone functions."""

import numpy as np
import pytest
import scipy.sparse as sp

from topo.topograph import TopOGraph
from topo.uom import consolidate_macros, find_components, louvain_micro, mbkm


class UoMTestGraph(TopOGraph):
    """Expose protected UoM helpers for focused unit tests."""

    def component_order(self) -> np.ndarray:
        return self._component_order()

    def block_diag_to_original_order(self, blocks) -> sp.csr_matrix:
        return self._block_diag_to_original_order(blocks)

    def fit_tiny_component(self, idx, n_i: int) -> None:
        self._fit_uom_tiny_component(idx, n_i)


def _assert_same_partition(labels: np.ndarray, groups: list[np.ndarray]) -> None:
    """Assert each supplied group has one unique label and groups differ."""
    group_labels = []
    for group in groups:
        uniq = np.unique(labels[group])
        assert uniq.size == 1
        group_labels.append(int(uniq[0]))

    assert len(set(group_labels)) == len(group_labels)


class TestLouvainMicro:
    """Test greedy Louvain-like modularity clustering."""

    def test_louvain_on_block_diagonal(self):
        """Two disconnected cliques should not mix labels across blocks."""
        n = 30
        A = np.zeros((2 * n, 2 * n), dtype=np.float32)
        A[:n, :n] = 1.0
        A[n:, n:] = 1.0
        np.fill_diagonal(A, 0.0)

        S = sp.csr_matrix(A)
        labels = louvain_micro(S, random_state=42)

        assert labels.shape == (2 * n,)
        assert labels.min() == 0

        # The important invariant: disconnected blocks must not be merged.
        first_block_labels = set(np.unique(labels[:n]).tolist())
        second_block_labels = set(np.unique(labels[n:]).tolist())
        assert first_block_labels.isdisjoint(second_block_labels)

    def test_louvain_micro_returns_valid_labels(self):
        S = sp.csr_matrix(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 0.1, 0],
                [0, 0, 0.1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        )

        labels = louvain_micro(S, random_state=0, max_passes=10)

        assert labels.shape == (5,)
        assert labels.min() == 0
        assert labels.max() < 5
        assert np.unique(labels).size >= 1


class TestFindComponents:
    """Test component discovery under the UoM hypothesis."""

    def test_find_components_connected(self):
        """A well-connected graph should return valid labels."""
        n = 50
        rng = np.random.default_rng(42)

        W = rng.random((n, n), dtype=np.float32)
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)

        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0
        P = sp.csr_matrix(W / row_sums[:, None])

        n_comp, labels = find_components(P, random_state=42)

        assert labels.shape == (n,)
        assert n_comp == np.unique(labels).size
        assert n_comp >= 1
        assert labels.min() == 0

    def test_find_components_disconnected(self):
        """A graph with 4+ disconnected blocks should use direct detection."""
        n = 15
        rings = []
        for _ in range(4):
            ring = sp.csr_matrix(
                np.roll(np.eye(n, dtype=np.float32), 1, axis=1)
                + np.roll(np.eye(n, dtype=np.float32), -1, axis=1)
            )
            rings.append(ring)

        W = sp.block_diag(rings, format="csr", dtype=np.float32)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        P = W.multiply(1.0 / row_sums[:, None]).tocsr()

        n_comp, labels = find_components(P, random_state=42)

        assert n_comp == 4
        assert labels.shape == (4 * n,)
        assert np.unique(labels).size == 4

        expected_groups = [
            np.arange(0, n),
            np.arange(n, 2 * n),
            np.arange(2 * n, 3 * n),
            np.arange(3 * n, 4 * n),
        ]
        _assert_same_partition(labels, expected_groups)

    def test_find_components_empty_graph_returns_one_component(self):
        n = 8
        P = sp.csr_matrix((n, n), dtype=np.float32)

        n_comp, labels = find_components(P, random_state=0)

        assert n_comp == 1
        assert labels.shape == (n,)
        assert np.all(labels == 0)


class TestMBKM:
    """Test MiniBatch KMeans wrapper."""

    def test_mbkm_returns_labels(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5)).astype(np.float32)

        labels = mbkm(X, n_clusters=3, random_state=0)

        assert labels.shape == (100,)
        assert labels.min() == 0
        assert np.unique(labels).size == 3

    def test_mbkm_clamps_clusters_to_samples(self):
        X = np.array([[0.0], [1.0]], dtype=np.float32)

        labels = mbkm(X, n_clusters=5, random_state=0)

        assert labels.shape == (2,)
        assert np.unique(labels).size <= 2


class TestConsolidateMacros:
    """Test conductance-based macro-component consolidation."""

    def test_consolidate_identity(self):
        """With two well-separated blocks, consolidation should keep them."""
        n = 30
        W = np.zeros((2 * n, 2 * n), dtype=np.float32)
        W[:n, :n] = 1.0
        W[n:, n:] = 1.0
        np.fill_diagonal(W, 0.0)

        W_sp = sp.csr_matrix(W)
        labels = np.array([0] * n + [1] * n, dtype=int)

        new_labels = consolidate_macros(W_sp, labels)

        assert new_labels.shape == (2 * n,)
        assert np.unique(new_labels).size == 2
        _assert_same_partition(new_labels, [np.arange(n), np.arange(n, 2 * n)])

    def test_consolidate_relabels_contiguously(self):
        W = sp.csr_matrix(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        labels = np.array([10, 10, 20, 20], dtype=int)

        new_labels = consolidate_macros(W, labels, max_iters=1)

        assert new_labels.min() == 0
        assert new_labels.max() == np.unique(new_labels).size - 1


class TestUoMAggregationHelpers:
    """Test UoM helper methods exposed through a small test subclass."""

    def test_uom_component_order_covers_samples_once(self):
        tg = UoMTestGraph(uom=True)
        tg.n = 5
        tg._init_uom_state()
        tg.uom_components_ = [np.array([2, 4]), np.array([0, 1, 3])]

        order = tg.component_order()

        assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
        np.testing.assert_array_equal(order, np.array([2, 4, 0, 1, 3]))

    def test_uom_component_order_rejects_duplicates(self):
        tg = UoMTestGraph(uom=True)
        tg.n = 4
        tg._init_uom_state()
        tg.uom_components_ = [np.array([0, 1]), np.array([1, 2])]

        with pytest.raises(ValueError, match="duplicates"):
            tg.component_order()

    def test_uom_component_order_rejects_incomplete_cover(self):
        tg = UoMTestGraph(uom=True)
        tg.n = 4
        tg._init_uom_state()
        tg.uom_components_ = [np.array([0, 1]), np.array([2])]

        with pytest.raises(ValueError, match="cover"):
            tg.component_order()

    def test_uom_block_diag_to_original_order(self):
        tg = UoMTestGraph(uom=True)
        tg.n = 4
        tg._init_uom_state()
        tg.uom_components_ = [np.array([2, 0]), np.array([3, 1])]

        B1 = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float32)
        B2 = sp.csr_matrix([[5, 6], [7, 8]], dtype=np.float32)

        M = tg.block_diag_to_original_order([B1, B2]).toarray()

        # Component [2, 0]: local 0 -> global 2, local 1 -> global 0.
        assert M[2, 2] == 1
        assert M[2, 0] == 2
        assert M[0, 2] == 3
        assert M[0, 0] == 4

        # Component [3, 1]: local 0 -> global 3, local 1 -> global 1.
        assert M[3, 3] == 5
        assert M[3, 1] == 6
        assert M[1, 3] == 7
        assert M[1, 1] == 8

        # Off-block entries must remain zero.
        assert M[0, 1] == 0
        assert M[1, 0] == 0
        assert M[2, 3] == 0
        assert M[3, 2] == 0

    def test_uom_tiny_component_list_alignment(self):
        tg = UoMTestGraph(uom=True)
        tg._init_uom_state()
        tg.uom_Z_list, tg.uom_msZ_list = [], []
        tg.uom_knn_Z_list, tg.uom_knn_msZ_list = [], []
        tg.uom_Kernel_Z_list, tg.uom_Kernel_msZ_list = [], []
        tg.uom_BaseKernel_list, tg.uom_knn_X_list = [], []
        tg.uom_DMEig_list, tg.uom_msDMEig_list = [], []
        tg.uom_eigenvalues_dm_list, tg.uom_eigenvalues_ms_list = [], []

        tg.fit_tiny_component(np.array([0]), 1)

        lists = [
            tg.uom_Z_list,
            tg.uom_msZ_list,
            tg.uom_knn_Z_list,
            tg.uom_knn_msZ_list,
            tg.uom_Kernel_Z_list,
            tg.uom_Kernel_msZ_list,
            tg.uom_BaseKernel_list,
            tg.uom_knn_X_list,
            tg.uom_DMEig_list,
            tg.uom_msDMEig_list,
            tg.uom_eigenvalues_dm_list,
            tg.uom_eigenvalues_ms_list,
        ]

        for value in lists:
            assert value is not None
            assert len(value) == 1

        assert tg.uom_Z_list is not None
        assert tg.uom_msZ_list is not None
        assert tg.uom_Z_list[0].shape == (1, 1)
        assert tg.uom_msZ_list[0].shape == (1, 1)

        assert tg.uom_eigenvalues_dm_list is not None
        assert tg.uom_eigenvalues_ms_list is not None
        np.testing.assert_allclose(tg.uom_eigenvalues_dm_list[0], np.ones(1))
        np.testing.assert_allclose(tg.uom_eigenvalues_ms_list[0], np.ones(1))

    def test_uom_aggregate_scaffold_to_original_order(self):
        tg = UoMTestGraph(uom=True)
        tg.n = 4
        tg._init_uom_state()
        tg.uom_components_ = [np.array([2, 0]), np.array([3, 1])]

        Z1 = np.array([[10, 11], [12, 13]], dtype=np.float32)
        Z2 = np.array([[20], [21]], dtype=np.float32)

        Z, slices = tg._aggregate_scaffold_to_original_order([Z1, Z2])

        assert Z.shape == (4, 3)
        assert slices == [(0, 2), (2, 3)]

        # Component [2, 0].
        np.testing.assert_array_equal(Z[2], np.array([10, 11, 0], dtype=np.float32))
        np.testing.assert_array_equal(Z[0], np.array([12, 13, 0], dtype=np.float32))

        # Component [3, 1].
        np.testing.assert_array_equal(Z[3], np.array([0, 0, 20], dtype=np.float32))
        np.testing.assert_array_equal(Z[1], np.array([0, 0, 21], dtype=np.float32))


def test_uom_same_size_components_build_independent_component_kernels():
    rng = np.random.default_rng(0)

    X1 = rng.normal(loc=-5.0, scale=0.1, size=(8, 3))
    X2 = rng.normal(loc=5.0, scale=0.1, size=(8, 3))
    X = np.vstack([X1, X2])

    tg = TopOGraph(
        uom=True,
        base_knn=3,
        graph_knn=3,
        min_eigs=4,
        id_min_components=2,
        id_max_components=4,
        base_kernel_version="bw_adaptive",
        graph_kernel_version="bw_adaptive",
        backend="sklearn",
        projection_methods=[],
        random_state=0,
    )

    tg.uom_comp_labels_ = np.array([0] * 8 + [1] * 8)
    tg.fit(X)

    Z = tg.spectral_scaffold(multiscale=False)
    msZ = tg.spectral_scaffold(multiscale=True)

    assert tg.P_of_Z.shape == (X.shape[0], X.shape[0])
    assert tg.P_of_msZ.shape == (X.shape[0], X.shape[0])
    assert Z.shape[0] == X.shape[0]  # pyright: ignore[reportOptionalSubscript]
    assert msZ.shape[0] == X.shape[0]  # pyright: ignore[reportOptionalSubscript]
