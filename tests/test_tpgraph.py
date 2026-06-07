"""Tests for graph construction and kernel modules."""

import numpy as np
import pytest
from scipy import sparse

from topo.base.ann import kNN
from topo.tpgraph import kernels
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.intrinsic_dim import (
    automated_scaffold_sizing,
    fsa_local,
    mle_local,
)
from topo.tpgraph.kernels import Kernel, compute_kernel


def test_compute_kernel(swiss_roll_data):
    X, _ = swiss_roll_data
    K = compute_kernel(X, n_neighbors=10, backend="sklearn")
    assert sparse.issparse(K)
    assert K.shape[0] == X.shape[0]


def test_kernel_estimator(swiss_roll_data):
    X, _ = swiss_roll_data
    ker = Kernel(n_neighbors=10, backend="sklearn").fit(X)
    assert ker.K is not None
    P = ker.diff_op()
    assert sparse.issparse(P)
    L = ker.laplacian()
    assert sparse.issparse(L)


def test_cknn_graph(swiss_roll_data):
    X, _ = swiss_roll_data
    res = cknn_graph(
        X, n_neighbors=10, weighted=None, return_densities=False, backend="sklearn"
    )
    assert len(res) == 2
    A, W = res
    assert sparse.issparse(A)
    assert sparse.issparse(W)


def test_automated_scaffold_sizing(swiss_roll_data):
    X, _ = swiss_roll_data
    n_comp = automated_scaffold_sizing(X, method="fsa", ks=[10, 20], backend="sklearn")
    assert isinstance(n_comp, int)
    assert n_comp >= 2


def test_local_intrinsic_dim(swiss_roll_data):
    X, _ = swiss_roll_data
    K = compute_kernel(X, n_neighbors=10, backend="sklearn")
    f_id = fsa_local(K, n_neighbors=10)
    m_id = mle_local(K, n_neighbors=10)
    assert len(f_id) == X.shape[0]
    assert len(m_id) == X.shape[0]


def test_kernel_helper_validation_and_sanitizing():
    dense = np.array([[1.0, np.inf], [-1.0, 2.0]])
    csr = kernels._as_csr_matrix(dense)
    arr = kernels._as_dense_array(csr)

    assert sparse.issparse(csr)
    np.testing.assert_allclose(arr, dense)
    kernels._check_2d_input(dense)
    kernels._check_square_matrix(dense)

    with pytest.raises(ValueError, match="2-D"):
        kernels._check_2d_input(np.array([1.0]))
    with pytest.raises(ValueError, match="square"):
        kernels._check_square_matrix(np.ones((2, 3)))

    sanitized = kernels._sanitize_sparse_data(csr)
    np.testing.assert_allclose(sanitized.toarray(), [[1.0, 0.0], [0.0, 2.0]])


def test_kernel_numeric_helpers():
    graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    res = kernels._safe_diffusion_operator_with_degree(graph, alpha=0.0)
    assert isinstance(res, tuple)
    P, D_left = res
    assert sparse.issparse(P)
    assert sparse.issparse(D_left)

    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = kernels._maybe_l2_normalize_rows(X.copy())
    assert isinstance(normalized, np.ndarray)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])

    assert kernels._cosine_knn_requires_unit_vectors("hnswlib")
    assert not kernels._cosine_knn_requires_unit_vectors("sklearn")
    np.testing.assert_allclose(
        kernels._cosine_distance_to_angle_from_sparse_triplets(
            np.array([0]), np.array([1]), np.array([0.0, 1.0, 2.0])
        ),
        [0.0, np.pi / 2, np.pi],
    )
    np.testing.assert_allclose(
        kernels._ensure_nonneg_and_finite(np.array([np.nan, -1.0, 2.0]), eps=0.1),
        [0.1, 0.1, 2.0],
    )
    np.testing.assert_allclose(kernels._adap_bw(graph, n_neighbors=2), [1.0, 1.0])


def test_cosine_use_angular_converts_adaptive_bandwidth_units():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    _K, densities = compute_kernel(
        X,
        metric="cosine",
        n_neighbors=2,
        backend="sklearn",
        use_angular=True,
        return_densities=True,
        symmetrize=False,
    )

    raw_bandwidth = kernels._adap_bw(densities["knn"], n_neighbors=2)
    expected = kernels._cosine_distance_to_angle_from_sparse_triplets(
        None, None, raw_bandwidth
    )
    np.testing.assert_allclose(densities["adaptive_bw"], expected)


@pytest.mark.parametrize("backend", ["sklearn"])
def test_knn_self_query_returns_requested_nonself_neighbors(backend):
    X = np.random.RandomState(0).randn(12, 4)

    graph = kNN(X, n_neighbors=3, backend=backend, metric="euclidean")

    assert graph.shape == (12, 12)
    np.testing.assert_allclose(graph.diagonal(), np.zeros(12))
    np.testing.assert_array_equal(graph.getnnz(axis=1), np.full(12, 3))


def test_kernel_parameterized_operator_caches_recompute_by_key():
    X = np.random.RandomState(1).randn(18, 3)
    ker = Kernel(n_neighbors=4, backend="sklearn", metric="euclidean").fit(X)

    L_norm = ker.laplacian("normalized")
    L_unnorm = ker.laplacian("unnormalized")
    assert not np.allclose(L_norm.toarray(), L_unnorm.toarray())

    P_alpha0 = ker.diff_op(anisotropy=0.0)
    P_alpha1 = ker.diff_op(anisotropy=1.0)
    assert not np.allclose(P_alpha0.toarray(), P_alpha1.toarray())


def test_shortest_paths_indices_and_landmark_guard():
    X = np.random.RandomState(2).randn(10, 2)
    ker = Kernel(n_neighbors=3, backend="sklearn", metric="euclidean").fit(X)

    subset = ker.shortest_paths(indices=[0, 2])

    assert subset.shape == (2, 10)
    assert np.isfinite(subset).any()
    with pytest.raises(NotImplementedError, match="landmark"):
        ker.shortest_paths(landmark=True, recompute=True)


def test_adaptive_density_ranks_constant_bandwidth_guard():
    adap_sd = np.ones(5)
    ranks = kernels._density_ranks(adap_sd, high=7)

    np.testing.assert_allclose(ranks, np.full(5, 7.0))


def test_resistance_distance_sparsify_and_impute_numerics():
    graph = sparse.csr_matrix(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    ker = Kernel(
        metric="precomputed",
        adaptive_bw=False,
        sigma=1.0,
        n_neighbors=2,
        laplacian_type="unnormalized",
    ).fit(graph)

    rd = ker.resistance_distance().toarray()
    np.testing.assert_allclose(rd, rd.T)
    np.testing.assert_allclose(np.diag(rd), np.zeros(4))
    assert np.isfinite(rd).all()
    assert (rd >= -1e-12).all()

    sparse_graph = ker.sparsify(epsilon=0.9, maxiter=1, random_state=0).tocsr()
    np.testing.assert_allclose(sparse_graph.toarray(), sparse_graph.toarray().T)
    assert (sparse_graph.data >= 0).all()

    Y = sparse.csr_matrix(np.eye(4))
    Y_imp = ker.impute(Y, t=2)
    assert Y_imp.shape == (4, 4)
    assert np.isfinite(Y_imp).all()
