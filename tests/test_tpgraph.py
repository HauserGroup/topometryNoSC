"""Tests for graph construction and kernel modules."""

import numpy as np
from scipy import sparse

from topo.tpgraph import kernels
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing
from topo.tpgraph.kernels import Kernel, compute_kernel


def test_compute_kernel(swiss_roll_data):
    X, _ = swiss_roll_data
    K = compute_kernel(X, n_neighbors=10)
    assert sparse.issparse(K)
    assert K.shape[0] == X.shape[0]


def test_kernel_estimator(swiss_roll_data):
    X, _ = swiss_roll_data
    ker = Kernel(n_neighbors=10).fit(X)
    assert ker.K is not None
    P = ker.diff_op()
    assert sparse.issparse(P)
    L = ker.laplacian()
    assert sparse.issparse(L)


def test_cknn_graph(swiss_roll_data):
    X, _ = swiss_roll_data
    A, W = cknn_graph(
        X, n_neighbors=10, weighted=None, return_densities=False, backend="sklearn"
    )
    assert sparse.issparse(A)
    assert sparse.issparse(W)


def test_automated_scaffold_sizing(swiss_roll_data):
    X, _ = swiss_roll_data
    n_comp = automated_scaffold_sizing(X, method="fsa", ks=[10, 20])
    assert isinstance(n_comp, int)
    assert n_comp >= 2


def test_kernel_helper_validation_and_sanitizing():
    dense = np.array([[1.0, np.inf], [-1.0, 2.0]])
    csr = kernels._as_csr_matrix(dense)
    arr = kernels._as_dense_array(csr)

    assert sparse.issparse(csr)
    np.testing.assert_allclose(arr, dense)
    kernels._check_2d_input(dense)
    kernels._check_square_matrix(dense)

    with np.testing.assert_raises_regex(ValueError, "2-D"):
        kernels._check_2d_input(np.array([1.0]))
    with np.testing.assert_raises_regex(ValueError, "square"):
        kernels._check_square_matrix(np.ones((2, 3)))

    sanitized = kernels._sanitize_sparse_data(csr)
    np.testing.assert_allclose(sanitized.toarray(), [[1.0, 0.0], [0.0, 2.0]])


def test_kernel_numeric_helpers():
    graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    P, D_left = kernels._safe_diffusion_operator_with_degree(graph, alpha=0.0)
    assert sparse.issparse(P)
    assert sparse.issparse(D_left)

    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = kernels._maybe_l2_normalize_rows(X.copy())
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
