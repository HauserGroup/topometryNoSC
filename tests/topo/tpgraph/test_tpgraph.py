"""Tests for graph construction and kernel modules."""

from typing import cast

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian

from topo.base.ann import kNN
from topo.topograph import TopOGraph
from topo.tpgraph import kernels
from topo.tpgraph.cknn import cknn_graph, cknn_ratio_matrix
from topo.tpgraph.intrinsic_dim import (
    automated_scaffold_sizing,
    fsa_local,
    mle_local,
)
from topo.tpgraph.kernels import Kernel, compute_kernel


def test_kernel_estimator(swiss_roll_data):
    X, _ = swiss_roll_data
    ker = Kernel(n_neighbors=10, backend="sklearn").fit(X)
    assert ker.K is not None
    P = ker.diff_op()
    assert sparse.issparse(P)
    L = ker.laplacian()
    assert sparse.issparse(L)


def _as_dense_array(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def test_cknn_exact_threshold_uses_strict_inequality():
    X = np.array([[0.0], [1.0], [3.0]])

    A = cknn_graph(X, scale_k=1, delta=1.0, exact=True)

    assert A[1, 0] == 0


def test_cknn_exact_threshold_positive_case():
    X = np.array([[0.0], [1.0], [3.0]])

    A = cknn_graph(X, scale_k=1, delta=1.01, exact=True)

    assert A[0, 1] == 1
    assert A[1, 0] == 1


def test_cknn_graph_properties_and_delta_monotonicity():
    X = np.random.default_rng(0).normal(size=(80, 2))

    A_small = cknn_graph(X, scale_k=8, delta=0.8, exact=True)
    A_large = cknn_graph(X, scale_k=8, delta=1.2, exact=True)

    assert sparse.issparse(A_small)
    np.testing.assert_array_equal(A_large.toarray(), A_large.T.toarray())
    assert set(np.unique(A_large.data)).issubset({1.0})
    np.testing.assert_array_equal(A_small.maximum(A_large).toarray(), A_large.toarray())


def test_cknn_candidate_mode_matches_exact_when_candidates_are_complete():
    X = np.random.default_rng(1).normal(size=(60, 4))

    exact = cknn_graph(X, scale_k=5, delta=1.1, exact=True)
    approx_complete = cknn_graph(
        X,
        scale_k=5,
        delta=1.1,
        exact=False,
        candidate_k=59,
        backend="sklearn",
    )

    np.testing.assert_array_equal(exact.toarray(), approx_complete.toarray())


def test_cknn_candidate_mode_basic_properties():
    X = np.random.default_rng(0).normal(size=(50, 3))
    A = cknn_graph(X, scale_k=5, delta=1.0, candidate_k=20, exact=False)

    assert (A != A.T).nnz == 0
    assert np.all(A.diagonal() == 0)
    assert set(np.unique(A.data)).issubset({1.0})


def test_cknn_sklearn_candidate_mode_uses_direct_neighbors():
    X = np.random.default_rng(0).normal(size=(50, 3))
    A = cknn_graph(
        X,
        scale_k=5,
        delta=1.1,
        candidate_k=20,
        exact=False,
        backend="sklearn",
    )
    assert (A != A.T).nnz == 0
    assert np.all(A.diagonal() == 0)
    assert set(np.unique(A.data)).issubset({1.0})


def test_cknn_unnormalized_laplacian_zero_modes_match_components():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=-5, scale=0.2, size=(30, 2))
    X2 = rng.normal(loc=5, scale=0.2, size=(30, 2))
    X = np.vstack([X1, X2])

    A = cknn_graph(X, scale_k=5, delta=1.0, exact=True)
    n_components, _ = connected_components(A, directed=False)
    L = cast(sparse.csr_matrix, laplacian(A, normed=False))
    evals = np.linalg.eigvalsh(L.toarray())

    assert np.sum(evals < 1e-8) == n_components


def test_cknn_ratio_matrix_threshold_matches_graph():
    X = np.random.default_rng(0).normal(size=(40, 2))

    ratio = cknn_ratio_matrix(X, scale_k=5, exact=True)
    A_from_ratio = ratio.copy()
    A_from_ratio.data = (A_from_ratio.data < 1.1).astype(np.float32)
    A_from_ratio.eliminate_zeros()
    A_from_ratio = A_from_ratio.minimum(A_from_ratio.T)

    A = cknn_graph(X, scale_k=5, delta=1.1, exact=True)

    np.testing.assert_array_equal(A_from_ratio.toarray(), A.toarray())


def test_kernel_cknn_returns_binary_sparse_graph():
    X = np.random.default_rng(2).normal(size=(40, 3))

    ker = Kernel(
        n_neighbors=5,
        metric="euclidean",
        backend="sklearn",
        cknn=True,
        cknn_delta=1.1,
        cknn_exact=True,
        laplacian_type="unnormalized",
    ).fit(X)
    assert sparse.issparse(ker.K)
    K = cast(sparse.csr_matrix, ker.K)
    np.testing.assert_array_equal(K.toarray(), K.T.toarray())
    assert set(np.unique(K.data)).issubset({1.0})


def test_topograph_cknn_smoke():
    X = np.random.default_rng(3).normal(size=(35, 4))
    tg = TopOGraph(
        base_knn=5,
        graph_knn=5,
        min_eigs=6,
        base_kernel_version="cknn",
        graph_kernel_version="cknn",
        base_metric="euclidean",
        backend="sklearn",
        projection_methods=[],
        cknn_exact=True,
    )

    tg.fit(X)

    assert tg.laplacian_type == "unnormalized"
    assert tg.base_kernel is not None
    assert tg.base_kernel.K is not None
    assert sparse.issparse(tg.base_kernel.K)
    assert set(np.unique(tg.base_kernel.K.data)).issubset({1.0})


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
    assert sparse.issparse(sanitized)
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

    raw_bandwidth = kernels._adap_bw(cast(np.ndarray, densities["knn"]), n_neighbors=2)
    expected = kernels._cosine_distance_to_angle_from_sparse_triplets(
        None, None, raw_bandwidth
    )
    np.testing.assert_allclose(cast(np.ndarray, densities["adaptive_bw"]), expected)


@pytest.mark.parametrize("backend", ["sklearn", "hnswlib"])
def test_knn_self_query_returns_requested_nonself_neighbors(backend):
    if backend == "hnswlib":
        pytest.importorskip("hnswlib")
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
    L_norm_arr = _as_dense_array(L_norm)
    L_unnorm_arr = _as_dense_array(L_unnorm)
    assert not np.allclose(L_norm_arr, L_unnorm_arr)

    P_alpha0 = ker.diff_op(anisotropy=0.0)
    P_alpha1 = ker.diff_op(anisotropy=1.0)
    P_alpha0_arr = _as_dense_array(P_alpha0)
    P_alpha1_arr = _as_dense_array(P_alpha1)
    assert not np.allclose(P_alpha0_arr, P_alpha1_arr)


def test_shortest_paths_caches_by_indices():
    X = np.random.RandomState(2).randn(10, 2)
    ker = kernels.Kernel(n_neighbors=3, backend="sklearn", metric="euclidean").fit(X)

    sp_all = ker.shortest_paths(indices=None)
    sp_subset = ker.shortest_paths(indices=[0, 2])

    assert sp_all is not None
    assert sp_subset is not None
    assert sp_all.shape == (10, 10)
    assert sp_subset.shape == (2, 10)


def test_shortest_paths_indices_and_landmark_guard():
    X = np.random.RandomState(2).randn(10, 2)
    ker = Kernel(n_neighbors=3, backend="sklearn", metric="euclidean").fit(X)

    subset = ker.shortest_paths(indices=[0, 2])

    assert subset.shape == (2, 10)
    assert np.isfinite(subset).any()
    with pytest.raises(NotImplementedError, match="landmark"):
        ker.shortest_paths(landmark=True, recompute=True)


def test_adaptive_density_ranks_constant_bandwidth_uses_neutral_fallback():
    adap_sd = np.ones(5)
    ranks = kernels._density_ranks(adap_sd, high=7)
    np.testing.assert_allclose(ranks, np.full(5, 4.5))


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

    rd_mat = ker.resistance_distance()
    rd = rd_mat.toarray() if sparse.issparse(rd_mat) else np.asarray(rd_mat)
    np.testing.assert_allclose(rd, rd.T)
    np.testing.assert_allclose(np.diag(rd), np.zeros(4))
    assert np.isfinite(rd).all()
    assert (rd >= -1e-12).all()

    sparse_graph = ker.sparsify(epsilon=0.9, maxiter=1, random_state=0)
    assert sparse.isspmatrix(sparse_graph)
    sparse_graph = sparse_graph.tocsr()
    np.testing.assert_allclose(sparse_graph.toarray(), sparse_graph.toarray().T)
    assert (sparse_graph.data >= 0).all()

    Y = sparse.csr_matrix(np.eye(4))
    Y_imp = ker.impute(Y, t=2)
    assert Y_imp.shape == (4, 4)
    assert np.isfinite(Y_imp).all()


def test_kernel_safe_refit_recompute_false():
    X_1 = np.random.RandomState(0).randn(10, 3)
    X_2 = np.random.RandomState(1).randn(20, 3)

    ker = kernels.Kernel(n_neighbors=2, backend="sklearn").fit(X_1)
    assert ker._K is not None
    orig_shape = ker._K.shape

    # Fitting new data with recompute=False should not crash if shape mismatch,
    # but the current implementation just returns self early.
    ker.fit(X_2, recompute=False)
    assert ker._K is not None
    assert ker._K.shape == orig_shape


# =============================================================================
# Step 1: CkNN Semantic Verification (from audit plan)
# =============================================================================


def test_cknn_is_not_plain_knn():
    """CkNN must remain distinct from ordinary kNN (threshold vs. candidate)."""
    X = np.array([[0.0], [1.0], [2.0], [10.0]])

    # CkNN with delta=1.01: connects if d < delta * sqrt(rho_i * rho_j)
    G_cknn = cknn_graph(X, scale_k=1, delta=1.01, exact=True)

    # Plain binary kNN: connects if in k-nearest neighbors
    G_knn = kNN(X, n_neighbors=1, backend="sklearn")
    G_knn = G_knn.astype(bool).astype(np.float32)

    # They should differ because CkNN is threshold-based, not k-based
    diff_count = cast(sparse.csr_matrix, G_cknn != G_knn).nnz
    assert diff_count > 0, "CkNN and plain kNN should produce different graphs"


def test_cknn_is_binary_symmetric_no_diagonal():
    """CkNN output must be binary, symmetric, and have no diagonal."""
    X = np.random.default_rng(46).normal(size=(20, 3))
    G = cknn_graph(X, scale_k=5, delta=1.0, exact=False, include_self=False)

    assert sparse.isspmatrix_csr(G), "Must be CSR sparse"
    assert np.all(G.data == 1.0), "Binary graph must have all data=1.0"
    assert G.diagonal().sum() == 0, "No self-loops allowed"
    assert (G != G.T).nnz == 0, "Graph must be symmetric"


def test_cknn_delta_monotonicity():
    """Larger delta should produce graphs with more edges (monotonicity)."""
    X = np.random.default_rng(47).normal(size=(30, 3))

    G_small = cknn_graph(X, scale_k=5, delta=0.8, exact=True)
    G_medium = cknn_graph(X, scale_k=5, delta=1.0, exact=True)
    G_large = cknn_graph(X, scale_k=5, delta=1.2, exact=True)

    # Larger delta allows more edges: small ⊆ medium ⊆ large
    diff_count = cast(sparse.csr_matrix, G_small != G_medium).nnz
    assert diff_count > 0, "Different delta should produce different graphs"
    assert G_small.nnz <= G_medium.nnz, "Larger delta should have more or equal edges"
    assert G_medium.nnz <= G_large.nnz, "Larger delta should have more or equal edges"


def test_cknn_ratio_matrix_threshold_semantics():
    """Verify ratio matrix definition: d_ij / sqrt(rho_i * rho_j)."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])

    ratio_graph = cknn_ratio_matrix(X, scale_k=2, metric="euclidean", exact=True)

    # Ratio matrix should have positive values (distances normalized by radii)
    assert np.all(ratio_graph.data > 0), "Ratio matrix should have positive values"
    # No diagonal
    assert ratio_graph.diagonal().sum() == 0, "Ratio matrix should have no diagonal"
    # Symmetric
    assert (ratio_graph != ratio_graph.T).nnz == 0, "Ratio matrix should be symmetric"


# =============================================================================
# Step 2: Exact kNN Graph Self-Neighbor and Row-Count Invariants
# =============================================================================


def test_exact_knn_graph_has_no_self_edges():
    """Exact kNN graph must have no diagonal entries."""
    X = np.random.default_rng(48).normal(size=(25, 4))
    k = 5
    G = kNN(X, n_neighbors=k, backend="sklearn")

    assert G.diagonal().sum() == 0, "kNN graph must have no self-edges"


def test_exact_knn_graph_exact_k_neighbors_per_row():
    """Exact kNN graph must have exactly k nonzero entries per row."""
    X = np.random.default_rng(49).normal(size=(30, 4))
    k = 5
    G = kNN(X, n_neighbors=k, backend="sklearn")

    row_counts = np.diff(G.indptr)
    assert np.all(row_counts == k), (
        f"Expected {k} neighbors per row, got {np.unique(row_counts)}"
    )


def test_exact_knn_connectivity_vs_distance_mode():
    """Document distance-graph behavior (current implementation returns distances)."""
    X = np.random.default_rng(50).normal(size=(15, 3))
    G = kNN(X, n_neighbors=4, backend="sklearn")

    # Current implementation returns distance graph
    assert G.data.dtype in [np.float32, np.float64], "Should be distance values"
    assert np.all(G.data >= 0), "Distances should be non-negative"
