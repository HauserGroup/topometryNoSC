import numpy as np
from scipy.sparse import csr_matrix

from topo.base.graph_matrix import as_csr_matrix, as_float32_csr, sparse_identity


def test_as_csr_matrix_preserves_dense_values():
    X = np.array([[0.0, 1.0], [2.0, 0.0]])
    Y = as_csr_matrix(X, "X")

    assert isinstance(Y, csr_matrix)
    np.testing.assert_allclose(Y.toarray(), X)


def test_as_csr_matrix_preserves_csr_values():
    X = csr_matrix([[0.0, 1.0], [2.0, 0.0]])
    Y = as_csr_matrix(X, "X")

    assert isinstance(Y, csr_matrix)
    np.testing.assert_allclose(Y.toarray(), X.toarray())


def test_as_float32_csr_converts_dtype_only():
    X = csr_matrix(np.array([[0.0, 1.5], [2.5, 0.0]], dtype=np.float64))
    Y = as_float32_csr(X, "X")

    assert isinstance(Y, csr_matrix)
    assert Y.dtype == np.float32
    np.testing.assert_allclose(Y.toarray(), X.toarray())


def test_sparse_identity():
    I = sparse_identity(3)

    assert isinstance(I, csr_matrix)
    assert I.dtype == np.float32
    np.testing.assert_allclose(I.toarray(), np.eye(3, dtype=np.float32))
