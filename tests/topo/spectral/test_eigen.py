"""Tests for eigendecomposition routines."""

import numpy as np
from scipy import sparse

from topo.spectral.eigen import EigenDecomposition, eigendecompose


def test_eigendecompose():
    G = np.random.rand(20, 20)
    G = (G + G.T) / 2
    evals, evecs = eigendecompose(G, n_components=3, eigensolver="dense")
    # eigendecompose intentionally requests one extra pair so callers can
    # optionally drop a trivial component.
    assert len(evals) == 4
    assert evecs.shape == (20, 4)


def test_eigen_decomposition_class():
    G = sparse.csr_matrix(np.ones((20, 20)) - np.eye(20))
    decomp = EigenDecomposition(n_components=3, method="DM", eigensolver="dense").fit(G)
    assert decomp.eigenvalues is not None
    assert len(decomp.eigenvalues) == 3
