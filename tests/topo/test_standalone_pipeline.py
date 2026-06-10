"""Integration test composing standalone estimators outside TopOGraph.

Mirrors the documented "custom pipeline" path (see docs/api/advanced.md and
notebooks/_example_utils.py): each building block must work on its own and
compose into a full embedding pipeline without the TopOGraph orchestrator.
This guards against breakage in standalone APIs that the TopOGraph tests do
not exercise.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_swiss_roll

from topo.base.ann import kNN
from topo.layouts.projector import Projector
from topo.spectral import LE
from topo.spectral.eigen import EigenDecomposition
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.kernels import Kernel


@pytest.fixture(scope="module")
def data():
    X, _ = make_swiss_roll(n_samples=200, noise=0.05, random_state=0)
    return np.asarray(X, dtype=np.float64)


def test_standalone_composition_knn_kernel_eigen_projection(data):
    """kNN -> Kernel -> EigenDecomposition -> LE init -> Projector."""
    n = data.shape[0]

    knn = kNN(data, n_neighbors=15, backend="sklearn")
    assert isinstance(knn, csr_matrix)
    assert knn.shape == (n, n)

    kernel = Kernel(n_neighbors=15, metric="euclidean", backend="sklearn")
    kernel.fit(data)
    assert kernel.P is not None

    eigen = EigenDecomposition(n_components=10, method="msDM", drop_first=True)
    eigen.fit(kernel)
    Z = np.asarray(eigen.transform())
    assert Z.shape[0] == n
    assert Z.shape[1] >= 2
    assert np.isfinite(Z).all()

    kernel_Z = Kernel(n_neighbors=15, metric="euclidean", backend="sklearn")
    kernel_Z.fit(Z)
    K_Z = csr_matrix(kernel_Z.K)

    init = np.asarray(LE(K_Z, n_eigs=2, laplacian_type="normalized", drop_first=True))
    assert init.shape == (n, 2)
    assert np.isfinite(init).all()

    projector = Projector(
        projection_method="MAP",
        n_components=2,
        n_neighbors=15,
        num_iters=30,
        init=init,
        random_state=42,
    )
    Y = np.asarray(projector.fit_transform(csr_matrix(kernel_Z.P)))
    assert Y.shape == (n, 2)
    assert np.isfinite(Y).all()


def test_standalone_cknn_kernel_pipeline(data):
    """CkNN kernel path: Kernel(cknn) on data -> eigenbasis on binary graph."""
    n = data.shape[0]

    W = cknn_graph(data, scale_k=10, delta=1.0, backend="sklearn")
    assert W.shape == (n, n)
    assert set(np.unique(W.data)).issubset({1.0})
    # symmetrize='or' default yields a symmetric adjacency
    assert (W != W.T).nnz == 0

    kernel = Kernel(n_neighbors=10, cknn=True, backend="sklearn")
    kernel.fit(data)

    eigen = EigenDecomposition(n_components=5, method="DM", drop_first=True)
    Z = np.asarray(eigen.fit_transform(kernel))
    assert Z.shape[0] == n
    assert np.isfinite(Z).all()
