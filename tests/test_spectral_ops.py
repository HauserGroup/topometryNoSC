"""Tests for graph spectral operators and lightweight layout kernels."""

import numpy as np
import pytest
from scipy import sparse

from topo.spectral import _spectral
from topo.spectral.umap_layouts import clip, rdist


def _path_graph():
    return sparse.csr_matrix(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )


class TestGraphOperators:
    def test_degree_dense_and_sparse(self):
        W_dense = np.asarray(_path_graph().toarray())
        W_sparse = _path_graph()

        np.testing.assert_allclose(_spectral.degree(W_dense).diagonal(), [1, 2, 1])
        np.testing.assert_allclose(
            _spectral.degree(W_sparse).diagonal(), np.array([1, 2, 1])
        )

    @pytest.mark.parametrize(
        "laplacian_type", ["unnormalized", "normalized", "random_walk"]
    )
    def test_graph_laplacian_sparse_shapes(self, laplacian_type):
        res = _spectral.graph_laplacian(
            _path_graph(), laplacian_type=laplacian_type, return_D=True
        )
        assert isinstance(res, tuple)
        L, D = res

        assert sparse.issparse(L)
        assert sparse.issparse(D)
        assert L.shape == (3, 3)
        assert D.shape == (3, 3)

    def test_graph_laplacian_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown laplacian"):
            _spectral.graph_laplacian(_path_graph(), laplacian_type="bad")

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_diffusion_operator_is_finite(self, symmetric):
        P = _spectral.diffusion_operator(
            _path_graph(),
            alpha=0.5,
            symmetric=symmetric,
            return_D_inv_sqrt=False,
        )

        assert isinstance(P, sparse.csr_matrix)
        P_dense = P.toarray()
        assert P.shape == (3, 3)
        assert np.isfinite(P_dense).all()
        if not symmetric:
            np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), np.ones(3))
        else:
            np.testing.assert_allclose(P_dense, P_dense.T)

    def test_diffusion_operator_can_return_similarity_transform(self):
        res = _spectral.diffusion_operator(
            _path_graph(), alpha=0.0, symmetric=True, return_D_inv_sqrt=True
        )
        assert isinstance(res, tuple)
        P, D_left = res

        assert sparse.issparse(P)
        assert sparse.issparse(D_left)

    def test_laplacian_eigenmaps_and_spectral_clustering(self):
        W = sparse.block_diag(
            [np.ones((3, 3)) - np.eye(3), np.ones((3, 3)) - np.eye(3)]
        )
        res = _spectral.LE(
            W.tocsr(), n_eigs=2, laplacian_type="normalized", return_evals=True
        )
        assert isinstance(res, tuple)
        evecs, evals = res

        assert evecs.shape == (6, 2)
        assert evals.shape == (2,)
        assert np.isfinite(evecs).all()

        labels = _spectral.spectral_clustering(evecs, random_state=0, n_iter_max=5)
        assert labels.shape == (6,)
        assert set(labels).issubset({0, 1})

    def test_spectral_clustering_validates_input(self):
        with pytest.raises(ValueError, match="2-D"):
            _spectral.spectral_clustering(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="zero-norm"):
            _spectral.spectral_clustering(np.zeros((3, 2)))


class TestUmapLayoutKernels:
    def test_clip_bounds_values(self):
        assert clip(5.0) == pytest.approx(4.0)
        assert clip(-5.0) == pytest.approx(-4.0)
        assert clip(0.25) == pytest.approx(0.25)

    def test_rdist_returns_squared_distance(self):
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([3.0, 4.0], dtype=np.float32)

        assert rdist(x, y) == pytest.approx(25.0)
