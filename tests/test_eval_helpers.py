"""Focused tests for evaluation score helpers."""

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA

from topo.eval import global_scores, local_scores, rmetric, topo_metrics


def _cycle_graph(n=5):
    rows = []
    cols = []
    data = []
    for i in range(n):
        for j in ((i - 1) % n, (i + 1) % n):
            rows.append(i)
            cols.append(j)
            data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


class TestGlobalScores:
    def test_global_loss_and_pca_score_for_pca_embedding(self):
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.2],
                [2.0, 0.5, 1.5],
            ]
        )
        Y = PCA(n_components=2).fit_transform(X)

        assert global_scores.global_loss_(X, Y) >= 0
        assert global_scores.global_score_pca(X, Y, Y_pca=Y) == pytest.approx(1.0)

    def test_global_score_laplacian_accepts_precomputed_graph(self):
        graph = _cycle_graph(6)
        Y = np.column_stack([np.cos(np.arange(6)), np.sin(np.arange(6))])

        score = global_scores.global_score_laplacian(
            graph, Y, data_is_graph=True, random_state=0
        )

        assert 0.0 <= score <= 1.0


class TestLocalScores:
    def test_geodesic_distance_is_symmetric_with_zero_diagonal(self):
        graph = _cycle_graph(5)
        dist = local_scores.geodesic_distance(graph, directed=False, n_jobs=1)

        np.testing.assert_allclose(dist, dist.T)
        np.testing.assert_allclose(np.diag(dist), np.zeros(5))

    def test_rank_correlations_match_for_same_graph(self):
        graph = _cycle_graph(5)

        assert local_scores.knn_spearman_r(graph, graph, n_jobs=1) == pytest.approx(1.0)
        assert local_scores.knn_kendall_tau(graph, graph, n_jobs=1) == pytest.approx(
            1.0
        )

    def test_geodesic_correlation_return_graphs_and_landmarks(self):
        graph = _cycle_graph(6)
        corr, data_graph, emb_graph = local_scores.geodesic_correlation(
            graph,
            graph,
            landmarks=np.array([0, 1, 2, 3]),
            n_jobs=1,
            return_graphs=True,
            random_state=0,
        )

        assert np.isfinite(corr)
        assert data_graph.shape == (4, 4)
        assert emb_graph.shape == (4, 4)

    def test_geodesic_correlation_validates_random_state_and_landmarks(self):
        graph = _cycle_graph(4)
        with pytest.raises(TypeError, match="random_state"):
            local_scores.geodesic_correlation(graph, graph, random_state=object())
        with pytest.raises(ValueError, match="landmarks"):
            local_scores.geodesic_correlation(graph, graph, landmarks=[0, 1])


class TestRiemannMetric:
    def test_metric_shapes_and_cached_accessors(self):
        Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        L = sparse.csgraph.laplacian(_cycle_graph(4), normed=False)

        H, G, Vh, S, Sinv = rmetric.riemann_metric(Y, L)
        assert H.shape == (4, 2, 2)
        assert G.shape == (4, 2, 2)
        assert Vh.shape == (4, 2, 2)
        assert S.shape == (4, 2)
        assert Sinv.shape == (4, 2)

        rm = rmetric.RiemannMetric(Y, L)
        assert rm.get_dual_rmetric().shape == (4, 2, 2)
        G2, vectors, hvals, gvals = rm.get_rmetric(return_svd=True)
        assert G2.shape == (4, 2, 2)
        assert vectors.shape == (4, 2, 2)
        assert hvals.shape == (4, 2)
        assert gvals.shape == (4, 2)
        assert rm.get_mdimG() == 2
        assert rm.get_detG().shape == (4,)

    def test_metric_small_helpers(self):
        A = sparse.csr_matrix([[0.0, 2.0], [1.0, 0.0]])
        np.testing.assert_allclose(rmetric._ensure_array(A), A.toarray())
        np.testing.assert_allclose(rmetric._symmetrize(A), [[0.0, 1.5], [1.5, 0.0]])
        np.testing.assert_allclose(
            rmetric._center(np.array([[1.0, 2.0], [3.0, 4.0]])),
            [[-1.0, -1.0], [1.0, 1.0]],
        )

        vals, vecs = rmetric.eigsorted(np.diag([1.0, 3.0]))
        np.testing.assert_allclose(vals, [3.0, 1.0])
        assert vecs.shape == (2, 2)

        G = rmetric._project_spd(np.array([[1.0, 2.0], [2.0, -1.0]]))
        assert np.linalg.eigvalsh(G).min() > 0
        a, b, theta = rmetric._ellipse_from_G(np.eye(2))
        assert a == pytest.approx(b)
        assert np.isfinite(theta)
        np.testing.assert_allclose(
            rmetric._scaling_values(np.repeat(np.eye(2)[None], 3, axis=0)), 1
        )

    def test_deformation_and_eccentricity(self):
        Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        L = sparse.csgraph.laplacian(_cycle_graph(4), normed=False)
        G = np.repeat(np.eye(2)[None], 4, axis=0)

        vals, limits = rmetric.calculate_deformation(Y, L, G_emb=G)
        np.testing.assert_allclose(vals, np.zeros(4))
        assert limits == pytest.approx((-0.0, 0.0))

        with pytest.warns(DeprecationWarning, match="deprecated"):
            ecc = rmetric.get_eccentricity(Y, L, G_emb=G)
        np.testing.assert_allclose(ecc, np.zeros(4))


class TestTopoMetricsHelpers:
    def test_diffusion_coordinate_roundtrip_helpers(self):
        graph = _cycle_graph(5)
        P = topo_metrics.get_P(graph)
        evals, evecs = topo_metrics._top_eigs_of_P(P, r=3)

        coords = topo_metrics.diffusion_coordinates(evals, evecs, t=1, r_use=2)
        dmat = topo_metrics.diffusion_distance_from_eigs(evals, evecs, t=1)
        upper = topo_metrics._upper_triangle_vec(dmat)

        assert P.shape == (5, 5)
        assert coords.shape == (5, 2)
        assert dmat.shape == (5, 5)
        assert upper.shape == (10,)

    def test_topk_support_ignores_self(self):
        row = sparse.csr_matrix([[0.0, 0.4, 0.2, 0.9]])
        support = topo_metrics._topk_support_from_row(row.data, row.indices, k=2)
        assert support == {1, 3}
