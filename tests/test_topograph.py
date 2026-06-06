"""Tests for the TopOGraph orchestrator."""

import os
import tempfile

import numpy as np
import pytest

from topo import TopOGraph, load_topograph, save_topograph


class TestTopOGraphInit:
    """Test TopOGraph instantiation."""

    def test_default_init(self):
        tg = TopOGraph()
        assert tg.base_knn == 30
        assert tg.graph_knn == 30
        assert tg.base_kernel_version == "bw_adaptive"
        assert tg.graph_kernel_version == "bw_adaptive"
        assert tg.verbosity == 0

    def test_custom_params(self):
        tg = TopOGraph(base_knn=10, graph_knn=20, base_kernel_version="cknn")
        assert tg.base_knn == 10
        assert tg.graph_knn == 20
        assert tg.base_kernel_version == "cknn"

    def test_invalid_kernel_version(self):
        with pytest.raises((ValueError, KeyError)):
            tg = TopOGraph(base_kernel_version="nonexistent")
            tg.fit(np.random.randn(50, 5))

    def test_repr(self):
        tg = TopOGraph()
        r = repr(tg)
        assert "TopOGraph" in r

    def test_repr_accepts_sklearn_char_limit(self):
        tg = TopOGraph()
        assert "TopOGraph" in tg.__repr__(N_CHAR_MAX=10)

    def test_empty_projection_methods_are_rejected_on_project(self):
        tg = TopOGraph(projection_methods=[])
        with pytest.raises(ValueError, match="No projection methods configured"):
            tg.project()


class TestTopOGraphFit:
    """Test TopOGraph.fit() pipeline."""

    def test_fit_sets_shape(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        assert fitted_topograph.n == X.shape[0]
        assert fitted_topograph.m == X.shape[1]

    def test_eigenbasis_exists(self, fitted_topograph):
        assert fitted_topograph.eigenbasis is not None
        assert fitted_topograph.eigenvalues is not None
        assert len(fitted_topograph.eigenvalues) > 0

    def test_eigenvalues_reasonable(self, fitted_topograph):
        evals = np.asarray(fitted_topograph.eigenvalues)
        # Raw eigenvalues from ARPACK may include small negative floating-point artifacts.
        # The positive eigenvalues (used for DM/msDM scaffolds) should be in (0, 1].
        pos_evals = evals[evals > 0]
        assert len(pos_evals) > 0
        assert np.all(pos_evals <= 1.0 + 1e-10)

    def test_scaffolds_shape(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        n = X.shape[0]
        Z = fitted_topograph.spectral_scaffold(multiscale=False)
        msZ = fitted_topograph.spectral_scaffold(multiscale=True)
        assert Z.shape[0] == n
        assert msZ.shape[0] == n
        assert Z.shape[1] > 0
        assert msZ.shape[1] > 0

    def test_base_kernel_exists(self, fitted_topograph):
        assert fitted_topograph.base_kernel is not None

    def test_graph_kernel_exists(self, fitted_topograph):
        assert fitted_topograph.graph_kernel is not None

    def test_intrinsic_dim_estimated(self, fitted_topograph):
        assert fitted_topograph.global_id is not None
        assert isinstance(
            fitted_topograph.global_id, (int, float, np.integer, np.floating)
        )
        assert fitted_topograph.global_id > 0

    def test_diffusion_operators(self, fitted_topograph):
        P_X = fitted_topograph.P_of_X
        P_msZ = fitted_topograph.P_of_msZ
        P_Z = fitted_topograph.P_of_Z
        assert P_X is not None
        assert P_msZ is not None
        assert P_Z is not None

    def test_runtimes_recorded(self, fitted_topograph):
        assert hasattr(fitted_topograph, "runtimes")
        assert isinstance(fitted_topograph.runtimes, dict)
        assert len(fitted_topograph.runtimes) > 0

    def test_fit_rejects_too_large_knn(self):
        X = np.random.RandomState(0).randn(5, 3)
        tg = TopOGraph(base_knn=5, graph_knn=2, projection_methods=[])
        with pytest.raises(ValueError, match="base_knn=5 must be smaller"):
            tg.fit(X)

    def test_fit_rejects_nonsquare_precomputed_input(self):
        X = np.ones((5, 3))
        tg = TopOGraph(base_metric="precomputed", projection_methods=[])
        with pytest.raises(ValueError, match="must be a square"):
            tg.fit(X)


class TestTopOGraphProjections:
    """Test projection / layout methods."""

    def test_map_layouts_exist(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        n = X.shape[0]
        # After fit(), MAP and PaCMAP layouts should exist
        assert fitted_topograph.msTopoMAP is not None
        assert fitted_topograph.msTopoMAP.shape == (n, 2)

    def test_pacmap_layouts_exist(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        n = X.shape[0]
        fitted_topograph.project(
            projection_method="PaCMAP", multiscale=True, num_iters=50
        )
        assert fitted_topograph.msTopoPaCMAP is not None
        assert fitted_topograph.msTopoPaCMAP.shape == (n, 2)

    def test_project_custom(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        n = X.shape[0]
        Y = fitted_topograph.project(
            projection_method="MAP", multiscale=True, num_iters=50
        )
        assert Y.shape == (n, 2)
        assert np.isfinite(Y).all()

    def test_pacmap_custom_init_array_workaround(self):
        """Test that PaCMAP accepts a custom numpy array initialization without crashing."""
        pytest.importorskip("pacmap")
        from topo.layouts.projector import Projector

        X = np.random.RandomState(42).rand(20, 5)
        init_arr = np.random.RandomState(42).rand(20, 2)
        proj = Projector(
            projection_method="PaCMAP",
            init=init_arr,
            num_iters=5,
        )
        Y = proj.fit_transform(X)
        assert Y.shape == (20, 2)

    def test_select_p_operator_rejects_invalid_name(self, fitted_topograph):
        with pytest.raises(ValueError, match="must be one of"):
            fitted_topograph._select_P_operator("bad")

    def test_spectral_selectivity_smooths_with_named_operator(self, fitted_topograph):
        result = fitted_topograph.spectral_selectivity(
            smooth_P="msZ", smooth_t=1, out_prefix="smoothed"
        )

        assert "EAS" in result
        assert "smoothed_EAS" in fitted_topograph.LocalScoresDict

    def test_find_ideal_projection_runs(self, fitted_topograph):
        # A very minimal grid search to test the machinery
        res = fitted_topograph.find_ideal_projection(
            min_dist_grid=[0.1],
            spread_grid=[1.0],
            initial_alpha_grid=[1.0],
            num_iters=10,
            verbosity=0,
        )
        assert "best_params" in res
        assert "best_score" in res


class TestTopOGraphSaveLoad:
    """Test save / load roundtrip."""

    def test_save_load_roundtrip(self, fitted_topograph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tg.pkl")
            save_topograph(fitted_topograph, path)
            assert os.path.exists(path)

            loaded = load_topograph(path)
            assert loaded.n == fitted_topograph.n
            assert loaded.m == fitted_topograph.m
            np.testing.assert_array_equal(
                loaded.eigenvalues, fitted_topograph.eigenvalues
            )

    def test_save_method(self, fitted_topograph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tg2.pkl")
            fitted_topograph.save(path)
            assert os.path.exists(path)
            loaded = load_topograph(path)
            assert loaded.n == fitted_topograph.n

    def test_save_does_not_mutate_live_neighbor_index(self, fitted_topograph):
        assert fitted_topograph.base_nbrs_class is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tg3.pkl")
            save_topograph(fitted_topograph, path, remove_base_class=True)
            loaded = load_topograph(path)

        assert fitted_topograph.base_nbrs_class is not None
        assert loaded.base_nbrs_class is None


def test_kernel_degree_lazily_builds_adjacency(fitted_topograph):
    K = fitted_topograph.base_kernel
    assert K is not None
    K._A = None
    K._degree = None

    deg = K.degree

    assert deg is not None
    assert K.A is not None
    assert deg.shape[0] == fitted_topograph.n


def test_sparse_umap_knn_conversion_handles_variable_degree():
    from scipy import sparse

    from topo.layouts.projector import _csr_to_fixed_knn

    K = sparse.csr_matrix(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )

    idx, dist = _csr_to_fixed_knn(K, n_neighbors=2)

    assert idx.shape == (4, 2)
    assert dist.shape == (4, 2)
    assert np.isfinite(dist[:, 0]).all()
