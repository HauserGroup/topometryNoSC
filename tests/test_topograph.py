"""Tests for the TopOGraph orchestrator."""

import os
import tempfile

import numpy as np
import pytest

from topo import TopOGraph, save_topograph, load_topograph


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
        evals = fitted_topograph.eigenvalues
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
