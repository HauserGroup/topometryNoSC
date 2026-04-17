"""Tests for topo.analysis standalone functions."""

import numpy as np

from topo import analysis


class TestFilterSignal:
    """Test diffusion-based signal filtering."""

    def test_filter_smooths(self, fitted_topograph, swiss_roll_data):
        _, color = swiss_roll_data
        P = fitted_topograph.P_of_msZ
        smoothed = analysis.filter_signal(color, P, t=3)
        assert smoothed.shape == color.shape
        assert np.isfinite(smoothed).all()
        # Smoothed signal should have lower variance than original
        assert np.var(smoothed) <= np.var(color) + 1e-10

    def test_filter_t0_identity(self, fitted_topograph, swiss_roll_data):
        _, color = swiss_roll_data
        P = fitted_topograph.P_of_msZ
        result = analysis.filter_signal(color, P, t=0)
        np.testing.assert_allclose(result, color, atol=1e-10)


class TestImpute:
    """Test diffusion-based imputation."""

    def test_impute_shape(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        P = fitted_topograph.P_of_msZ
        imputed = analysis.impute(X, P, t=3)
        assert imputed.shape == X.shape
        assert np.isfinite(imputed).all()


class TestSpectralSelectivity:
    """Test per-sample spectral selectivity diagnostics."""

    def test_selectivity_returns_dict(self, fitted_topograph):
        Z = fitted_topograph.spectral_scaffold(multiscale=True)
        evals = fitted_topograph.eigenvalues
        # Only pass eigenvalues matching scaffold width
        pos_evals = evals[evals > 0][: Z.shape[1]]
        result = analysis.spectral_selectivity(Z, evals=pos_evals, k_neighbors=10)
        assert isinstance(result, dict)
        expected_keys = {"EAS", "RayScore", "LAC", "axis", "axis_sign", "radius"}
        assert expected_keys.issubset(result.keys())

    def test_selectivity_shapes(self, fitted_topograph, swiss_roll_data):
        X, _ = swiss_roll_data
        n = X.shape[0]
        Z = fitted_topograph.spectral_scaffold(multiscale=True)
        evals = fitted_topograph.eigenvalues
        pos_evals = evals[evals > 0][: Z.shape[1]]
        result = analysis.spectral_selectivity(Z, evals=pos_evals, k_neighbors=10)
        assert result["EAS"].shape == (n,)
        assert result["RayScore"].shape == (n,)


class TestRiemannDiagnostics:
    """Test Riemannian distortion diagnostics."""

    def test_diagnostics_returns_dict(self, fitted_topograph):
        Y = fitted_topograph.msTopoMAP
        L = fitted_topograph.P_of_msZ
        result = analysis.riemann_diagnostics(Y, L)
        assert isinstance(result, dict)

    def test_diagnostics_keys(self, fitted_topograph):
        Y = fitted_topograph.msTopoMAP
        L = fitted_topograph.P_of_msZ
        result = analysis.riemann_diagnostics(Y, L)
        # Should contain metric and/or scalar info
        assert len(result) > 0
