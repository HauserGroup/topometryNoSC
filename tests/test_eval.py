"""Tests for evaluation metrics (RDC, EMD, Procrustes, etc.)."""

import numpy as np

from topo.eval.topo_metrics import (
    diffusion_rank_biased_overlap,
    multiscale_diffusion_emd,
    rank_diffusion_correlation,
    rowwise_js_similarity,
    sparse_neighborhood_f1,
    spectral_procrustes,
    topo_preserve_score,
)


def test_topo_metrics(fitted_topograph):
    Px = fitted_topograph.P_of_msZ
    Py = fitted_topograph.P_of_Z

    rdc = rank_diffusion_correlation(Px, Py, times=(1,), r=10)
    assert np.isfinite(rdc)

    emd = multiscale_diffusion_emd(Px, Py, times=(1,), r=10)
    assert np.isfinite(emd)

    sp = spectral_procrustes(Px, Py, times=(1,), r=10)
    assert np.isfinite(sp)

    rbo = diffusion_rank_biased_overlap(Px, Py, times=(1,), r=10, k_max=10)
    assert np.isfinite(rbo)

    pjs = rowwise_js_similarity(Px, Py)
    assert np.isfinite(pjs)

    pf1 = sparse_neighborhood_f1(Px, Py, k=10)
    assert np.isfinite(pf1)

    score, parts = topo_preserve_score(Px, Py, times=(1,), r=10)
    assert np.isfinite(score)
    assert "PF1" in parts
