"""Tests for approximate nearest neighbors wrappers."""

import importlib.util

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from topo.base.ann import HNSWlibTransformer, NMSlibTransformer, _resolve_n_jobs, kNN


def test_kNN_sklearn():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    res = kNN(X, n_neighbors=2, backend="sklearn")
    assert isinstance(res, csr_matrix)
    assert res.shape == (3, 3)


def test_resolve_n_jobs():
    assert _resolve_n_jobs(None) == 1
    assert _resolve_n_jobs("2") == 2
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(-1) > 0


def test_hnswlib_transformer_raises_or_fits():
    X = np.random.rand(20, 3)
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            HNSWlibTransformer(n_neighbors=2).fit(X)
        return

    model = HNSWlibTransformer(n_neighbors=2).fit(X)
    knn = model.transform(X)
    assert knn.shape == (20, 20)


def test_nmslib_transformer_raises_or_fits():
    X = np.random.rand(20, 3)

    if importlib.util.find_spec("nmslib") is None:
        with pytest.raises(ImportError):
            NMSlibTransformer(n_neighbors=2).fit(X)
        return

    model = NMSlibTransformer(n_neighbors=2).fit(X)
    knn = model.transform(X)
    assert knn.shape == (20, 20)
