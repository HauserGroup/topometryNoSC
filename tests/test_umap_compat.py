"""Regression tests for the UMAP compatibility adapter."""

import numpy as np
import pytest
from scipy import sparse

from topo._compat.umap import (
    check_umap_version,
    find_umap_ab_params,
    fuzzy_graph_from_data,
    fuzzy_graph_from_knn,
    validate_knn_for_umap,
)


def _assert_sparse_allclose(actual, expected):
    diff = (actual.tocsr() - expected.tocsr()).tocoo()
    if diff.nnz:
        np.testing.assert_allclose(diff.data, np.zeros_like(diff.data), atol=1e-6)


def test_umap_dependency_importable_and_supported():
    import umap
    from umap.umap_ import find_ab_params, fuzzy_simplicial_set

    assert umap is not None
    assert find_ab_params is not None
    assert fuzzy_simplicial_set is not None
    check_umap_version()


def test_find_ab_params_parity():
    from umap.umap_ import find_ab_params

    expected = find_ab_params(spread=1.0, min_dist=0.1)
    actual = find_umap_ab_params(spread=1.0, min_dist=0.1)

    np.testing.assert_allclose(actual, expected)


def test_fuzzy_graph_from_knn_matches_umap():
    from umap.umap_ import fuzzy_simplicial_set

    X = np.zeros((4, 1), dtype=np.float32)
    indices = np.array(
        [[1, 2], [0, 2], [1, 3], [2, 1]],
        dtype=np.int32,
    )
    dists = np.array(
        [[0.25, 0.75], [0.25, 0.5], [0.5, 0.6], [0.6, 0.9]],
        dtype=np.float32,
    )

    actual_graph, actual_sigmas, actual_rhos = fuzzy_graph_from_knn(
        X,
        knn_indices=indices,
        knn_dists=dists,
        n_neighbors=2,
        random_state=42,
        metric="euclidean",
    )
    expected = fuzzy_simplicial_set(
        X=X,
        n_neighbors=2,
        random_state=42,
        metric="euclidean",
        knn_indices=indices,
        knn_dists=dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=False,
        return_dists=False,
    )
    expected_graph, expected_sigmas, expected_rhos = expected[:3]

    _assert_sparse_allclose(actual_graph, expected_graph)
    np.testing.assert_allclose(actual_sigmas, expected_sigmas)
    np.testing.assert_allclose(actual_rhos, expected_rhos)


def test_fuzzy_graph_from_data_matches_umap():
    from umap.umap_ import fuzzy_simplicial_set

    X = np.random.RandomState(0).randn(12, 3).astype(np.float32)

    actual_graph, actual_sigmas, actual_rhos = fuzzy_graph_from_data(
        X,
        n_neighbors=4,
        random_state=42,
        metric="euclidean",
    )
    expected = fuzzy_simplicial_set(
        X=X,
        n_neighbors=4,
        random_state=42,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=False,
        return_dists=False,
    )
    expected_graph, expected_sigmas, expected_rhos = expected[:3]

    _assert_sparse_allclose(actual_graph, expected_graph)
    np.testing.assert_allclose(actual_sigmas, expected_sigmas)
    np.testing.assert_allclose(actual_rhos, expected_rhos)


def test_validate_knn_for_umap_rejects_bad_arrays():
    indices = np.array([[1, -1], [0, 1]], dtype=np.int32)
    dists = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="missing"):
        validate_knn_for_umap(indices, dists, n_samples=2, n_neighbors=2)

    with pytest.raises(ValueError, match="nondecreasing"):
        validate_knn_for_umap(
            np.array([[1, 0], [0, 1]], dtype=np.int32),
            np.array([[0.2, 0.1], [0.1, 0.2]], dtype=np.float32),
            n_samples=2,
            n_neighbors=2,
        )


def test_kernel_fuzzy_delegates_to_umap_graph():
    from topo.tpgraph.kernels import Kernel

    X = np.random.RandomState(1).randn(15, 3)
    ker = Kernel(
        n_neighbors=4,
        metric="euclidean",
        backend="sklearn",
        fuzzy=True,
    ).fit(X)

    assert sparse.issparse(ker.K)
    assert ker.umap_sigmas_ is ker.sigma_
    assert ker.umap_rhos_ is ker.rho_


def test_projector_umap_uses_upstream_estimator_and_is_deterministic():
    from umap import UMAP

    from topo.layouts.projector import Projector

    X = np.random.RandomState(2).randn(20, 4)
    kwargs = dict(
        projection_method="UMAP",
        n_components=2,
        n_neighbors=5,
        nbrs_backend="sklearn",
        num_iters=10,
        random_state=42,
        init="random",
    )

    first = Projector(**kwargs).fit(X)
    second = Projector(**kwargs).fit(X)

    assert isinstance(first.estimator_, UMAP)
    assert first.Y_.shape == (20, 2)
    np.testing.assert_allclose(first.Y_, second.Y_)


def test_no_local_umap_graph_helper_definitions():
    root = "src/topo"
    forbidden = (
        "def smooth_knn_dist",
        "def compute_membership_strengths",
        "def fuzzy_simplicial_set",
        "def find_ab_params",
    )
    from pathlib import Path

    for path in Path(root).rglob("*.py"):
        if path.as_posix().endswith("src/topo/_compat/umap.py"):
            continue
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} remains in {path}"
