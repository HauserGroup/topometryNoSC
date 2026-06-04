"""Tests for pipeline mixin control-flow guards."""

import numpy as np
import pytest
from scipy import sparse

from topo._pipeline import eigen as eigen_pipeline
from topo._pipeline.eigen import EigenBuildMixin
from topo._pipeline.graph import GraphBuildMixin
from topo._pipeline.layout import LayoutBuildMixin


class DummyGraphBuilder(GraphBuildMixin):
    def __init__(self):
        self.n = 0
        self.m = 0
        self.verbosity = 0
        self.base_knn = 2
        self.base_metric = "precomputed"
        self.n_jobs = 1
        self.backend = "sklearn"
        self.bases_graph_verbose = False
        self.runtimes = {}
        self.base_kernel_version = "dummy"
        self.low_memory = True
        self.BaseKernelDict = {}
        self.base_kernel = None
        self.base_nbrs_class = None
        self.base_knn_graph = None
        self.build_kernel_calls = []

    def _build_kernel(self, *args, **kwargs):
        self.build_kernel_calls.append((args, kwargs))
        updated = dict(args[3])
        updated[args[2]] = "kernel"
        return "kernel", updated


class DummyEigenBuilder(EigenBuildMixin):
    def __init__(self):
        self.id_max_components = 10
        self.id_method = "fsa"
        self.id_ks = [3]
        self.backend = "sklearn"
        self.id_metric = "euclidean"
        self.n_jobs = 1
        self.id_quantile = 0.8
        self.id_min_components = 2
        self.id_headroom = 1.0
        self.random_state = 0
        self._id_details = {}
        self._scaffold_components_ms = None
        self._scaffold_components_dm = None
        self.n_eigs = 2
        self.global_dimensionality = None
        self.local_dimensionality = None


class DummyLayoutBuilder(LayoutBuildMixin):
    def __init__(self):
        self.projection_methods = []
        self.graph_kernel_version = "gk"
        self.base_kernel_version = "bk"
        self.ProjectionDict = {}
        self._kernel_msZ = None
        self._kernel_Z = None
        self.random_state = 0
        self.laplacian_type = "normalized"
        self.eigen_tol = 0
        self.runtimes = {}
        self.SpecLayout = None
        self.graph_knn = 2
        self.P_of_msZ = None
        self.P_of_Z = None
        self.graph_metric = "euclidean"
        self.uom_enabled = False
        self.msZ_uom = None
        self.Z_uom = None
        self.EigenbasisDict = {}
        self.n_jobs = 1
        self.backend = "sklearn"
        self.layout_verbose = False
        self.verbosity = 0
        self.msTopoMAP_snapshots = []
        self.TopoMAP_snapshots = []
        self.uom_eigenvalues_ms_list = []
        self._uom_active_mode = "msDM"
        self.uom_eigenvalues_dm_list = []
        self.uom_components_ = None
        self.eigenbasis = None
        self.base_kernel = None


def test_graph_build_base_graph_accepts_precomputed_matrix():
    builder = DummyGraphBuilder()
    X = sparse.eye(4, format="csr")

    builder._build_base_graph(X)

    assert builder.n == 4
    assert builder.m == 4
    assert builder.base_knn_graph.shape == (4, 4)


def test_graph_build_base_graph_rejects_missing_or_nonsquare_input():
    builder = DummyGraphBuilder()
    with pytest.raises(ValueError, match="no base_kernel"):
        builder._build_base_graph(None)

    with pytest.raises(ValueError, match="must be square"):
        builder._build_base_graph(np.ones((4, 2)))


def test_graph_build_base_kernel_uses_cache_or_builder():
    builder = DummyGraphBuilder()
    builder.BaseKernelDict["dummy"] = "cached"
    builder._build_base_kernel(np.ones((3, 2)))
    assert builder.base_kernel == "cached"
    assert builder.build_kernel_calls == []

    builder.base_kernel_version = "new"
    builder._build_base_kernel(np.ones((3, 2)))
    assert builder.base_kernel == "kernel"
    assert builder.BaseKernelDict == {"dummy": "cached", "new": "kernel"}


def test_automated_sizing_updates_component_state(monkeypatch):
    builder = DummyEigenBuilder()

    def fake_sizing(*args, **kwargs):
        return 5, {"local_id": np.array([2.0, 3.0])}

    monkeypatch.setattr(eigen_pipeline, "automated_scaffold_sizing", fake_sizing)
    builder._automated_sizing(np.ones((8, 3)))

    assert builder.n_eigs == 5
    assert builder.global_dimensionality == 5
    assert builder._scaffold_components_ms == 5
    assert builder._scaffold_components_dm == 5
    np.testing.assert_array_equal(builder.local_dimensionality, [2.0, 3.0])
    assert "fsa" in builder._id_details


def test_layout_get_projection_standard_and_uom_keys():
    layout = DummyLayoutBuilder()
    standard_key = "MAP of gk from msDM with bk"
    layout.ProjectionDict[standard_key] = np.ones((3, 2))
    assert layout._get_projection("MAP", multiscale=True).shape == (3, 2)

    layout.ProjectionDict.clear()
    uom_key = "t-SNE of UoM DM with bk"
    layout.ProjectionDict[uom_key] = np.zeros((3, 2))
    assert layout._get_projection("t-SNE", multiscale=False).shape == (3, 2)

    with pytest.raises(AttributeError, match="embedding unavailable"):
        layout._get_projection("MAP", multiscale=False)


def test_layout_spectral_layout_requires_graph():
    layout = DummyLayoutBuilder()
    with pytest.raises(ValueError, match="No graph kernel"):
        layout.spectral_layout()
