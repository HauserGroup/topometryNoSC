"""Tests for packaging-level machinery: optional deps, backends, logging.

These guard the refactor's invariants: the core package imports without the
optional dependencies, missing extras raise actionable errors, and backend
detection is centralised.
"""

import builtins
import importlib
import logging
import sys

import pytest

import topo
from topo import _logging, _optional


class TestVersion:
    def test_version_is_exposed(self):
        assert isinstance(topo.__version__, str)
        assert topo.__version__


class TestOptionalHelpers:
    def test_has_known_present_module(self):
        assert _optional.has("numpy") is True

    def test_has_missing_module(self):
        assert _optional.has("definitely_not_a_real_module_xyz") is False

    def test_optional_import_present(self):
        np = _optional.optional_import("numpy")
        assert np is not None
        assert np.__name__ == "numpy"

    def test_optional_import_missing_returns_none(self):
        assert _optional.optional_import("definitely_not_a_real_module_xyz") is None

    def test_require_present_returns_module(self):
        assert _optional.require("numpy").__name__ == "numpy"

    def test_require_missing_raises_friendly_error(self):
        with pytest.raises(ImportError) as exc:
            _optional.require("definitely_not_a_real_module_xyz", purpose="testing")
        assert "definitely_not_a_real_module_xyz" in str(exc.value)

    @pytest.mark.optional_deps
    def test_require_names_extra_for_known_optional(self, monkeypatch):
        # Force matplotlib to look missing and assert the hint names the extra.
        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(ImportError) as exc:
            _optional.require("matplotlib", purpose="plotting")
        assert "topometry-nosc[plot]" in str(exc.value)


class TestBackendSelection:
    def test_best_ann_backend_falls_back_to_sklearn(self, monkeypatch):
        monkeypatch.setattr(_optional, "has", lambda name: False)
        assert _optional.best_ann_backend("hnswlib") == "sklearn"

    def test_best_ann_backend_honours_preferred_when_present(self, monkeypatch):
        monkeypatch.setattr(_optional, "has", lambda name: name == "annoy")
        assert _optional.best_ann_backend("annoy") == "annoy"

    def test_best_ann_backend_picks_first_available(self, monkeypatch):
        monkeypatch.setattr(_optional, "has", lambda name: name == "nmslib")
        # preferred missing -> first available in preference order
        assert _optional.best_ann_backend("hnswlib") == "nmslib"

    def test_available_ann_backends_is_subset_in_order(self, monkeypatch):
        monkeypatch.setattr(_optional, "has", lambda name: name in {"hnswlib", "faiss"})
        assert _optional.available_ann_backends() == ["hnswlib", "faiss"]


class TestLoggingConfig:
    def test_configure_sets_info_when_verbose(self):
        _logging.configure(verbose=True)
        assert _logging.logger.level == logging.INFO

    def test_configure_sets_debug_for_high_verbosity(self):
        _logging.configure(verbose=2)
        assert _logging.logger.level == logging.DEBUG

    def test_configure_quiet_by_default(self):
        _logging.configure(verbose=False)
        assert _logging.logger.level == logging.WARNING

    def test_configure_attaches_single_handler(self):
        _logging.configure(verbose=True)
        n_handlers = len(_logging.logger.handlers)
        _logging.configure(verbose=False)
        assert len(_logging.logger.handlers) == n_handlers

    def test_configure_logging_is_public(self):
        assert topo.configure_logging is _logging.configure


class TestNoEagerOptionalImports:
    """Importing core modules must not pull in optional deps at module load."""

    @pytest.mark.parametrize(
        "module",
        [
            "topo",
            "topo.tpgraph.kernels",
            "topo.spectral._spectral",
            "topo.spectral.eigen",
            "topo.base.ann",
        ],
    )
    def test_core_module_imports_without_matplotlib(self, module, monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib is blocked for this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        # Drop cached modules so the import actually re-executes.
        for mod in list(sys.modules):
            if mod == module or mod.startswith(module + "."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        importlib.import_module(module)
