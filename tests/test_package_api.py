"""Tests for package-level lazy exports."""

import pytest

import topo
import topo.eval as topo_eval
import topo.layouts as topo_layouts
import topo.spectral as topo_spectral
import topo.tpgraph as topo_tpgraph


def test_topo_lazy_exports_and_dir():
    assert topo.TopOGraph.__name__ == "TopOGraph"
    assert "TopOGraph" in dir(topo)
    with pytest.raises(AttributeError, match="has no attribute"):
        topo.__getattr__("not_exported")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (topo_eval, "geodesic_correlation"),
        (topo_layouts, "Projector"),
        (topo_spectral, "EigenDecomposition"),
        (topo_tpgraph, "Kernel"),
    ],
)
def test_subpackage_lazy_exports(module, name):
    exported = getattr(module, name)
    assert exported is not None
    assert name in dir(module)
    with pytest.raises(AttributeError, match="has no attribute"):
        module.__getattr__("not_exported")
