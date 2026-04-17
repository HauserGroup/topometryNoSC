"""Shared fixtures for TopOMetry tests."""

import pytest
from sklearn.datasets import make_swiss_roll


@pytest.fixture(scope="session")
def swiss_roll_data():
    """Small Swiss-roll dataset for fast tests."""
    X, color = make_swiss_roll(n_samples=300, noise=0.5, random_state=42)
    return X, color


@pytest.fixture(scope="session")
def fitted_topograph(swiss_roll_data):
    """A TopOGraph fitted on the small Swiss-roll."""
    from topo import TopOGraph

    X, _ = swiss_roll_data
    tg = TopOGraph(
        base_knn=15,
        graph_knn=15,
        verbosity=0,
        base_metric="euclidean",
    )
    tg.fit(X)
    return tg
