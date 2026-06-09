"""Thin wrappers around scipy.sparse.csgraph.

All generic sparse graph algorithms should go through this module so that graph
semantics, zero-degree handling, and return formats remain consistent across the
package.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.csgraph import (
    connected_components as scipy_connected_components,
)
from scipy.sparse.csgraph import (
    laplacian as scipy_laplacian,
)
from scipy.sparse.csgraph import (
    shortest_path as scipy_shortest_path,
)

from topo.base.graph_matrix import as_csr_matrix


def as_csr_graph(graph) -> csr_matrix:
    """Return graph as CSR sparse matrix."""
    return as_csr_matrix(graph, "graph")


def graph_connected_components(
    graph,
    *,
    directed: bool = False,
    connection: str = "weak",
    return_labels: bool = True,
):
    """Return connected components using SciPy csgraph."""
    return scipy_connected_components(
        as_csr_graph(graph),
        directed=directed,
        connection=connection,
        return_labels=return_labels,
    )


def graph_shortest_paths(
    graph,
    *,
    directed: bool = False,
    unweighted: bool = False,
    indices=None,
    method: str = "D",
):
    """Return sparse-graph shortest-path distances using SciPy csgraph."""
    return scipy_shortest_path(
        as_csr_graph(graph),
        directed=directed,
        unweighted=unweighted,
        indices=indices,
        method=method,
    )


def _safe_inverse(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    mask = values > 0
    out[mask] = 1.0 / values[mask]
    return out


def graph_laplacian(
    graph,
    *,
    laplacian_type: str = "normalized",
    return_D: bool = False,
):
    """Return graph Laplacian with package-standard laplacian_type names.

    Supported types:

    - "unnormalized": D - W, delegated to scipy.sparse.csgraph.laplacian.
    - "normalized" / "symmetric": I - D^-1/2 W D^-1/2, delegated to SciPy.
    - "random_walk" / "rw": I - D^-1 W, implemented here because SciPy's
      normalized Laplacian is the symmetric normalized Laplacian.

    Isolated nodes are handled without inf/nan values.
    """
    W = as_csr_graph(graph)
    laplacian_type = str(laplacian_type).lower()

    if laplacian_type in {"unnormalized", "un", "none"}:
        L = as_csr_graph(scipy_laplacian(W, normed=False, return_diag=False))
        if return_D:
            degree = np.asarray(W.sum(axis=1)).ravel()
            return L, diags(degree, offsets=0, format="csr")
        return L

    if laplacian_type in {"normalized", "symmetric", "sym"}:
        if return_D:
            result = scipy_laplacian(W, normed=True, return_diag=True)
            L, diag = result[0], result[1]
            return as_csr_graph(L), diags(np.asarray(diag), offsets=0, format="csr")
        return as_csr_graph(scipy_laplacian(W, normed=True, return_diag=False))

    if laplacian_type in {"random_walk", "rw"}:
        n = W.shape[0]  # type: ignore[index]
        degree = np.asarray(W.sum(axis=1)).ravel()  # type: ignore[union-attr]
        inv_degree = _safe_inverse(degree)
        P = diags(inv_degree, offsets=0, format="csr") @ W
        I = identity(n, format="csr")
        L = as_csr_graph(I) - P

        isolated = degree == 0
        if np.any(isolated):
            L = L.tolil()
            L[isolated, :] = 0.0
            L = L.tocsr()

        if return_D:
            return L, diags(degree, offsets=0, format="csr")
        return L

    raise ValueError(f"Unknown laplacian_type={laplacian_type!r}.")
