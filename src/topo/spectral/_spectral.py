"""Sparse spectral graph operators.

This module provides the spectral graph primitives used by the rest of the
package:

- graph degree helpers;
- Laplacian Eigenmaps after Belkin and Niyogi;
- anisotropic diffusion-map operators after Coifman and Lafon.

Graph Laplacian construction is delegated to
``topo._compat.scipy_graph.graph_laplacian``, which centralizes the package's
Laplacian naming and zero-degree-node conventions. Diffusion-map normalization
is kept here because the ``alpha`` and ``semi_aniso`` behavior is specific to
TopoMetry-style operators.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackError, eigsh

from topo._compat.scipy_graph import graph_laplacian

logger = logging.getLogger(__name__)


def _as_square_csr(W: ArrayLike | sparse.spmatrix, name: str = "W") -> csr_matrix:
    """Return ``W`` as a square CSR sparse matrix."""
    W_csr = csr_matrix(W)
    shape = W_csr.shape
    if shape is None:
        raise ValueError(f"{name} must have a valid shape.")
    n_rows, n_cols = int(shape[0]), int(shape[1])

    if n_rows != n_cols:
        raise ValueError(f"{name} must be square.")

    return W_csr


def _csr_shape(W: csr_matrix) -> tuple[int, int]:
    """Return CSR matrix shape as plain Python ints."""
    shape = W.shape
    if shape is None:
        raise ValueError("W must have a valid shape.")
    return int(shape[0]), int(shape[1])


def degree_vector(W: ArrayLike | sparse.spmatrix) -> NDArray[np.float64]:
    """Return row-sum graph degrees as a 1-D vector.

    Parameters
    ----------
    W
        Dense or sparse graph adjacency/affinity matrix.

    Returns
    -------
    ndarray of shape ``(n_nodes,)``
        Degree vector where each entry is the row sum of the corresponding node.
    """
    W_csr = csr_matrix(W)
    return np.asarray(W_csr.sum(axis=1), dtype=float).ravel()


def degree_matrix(W: ArrayLike | sparse.spmatrix) -> csr_matrix:
    """Return graph degrees as a CSR diagonal matrix.

    Parameters
    ----------
    W
        Dense or sparse graph adjacency/affinity matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Diagonal matrix with graph degrees on the diagonal.
    """
    d = degree_vector(W)
    return csr_matrix(sparse.diags(d, offsets=0, format="csr"))


def degree(W: ArrayLike | sparse.spmatrix) -> csr_matrix:
    """Return graph degrees as a CSR diagonal matrix.

    This is kept as the public compatibility name for degree-matrix
    construction. New code should prefer ``degree_vector`` or
    ``degree_matrix`` for clarity.
    """
    return degree_matrix(W)


def _safe_inverse(values: ArrayLike) -> NDArray[np.float64]:
    """Return elementwise inverse, using zero where values are non-positive."""
    values_arr = np.asarray(values, dtype=float)
    out = np.zeros_like(values_arr, dtype=float)
    mask = values_arr > 0
    out[mask] = 1.0 / values_arr[mask]
    return out


def _safe_inverse_power(values: ArrayLike, power: float) -> NDArray[np.float64]:
    """Return ``values ** -power``, using zero where values are non-positive."""
    values_arr = np.asarray(values, dtype=float)
    out = np.zeros_like(values_arr, dtype=float)
    mask = values_arr > 0
    out[mask] = values_arr[mask] ** (-float(power))
    return out


def _sparse_diffusion(
    W: ArrayLike | sparse.spmatrix,
    alpha: float = 0.0,
    semi_aniso: bool = False,
) -> csr_matrix:
    """Return the row-stochastic anisotropic diffusion operator.

    For ``alpha > 0``, the affinity matrix is first density-normalized as

    ``W_alpha = D^-alpha W D^-alpha``.

    The resulting matrix is then row-normalized. If ``semi_aniso=True``, the
    row normalization computed from ``W_alpha`` is applied to the original
    affinity matrix instead of to ``W_alpha``.
    """
    W_csr = _as_square_csr(W)
    alpha = max(float(alpha), 0.0)

    if alpha > 0:
        d = degree_vector(W_csr)
        d_alpha_inv = _safe_inverse_power(d, alpha)
        D_alpha_inv = csr_matrix(sparse.diags(d_alpha_inv, format="csr"))

        W_alpha = csr_matrix(D_alpha_inv @ W_csr @ D_alpha_inv)
        d_alpha = degree_vector(W_alpha)
        D_alpha_row_inv = csr_matrix(sparse.diags(_safe_inverse(d_alpha), format="csr"))

        base = W_csr if semi_aniso else W_alpha
        return csr_matrix(D_alpha_row_inv @ base)

    D_inv = csr_matrix(sparse.diags(_safe_inverse(degree_vector(W_csr)), format="csr"))
    return csr_matrix(D_inv @ W_csr)


def _sparse_diffusion_symmetric(
    W: ArrayLike | sparse.spmatrix,
    alpha: float = 0.0,
    semi_aniso: bool = False,
    return_D_inv_sqrt: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    """Return a symmetric anisotropic diffusion operator.

    The non-symmetric row-stochastic diffusion operator is conjugated into a
    symmetric form suitable for symmetric eigensolvers. When
    ``return_D_inv_sqrt=True``, the inverse square-root degree matrix used for
    this conjugation is returned as the second element.
    """
    W_csr = _as_square_csr(W)
    alpha = max(float(alpha), 0.0)

    if alpha > 0:
        d = degree_vector(W_csr)
        d_alpha_inv = _safe_inverse_power(d, alpha)
        D_alpha_inv = csr_matrix(sparse.diags(d_alpha_inv, format="csr"))

        W_alpha = csr_matrix(D_alpha_inv @ W_csr @ D_alpha_inv)
        d_alpha = degree_vector(W_alpha)
        D_alpha_row_inv = csr_matrix(sparse.diags(_safe_inverse(d_alpha), format="csr"))

        base = W_csr if semi_aniso else W_alpha
        P = csr_matrix(D_alpha_row_inv @ base)
        d_right = degree_vector(base)
    else:
        d_right = degree_vector(W_csr)
        D_inv = csr_matrix(sparse.diags(_safe_inverse(d_right), format="csr"))
        P = csr_matrix(D_inv @ W_csr)

    D_sqrt = csr_matrix(sparse.diags(np.sqrt(np.maximum(d_right, 0.0)), format="csr"))
    D_inv_sqrt = csr_matrix(
        sparse.diags(_safe_inverse_power(d_right, 0.5), format="csr")
    )
    P_sym = csr_matrix(D_sqrt @ P @ D_inv_sqrt)

    if return_D_inv_sqrt:
        return P_sym, D_inv_sqrt
    return P_sym


def LE(
    W: ArrayLike | sparse.spmatrix,
    n_eigs: int = 10,
    laplacian_type: str = "random_walk",
    drop_first: bool = True,
    return_evals: bool = False,
    eigen_tol: float = 0.0,
    random_state=None,
):
    """Compute a Laplacian Eigenmaps embedding from an affinity graph.

    Laplacian Eigenmaps were introduced by Belkin and Niyogi as a spectral
    embedding method based on eigenvectors of a graph Laplacian:
    https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf

    Parameters
    ----------
    W
        Dense or sparse graph adjacency/affinity matrix. ``W`` must be square.
        No symmetrization is performed here; callers should pass a symmetric
        affinity matrix when using a symmetric Laplacian.
    n_eigs
        Number of non-trivial eigenvectors to return.
    laplacian_type
        Laplacian type understood by ``topo._compat.scipy_graph.graph_laplacian``.
        Common values are ``"unnormalized"``, ``"normalized"``, and
        ``"random_walk"``.
    drop_first
        Whether to drop the first eigenvector, which is typically the trivial
        constant mode for connected graphs.
    return_evals
        If ``True``, return ``(eigenvectors, eigenvalues)``.
    eigen_tol
        Tolerance passed to ``scipy.sparse.linalg.eigsh``.
    random_state
        Accepted for API compatibility. The current implementation uses ARPACK
        through ``eigsh`` and does not use randomness.

    Returns
    -------
    ndarray or tuple[ndarray, ndarray]
        Eigenvectors sorted by ascending eigenvalue. If ``return_evals=True``,
        also returns the corresponding eigenvalues.

    Raises
    ------
    ValueError
        If ``W`` is not square or if the requested number of eigenvectors is
        invalid.
    RuntimeError
        If ARPACK fails to compute the requested eigendecomposition.
    """
    del random_state

    W_csr = _as_square_csr(W)
    n_nodes, _ = _csr_shape(W_csr)

    n_eigs = int(n_eigs)
    if n_eigs < 1:
        raise ValueError("n_eigs must be >= 1.")

    k = n_eigs + int(drop_first)
    if k >= n_nodes:
        raise ValueError("n_eigs + drop_first must be smaller than n_nodes.")

    L = csr_matrix(graph_laplacian(W_csr, laplacian_type=laplacian_type))

    try:
        eigsh_tol: Any = float(eigen_tol)
        evals, evecs = eigsh(
            L,
            k=k,
            which="SM",
            tol=eigsh_tol,
        )
    except ArpackError as exc:
        raise RuntimeError(
            "Laplacian Eigenmaps eigendecomposition failed. "
            "The graph may be disconnected, ill-conditioned, or have too small "
            "an eigengap."
        ) from exc

    evals = np.real(evals)
    evecs = np.real(evecs)

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    norms = np.linalg.norm(evecs, axis=0)
    nonzero = norms > 0
    evecs[:, nonzero] /= norms[nonzero]

    if drop_first:
        evals = evals[1:]
        evecs = evecs[:, 1:]

    if return_evals:
        return evecs, evals
    return evecs


def diffusion_operator(
    W: ArrayLike | sparse.spmatrix,
    alpha: float = 1.0,
    symmetric: bool = False,
    semi_aniso: bool = False,
    *,
    return_D_inv_sqrt: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    """Compute a sparse diffusion-map operator from an affinity graph.

    This implements the anisotropic normalization used in diffusion maps, following
    Coifman and Lafon:
    https://doi.org/10.1016/j.acha.2006.04.006

    Parameters
    ----------
    W
        Dense or sparse graph adjacency/affinity matrix. ``W`` must be square.
        No symmetrization is performed here.
    alpha
        Diffusion-maps anisotropy parameter. For ``alpha > 0``, the affinity is
        reweighted by ``D^-alpha W D^-alpha`` before row normalization.
        Negative values are treated as ``0``.
    symmetric
        If ``True``, return the symmetric conjugate form of the diffusion operator.
        This is useful when downstream eigensolvers require a symmetric operator.
        The symmetric and row-stochastic forms are related by a diagonal similarity
        transform under the usual diffusion-map construction.
    semi_aniso
        If ``True``, compute the density correction from the anisotropically
        reweighted affinity but apply the resulting row normalization to the
        original affinity matrix.
    return_D_inv_sqrt
        If ``True``, also return the inverse square-root degree matrix used to
        construct the symmetric operator. This option requires ``symmetric=True``.

    Returns
    -------
    scipy.sparse.csr_matrix or tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        The diffusion operator. If ``return_D_inv_sqrt=True``, returns
        ``(P_symmetric, D_inv_sqrt)``.

    Notes
    -----
    The return type is always sparse CSR, regardless of the input format.
    """
    W_csr = _as_square_csr(W)

    if symmetric:
        return _sparse_diffusion_symmetric(
            W_csr,
            alpha=alpha,
            semi_aniso=semi_aniso,
            return_D_inv_sqrt=return_D_inv_sqrt,
        )

    if return_D_inv_sqrt:
        raise ValueError("return_D_inv_sqrt=True requires symmetric=True.")

    return _sparse_diffusion(W_csr, alpha=alpha, semi_aniso=semi_aniso)
