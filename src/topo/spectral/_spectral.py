#####################################
# Author: David S Oliveira
# University of Oxford
# contact: david.oliveira[at]dpag[dot]ox[dot]ac[dot]uk
# License: MIT
######################################
# Clearly defining laplacian-type operators and spectral decompositions
"""Laplacian-type operators and spectral decompositions.

Dense and sparse implementations of graph degree, the (un)normalized /
random-walk graph Laplacians, anisotropic diffusion operators and the
Laplacian-eigenmaps (``LE``) layout that the rest of the package builds on.
"""

import logging
from typing import cast
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError
from scipy.sparse import csc_matrix
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


def degree(W):
    """Compute degree matrix (sparse or dense -> sparse)."""
    W_csr = cast(
        sparse.csr_matrix, sparse.csr_matrix(W) if not sparse.issparse(W) else W.tocsr()
    )
    N = W_csr.shape[0]  # type: ignore[index]
    D = np.ravel(W_csr.sum(axis=1))
    return sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])


def _diffusion_operator_asymmetric(W, alpha: float = 0, semi_aniso=False):
    """Compute asymmetric diffusion operator (sparse matrices only)."""
    W_csr = cast(
        sparse.csr_matrix, sparse.csr_matrix(W) if not sparse.issparse(W) else W.tocsr()
    )
    N = W_csr.shape[0]  # type: ignore[index]
    D = np.ravel(W_csr.sum(axis=1))
    if alpha > 0:
        D[D != 0] = D[D != 0] ** (-alpha)
        Dinva = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
        Wa = Dinva.dot(W_csr).dot(Dinva)
        Da = np.ravel(Wa.sum(axis=1))
        Da[Da != 0] = 1 / Da[Da != 0]
        if semi_aniso:
            P = sparse.csr_matrix((Da, (range(N), range(N))), shape=[N, N]).dot(W_csr)
        else:
            P = sparse.csr_matrix((Da, (range(N), range(N))), shape=[N, N]).dot(Wa)
    else:
        D[D != 0] = 1 / D[D != 0]
        Dd = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
        P = Dd.dot(W_csr)
    return P


def _diffusion_operator_symmetric(
    W, alpha: float = 0, semi_aniso=False, return_D_inv_sqrt=False
):
    """Compute symmetric diffusion operator (sparse matrices only)."""
    if alpha < 0:
        alpha = 0
    W_csr = cast(
        sparse.csr_matrix, sparse.csr_matrix(W) if not sparse.issparse(W) else W.tocsr()
    )
    N = W_csr.shape[0]  # type: ignore[index]
    D = np.ravel(W_csr.sum(axis=1))
    if alpha > 0:
        D[D != 0] = D[D != 0] ** (-alpha)
        Dinva = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
        Wa = Dinva.dot(W_csr).dot(Dinva)
        Da = np.ravel(Wa.sum(axis=1))
        Da[Da != 0] = 1 / Da[Da != 0]
        Dalpha_inv = sparse.csr_matrix((Da, (range(N), range(N))), shape=[N, N])
        if semi_aniso:
            Pa = Dalpha_inv.dot(W_csr)
            D_right = np.ravel(W_csr.sum(axis=1))
        else:
            Pa = Dalpha_inv.dot(Wa)
            D_right = np.ravel(Wa.sum(axis=1))
    else:
        D[D != 0] = 1 / D[D != 0]
        Pa = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W_csr)
        D_right = np.ravel(W_csr.sum(axis=1))
    D_left = D_right.copy()
    D_right[D_right != 0] = np.sqrt(D_right[D_right != 0])
    D_left[D_left != 0] = 1 / np.sqrt(D_left[D_left != 0])
    D_right = sparse.csr_matrix((D_right, (range(N), range(N))), shape=[N, N])
    D_left = sparse.csr_matrix((D_left, (range(N), range(N))), shape=[N, N])
    Psym = D_right.dot(Pa).dot(D_left)
    if return_D_inv_sqrt:
        return Psym, D_left
    else:
        return Psym


def graph_laplacian(W, laplacian_type="normalized", return_D=False):
    """Compute the graph Laplacian of an adjacency/affinity graph ``W``.

    For a friendly reference, see this material from James Melville:
    https://jlmelville.github.io/smallvis/spectral.html

    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.
        No further symmetrization is performed, so make sure to symmetrize W if necessary (usually done additively with W = (W + W.T)/2 ).

    laplacian_type : str, default='random_walk'
        The type of laplacian to use. Can be 'unnormalized', 'normalized' or 'random_walk'.

    return_D : bool, default=False
        Whether to also return a degree matrix with the Laplacian in a tuple

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The graph Laplacian.
    """
    from topo._compat.scipy_graph import graph_laplacian as _graph_laplacian

    return _graph_laplacian(W, laplacian_type=laplacian_type, return_D=return_D)


def LE(
    W,
    n_eigs=10,
    laplacian_type="random_walk",
    drop_first=True,
    return_evals=False,
    eigen_tol: float = 0,
    random_state=None,
):
    """Compute [Laplacian Eigenmaps](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) of an adjacency or affinity graph W.

    The graph W can be a sparse matrix or a dense matrix. It is assumed to be symmetric (no further symmetrization is performed, be sure it is),
    and with zero diagonal (all diagonal elements are 0). The eigenvectors associated with the smallest eigenvalues
    form a new orthonormal basis which represents the graph in the feature space and are useful for denoising and clustering.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.

    n_eigs : int, default=10
        The number of eigenvectors to compute.

    laplacian_type : str, default='random_walk'
        The type of laplacian to use. Can be 'unnormalized', 'normalized', or 'random_walk'.

    drop_first : bool, default=True
        Whether to drop the first eigenvector.

    return_evals : bool, default=False
        Whether to return the eigenvalues. If True, returns a tuple of (eigenvectors, eigenvalues).

    eigen_tol : float, default=0
        The tolerance for the eigendecomposition.

    random_state : int, default=None
        The random state for the eigendecomposition in scipy.sparse.linalg.lobpcg() if the data has more than
        a million samples.

    Returns
    -------
    evecs : np.ndarray of shape (W.shape[0], n_eigs)
        The eigenvectors of the graph Laplacian, sorted by ascending eigenvalues.

    If return_evals:
        evecs, evals : tuple of ndarrays
        The eigenvectors and associated eigenvalues, sorted by ascending eigenvalues.

    """
    random_state = check_random_state(random_state)
    if n_eigs > np.shape(W)[0]:
        raise ValueError("n_eigs must be less than or equal to the number of nodes.")
    # Compute graph Laplacian
    L = graph_laplacian(W, laplacian_type)
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)  # for ARPACK efficiency
    L = cast(sparse.csr_matrix, L)
    shape = L.shape
    if shape is None:
        raise ValueError("Graph Laplacian must have a valid shape.")
    n_nodes = int(shape[0])
    # Add one more eig if drop_first is True
    if drop_first:
        n_eigs = n_eigs + 1
    # Compute eigenvalues and eigenvectors
    try:
        if n_nodes < 1000000:
            evals, evecs = sparse.linalg.eigsh(
                L, k=n_eigs, which="SM", tol=eigen_tol, maxiter=n_nodes * 5
            )
        else:
            evals, evecs = sparse.linalg.lobpcg(
                L,
                random_state.normal(size=(n_nodes, n_eigs)),
                largest=False,
                tol=1e-8,
            )
    except sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral decomposition FAILED! This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data."
        )
        return None
    evals = np.real(evals)
    evecs = np.real(evecs)
    # Sort eigenvalues and eigenvectors in ascending order
    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Normalize
    for i in range(evecs.shape[1]):
        evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
    # Return embedding and evals
    if drop_first:
        evecs = evecs[:, 1:]
        evals = evals[1:]
    if return_evals:
        return evecs, evals
    else:
        return evecs


def plain_spectral_embedding(
    affinity_matrix,
    n_components=10,
    eigen_solver="auto",
    random_state=None,
    drop_first=True,
):
    """Compute plain Laplacian Eigenmaps via sklearn for efficiency.

    Delegates to sklearn.manifold.SpectralEmbedding for standard
    normalized Laplacian Eigenmaps case. Drops first (zero) eigenvector
    and returns eigenvectors corresponding to smallest eigenvalues.

    Parameters
    ----------
    affinity_matrix : scipy.sparse.csr_matrix or np.ndarray
        Precomputed affinity or similarity matrix.
    n_components : int, default=10
        Number of eigenvectors to return.
    eigen_solver : str, default='auto'
        Eigendecomposition solver ('auto', 'arpack', 'dense').
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    drop_first : bool, default=True
        Whether to drop the first (zero) eigenvector.

    Returns
    -------
    embedding : np.ndarray of shape (n_samples, n_components)
        Spectral embedding coordinates.
    """
    from sklearn.manifold import SpectralEmbedding

    n_comps_req = n_components + 1 if drop_first else n_components
    embedding = SpectralEmbedding(
        n_components=n_comps_req,
        affinity="precomputed",
        eigen_solver=eigen_solver,
        random_state=random_state,
    ).fit_transform(affinity_matrix)

    if drop_first:
        embedding = embedding[:, 1:]

    return embedding


def diffusion_operator(
    W, alpha=1.0, symmetric=False, semi_aniso=False, return_D_inv_sqrt=False
):
    """Compute the [diffusion operator](https://doi.org/10.1016/j.acha.2006.04.006).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.
        No further symmetrization is performed, so make sure to symmetrize W if necessary (usually done additively with W = (W + W.T)/2 ).

    alpha : float, default=1.0
        Anisotropy to apply. 'Alpha' in the diffusion maps literature.

    symmetric : bool, default=True
        Whether to use a symmetric version of the diffusion operator. This is particularly useful to yield a symmetric operator
        when using anisotropy (alpha > 0), as the diffusion operator P would be assymetric otherwise, which can be problematic
        during matrix decomposition. Eigenvalues are the same of the assymetric version, and the eigenvectors of the original assymetric
        operator can be obtained by left multiplying by D_inv_sqrt (returned if `return_D_inv_sqrt` set to True).

    semi_aniso : bool, default=False
        Whether to use semi-anisotropic diffusion. This reweights the original kernel  (not the renormalized kernel) by the renormalized degree.

    return_D_inv_sqrt : bool, default=False
        Whether to return a tuple of diffusion operator P and inverse square root of the degree matrix.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        The graph diffusion operator.
    """
    if symmetric:
        if return_D_inv_sqrt:
            return _diffusion_operator_symmetric(
                W, alpha, semi_aniso=semi_aniso, return_D_inv_sqrt=True
            )
        else:
            return _diffusion_operator_symmetric(
                W, alpha, semi_aniso=semi_aniso, return_D_inv_sqrt=False
            )
    else:
        return _diffusion_operator_asymmetric(W, alpha, semi_aniso)


def spectral_clustering(
    init, max_svd_restarts=50, n_iter_max=50, random_state=None, copy=True
):
    """
    Search for a partition matrix (clustering) which is closest to the eigenvector embedding.

    Parameters
    ----------
    init : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.
    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails
    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached
    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.


    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------
    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf


    Notes
    -----
    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.
    """
    random_state = check_random_state(random_state)

    vectors = (
        np.asarray(init, dtype=float).copy() if copy else np.asarray(init, dtype=float)
    )
    if vectors.ndim != 2:
        raise ValueError(
            "init must be a 2-D dense array of shape (n_samples, n_clusters)."
        )

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    norm_ones = np.sqrt(n_samples)

    for i in range(n_components):
        col_norm = np.linalg.norm(vectors[:, i])
        if col_norm == 0:
            raise ValueError("init contains a zero-norm eigenvector column.")
        vectors[:, i] = (vectors[:, i] / col_norm) * norm_ones

        if vectors[0, i] != 0:
            vectors[:, i] *= -np.sign(vectors[0, i])

    row_norms = np.sqrt((vectors**2).sum(axis=1))
    if np.any(row_norms == 0):
        raise ValueError("init contains a zero-norm row after column normalization.")
    vectors = vectors / row_norms[:, np.newaxis]

    svd_restarts = 0
    has_converged = False
    labels: np.ndarray = np.zeros(0, dtype=int)

    while (svd_restarts < max_svd_restarts) and not has_converged:
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        c = np.zeros(n_samples)
        for j in range(1, n_components):
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T @ vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                logger.warning("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")

    return labels
