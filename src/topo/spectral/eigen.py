#####################################
# Author: David S Oliveira
######################################
# Defining eigendecomposition routines for kernels in a scikit-learn fashion
"""Eigendecomposition transformers for kernels and operators.

Provides :func:`eigendecompose` and the scikit-learn-style
:class:`EigenDecomposition` transformer, which turns a kernel, Laplacian or
diffusion operator into a (multiscale) diffusion-map / Laplacian-eigenmap
embedding, plus spectral-layout helpers for disconnected graphs.
"""

import logging
from typing import Any
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from topo.spectral import LE, diffusion_operator, graph_laplacian
from topo.tpgraph.kernels import Kernel

logger = logging.getLogger(__name__)

EIGEN_SOLVERS = ["dense", "arpack", "lobpcg"]
try:
    from pyamg import smoothed_aggregation_solver

    PYAMG_LOADED = True
    EIGEN_SOLVERS.append("amg")
except ImportError:
    PYAMG_LOADED = False
    smoothed_aggregation_solver: Any = None


def _diffusion_operator_with_degree(W, alpha) -> tuple[Any, Any]:
    """Return symmetric diffusion operator and D^{-1/2} with explicit tuple validation."""
    result = diffusion_operator(
        W,
        alpha=alpha,
        symmetric=True,
        return_D_inv_sqrt=True,
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "diffusion_operator(..., return_D_inv_sqrt=True) must return "
            "a tuple of (operator, D_inv_sqrt)."
        )

    return result


def _safe_msdm_weights(evals):
    """Return stable multiscale diffusion-map weights λ / (1 - λ)."""
    eig_vals = np.asarray(evals, dtype=float).ravel()
    denom = 1.0 - eig_vals

    eps = np.finfo(float).eps
    small = np.abs(denom) < eps
    signs = np.sign(denom[small])
    signs[signs == 0] = 1.0
    denom[small] = signs * eps

    return eig_vals / denom


def eigendecompose(
    G,
    n_components=8,
    eigensolver="arpack",
    largest=True,
    eigen_tol=1e-4,
    random_state=None,
    verbose=False,
):
    """
    Eigendecomposition of a square graph/operator matrix.

    Returns
    -------
    evals : ndarray of shape (k,)
    evecs : ndarray of shape (n_vertices, k)
    """
    if G is None:
        raise ValueError("G cannot be None.")

    if not hasattr(G, "shape") or len(G.shape) != 2:
        raise ValueError("G must be a 2-D square matrix.")

    N, M = G.shape
    if N != M:
        raise ValueError(f"G must be square; got shape {G.shape}.")

    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    # One extra component is requested so callers can drop the trivial one.
    k = n_components + 1

    if N < 2:
        raise ValueError("G must contain at least two vertices.")

    # scipy.sparse.linalg.eigsh requires k < N.
    k = min(k, N - 1)

    if eigensolver not in EIGEN_SOLVERS:
        raise ValueError(
            f"Unknown eigensolver {eigensolver!r}. Expected one of {EIGEN_SOLVERS}."
        )

    random_state = check_random_state(random_state)

    if eigensolver == "dense":
        if sparse.issparse(G):
            if verbose:
                logger.info(
                    "Converting sparse input to dense array for dense eigensolver."
                )
            G_dense = G.toarray()
        else:
            G_dense = np.asarray(G, dtype=float)

        if not np.isfinite(G_dense).all():
            raise ValueError("G contains NaN or infinite values.")

        evals_all, evecs_all = eigh(G_dense)
        order = np.argsort(evals_all)
        if largest:
            order = order[::-1]
        order = order[:k]

        evals = np.real(evals_all[order])
        evecs = np.real(evecs_all[:, order])
        return evals, evecs

    if not sparse.issparse(G):
        if verbose:
            logger.info("Converting dense input to CSR matrix for sparse eigensolver.")
        G = sparse.csr_matrix(G)
    elif not isinstance(G, sparse.csr_matrix):
        G = G.tocsr()

    G = G.astype(float)
    if not np.isfinite(G.data).all():
        raise ValueError("G contains NaN or infinite values.")

    if eigensolver == "arpack":
        which = "LM" if largest else "SM"
        evals, evecs = sparse.linalg.eigsh(
            G,
            k=k,
            which=which,
            tol=eigen_tol,
            maxiter=max(100, N * 5),
        )

    elif eigensolver == "lobpcg":
        X = random_state.normal(size=(N, k))
        evals, evecs = sparse.linalg.lobpcg(
            G,
            X,
            largest=largest,
            tol=eigen_tol,
            maxiter=max(20, N // 5),
        )

    elif eigensolver == "amg":
        if not PYAMG_LOADED:
            raise ImportError(
                'Using "amg" as eigensolver requires pyamg. '
                "Install it with `pip install pyamg`."
            )

        np.random.set_state(random_state.get_state())

        ml = smoothed_aggregation_solver(G)
        M_prec = ml.aspreconditioner()

        X = random_state.normal(size=(N, k))
        X[:, 0] = np.asarray(G.diagonal()).ravel()

        evals, evecs = sparse.linalg.lobpcg(
            G,
            X,
            M=M_prec,
            largest=largest,
            tol=eigen_tol,
            maxiter=max(20, N // 5),
        )

    else:
        raise ValueError(f"Unhandled eigensolver: {eigensolver!r}")

    evals = np.real(evals)
    evecs = np.real(evecs)

    order = np.argsort(evals)
    if largest:
        order = order[::-1]

    evals = evals[order][:k]
    evecs = evecs[:, order][:, :k]

    return evals, evecs


class EigenDecomposition(BaseEstimator, TransformerMixin):
    """Scikit-learn flavored transformer for eigendecomposing sparse symmetric matrices.

    Computes and explores the associated eigenvectors and eigenvalues.
    Takes as main input a `topo.tpgraph.Kernel()` object or a symmetric matrix, which can be either an adjacency/affinity matrix,
    a kernel, a graph laplacian, or a diffusion operator.

    Parameters
    ----------
    n_components : int (optional, default 10).
        Number of eigenpairs to be computed.

    method : string (optional, default 'DM').
        Method for organizing the eigendecomposition. Can be either 'top', 'bottom', 'msDM', 'DM' or 'LE'.
        * 'top' : computes the top eigenpairs of the matrix.
        * 'bottom' : computes the bottom eigenpairs of the matrix.
        * 'msDM' : computes the eigenpairs of the diffusion operator on the matrix, and multiscales them. If a `Kernel()` object is provided, will use the computed diffusion operator if available.
        * 'DM' : computes the eigenpairs of the diffusion operator on the matrix. If a `Kernel()` object is provided, will use the computed diffusion operator if available.
        * 'LE' : computes the eigenpairs of the graph laplacian on the matrix. If a `Kernel()` object is provided, will use the computed graph laplacian if available.

    eigensolver : string (optional, default 'arpack').
        Method for computing the eigendecomposition. Can be either 'arpack', 'lobpcg', 'amg' or 'dense'.
        * 'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.
            This method should be avoided for large problems.
        * 'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
        * 'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        * 'amg' :
            Algebraic Multigrid solver (requires ``pyamg`` to be installed)
            It can be faster on very large, sparse problems, but requires
            setting a random seed for better reproducibility.

    laplacian_type : string (optional, default 'normalized')
        The type of Laplacian to compute. Possible values are: 'normalized', 'unnormalized', 'random_walk' and 'geometric'.

    anisotropy : float (optional, default 0).
        The anisotropy (alpha) parameter in the diffusion maps literature for kernel reweighting.

    eigen_tol : float (optional, default 0.0).
        Error tolerance for the eigenvalue solver. If 0, machine precision is used.


    t : int (optional, default 1).
        Time parameter for the diffusion operator, if 'method' is 'DM'. The diffusion operator will be powered by t. Ignored for other methods.

    return_evals : bool (optional, default False).
        Whether to return the eigenvalues along with the eigenvectors.

    random_state : int or numpy.random.RandomState() (optional, default None).
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.

    """

    def __init__(
        self,
        n_components=10,
        method="DM",
        eigensolver="arpack",
        eigen_tol=1e-4,
        drop_first=True,
        weight=True,
        laplacian_type="random_walk",
        anisotropy=1,
        t=1,
        random_state=None,
        return_evals=False,
        estimate_eigengap=True,
        enforce_min_eigs=True,
        verbose=False,
    ):
        self.n_components = n_components
        self.method = method
        self.eigensolver = eigensolver
        self.eigen_tol = eigen_tol
        self.drop_first = drop_first
        self.laplacian_type = laplacian_type
        self.weight = weight
        self.t = t
        self.anisotropy = anisotropy
        self.random_state = random_state
        self.verbose = verbose
        self.eigenvalues = None
        self.eigenvectors = None
        self.laplacian = None
        self.diffusion_operator = None
        self.embedding = None
        self.powered_operator = None
        self.N = None
        self.D_inv_sqrt_ = None
        self.return_evals = return_evals
        self.estimate_eigengap = estimate_eigengap
        self.eigengap = None
        self.enforce_min_eigs = enforce_min_eigs

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:  # type: ignore[override]
        """Return a short summary of the fitted state and decomposition method."""
        if self.eigenvectors is not None:
            if self.N is not None:
                msg = "EigenDecomposition() estimator fitted with %i samples" % (self.N)
            else:
                msg = "EigenDecomposition() estimator without fitted data."
        else:
            msg = "EigenDecomposition() estimator without any fitted data."
        if self.eigenvectors is not None:
            if self.method == "DM":
                msg += " using Diffusion Maps"
            elif self.method == "msDM":
                msg += " using multiscale Diffusion Maps"
            elif self.method == "LE":
                msg += " using Laplacian Eigenmaps"
            elif self.method == "top":
                msg += " using top eigenpairs"
            elif self.method == "bottom":
                msg += " using bottom eigenpairs"
            if self.weight:
                msg += ", weighted by the square root of the eigenvalues"
            msg += "."
        return msg

    def fit(self, X):
        """Compute the eigendecomposition of kernel matrix ``X`` per ``method``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Matrix to be decomposed. Should generally be an adjacency, affinity/kernel/similarity, Laplacian matrix or a diffusion-type operator.

        Returns
        -------
        self : object
            Returns the instance itself, with eigenvectors stored at EigenDecomposition.eigenvectors
            and eigenvalues stored at EigenDecomposition.eigenvalues. If method is 'DM', stores Diffusion Maps at EigenDecomposition.embedding.
            If 'method' is 'DM' or 'LE', the diffusion operator or graph laplacian is stored at EigenDecomposition.diffusion_operator
            or EigenDecomposition.graph_laplacian, respectively.
        """
        symmetric = False
        if self.method not in ["msDM", "DM", "LE", "top", "bottom"]:
            raise ValueError(
                "Method must be one of 'msDM','DM', 'LE', 'top', 'bottom'."
            )
        if self.method in ["msDM", "DM", "top"]:
            largest = True
        else:
            largest = False
        target: Any = None
        if isinstance(X, Kernel):
            self.N = X.N
            if self.method == "DM" or self.method == "msDM":
                self.diffusion_operator = X.P
                if X.D_inv_sqrt_ is not None:
                    self.D_inv_sqrt_ = X.D_inv_sqrt_
                    symmetric = True
                else:
                    symmetric = False
                target = self.diffusion_operator
            elif self.method == "LE":
                self.laplacian = X.L
                target = self.laplacian
            else:
                target = X.K
        else:
            if not hasattr(X, "shape") or len(X.shape) != 2:
                raise ValueError("X must be a Kernel or a 2-D square matrix.")

            if X.shape[0] != X.shape[1]:
                raise ValueError(f"X must be square; got shape {X.shape}.")

            self.N = X.shape[0]

            if sparse.issparse(X):
                X = X.tocsr()
            else:
                X = np.asarray(X, dtype=float)

            if self.method in ["DM", "msDM"]:
                # Use the symmetric diffusion operator by default. It is numerically
                # better for eigendecomposition, and the right eigenvectors are
                # recovered below through D_inv_sqrt_.
                self.diffusion_operator, self.D_inv_sqrt_ = (
                    _diffusion_operator_with_degree(
                        X,
                        alpha=self.anisotropy,
                    )
                )
                symmetric = True
                target = self.diffusion_operator

        if target is None:
            raise ValueError("Could not determine matrix/operator to decompose.")

        evals, evecs = eigendecompose(
            target,
            eigensolver=self.eigensolver,
            n_components=self.n_components,
            largest=largest,
            eigen_tol=self.eigen_tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        if self.drop_first:
            evals = evals[1:]
            evecs = evecs[:, 1:]

        if self.estimate_eigengap:
            max_eigs = int(np.sum(evals > 0, axis=0))

            if len(evals) < 2:
                self.eigengap = max_eigs
            elif self.method not in ["LE", "top", "bottom"]:
                first_diff = np.diff(evals)
                eg = int(np.argmax(first_diff) + 1)
                self.eigengap = max_eigs if max_eigs == len(evals) else eg
            else:
                self.eigengap = max_eigs

        # Normalize eigenvectors if DM/msDM; store, but DO NOT build embedding here.
        if self.method in ["DM", "msDM"]:
            if symmetric and self.D_inv_sqrt_ is not None:
                assert isinstance(self.D_inv_sqrt_, (np.ndarray, sparse.spmatrix))
                evecs = self.D_inv_sqrt_.dot(evecs)  # type: ignore[union-attr]
            assert evecs is not None
            for i in range(evecs.shape[1]):
                norm = np.linalg.norm(evecs[:, i])
                if norm == 0:
                    raise ValueError("Encountered a zero-norm eigenvector.")
                evecs[:, i] = evecs[:, i] / norm

        self.eigenvectors = evecs
        self.eigenvalues = evals
        return self

    def rescale(self, use_eigs=50):
        """
        Re-compute the msDM embedding using a different number of eigenvectors.

        Parameters
        ----------
        use_eigs : int, default 50
            Number of eigenvectors to include in the embedding
            (must be ≤ the number retained during ``fit``).
        """
        if self.eigenvectors is None or self.eigenvalues is None:
            raise ValueError("The estimator has not been fitted yet.")
        if self.method != "msDM":
            raise ValueError(
                "Rescaling is only available for multiscale diffusion maps."
            )
        if use_eigs > self.eigenvectors.shape[1]:
            raise ValueError("Cannot rescale to more eigenvectors than are available.")
        use_eigs = int(use_eigs)
        weights = _safe_msdm_weights(self.eigenvalues[:use_eigs])
        self.embedding = self.eigenvectors[:, :use_eigs] * weights
        return self

    def results(self, return_evals=None):
        """
        Return the fitted representation.

        For DM/msDM, this computes the embedding from stored eigenpairs if needed.
        """
        if self.eigenvectors is None:
            raise ValueError("The estimator has not been fitted yet.")

        if return_evals is None:
            return_evals = self.return_evals

        if self.method in ["DM", "msDM"]:
            original_return_evals = self.return_evals
            self.return_evals = False
            try:
                embedding = self.transform()
            finally:
                self.return_evals = original_return_evals

            if return_evals:
                return embedding, self.eigenvalues
            return embedding

        if return_evals:
            return self.eigenvectors, self.eigenvalues
        return self.eigenvectors

    def transform(self, X=None):
        """Return the current representation.

        For DM/msDM, compute the embedding from stored eigenpairs:

        * ``DM``  : ``evecs * (evals ** t)``
        * ``msDM``: ``evecs[:, :use] * (λ / (1 - λ))`` where *use* counts
          the positive-eigenvalue components.
        """
        if self.eigenvectors is None:
            raise ValueError("The estimator has not been fitted yet.")

        # Return eigenvectors/evals for non-diffusion methods.
        if self.method not in ["DM", "msDM"]:
            if self.return_evals:
                return self.eigenvectors, self.eigenvalues
            else:
                return self.eigenvectors

        evecs = self.eigenvectors
        evals = self.eigenvalues
        assert evals is not None

        if self.method == "DM":
            t = int(self.t) if (self.t is not None and self.t > 1) else 1
            lam = evals**t  # apply diffusion time here (no powering in fit)
            emb = evecs * lam
            self.embedding = emb

        elif self.method == "msDM":
            # msDM scaling: weight each component by λ / (1 - λ), using only
            # positive-eigenvalue components (all of them after drop_first).
            use_eigs = int(np.sum(evals > 0, axis=0))
            if use_eigs == 0:
                use_eigs = len(evals)  # fallback: keep all
            weights = _safe_msdm_weights(evals[:use_eigs])
            self.embedding = evecs[:, :use_eigs] * weights

        if self.return_evals:
            return self.embedding, self.eigenvalues
        else:
            return self.embedding

    def fit_transform(self, X=None, y=None, **fit_params):  # type: ignore[override]
        """Fit the model on ``X`` and return the resulting representation."""
        if X is None:
            raise ValueError("X is required for fit_transform().")
        return self.fit(X).transform()

    def spectral_layout(self, X, laplacian_type="normalized", return_evals=False):
        """Compute the spectral embedding of a graph.

        Calls specialized routines if the graph has several connected components.

        Parameters
        ----------
        X : sparse matrix
            The (weighted) adjacency matrix of the graph as a sparse matrix.
        laplacian_type : string (optional, default 'normalized').
            The type of laplacian to use. Can be 'unnormalized', 'symmetric' or 'random_walk'.
        return_evals : bool
            Whether to also return the eigenvalues of the laplacian.

        Returns
        -------
        embedding: array of shape (n_vertices, dim)
            The spectral embedding of the graph.

        evals: array of shape (dim,)
            The eigenvalues of the laplacian of the graph. Only returned if return_evals is True.
        """
        if sparse.issparse(X):
            graph = X.tocsr()
        else:
            graph = sparse.csr_matrix(np.asarray(X, dtype=float))

        n_components, labels = sparse.csgraph.connected_components(
            graph, directed=False
        )

        if n_components > 1:
            return multi_component_layout(
                graph,
                n_components,
                labels,
                self.n_components,
                laplacian_type,
                self.random_state,
                self.eigen_tol,
                return_evals,
            )

        result = LE(
            graph,
            n_eigs=self.n_components,
            laplacian_type=laplacian_type,
            eigen_tol=self.eigen_tol,
            return_evals=return_evals,
        )
        if result is None:
            raise ValueError("Spectral decomposition failed.")
        return result

    def plot_eigenspectrum(self):
        """Plot the eigenspectrum (eigenvalue versus index)."""
        if self.eigenvalues is None:
            raise ValueError("The estimator has not been fitted yet.")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")
        plt.plot(range(0, len(self.eigenvalues)), self.eigenvalues)
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.show()


def spectral_layout(
    graph,
    dim,
    random_state,
    laplacian_type="normalized",
    eigen_tol=10e-4,
    return_evals=False,
):
    """Compute the spectral embedding of a graph.

    This is simply the eigenvectors of the (normalized) Laplacian of the graph.

    Parameters
    ----------
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim: int
        The dimension of the space into which to embed.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    random_state = check_random_state(random_state)

    if sparse.issparse(graph):
        graph = graph.tocsr()
    else:
        graph = sparse.csr_matrix(np.asarray(graph, dtype=float))

    n_components, labels = sparse.csgraph.connected_components(graph, directed=False)

    if n_components > 1:
        return multi_component_layout(
            graph,
            n_components,
            labels,
            dim,
            laplacian_type,
            random_state,
            eigen_tol,
            return_evals,
        )

    else:
        result = LE(
            graph,
            n_eigs=dim,
            laplacian_type=laplacian_type,
            eigen_tol=eigen_tol,
            return_evals=return_evals,
        )

        if result is None:
            raise ValueError("Spectral decomposition failed.")

        return result


def component_layout(
    W,
    n_components,
    component_labels,
    dim,
    laplacian_type="normalized",
    eigen_tol=10e-4,
    return_evals=False,
):
    """Compute a meta-layout for connected components."""
    if dim < 1:
        raise ValueError("dim must be >= 1.")

    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    if n_components == 1:
        component_embedding = np.zeros((1, dim), dtype=np.float64)
        evals = np.zeros(dim, dtype=np.float64)
        if return_evals:
            return component_embedding, evals
        return component_embedding

    component_labels = np.asarray(component_labels)

    if sparse.issparse(W):
        W_csr = W.tocsr()
    else:
        W_csr = sparse.csr_matrix(np.asarray(W, dtype=float))

    distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)

    for c_i in range(n_components):
        rows = component_labels == c_i
        dm_i = W_csr[rows, :]

        for c_j in range(c_i + 1, n_components):
            cols = component_labels == c_j
            block = dm_i[:, cols]

            if block.nnz == 0:
                dist = 1.0
            else:
                positive = block.data[block.data > 0]
                dist = float(positive.min()) if positive.size > 0 else 1.0

            distance_matrix[c_i, c_j] = dist
            distance_matrix[c_j, c_i] = dist

    affinity_matrix = np.exp(-(distance_matrix**2))
    np.fill_diagonal(affinity_matrix, 0.0)

    n_eigs = min(dim, max(1, n_components - 1))

    result = LE(
        affinity_matrix,
        n_eigs=n_eigs,
        laplacian_type=laplacian_type,
        eigen_tol=eigen_tol,
        return_evals=True,
    )
    if result is None:
        raise ValueError("Spectral decomposition failed for component layout.")

    component_embedding, evals = result

    if component_embedding.shape[1] < dim:
        pad = np.zeros(
            (component_embedding.shape[0], dim - component_embedding.shape[1]),
            dtype=component_embedding.dtype,
        )
        component_embedding = np.hstack([component_embedding, pad])

    scale = np.max(np.abs(component_embedding))
    if scale > 0:
        component_embedding = component_embedding / scale

    if return_evals:
        return component_embedding, evals
    return component_embedding


def multi_component_layout(
    graph,
    n_components,
    component_labels,
    dim,
    laplacian_type,
    random_state,
    eigen_tol,
    return_eval_list,
):
    """Compute a spectral layout for a graph with multiple connected components."""
    if dim < 1:
        raise ValueError("dim must be >= 1.")

    random_state = check_random_state(random_state)
    component_labels = np.asarray(component_labels)

    if sparse.issparse(graph):
        graph_csr = graph.tocsr()
    else:
        graph_csr = sparse.csr_matrix(np.asarray(graph, dtype=float))

    result = np.empty((graph_csr.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            graph_csr,
            n_components,
            component_labels,
            dim,
            laplacian_type,
            eigen_tol=eigen_tol,
            return_evals=False,
        )
    else:
        k_meta = int(np.ceil(n_components / 2.0))
        if k_meta > dim:
            base = np.eye(k_meta, dtype=float)[:, :dim]
        else:
            base = np.hstack(
                [np.eye(k_meta, dtype=float), np.zeros((k_meta, dim - k_meta))]
            )
        meta_embedding = np.vstack([base, -base])[:n_components]

    meta_embedding = np.asarray(meta_embedding, dtype=float)

    evals_list = []

    for label in range(n_components):
        mask = component_labels == label
        component_graph = graph_csr[mask, :][:, mask].tocoo()

        distances = pairwise_distances(
            np.asarray(meta_embedding[label], dtype=float).reshape(1, -1),
            meta_embedding,
        )
        positive_distances = distances[distances > 0.0]
        data_range = (
            float(positive_distances.min() / 2.0)
            if positive_distances.size > 0
            else 1.0
        )

        if component_graph.shape[0] < max(3, 2 * dim):
            result[mask] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            evals_list.append(np.full(dim, np.nan, dtype=float))
            continue

        L = graph_laplacian(component_graph, laplacian_type)
        k_eigs = min(dim + 1, component_graph.shape[0] - 1)

        try:
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                L,
                k=k_eigs,
                which="SM",
                tol=eigen_tol,
                v0=np.ones(component_graph.shape[0]),
                maxiter=max(100, component_graph.shape[0] * 2),
            )

            order = np.argsort(eigenvalues)
            order = order[1 : dim + 1]

            component_embedding = eigenvectors[:, order]

            if component_embedding.shape[1] < dim:
                pad = np.zeros(
                    (component_embedding.shape[0], dim - component_embedding.shape[1]),
                    dtype=component_embedding.dtype,
                )
                component_embedding = np.hstack([component_embedding, pad])

            max_abs = np.max(np.abs(component_embedding))
            expansion = data_range / max_abs if max_abs > 0 else 1.0
            component_embedding = component_embedding * expansion

            result[mask] = component_embedding + meta_embedding[label]

            component_evals = eigenvalues[order]
            if component_evals.shape[0] < dim:
                component_evals = np.pad(
                    component_evals,
                    (0, dim - component_evals.shape[0]),
                    constant_values=np.nan,
                )
            evals_list.append(component_evals)

        except sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral decomposition failed for one connected component; "
                "falling back to a random local initialization for that component."
            )
            result[mask] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            evals_list.append(np.full(dim, np.nan, dtype=float))

    if return_eval_list:
        return result, evals_list
    return result
