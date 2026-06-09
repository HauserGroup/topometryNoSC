"""Eigendecomposition transformers for kernels and spectral operators.

This module delegates numerical eigendecomposition to SciPy and only handles
operator selection plus Diffusion Maps / multiscale Diffusion Maps weighting.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackError, eigsh

from topo.spectral._spectral import diffusion_operator, graph_laplacian
from topo.tpgraph.kernels import Kernel

EIGEN_SOLVERS = {"auto", "dense", "arpack"}


def _shape_2d(matrix: Any, name: str) -> tuple[int, int]:
    """Return a validated 2-D shape as plain Python ints."""
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")
    return int(shape[0]), int(shape[1])


def _as_square_matrix(matrix: Any, name: str) -> Any:
    """Validate that ``matrix`` is square and return it unchanged."""
    n_rows, n_cols = _shape_2d(matrix, name)
    if n_rows != n_cols:
        raise ValueError(f"{name} must be square; got shape {(n_rows, n_cols)}.")
    return matrix


def _as_csr_matrix(matrix: Any, name: str) -> csr_matrix:
    """Return ``matrix`` as a square CSR matrix."""
    _as_square_matrix(matrix, name)
    return (
        matrix.tocsr()
        if sparse.issparse(matrix)
        else csr_matrix(np.asarray(matrix, dtype=float))
    )


def _eigsh_tol(value: float) -> Any:
    """Return eigsh tolerance while isolating SciPy typing-stub limitations."""
    return float(value)


def _diffusion_operator_with_degree(
    W: ArrayLike | sparse.spmatrix,
    alpha: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Return symmetric diffusion operator and ``D^{-1/2}``."""
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

    P, D_inv_sqrt = result
    return csr_matrix(P), csr_matrix(D_inv_sqrt)


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
    G: ArrayLike | sparse.spmatrix,
    n_components: int = 8,
    eigensolver: str = "auto",
    largest: bool = True,
    eigen_tol: float = 1e-4,
    random_state=None,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return sorted eigenpairs of a square symmetric matrix/operator.

    Dense inputs use ``scipy.linalg.eigh``. Sparse inputs use
    ``scipy.sparse.linalg.eigsh`` unless ``eigensolver="dense"`` is requested.

    ``random_state`` and ``verbose`` are accepted for compatibility and ignored.
    """
    del random_state, verbose

    if G is None:
        raise ValueError("G cannot be None.")

    shape = getattr(G, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("G must be a 2-D square matrix.")

    n_vertices, n_cols = int(shape[0]), int(shape[1])
    if n_vertices != n_cols:
        raise ValueError(f"G must be square; got shape {(n_vertices, n_cols)}.")
    if n_vertices < 2:
        raise ValueError("G must contain at least two vertices.")

    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    k = min(n_components + 1, n_vertices - 1)

    eigensolver = str(eigensolver).lower()
    if eigensolver not in EIGEN_SOLVERS:
        raise ValueError(
            f"eigensolver must be one of {sorted(EIGEN_SOLVERS)}; got {eigensolver!r}."
        )

    use_dense = eigensolver == "dense" or (
        eigensolver == "auto" and not sparse.issparse(G)
    )

    if use_dense:
        G_dense = (
            _as_csr_matrix(G, "G").toarray()
            if sparse.issparse(G)
            else np.asarray(G, dtype=float)
        )

        if not np.isfinite(G_dense).all():
            raise ValueError("G contains NaN or infinite values.")

        evals_all, evecs_all = eigh(G_dense)
        order = np.argsort(evals_all)
        if largest:
            order = order[::-1]
        order = order[:k]

        return (
            np.asarray(np.real(evals_all[order]), dtype=float),
            np.asarray(np.real(evecs_all[:, order]), dtype=float),
        )

    G_csr = _as_csr_matrix(G, "G")
    G_csr = G_csr.astype(float)

    if not np.isfinite(G_csr.data).all():
        raise ValueError("G contains NaN or infinite values.")

    try:
        evals, evecs = eigsh(
            G_csr,
            k=k,
            which="LM" if largest else "SM",
            tol=float(eigen_tol),  # type: ignore
        )
    except ArpackError as exc:
        raise RuntimeError(
            "Sparse eigendecomposition failed. The operator may be "
            "ill-conditioned, disconnected, or have too small an eigengap."
        ) from exc

    evals = np.asarray(np.real(evals), dtype=float)
    evecs = np.asarray(np.real(evecs), dtype=float)

    order = np.argsort(evals)
    if largest:
        order = order[::-1]

    return evals[order], evecs[:, order]


class EigenDecomposition:
    """Transformer for eigendecomposing kernels and spectral operators.

    Use this when you already have a kernel, Laplacian, adjacency/affinity matrix,
    or diffusion operator and want spectral coordinates. The numerical
    eigendecomposition is delegated to SciPy:

    * dense inputs use ``scipy.linalg.eigh`` when ``eigensolver="dense"`` or when
    ``eigensolver="auto"`` receives a dense matrix;
    * sparse inputs use ``scipy.sparse.linalg.eigsh`` when
    ``eigensolver="arpack"`` or when ``eigensolver="auto"`` receives a sparse
    matrix.

    The main input can be either a ``topo.tpgraph.Kernel`` object or a square matrix.
    For ``Kernel`` inputs, the transformer reuses the fitted kernel's stored
    operators when possible.

    Parameters
    ----------
    n_components : int, default=10
        Number of non-trivial components to return. One extra eigenpair is computed
        internally so the trivial first component can be dropped when
        ``drop_first=True``.

    method : {'top', 'bottom', 'msDM', 'DM', 'LE'}, default='DM'
        Method for organizing the eigendecomposition.

        * ``'top'``:
        compute the largest eigenpairs of the input matrix/operator.
        * ``'bottom'``:
        compute the smallest eigenpairs of the input matrix/operator.
        * ``'DM'``:
        compute Diffusion Maps coordinates from a diffusion operator. If a
        ``Kernel`` object is provided, its fitted diffusion operator is reused
        when available.
        * ``'msDM'``:
        compute multiscale Diffusion Maps coordinates by weighting diffusion
        components by ``lambda / (1 - lambda)``.
        * ``'LE'``:
        compute Laplacian Eigenmaps coordinates from a graph Laplacian. If a
        ``Kernel`` object is provided, its fitted Laplacian is reused when
        available.

    eigensolver : {'auto', 'dense', 'arpack'}, default='auto'
        Solver policy.

        * ``'auto'``:
        use dense eigendecomposition for dense inputs and ARPACK for sparse inputs.
        * ``'dense'``:
        use ``scipy.linalg.eigh`` on a dense array. Avoid for large graphs.
        * ``'arpack'``:
        use ``scipy.sparse.linalg.eigsh`` for partial sparse eigendecomposition.

    eigen_tol : float, default=1e-4
        Tolerance passed to ``scipy.sparse.linalg.eigsh``. Ignored by the dense path.

    drop_first : bool, default=True
        Whether to drop the first eigenpair, typically the trivial component.

    laplacian_type : str, default='random_walk'
        Laplacian type used when ``method='LE'`` and the input is a matrix.
        Common values are ``'normalized'``, ``'unnormalized'``, and
        ``'random_walk'``.

    anisotropy : float, default=1
        Diffusion-maps anisotropy parameter, usually denoted ``alpha``.

    t : int, default=1
        Diffusion time used when ``method='DM'``. Ignored by other methods.

    random_state : optional
        Accepted for API compatibility. The simplified eigensolver path does not
        use randomness.

    return_evals : bool, default=False
        Whether ``results()`` should return eigenvalues along with the representation.

    estimate_eigengap : bool, default=True
        Whether to store a simple eigengap estimate after fitting.

    verbose : bool, default=False
        Accepted for API compatibility. The simplified eigensolver path does not
        emit verbose solver diagnostics.
    """

    def __init__(
        self,
        n_components=10,
        method="DM",
        eigensolver="auto",
        eigen_tol=1e-4,
        drop_first=True,
        laplacian_type="random_walk",
        anisotropy=1,
        t=1,
        random_state=None,
        return_evals=False,
        estimate_eigengap=True,
        verbose=False,
    ):
        self.n_components = n_components
        self.method = method
        self.eigensolver = eigensolver
        self.eigen_tol = eigen_tol
        self.drop_first = drop_first
        self.laplacian_type = laplacian_type
        self.t = t
        self.anisotropy = anisotropy
        self.random_state = random_state
        self.verbose = verbose
        self.eigenvalues = None
        self.eigenvectors = None
        self.laplacian = None
        self.diffusion_operator = None
        self.embedding = None
        self.N = None
        self.D_inv_sqrt_ = None
        self.return_evals = return_evals
        self.estimate_eigengap = estimate_eigengap
        self.eigengap = None

    def __repr__(self) -> str:
        """Return a short fitted-state summary."""
        status = (
            f"fitted with {self.N} samples"
            if self.eigenvectors is not None and self.N is not None
            else "not fitted"
        )
        return f"EigenDecomposition(method={self.method!r}, {status})"

    def fit(self, X):
        """Compute the eigendecomposition of kernel matrix ``X`` per ``method``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Matrix to be decomposed. Should generally be an adjacency, affinity/kernel/similarity, Laplacian matrix or a diffusion-type operator.

        Returns
        -------
        self : EigenDecomposition
            The fitted instance itself. Eigenvectors and eigenvalues are stored
            as attributes. If method is 'DM' or 'msDM', the diffusion operator
            is cached as well.
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
            X = _as_square_matrix(X, "X")
            n_samples, _ = _shape_2d(X, "X")
            self.N = n_samples

            X_matrix = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=float)

            if self.method in ["DM", "msDM"]:
                # Use the symmetric diffusion operator by default. It is numerically
                # better for eigendecomposition, and the right eigenvectors are
                # recovered below through D_inv_sqrt_.
                self.diffusion_operator, self.D_inv_sqrt_ = (
                    _diffusion_operator_with_degree(
                        X_matrix,
                        alpha=float(self.anisotropy),
                    )
                )
                symmetric = True
                target = self.diffusion_operator

            elif self.method == "LE":
                self.laplacian = graph_laplacian(
                    X_matrix,
                    laplacian_type=self.laplacian_type,
                )
                target = self.laplacian

            else:
                target = X_matrix

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
        evals = np.asarray(evals)
        evecs = np.asarray(evecs)

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
                evecs = np.asarray(csr_matrix(self.D_inv_sqrt_) @ evecs, dtype=float)

            norms = np.linalg.norm(evecs, axis=0)
            if np.any(norms == 0):
                raise ValueError("Encountered a zero-norm eigenvector.")
            evecs = evecs / norms

        self.eigenvectors = evecs
        self.eigenvalues = evals
        return self

    def rescale(self, use_eigs=50):
        """
        Re-compute the msDM embedding using a different number of eigenvectors.

        Parameters
        ----------
        use_eigs : int, default=50
            Number of eigenvectors to include in the embedding
            (must be ≤ the number retained during ``fit``).

        Returns
        -------
        self : EigenDecomposition
            The modified instance with the scaled embedding updated.
        """
        if self.eigenvectors is None or self.eigenvalues is None:
            raise ValueError("The estimator has not been fitted yet.")
        if self.method != "msDM":
            raise ValueError(
                "Rescaling is only available for multiscale diffusion maps."
            )
        use_eigs = int(use_eigs)
        if use_eigs < 1:
            raise ValueError("use_eigs must be >= 1.")
        if use_eigs > self.eigenvectors.shape[1]:
            raise ValueError("Cannot rescale to more eigenvectors than are available.")

        weights = _safe_msdm_weights(self.eigenvalues[:use_eigs])
        self.embedding = self.eigenvectors[:, :use_eigs] * weights
        return self

    def results(self, return_evals=None):
        """Return the fitted spectral representation.

        For ``DM`` and ``msDM``, this computes the embedding from stored eigenpairs
        if needed. For ``LE``, ``top``, and ``bottom``, it returns the fitted
        eigenvectors.
        """
        if self.eigenvectors is None:
            raise ValueError("The estimator has not been fitted yet.")

        include_evals = (
            self.return_evals if return_evals is None else bool(return_evals)
        )
        representation = self._represent()

        if include_evals:
            return representation, self.eigenvalues
        return representation

    def transform(self, X=None):
        """Return the current representation.

        ``X`` is ignored; the representation is computed from the eigenpairs
        stored during `fit`. Present for scikit-learn compatibility.

        Parameters
        ----------
        X : None
            Ignored.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            The computed spectral representation.
        """
        return self._represent()

    def _represent(self):
        """Build the representation from stored eigenpairs.

        For non-diffusion methods, the representation is the eigenvector matrix.

        For diffusion methods:

        * ``DM`` uses ``evecs * (evals ** t)``.
        * ``msDM`` uses ``evecs[:, :use] * (lambda / (1 - lambda))`` where
        ``use`` is the number of positive-eigenvalue components.
        """
        if self.eigenvectors is None:
            raise ValueError("The estimator has not been fitted yet.")

        if self.method not in {"DM", "msDM"}:
            return self.eigenvectors

        if self.eigenvalues is None:
            raise ValueError("The estimator has no fitted eigenvalues.")

        if self.method == "DM":
            t = int(self.t) if self.t is not None and self.t > 1 else 1
            self.embedding = self.eigenvectors * (self.eigenvalues**t)
            return self.embedding

        use_eigs = int(np.sum(self.eigenvalues > 0))
        if use_eigs == 0:
            use_eigs = len(self.eigenvalues)

        weights = _safe_msdm_weights(self.eigenvalues[:use_eigs])
        self.embedding = self.eigenvectors[:, :use_eigs] * weights
        return self.embedding

    def fit_transform(self, X):
        """Fit the model on ``X`` and return the resulting representation."""
        if X is None:
            raise ValueError("X is required for fit_transform().")
        return self.fit(X)._represent()
