"""Thin wrappers around sklearn manifold estimators."""

from typing import Literal

from sklearn.manifold import SpectralEmbedding


def spectral_embedding_from_affinity(
    affinity,
    *,
    n_components: int = 2,
    random_state=None,
    eigen_solver: Literal["arpack", "lobpcg", "amg"] | None = None,
    n_jobs: int = -1,
):
    """Compute Laplacian Eigenmaps from a precomputed affinity matrix.

    Parameters
    ----------
    affinity : array-like or sparse matrix
        Precomputed affinity (similarity) matrix.

    n_components : int, default=2
        Number of dimensions to embed into.

    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        Eigendecomposition solver.

    n_jobs : int | None, default=None
        Number of parallel jobs.

    Returns
    -------
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional embedding.
    """
    model = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=random_state,
        eigen_solver=eigen_solver,
        n_jobs=n_jobs,
    )
    return model.fit_transform(affinity)
