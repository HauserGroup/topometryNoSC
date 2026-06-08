"""Isomap wrapper around sklearn.manifold.Isomap."""

from typing import Any, Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import Isomap as SklearnIsomap


class Isomap(BaseEstimator, TransformerMixin):
    """Scikit-learn Isomap wrapper preserving TopoMetry's local class name.

    Classical Isomap: geodesic distances on a neighborhood graph followed by
    eigendecomposition of the centered distance matrix.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions to embed into.

    n_neighbors : int, default=5
        Number of neighbors used to build the kNN graph.

    metric : str, default='minkowski'
        Distance metric.

    n_jobs : int | None, default=None
        Number of parallel jobs.

    eigen_solver : str, default='auto'
        Eigendecomposition solver.

    path_method : str, default='auto'
        Shortest-path algorithm.

    neighbors_algorithm : str, default='auto'
        Nearest-neighbor search algorithm.

    p : int, default=2
        Minkowski metric parameter.
    """

    n_components: int
    n_neighbors: int
    metric: str
    n_jobs: int | None
    eigen_solver: Literal["auto", "arpack", "dense"]
    path_method: Literal["auto", "FW", "D"]
    neighbors_algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"]
    p: int
    kwargs: Any
    estimator_: Any
    embedding_: Any

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        n_jobs: int | None = None,
        eigen_solver: Literal["auto", "arpack", "dense"] = "auto",
        path_method: Literal["auto", "FW", "D"] = "auto",
        neighbors_algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "auto",
        p: int = 2,
        **kwargs: Any,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.eigen_solver = eigen_solver
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.p = p
        self.kwargs = kwargs
        self.estimator_ = None
        self.embedding_ = None

    def fit(self, X, y=None):  # noqa: ARG002
        """Fit the Isomap embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : ignored
            Not used, present here for API consistency.

        Returns
        -------
        self
        """
        metric = self.metric
        neighbors_algorithm = self.neighbors_algorithm

        if metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "Isomap with metric='precomputed' requires a square matrix."
                )
            neighbors_algorithm = "auto"

        self.estimator_ = SklearnIsomap(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            path_method=self.path_method,
            neighbors_algorithm=neighbors_algorithm,
            metric=metric,
            p=self.p,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )
        self.embedding_ = self.estimator_.fit_transform(X)
        return self

    def transform(self, X=None):
        """Transform the data or return the fitted embedding.

        Parameters
        ----------
        X : array-like, optional
            New data to transform. If None, returns the fitted embedding.

        Returns
        -------
        Y : ndarray, shape (n_samples, n_components)
            Embedded coordinates.
        """
        if self.embedding_ is None:
            raise ValueError("Isomap has not been fitted yet.")
        if X is None:
            return self.embedding_
        if self.estimator_ is None:
            raise ValueError("Isomap has not been fitted yet.")
        return self.estimator_.transform(X)

    def fit_transform(self, X, y=None, **_fit_params):
        """Fit the embedding and return coordinates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : ignored
            Not used, present here for API consistency.

        **_fit_params : dict
            Additional fit parameters (ignored, for sklearn compatibility).

        Returns
        -------
        Y : ndarray, shape (n_samples, n_components)
            Embedded coordinates.
        """
        fitted = self.fit(X, y=y)
        if fitted.embedding_ is None:
            raise ValueError("Failed to fit Isomap embedding.")
        return fitted.embedding_
