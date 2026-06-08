######################################
# Defining a projection class in a scikit-learn fashion to handle all projection methods
"""Unified projection interface.

The `Projector` dispatches to the available projection backends
(MAP, Isomap, t-SNE, UMAP, PaCMAP, TriMAP, MDE, …) behind a single
scikit-learn-style estimator, with optional landmarks and checkpointing.
"""

import logging
import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np

# dumb warning, suggests lilmatrix but it doesnt work
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from topo._compat.umap import validate_knn_for_umap
from topo._optional import has, require
from topo.base.ann import kNN
from topo.layouts.isomap import Isomap
from topo.layouts.map import fuzzy_embedding
from topo.spectral.eigen import spectral_layout
from topo.tpgraph.kernels import Kernel
from topo.utils._utils import get_landmark_indices

warnings.simplefilter("ignore", SparseEfficiencyWarning)


@runtime_checkable
class _SupportsTransform(Protocol):
    def transform(self, X: np.ndarray | csr_matrix) -> np.ndarray:
        raise NotImplementedError


@runtime_checkable
class _SupportsFitTransform(Protocol):
    def fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _n_rows(matrix_like: object, name: str) -> int:
    shape = getattr(matrix_like, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError(f"{name} must expose a row dimension.")
    return int(shape[0])


def _estimator_fit_transform(
    estimator: object | None, *args: Any, **kwargs: Any
) -> np.ndarray:
    if not isinstance(estimator, _SupportsFitTransform):
        raise ValueError(
            "The selected projection estimator does not support fit_transform."
        )
    return np.asarray(estimator.fit_transform(*args, **kwargs))


class Projector(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible class that handles all projection methods.

    Use this when you already have a spectral representation or graph and only want the final 2-D/3-D layout.
    Ideally, it takes in either a orthonormal eigenbasis or a graph kernel learned from such an eigenbasis.
    It is included in TopoMetry to allow custom `TopOGraph`-like pipelines (projection is the final step).

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions to optimize the layout to. Usually 2 or 3 if you're into visualizing data.

    projection_method : str, default='Isomap'
        Which projection method to use. `UMAP` delegates to `umap-learn`; `MAP`
        is TopoMetry's local checkpoint-aware graph-layout optimizer. Current options are:
            * 'Isomap' - one of the first manifold learning methods
            * ['t-SNE'](https://github.com/DmitryUlyanov/Multicore-TSNE) - a classic manifold learning method
            * 'MAP' - local checkpoint-aware graph-layout optimization
            * ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html) - upstream `umap-learn` estimator
            * ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations
            * ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets
            * 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors
            * 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances
            * ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance
        These are frankly quite direct to add, so feel free to make a feature request if your favorite method is not listed here.

    metric : str, default='euclidean'
        The metric to use when computing distances.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances ('precomputed').

    n_neighbors : int, default=10
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    landmarks : int or np.ndarray, default=None
        If passed as `int`, will obtain the number of landmarks. If passed as `np.ndarray`, will use the specified indexes in the array.
        Any value other than `None` will result in only the specified landmarks being used in the layout optimization, and will
        populate the Projector.landmarks_ slot.

    landmark_method : str, default='kmeans'
        The method to use for selecting landmarks. If `landmarks` is passed as an `int`, this will be used to select the landmarks.
        Can be either 'kmeans' or 'random'.

    num_iters : int, default=1000
        Most (if not all) methods optimize the layout up to a limit number of iterations. Use this parameter to set this number.

    keep_estimator : bool, default=False
        Whether to keep the used estimator as Projector.estimator_ after fitting. Useful if you want to use it later (e.g. UMAP
        allows inverse transforms and out-of-sample mapping).

    """

    def __init__(
        self,
        n_components=2,
        projection_method="MAP",
        metric="euclidean",
        n_neighbors=10,
        n_jobs=1,
        landmarks=None,
        landmark_method="kmeans",
        num_iters=800,
        init: str | np.ndarray = "spectral",
        nbrs_backend="hnswlib",
        keep_estimator=False,
        random_state=None,
        verbose=False,
        # ---- NEW: checkpointing passthrough to MAP ----
        save_every=None,  # int or None: store Y every `save_every` epochs
        save_limit=None,  # cap snapshots kept in memory
        save_callback=None,  # callable(epoch:int, Y:np.ndarray) -> None
        include_init_snapshot=True,  # store epoch=0 (post-init) snapshot
    ):
        self.n_components = n_components
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.projection_method = projection_method
        self.landmarks = landmarks
        self.landmark_method = landmark_method
        self.num_iters = num_iters
        self.n_jobs = n_jobs
        self.init = init
        self.nbrs_backend = nbrs_backend
        self.keep_estimator = keep_estimator
        self.random_state = random_state
        self.verbose = verbose
        self.Y_: np.ndarray | None = None
        self.landmarks_: np.ndarray | None = None
        self.N: int | None = None
        self.M: int | None = None
        self.init_Y_: np.ndarray | None = None
        self.estimator_: object | None = None
        # NEW: checkpointing options (only used by MAP)
        self.save_every = save_every
        self.save_limit = save_limit
        self.save_callback = save_callback
        self.include_init_snapshot = include_init_snapshot
        # NEW: holders for aux/checkpoints returned by MAP
        self.Y_aux_ = None
        self.checkpoints_ = None

    def __repr__(self, N_CHAR_MAX=700):
        """Return a short summary of the fitted state and projection method."""
        if self.Y_ is not None:
            if self.metric == "precomputed":
                msg = "Projector() estimator fitted with precomputed distance matrix"
            elif (self.N is not None) and (self.M is not None):
                msg = (
                    "Projector() estimator fitted with %i samples and %i observations"
                    % (self.N, self.M)
                )
            else:
                msg = "Projector() estimator fitted"
        else:
            msg = "Kernel() estimator without any fitted data."

        method_msg = " using %s" % self.projection_method

        msg = msg + method_msg
        return msg

    def _parse_backend(self):
        from topo._optional import has

        if self.nbrs_backend not in {"hnswlib", "sklearn"}:
            raise ValueError(
                f"Invalid backend: {self.nbrs_backend!r}. Must be 'hnswlib' or 'sklearn'."
            )

        if self.nbrs_backend == "hnswlib" and not has("hnswlib"):
            warnings.warn(
                "HNSWlib not installed; falling back to scikit-learn. "
                "Install it with: pip install topometry-nosc[ann]",
                stacklevel=2,
            )
            self.nbrs_backend = "sklearn"

    def fit(self, X: np.ndarray | csr_matrix | Kernel, **kwargs: Any) -> "Projector":
        """Run the desired projection method on the specified data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or topo.Kernel() class.
            The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices or a `topo.Kernel()` object.
            If precomputed, assumed to be a square symmetric semidefinite matrix.

        **kwargs
            Additional keyword arguments for the desired projection method.

        Returns
        -------
        self : Projector
            The fitted Projector instance with populated `Y_` attribute.
        """
        self.random_state = check_random_state(self.random_state)

        # --- CHECKPOINTING KWARG HANDLING ---
        # Snapshot keys that only MAP understands
        snapshot_keys = (
            "save_every",
            "save_limit",
            "save_callback",
            "include_init_snapshot",
        )

        if self.projection_method == "MAP":
            # 1) Accept any explicit kwarg overrides, but
            # 2) Fallback to constructor attributes if not provided in kwargs.
            map_checkpointing = {}
            for k in snapshot_keys:
                if k in kwargs:  # explicit call-time override
                    map_checkpointing[k] = kwargs.pop(k)
            # Fallbacks to instance attributes set in __init__
            if "save_every" not in map_checkpointing:
                map_checkpointing["save_every"] = self.save_every
            if "save_limit" not in map_checkpointing:
                map_checkpointing["save_limit"] = self.save_limit
            if "save_callback" not in map_checkpointing:
                map_checkpointing["save_callback"] = self.save_callback
            if "include_init_snapshot" not in map_checkpointing:
                map_checkpointing["include_init_snapshot"] = self.include_init_snapshot
        else:
            # Strip snapshot kwargs so other estimators (PaCMAP, etc.) never see them
            for k in snapshot_keys:
                kwargs.pop(k, None)
            map_checkpointing = {}
        # Check inputs
        if self.projection_method not in [
            "Isomap",
            "t-SNE",
            "MAP",
            "UMAP",
            "PaCMAP",
            "TriMAP",
            "IsomorphicMDE",
            "IsometricMDE",
            "NCVis",
        ]:
            raise ValueError(
                "'projection_method' must be one of 'Isomap', 't-SNE', 'MAP', 'UMAP', 'PaCMAP', 'TriMAP', 'IsomorphicMDE', 'IsometricMDE' or 'NCVis'."
            )

        if self.landmarks is not None:
            if isinstance(self.landmarks, int):
                self.landmarks_ = get_landmark_indices(
                    X,
                    n_landmarks=self.landmarks,
                    method=self.landmark_method,
                    random_state=self.random_state,
                )
            elif isinstance(self.landmarks, np.ndarray):
                self.landmarks_ = self.landmarks
            else:
                raise ValueError(
                    "'landmarks' must be either an integer or a numpy array."
                )

        K: Any
        if isinstance(X, Kernel):
            if X.N is None:
                raise ValueError("Kernel must be fitted before projection.")
            source_n_samples = int(X.N)
            self.M = X.M
            kernel_P = X.P
            if self.landmarks_ is not None and self.projection_method != "Isomap":
                self.N = len(self.landmarks_)
                K = kernel_P[np.ix_(self.landmarks_, self.landmarks_)].copy()
            else:
                self.N = X.N
                K = kernel_P.copy()
        else:
            source_n_samples = _n_rows(X, "X")
            if self.metric != "precomputed":
                if issparse(X):
                    X_fit = csr_matrix(X).toarray()
                else:
                    X_fit = np.asarray(X)
                if self.landmarks_ is not None and self.projection_method != "Isomap":
                    X_fit = X_fit[self.landmarks_, :]
                K = kNN(
                    X_fit,
                    metric=self.metric,
                    n_neighbors=self.n_neighbors,
                    n_jobs=self.n_jobs,
                    backend=self.nbrs_backend,
                )
            else:
                X_precomputed = X
                if self.landmarks_ is not None and self.projection_method != "Isomap":
                    K = X_precomputed[np.ix_(self.landmarks_, self.landmarks_)].copy()
                else:
                    K = X_precomputed.copy()

        if isinstance(self.init, np.ndarray):
            self.init_Y_ = self.init
        else:
            if self.init == "spectral":
                try:
                    self.init_Y_ = np.asarray(
                        spectral_layout(
                            K,
                            self.n_components,
                            self.random_state,
                            laplacian_type="random_walk",
                            eigen_tol=10e-4,
                            return_evals=False,
                        )
                    )
                except Exception:
                    warnings.warn(
                        "Multicomponent spectral layout initialization failed, falling back to simple spectral layout..."
                    )
                    from topo.spectral.eigen import EigenDecomposition

                    self.init_Y_ = np.asarray(
                        EigenDecomposition(
                            n_components=self.n_components
                        ).fit_transform(K)
                    )
            else:
                self.init_Y_ = self.random_state.randn(
                    _n_rows(K, "projection graph"), self.n_components
                )
        # Fit the desired method
        if self.projection_method == "Isomap":
            isomap_est = Isomap(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                metric="precomputed",
                n_jobs=self.n_jobs,
            )
            self.Y_ = isomap_est.fit_transform(K)

        elif self.projection_method == "t-SNE":
            if has("MulticoreTSNE"):
                tsne_cls = require("MulticoreTSNE").MulticoreTSNE
                self.estimator_ = tsne_cls(
                    n_components=self.n_components,
                    metric="precomputed",
                    n_iter=self.num_iters,
                )
            else:
                from sklearn.manifold import TSNE as SklearnTSNE

                self.estimator_ = SklearnTSNE(
                    n_components=int(self.n_components),
                    metric="precomputed",
                    max_iter=int(self.num_iters),
                    init="random",
                )
            self.Y_ = _estimator_fit_transform(self.estimator_, K)

        elif self.projection_method == "MAP":
            Y, Y_aux = fuzzy_embedding(
                K,
                n_components=self.n_components,
                init=self.init_Y_,
                n_epochs=self.num_iters,
                random_state=self.random_state,
                metric=self.metric,
                verbose=self.verbose,
                # --- pass checkpointing settings (from kwargs or constructor) ---
                save_every=map_checkpointing.get("save_every"),
                save_limit=map_checkpointing.get("save_limit"),
                save_callback=map_checkpointing.get("save_callback"),
                include_init_snapshot=map_checkpointing.get(
                    "include_init_snapshot", True
                ),
                # any additional kwargs for fuzzy_embedding
                **kwargs,
            )
            self.Y_ = Y
            self.aux_ = Y_aux
            self.Y_aux_ = Y_aux
            self.checkpoints_ = (
                Y_aux.get("checkpoints") if isinstance(Y_aux, dict) else None
            )

        elif self.projection_method == "UMAP":
            from umap import UMAP

            # UMAP's estimator expects precomputed_knn as
            # (indices, distances, search_index). We reuse TopoMetry's graph
            # and set search_index=None because out-of-sample UMAP transform is
            # unavailable for this precomputed graph path.
            if issparse(K):
                knn_indices, knn_dists = _csr_to_fixed_knn(K, self.n_neighbors)
                n_samples = knn_indices.shape[0]
                if source_n_samples != n_samples:
                    raise ValueError(
                        "X and precomputed_knn must have the same number of samples."
                    )
                knn_indices, knn_dists = validate_knn_for_umap(
                    knn_indices,
                    knn_dists,
                    n_samples=n_samples,
                    n_neighbors=knn_indices.shape[1],
                )
                precomputed_knn = (knn_indices, knn_dists, None)
                k_nbrs = knn_indices.shape[1]
            elif isinstance(K, tuple):
                if len(K) not in {2, 3}:
                    raise ValueError(
                        "precomputed_knn must be a 2- or 3-tuple of "
                        "(indices, distances[, search_index])."
                    )
                knn_indices, knn_dists = validate_knn_for_umap(
                    K[0],
                    K[1],
                    n_samples=source_n_samples,
                    n_neighbors=self.n_neighbors,
                )
                search_index = K[2] if len(K) == 3 else None
                precomputed_knn = (knn_indices, knn_dists, search_index)
                k_nbrs = knn_indices.shape[1]
            else:
                precomputed_knn = K
                k_nbrs = self.n_neighbors
            self.estimator_ = UMAP(
                n_components=self.n_components,
                precomputed_knn=precomputed_knn,
                n_neighbors=k_nbrs,
                init=self.init_Y_,  # type: ignore[arg-type]
                n_epochs=self.num_iters,
                random_state=self.random_state,
                low_memory=True,
                verbose=self.verbose,
                **kwargs,
            )
            self.Y_ = _estimator_fit_transform(self.estimator_, X)

        elif self.projection_method == "PaCMAP":
            pacmap = require("pacmap", purpose="PaCMAP layout")
            logging.getLogger("pacmap").setLevel(logging.ERROR)
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="pacmap"
            )  # PaCMAP is way too verbose...
            if self.metric == "cosine":
                metric = "angular"
            else:
                metric = self.metric

            # Workaround for PaCMAP <=0.7.0 bug where `init == "pca"` crashes on ndarrays
            class _SafePacmapInit(np.ndarray):
                def __eq__(self, other):
                    if isinstance(other, str):
                        return False
                    return super().__eq__(other)

                def __ne__(self, other):
                    if isinstance(other, str):
                        return True
                    return super().__ne__(other)

            safe_init = self.init_Y_
            if isinstance(safe_init, np.ndarray):
                safe_init = safe_init.view(_SafePacmapInit)

            self.estimator_ = pacmap.PaCMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                apply_pca=False,
                distance=metric,
                num_iters=self.num_iters,
                verbose=self.verbose,
                **kwargs,
            )
            self.Y_ = _estimator_fit_transform(self.estimator_, X, init=safe_init)

        elif self.projection_method == "TriMAP":
            trimap = require("trimap", purpose="TriMAP layout")

            if self.metric == "cosine":
                metric = "angular"
            else:
                metric = self.metric
            self.estimator_ = trimap.TRIMAP(
                n_dims=self.n_components,
                distance=metric,
                n_iters=self.num_iters,
                verbose=self.verbose,
                **kwargs,
            )
            self.Y_ = _estimator_fit_transform(self.estimator_, X, init=self.init_Y_)

        elif self.projection_method == "IsomorphicMDE":
            pymde = require("pymde", purpose="IsomorphicMDE layout")
            from pymde import preprocess  # type: ignore

            attractive_penalty = pymde.penalties.Log1p
            repulsive_penalty = pymde.penalties.Log
            loss = pymde.losses.Absolute
            graph = preprocess.graph.Graph(K)
            mde_estimator = IsomorphicMDE(
                graph,
                attractive_penalty=attractive_penalty,
                repulsive_penalty=repulsive_penalty,
                embedding_dim=self.n_components,
                n_neighbors=self.n_neighbors,
                init="quadratic",
                verbose=self.verbose,
                **kwargs,
            )
            self.estimator_ = mde_estimator

            self.Y_ = np.asarray(
                mde_estimator.embed(
                    max_iter=self.num_iters,
                    memory_size=10,
                    eps=10e-4,
                    verbose=self.verbose,
                )
            )

        elif self.projection_method == "IsometricMDE":
            pymde = require("pymde", purpose="IsometricMDE layout")
            from pymde import preprocess  # type: ignore

            attractive_penalty = pymde.penalties.Log1p
            repulsive_penalty = pymde.penalties.Log
            loss = pymde.losses.Absolute
            graph = preprocess.graph.Graph(K)
            max_distance = 5e7
            mde_estimator = IsometricMDE(
                graph,
                embedding_dim=self.n_components,
                loss=loss,
                constraint=None,
                max_distances=max_distance,
                verbose=self.verbose,
                **kwargs,
            )
            self.estimator_ = mde_estimator
            self.Y_ = np.asarray(
                mde_estimator.embed(
                    max_iter=self.num_iters, memory_size=1, verbose=self.verbose
                )
            )

        elif self.projection_method == "NCVis":
            ncvis = require("ncvis", purpose="NCVis layout")
            self.estimator_ = ncvis.NCVis(
                d=self.n_components,
                n_neighbors=self.n_neighbors,
                distance=self.metric,
                n_epochs=self.num_iters,
                n_threads=self.n_jobs,
                **kwargs,
            )
            self.Y_ = _estimator_fit_transform(self.estimator_, X)
        return self

    def transform(self, X: np.ndarray | csr_matrix | None = None) -> np.ndarray:
        """Return the projection, using the backend's transform when available.

        If the desired method does not have a transform method, returns the results from the fit method.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
            Projection results.
        """
        if self.Y_ is None:
            raise ValueError("Projector has not been fitted yet.")
        if self.projection_method == "UMAP" and X is not None:
            if not isinstance(self.estimator_, _SupportsTransform):
                raise ValueError(
                    "The fitted UMAP estimator does not support transform."
                )
            return np.asarray(self.estimator_.transform(X))
        return self.Y_

    def fit_transform(self, X, y=None, **kwargs) -> np.ndarray:
        """Fit the projection and return the embedding.

        If the desired method does not have a fit_transform method, returns the results from the fit method.

        Parameters
        ----------
        X : array-like or Kernel, shape (n_samples, n_features)
            The input data to fit and transform.
        y : None
            Ignored.
        **kwargs
            Additional arguments passed to the underlying backend.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
            Projection results.
        """
        self.fit(X, **kwargs)
        if self.Y_ is None:
            raise ValueError("Projector has not been fitted yet.")
        return self.Y_


# Check if pymde is installed
_HAS_PYMDE = has("pymde")


# Define custom pymde problems
if _HAS_PYMDE:

    def _remove_anchor_anchor_edges(edges, data, anchors):
        # exclude edges in which both items are anchors, since these
        # items are already pinned in place by the anchor constraint
        # NOTICE: This is exactly as implemented by Akshay Agrawal, at least for now.
        neither_anchors_mask = ~(
            (edges[:, 0][..., None] == anchors).any(-1)
            * (edges[:, 1][..., None] == anchors).any(-1)
        )
        edges = edges[neither_anchors_mask]
        data = data[neither_anchors_mask]
        return edges, data

    def IsomorphicMDE(
        data,
        attractive_penalty=None,
        repulsive_penalty=None,
        embedding_dim=2,
        constraint=None,
        n_neighbors=None,
        repulsive_fraction=None,
        max_distance=None,
        init="quadratic",
        eps=1e-04,
        max_iter=100,
        memory_size=1,
        print_every=None,
        device="cpu",
        verbose=False,
        **kwargs,
    ):
        # Inherits from pymde.recipes.preserve_neighbors()
        """Construct an MDE problem designed to preserve local structure.

        This function constructs an MDE problem for preserving the
        local structure of original data. This MDE problem is well-suited for
        visualization (using ``embedding_dim`` 2 or 3), but can also be used to
        generate features for machine learning tasks (with ``embedding_dim`` = 10,
        50, or 100, for example). It yields embeddings in which similar items
        are near each other, and dissimilar items are not near each other.
        The original data can either be a data matrix, or a graph.
        Data matrices should be torch Tensors, NumPy arrays, or scipy sparse
        matrices; graphs should be instances of ``pymde.Graph``.
        The MDE problem uses distortion functions derived from weights (i.e.,
        penalties).
        To obtain an embedding, call the ``embed`` method on the returned ``MDE``
        object. To plot it, use ``pymde.plot``.
        .. code:: python3
            embedding = pymde.preserve_neighbors(data).embed()
            pymde.plot(embedding)
        Arguments
        ---------
        data: {torch.Tensor, numpy.ndarray, scipy.sparse matrix} or pymde.Graph
            The original data, a data matrix of shape ``(n_items, n_features)`` or
            a graph. Neighbors are computed using Euclidean distance if the data is
            a matrix, or the shortest-path metric if the data is a graph.
        embedding_dim: int
            The embedding dimension. Use 2 or 3 for visualization.
        attractive_penalty: pymde.Function class (or factory)
            Callable that constructs a distortion function, given positive
            weights. Typically one of the classes from ``pymde.penalties``,
            such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
            ``pymde.penalties.Quadratic``.
        repulsive_penalty: pymde.Function class (or factory)
            Callable that constructs a distortion function, given negative
            weights. (If ``None``, only positive weights are used.) For example,
            ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
        constraint: pymde.constraints.Constraint, optional
            Embedding constraint, like ``pymde.Standardized()`` or
            ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
            constraint when a repulsive penalty is provided, otherwise defaults to
            ``pymde.Standardized()``.
        n_neighbors: int, optional
            The number of nearest neighbors to compute for each row (item) of
            ``data``. A sensible value is chosen by default, depending on the
            number of items.
        repulsive_fraction: float, optional
            How many repulsive edges to include, relative to the number
            of attractive edges. ``1`` means as many repulsive edges as attractive
            edges. The higher this number, the more uniformly spread out the
            embedding will be. Defaults to ``0.5`` for standardized embeddings, and
            ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
            is ignored.)
        max_distance: float, optional
            If not None, neighborhoods are restricted to have a radius
            no greater than ``max_distance``.
        init: str or np.ndarray, default='quadratic'
            Initialization strategy; np.ndarray, 'quadratic' or 'random'.
        eps :float, optional
            Residual norm threshold; quit when the residual norm is smaller than eps.
        max_iter: int
            Maximum number of iterations.
        memory_size : int
            The quasi-Newton memory. Larger values may lead to more stable behavior, but will increase the amount of time each iteration takes.
        print_every : int, optional
            Print verbose output every print_every iterations.
        device: str, optional
            Device for the embedding (eg, 'cpu', 'cuda').
        verbose: bool
            If ``True``, print verbose output.

        Returns
        -------
        pymde.MDE
            A ``pymde.MDE`` object, based on the original data.
        """
        import torch  # type: ignore
        from pymde import constraints, preprocess, problem, quadratic  # type: ignore
        from pymde.functions import penalties  # type: ignore

        if attractive_penalty is None:
            attractive_penalty = penalties.Log1p

        if repulsive_penalty is None:
            repulsive_penalty = penalties.Log

        if isinstance(data, preprocess.graph.Graph):
            n = data.n_items
        elif data.shape[0] <= 1:
            raise ValueError("The data matrix must have at least two rows.")
        else:
            n = data.shape[0]

        if n_neighbors is None:
            # target included edges to be ~1% of total number of edges
            n_choose_2 = n * (n - 1) / 2
            n_neighbors = int(max(min(15, n_choose_2 * 0.01 / n), 5))

        if n_neighbors > n:
            problem.LOGGER.warning(
                f"Requested n_neighbors {n_neighbors} > number of items {n}."
                f" Setting n_neighbors to {n - 1}"
            )
            n_neighbors = n - 1

        if constraint is None and repulsive_penalty is not None:
            constraint = constraints.Centered()
        elif constraint is None and repulsive_penalty is None:
            constraint = constraints.Standardized()
        assert constraint is not None

        # enforce a max distance, otherwise may very well run out of memory
        # when n_items is large
        if max_distance is None:
            max_distance = (3 * torch.quantile(data.distances, 0.75)).item()
        if verbose:
            problem.LOGGER.info(
                f"Computing {n_neighbors}-nearest neighbors, with "
                f"max_distance={max_distance}"
            )
        knn_graph = preprocess.generic.k_nearest_neighbors(
            data,
            k=n_neighbors,
            max_distance=max_distance,
            verbose=verbose,
        )
        edges = knn_graph.edges.to(device)
        weights = knn_graph.weights.to(device)

        if isinstance(constraint, constraints.Anchored):
            # remove anchor-anchor edges before generating intialization
            edges, weights = _remove_anchor_anchor_edges(
                edges, weights, constraint.anchors
            )

        # DS: add multicomponent spectral initialization
        if isinstance(init, np.ndarray):
            X_init = torch.tensor(init)
        elif init == "quadratic":
            if verbose:
                problem.LOGGER.info("Computing quadratic initialization.")
            X_init = quadratic.spectral(n, embedding_dim, edges, weights, device=device)
            if not isinstance(
                constraint, (constraints._Centered, constraints._Standardized)
            ):
                constraint.project_onto_constraint(X_init, inplace=True)
        elif init == "random":
            X_init = constraint.initialization(n, embedding_dim, device)
        else:
            raise ValueError(
                f"Unsupported value '{init}' for keyword argument `init`; "
                "the supported values are 'quadratic' and 'random', or a np.ndarray of shape (n_items, embedding_dim)."
            )

        if repulsive_penalty is not None:
            if repulsive_fraction is None:
                if isinstance(constraint, constraints._Standardized):
                    # standardization constraint already implicity spreads,
                    # so use a lower replusion
                    repulsive_fraction = 0.5
                else:
                    repulsive_fraction = 1

            n_choose_2 = int(n * (n - 1) / 2)
            n_repulsive = int(repulsive_fraction * (edges.shape[0]))
            # cannot sample more edges than there are available
            n_repulsive = min(n_repulsive, n_choose_2 - edges.shape[0])

            negative_edges = preprocess.sample_edges(n, n_repulsive, exclude=edges).to(
                device
            )

            negative_weights = -torch.ones(
                negative_edges.shape[0], dtype=X_init.dtype, device=device
            )

            if isinstance(constraint, constraints.Anchored):
                negative_edges, negative_weights = _remove_anchor_anchor_edges(
                    negative_edges, negative_weights, constraint.anchors
                )

            edges = torch.cat([edges, negative_edges])
            weights = torch.cat([weights, negative_weights])

            f = penalties.PushAndPull(
                weights,
                attractive_penalty=attractive_penalty,
                repulsive_penalty=repulsive_penalty,
            )
        else:
            f = attractive_penalty(weights)

        if eps is None:
            eps = 1e-6 * torch.max(weights).item()
        mde = problem.MDE(
            n_items=n,
            embedding_dim=embedding_dim,
            edges=edges,
            distortion_function=f,
            constraint=constraint,
            device=device,
        )
        mde._X_init = X_init

        # Won't need to cache the graph - we have already computed it and cached with TopoMetry

        distances = mde.distances(mde._X_init)
        if (distances == 0).any():
            # pathological scenario in which at least two points overlap can yield
            # non-differentiable average distortion. perturb the initialization to
            # mitigate.
            x_init = mde._X_init
            mde._X_init += 1e-4 * torch.randn(
                x_init.shape,
                device=x_init.device,
                dtype=x_init.dtype,
            )
        return mde

    def IsometricMDE(
        data,
        embedding_dim=2,
        loss=None,
        constraint=None,
        max_distances=5e7,
        device="cpu",
        verbose=False,
    ):
        # Inherits from pymde.recipes.preserve_distances()
        """Construct an MDE problem based on original distances.

        This function constructs an MDE problem for preserving pairwise
        distances between items. This can be useful for preserving the global
        structure of the data.
        The data can be specified with either a data matrix (a NumPy array, torch
        Tensor, or sparse matrix), or a ``pymde.Graph`` instance encoding the
        distances:
            A NumPy array, torch tensor, or sparse matrix is interpreted as a
            collection of feature vectors: each row gives the feature vector for an
            item. The original distances are the Euclidean distances between the
            feature vectors.
            A ``pymde.Graph`` instance is interpreted as encoding all (n_items
            choose 2) distances: the distance between i and j is taken to be the
            length of the shortest path connecting i and j.
        When the number of items n_items is large, the total number of pairs will
        be very large. When this happens, instead of computing all pairs of
        distances, this function will sample a subset uniformly at random. The
        maximum number of distances to compute is specified by the parameter
        ``max_distances``. Depending on how many items you have (and how much
        memory your machine has), you may need to adjust this parameter.
        To obtain an embedding, call the ``embed`` method on the returned object.
        To plot it, use ``pymde.plot``.
        For example:
        .. code:: python3
            embedding = pymde.preserve_distances(data).embed()
            pymde.plot(embedding)
        Arguments
        ---------
        data: {np.ndarray, torch.Tensor, scipy.sparse matrix} or pymde.Graph
            The original data, a data matrix of shape ``(n_items, n_features)`` or
            a graph.
        embedding_dim: int
            The embedding dimension.
        loss: pymde.Function class (or factory)
            Callable that constructs a distortion function, given
            original distances. Typically one of the classes defined in
            ``pymde.losses``, such as ``pymde.losses.Absolute``, or
            ``pymde.losses.WeightedQuadratic``.
        constraint: pymde.constraints.Constraint, optional
            Embedding constraint, such as ``pymde.Standardized()`` or
            ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
            constraint. Note: when the constraint is ``pymde.Standardized()``,
            the original distances will be scaled by a constant (because the
            standardization constraint puts a limit on how large any one
            distance can be).
        max_distances: int
            Maximum number of distances to compute.
        device: str, optional
            Device for the embedding (eg, 'cpu', 'cuda').
        verbose: bool
            If ``True``, print verbose output.

        Returns
        -------
        pymde.MDE
            A ``pymde.MDE`` instance, based on preserving the original distances.
        """
        import torch  # type: ignore
        from pymde import constraints, preprocess, problem  # type: ignore
        from pymde.functions import losses  # type: ignore
        from scipy.sparse import issparse

        if loss is None:
            loss = losses.Absolute

        if not isinstance(
            data, (np.ndarray, torch.Tensor, preprocess.graph.Graph)
        ) and not issparse(data):
            raise ValueError(
                "`data` must be a np.ndarray/torch.Tensor/scipy.sparse matrix"
                ", or a pymde.Graph."
            )

        if isinstance(data, preprocess.graph.Graph):
            n_items = data.n_items
        else:
            n_items = data.shape[0]
        n_all_edges = (n_items) * (n_items - 1) / 2
        retain_fraction = max_distances / n_all_edges

        if isinstance(data, preprocess.graph.Graph):
            edges = data.edges.to(device)
            deviations = data.distances.to(device)
        else:
            graph = preprocess.generic.distances(
                data, retain_fraction=retain_fraction, verbose=verbose
            )
            edges = graph.edges.to(device)
            deviations = graph.distances.to(device)

        if constraint is None:
            constraint = constraints.Centered()
        elif isinstance(constraint, constraints._Standardized):
            deviations = preprocess.scale(
                deviations, constraint.natural_length(n_items, embedding_dim)
            )
        elif isinstance(constraint, constraints.Anchored):
            edges, deviations = _remove_anchor_anchor_edges(
                edges, deviations, constraint.anchors
            )

        mde = problem.MDE(
            n_items=n_items,
            embedding_dim=embedding_dim,
            edges=edges,
            distortion_function=loss(deviations),
            constraint=constraint,
            device=device,
        )
        return mde


def _csr_to_fixed_knn(K, n_neighbors):
    """Convert a CSR sparse graph to UMAP's fixed-width precomputed_knn tuple."""
    K = K.tocsr()
    n_pts = K.shape[0]
    k = min(int(n_neighbors), max(1, K.shape[1] - 1))

    knn_indices = np.full((n_pts, k), -1, dtype=np.int32)
    knn_dists = np.full((n_pts, k), np.inf, dtype=np.float32)

    for i in range(n_pts):
        start, end = K.indptr[i], K.indptr[i + 1]
        row_idx = K.indices[start:end]
        row_dist = K.data[start:end]

        keep = row_idx != i
        row_idx = row_idx[keep]
        row_dist = row_dist[keep]

        if row_idx.size == 0:
            continue

        order = np.argsort(row_dist, kind="mergesort")[:k]
        width = order.size
        knn_indices[i, :width] = row_idx[order].astype(np.int32, copy=False)
        knn_dists[i, :width] = row_dist[order].astype(np.float32, copy=False)

    return knn_indices, knn_dists
