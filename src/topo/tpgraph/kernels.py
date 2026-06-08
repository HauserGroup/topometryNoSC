#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Defining graph kernels in a scikit-learn fashion
"""Graph-kernel construction and graph operators.

Provides `compute_kernel` and the scikit-learn-style `Kernel`
estimator, which build neighborhood-graph matrices (adaptive-bandwidth
affinities, fuzzy simplicial sets, binary continuous kNN graphs) and expose
graph operators: adjacency, Laplacian, diffusion operator, shortest paths,
sparsification and imputation.
"""

import logging
import warnings
from typing import cast

import numpy as np

# dumb warning, suggests lilmatrix but it doesnt work
from scipy.sparse import (
    SparseEfficiencyWarning,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    diags,
    find,
    issparse,
    tril,
)
from scipy.spatial import procrustes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize as _l2_normalize_rows
from sklearn.utils import check_random_state

from topo._compat.umap import fuzzy_graph_from_knn
from topo.base.ann import kNN
from topo.base.dists import pairwise_distances
from topo.base.graph_matrix import get_indices_distances_from_sparse_matrix
from topo.spectral._spectral import degree as compute_degree
from topo.spectral._spectral import diffusion_operator, graph_laplacian
from topo.tpgraph.cknn import cknn_graph

warnings.simplefilter("ignore", SparseEfficiencyWarning)

logger = logging.getLogger(__name__)


def _as_csr_matrix(X, *, dtype=float) -> csr_matrix:
    """Return X as a CSR matrix."""
    if issparse(X):
        return X.tocsr().astype(dtype)
    return csr_matrix(np.asarray(X, dtype=dtype))


def _ensure_csr_matrix(X, *, dtype=float) -> csr_matrix:
    """Normalize sparse/dense matrix results to CSR without changing semantics."""
    if issparse(X):
        return X.tocsr().astype(dtype)
    return csr_matrix(np.asarray(X, dtype=dtype))


def _as_dense_array(X, *, dtype=float) -> np.ndarray:
    """Return X as a dense ndarray."""
    if issparse(X):
        return X.toarray().astype(dtype)
    return np.asarray(X, dtype=dtype)


def _check_2d_input(X, name="X"):
    """Validate a 2-D matrix-like input."""
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError(f"{name} must be a 2-D array-like or sparse matrix.")
    if X.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")


def _check_square_matrix(X, name="X"):
    """Validate a square matrix-like input."""
    _check_2d_input(X, name=name)
    if X.shape[0] != X.shape[1]:
        raise ValueError(f"{name} must be square; got shape {X.shape}.")


def _sanitize_sparse_data(M):
    """Set non-finite sparse data to zero and remove explicit zeros."""
    M = M.tocsr()
    M.data = np.where(np.isfinite(M.data), M.data, 0.0)
    M.data = np.maximum(M.data, 0.0)
    M.eliminate_zeros()
    return M


def _safe_diffusion_operator_with_degree(W, alpha, semi_aniso=False):
    """Return symmetric diffusion operator and D^{-1/2} with tuple validation."""
    result = diffusion_operator(
        W,
        alpha=alpha,
        semi_aniso=semi_aniso,
        symmetric=True,
        return_D_inv_sqrt=True,
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "diffusion_operator(..., return_D_inv_sqrt=True) must return "
            "a tuple of (operator, D_inv_sqrt)."
        )

    return result


def _maybe_l2_normalize_rows(X: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
    """Return ``X`` with row-wise L2 normalization if possible.

    Works for dense (ndarray) and CSR/CSC/COO sparse matrices.
    """
    try:
        normalized = _l2_normalize_rows(X, norm="l2", axis=1, copy=False)
    except Exception:
        # Fall back to a safe copy if in-place fails
        normalized = _l2_normalize_rows(X, norm="l2", axis=1, copy=True)
    if issparse(normalized):
        return csr_matrix(normalized)
    return np.asarray(normalized)


def _cosine_distance_to_angle_from_sparse_triplets(x_idx, y_idx, dists):
    """Convert sparse cosine-distance triplets to angles (radians).

    Given triplets of cosine *distance* d = 1 - cos in [0, 2],
    converts to angle θ = arccos(cos) with cos = 1 - d.
    Returns in-place modified dists (angles in radians).
    """
    # cos = 1 - d
    # clamp to [-1, 1] before arccos
    cos_vals = 1.0 - dists
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return np.arccos(cos_vals)


def _ensure_nonneg_and_finite(arr, eps=0.0):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if eps > 0.0:
        arr = np.maximum(arr, eps)
    else:
        arr = np.maximum(arr, 0.0)
    return arr


def _adap_bw(K, n_neighbors):
    """Adaptive bandwidth from the median local neighbor distance."""
    K = K.tocsr()
    median_k = max(1, int(np.floor(n_neighbors / 2)))
    adap_sd = np.zeros(K.shape[0], dtype=float)

    for i in range(K.shape[0]):
        row_data = K.data[K.indptr[i] : K.indptr[i + 1]]
        row_data = row_data[np.isfinite(row_data)]
        row_data = row_data[row_data > 0]

        if row_data.size == 0:
            adap_sd[i] = 1.0
            continue

        sorted_row = np.sort(row_data)
        idx = min(median_k - 1, sorted_row.size - 1)
        adap_sd[i] = sorted_row[idx]

    positive = adap_sd[adap_sd > 0]
    fallback = float(np.median(positive)) if positive.size > 0 else 1.0
    adap_sd[adap_sd <= 0] = fallback

    return adap_sd


def _density_ranks(adap_sd, high):
    """Interpolate adaptive bandwidths to density ranks with a neutral constant guard."""
    adap_sd = np.asarray(adap_sd, dtype=float)
    lo = float(np.nanmin(adap_sd))
    hi = float(np.nanmax(adap_sd))
    if hi <= lo + 1e-7:
        return np.full_like(adap_sd, fill_value=(2.0 + float(high)) / 2.0, dtype=float)
    return np.interp(adap_sd, (lo, hi), (2, high))


def _compute_cknn_kernel(
    X,
    n_neighbors: int,
    cknn_delta: float,
    cknn_candidate_neighbors,
    cknn_exact: bool,
    metric: str,
    backend: str,
    n_jobs: int,
    verbose: bool,
    return_densities: bool,
    kwargs: dict,
) -> tuple[csr_matrix, dict]:
    """Compute continuous k-nearest neighbors kernel (binary unweighted)."""
    W = cknn_graph(
        X,
        scale_k=n_neighbors,
        delta=cknn_delta,
        metric=metric,
        candidate_k=cknn_candidate_neighbors,
        exact=cknn_exact,
        include_self=False,
        symmetrize="or",
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    dens_dict: dict = {}
    if return_densities:
        dens_dict["unweighted_adjacency"] = W
    return W, dens_dict


def _prepare_knn_input(X, metric: str, backend: str, pairwise: bool):
    """Prepare input data for KNN computation (normalize if needed)."""
    X_for_knn = X
    if (metric == "cosine") and (not pairwise) and backend == "hnswlib":
        X_for_knn = _maybe_l2_normalize_rows(X)
    if issparse(X_for_knn):
        X_for_knn = csr_matrix(X_for_knn)
    else:
        X_for_knn = np.asarray(X_for_knn)
    return X_for_knn


def _compute_knn_distance_graph(
    X_prep,
    metric: str,
    n_neighbors: int,
    pairwise: bool,
    backend: str,
    n_jobs: int,
    kwargs: dict,
) -> csr_matrix:
    """Compute KNN or pairwise distance graph."""
    if pairwise:
        K_dense = pairwise_distances(X_prep, metric)
        K = csr_matrix(K_dense)
    else:
        K = kNN(
            X_prep,
            metric=metric,
            n_neighbors=n_neighbors,
            backend=backend,
            n_jobs=n_jobs,
            **kwargs,
        )
    return K


def _compute_fuzzy_kernel_from_knn(
    K, N: int, n_neighbors: int, random_state, verbose: bool
):
    """Compute fuzzy simplicial set kernel from KNN graph."""
    knn_indices, knn_dists = get_indices_distances_from_sparse_matrix(
        K, n_neighbors=n_neighbors
    )
    result = fuzzy_graph_from_knn(
        np.zeros((N, 1), dtype=np.float32),
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        n_neighbors=n_neighbors,
        metric="precomputed",
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        random_state=random_state,
        verbose=verbose,
    )
    W, sigmas, rhos = result[:3]  # type: ignore[misc]
    return W, sigmas, rhos


def _compute_adaptive_bandwidth_kernel(
    K: csr_matrix,
    metric: str,
    use_angular: bool,
    k: int,
    adaptive_bw: bool,
    sigma,
    alpha_decaying: bool,
    square_distances: bool,
    expand_nbr_search: bool,
    new_K,
    adap_sd,
    adap_sd_new,
    pm,
    pm_new,
    new_k,
    N: int,
    backend: str,
    n_jobs: int,
    X,
    kwargs: dict,
) -> tuple[csr_matrix, dict]:
    """Compute adaptive bandwidth kernel and return W and density diagnostics."""
    dens_dict: dict = {}
    active_K = new_K if expand_nbr_search and new_K is not None else K
    x, y, dists = find(active_K)

    # Convert cosine distance to angle if needed
    if metric == "cosine" and use_angular:
        dists = _cosine_distance_to_angle_from_sparse_triplets(x, y, dists)

    # Numerical guards for distances
    if metric == "cosine" and not use_angular:
        dists = np.clip(dists, 0.0, 2.0)
    elif metric == "cosine" and use_angular:
        dists = np.clip(dists, 0.0, np.pi)
    else:
        dists = np.maximum(dists, 0.0)

    # Normalize distances
    if adaptive_bw:
        assert adap_sd is not None
        assert pm is not None
        # Select active bandwidth based on expand_nbr_search
        if expand_nbr_search:
            assert adap_sd_new is not None
            active_adap_sd = adap_sd_new
            active_pm = pm_new
            active_k = new_k
        else:
            active_adap_sd = adap_sd
            active_pm = pm
            active_k = k

        assert active_adap_sd is not None
        assert active_k is not None

        if alpha_decaying:
            assert active_pm is not None
            base = dists / (active_adap_sd[x] + 1e-10)
            expo = np.power(2, ((active_k - active_pm[x]) / active_pm[x]))
            d_scaled = np.power(base, expo)
        else:
            d_scaled = dists / (active_adap_sd[x] + 1e-10)
        if square_distances:
            d_scaled = d_scaled**2
    else:
        if sigma is not None:
            if sigma == 0:
                sigma = 1e-10
            d_scaled = np.asarray(dists, dtype=np.float64) / sigma
            if square_distances:
                d_scaled = d_scaled**2
        else:
            d_scaled = dists

    d_scaled_float = np.asarray(d_scaled, dtype=np.float64)
    W = csr_matrix((np.exp(-d_scaled_float), (x, y)), shape=[N, N])
    return W, dens_dict


def compute_kernel(
    X,
    metric="cosine",
    n_neighbors=10,
    fuzzy=False,
    cknn=False,
    cknn_delta=1.0,
    cknn_candidate_neighbors=None,
    cknn_exact=False,
    pairwise=False,
    sigma=None,
    adaptive_bw=True,
    expand_nbr_search=False,
    alpha_decaying=False,
    return_densities=False,
    symmetrize=True,
    backend="hnswlib",
    n_jobs=-1,
    verbose=False,
    use_angular=True,
    square_distances=True,
    random_state=None,
    **kwargs,
) -> csr_matrix | tuple[csr_matrix, dict[str, csr_matrix | np.ndarray | int]]:
    """
    Compute a kernel matrix from a set of points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features).
        The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
        If precomputed, assumed to be a square symmetric semidefinite matrix.

    metric : string, default='cosine'
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances.

    n_neighbors : int, default=10
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    fuzzy : bool, default=False
        Whether to build the kernel matrix using fuzzy simplicial sets, similarly to UMAP.
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.
        If set to `True` at the same time that `cknn` is set to `True`, the `cknn` parameter is ignored.

    cknn : bool, default=False
        Whether to build the binary, unweighted continuous k-nearest-neighbors graph.
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.

    cknn_delta : float, default=1.0
        Unitless CkNN edge threshold. Ignored if `cknn` is ``False``.

    cknn_candidate_neighbors : int or None, default=None
        Number of candidate neighbors tested in approximate CkNN mode. Ignored
        when ``cknn_exact=True``.

    cknn_exact : bool, default=False
        If True, threshold all pairwise distances for CkNN construction.

    pairwise : bool, default=False
        Whether to compute the kernel using dense pairwise distances.
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float, default=None
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool, default=True
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool, default=False
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool, default=False
        Whether to use an adaptively decaying kernel.

    return_densities : bool, default=False
        Whether to return the bandwidth metrics as a dictinary. If set to `True`, the function
        returns a tuple containing the kernel matrix and a dictionary containing the
        bandwidth metric.

    symmetrize : bool, default=True
        Whether to symmetrize the kernel matrix after normalizations.

    backend : str, default='hnswlib'
        Which backend to use for neighborhood computations. Defaults to 'hnswlib'.
        Options are 'hnswlib' and 'sklearn'.

    n_jobs : int, default=1
        The number of jobs to use for parallel computations. If set to -1, all available cores are used.

    verbose : bool, default=False
        Whether to print progress messages.

    **kwargs : dict, optional
        Additional arguments to be passed to the nearest-neighbors backend.

    Returns
    -------
    K : array-like, shape (n_samples, n_samples)
        The kernel matrix.

    densities : dict, optional (if `return_densities` is set to `True`)
        If `fuzzy` and `cknn` are `False`, is a dictionary containing the bandwidth metrics.
        If `fuzzy` is set to `True`, the dictionary contains sigma and rho estimates.
        If `cknn` is set to `True`, the dictionary contains the binary adjacency.
    """
    _check_2d_input(X)

    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")

    N = X.shape[0]
    if metric == "precomputed":
        _check_square_matrix(X)
    else:
        if n_neighbors >= N:
            raise ValueError(
                f"n_neighbors={n_neighbors} must be smaller than n_samples={N}."
            )

    if n_jobs == -1:
        from joblib import cpu_count

        n_jobs = cpu_count()
    k = n_neighbors

    # CkNN is mutually exclusive with fuzzy (CkNN takes precedence)
    if cknn and not fuzzy:
        if cknn_delta <= 0:
            raise ValueError("cknn_delta must be positive.")
        return _compute_cknn_kernel(
            X,
            n_neighbors,
            cknn_delta,
            cknn_candidate_neighbors,
            cknn_exact,
            metric,
            backend,
            n_jobs,
            verbose,
            return_densities,
            kwargs,
        )

    # Prepare input (precomputed or KNN-ready)
    if metric == "precomputed":
        K = _as_csr_matrix(X)
        K = _sanitize_sparse_data(K)
        expand_nbr_search = False
        dens_dict = {}
    else:
        X_for_knn = _prepare_knn_input(X, metric, backend, pairwise)
        K = _compute_knn_distance_graph(
            X_for_knn, metric, k, pairwise, backend, n_jobs, kwargs
        )
        dens_dict = {}
        if return_densities:
            dens_dict["knn"] = K

    # Fuzzy simplicial set path
    if fuzzy:
        W, sigmas, rhos = _compute_fuzzy_kernel_from_knn(
            K, N, n_neighbors, random_state, verbose
        )
        if return_densities:
            dens_dict["sigma"] = sigmas
            dens_dict["rho"] = rhos
    else:
        # Adaptive bandwidth path: compute bandwidths and optionally expand search
        new_k = None
        adap_sd = None
        pm = None
        adap_sd_new = None
        pm_new = None
        new_K = None

        if adaptive_bw:
            adap_sd = _adap_bw(K, k)
            if metric == "cosine" and use_angular:
                adap_sd = _cosine_distance_to_angle_from_sparse_triplets(
                    None, None, adap_sd
                )
            pm = _density_ranks(adap_sd, k)
            if return_densities:
                dens_dict["omega"] = pm
                dens_dict["adaptive_bw"] = adap_sd

            # Expand neighborhood search if requested
            if expand_nbr_search:
                new_k = max(k + 1, int(np.ceil(k + (k - float(pm.max())))))
                new_k = min(new_k, N - 1)

                if new_k <= k:
                    expand_nbr_search = False
                else:
                    new_K = kNN(
                        X,
                        metric=metric,
                        n_neighbors=new_k,
                        backend=backend,
                        n_jobs=n_jobs,
                        **kwargs,
                    )
                    adap_sd_new = _adap_bw(new_K, new_k)
                    if metric == "cosine" and use_angular:
                        adap_sd_new = _cosine_distance_to_angle_from_sparse_triplets(
                            None, None, adap_sd_new
                        )
                    pm_new = _density_ranks(adap_sd_new, new_k)
                    if return_densities:
                        dens_dict["expanded_k_neighbor"] = new_k
                        dens_dict["omega_nbr_expanded"] = pm_new
                        dens_dict["adaptive_bw_nbr_expanded"] = adap_sd_new
                        dens_dict["expanded_neighborhood_graph"] = new_K
                        dens_dict["knn_expanded"] = new_K

        # Compute adaptive bandwidth kernel
        W, _ = _compute_adaptive_bandwidth_kernel(
            K,
            metric,
            use_angular,
            k,
            adaptive_bw,
            sigma,
            alpha_decaying,
            square_distances,
            expand_nbr_search,
            new_K,
            adap_sd,
            adap_sd_new,
            pm,
            pm_new,
            new_k,
            N,
            backend,
            n_jobs,
            X,
            kwargs,
        )

    # Finalize kernel: handle NaN/Inf, symmetrize
    W = _ensure_csr_matrix(W)
    W.data = _ensure_nonneg_and_finite(W.data, eps=0.0)

    if symmetrize:
        W = (W + W.T) / 2  # type: ignore

    W = _ensure_csr_matrix(W)
    W.data = np.where(np.isfinite(W.data), W.data, 0.0)

    if not return_densities:
        return W
    else:
        return W, dens_dict


class Kernel(BaseEstimator, TransformerMixin):
    """Scikit-learn flavored estimator for computing a kernel matrix from points.

    Includes functions
    for computing the kernel matrix with a variety of methods (adaptive bandwidth, fuzzy simplicial sets,
    continuous k-nearest-neighbors, etc) and performing operations
    on the resulting graph, such as obtaining its Laplacian, sparsifying it, filtering and interpolating signals,
    obtaining diffusion operators, imputing missing values and computing shortest paths.

    Use this only when you want to manually construct a graph or diffusion operator.

    **Example**
    ```python
    from topo.tpgraph import Kernel
    from topo.spectral import EigenDecomposition
    from topo.layouts import Projector
    kernel = Kernel(n_neighbors=30, metric="cosine").fit(X)
    Z = EigenDecomposition(n_components=64, method="msDM").fit_transform(kernel)
    Y = Projector(projection_method="MAP").fit_transform(Z)
    ```

    Parameters
    ----------
    metric : str, default='cosine'
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others, depending on the chosen nearest-neighbors backend. Accepts precomputed distances as 'precomputed'.

    n_neighbors : int, default=10
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    fuzzy : bool, default=False
        Whether to build the kernel matrix using fuzzy simplicial sets, similarly to UMAP.
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.
        If set to `True` at the same time that `cknn` is set to `True`, the `cknn` parameter is ignored.

    cknn : bool, default=False
        Whether to build the binary, unweighted continuous k-nearest-neighbors graph.
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.

    pairwise : bool, default=False
        Whether to compute the kernel using dense pairwise distances.
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float or None, default=None
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool, default=True
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool, default=False
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool, default=False
        Whether to use an adaptively decaying kernel.

    anisotropy : float, default=1.0
        The anisotropy (alpha) parameter in the diffusion maps literature for kernel reweighting.

    semi_aniso : bool, default=False
        Whether to use semi-anisotropic diffusion. This reweights the original kernel (not the renormalized kernel) by the renormalized degree.

    symmetrize : bool, default=True
        Whether to symmetrize the kernel matrix after normalizations.

    backend : {"hnswlib", "sklearn"}, default="hnswlib"
        Which backend to use for k-nearest-neighbor computations. Defaults to 'hnswlib'.
        Options are 'hnswlib' and 'sklearn'.

    n_jobs : int, default=-1
        The number of jobs to use for parallel computations. If -1, all CPUs are used.
        Parallellization (multiprocessing) is ***highly*** recommended whenever possible.

    laplacian_type : {"normalized", "unnormalized", "random_walk"}, default="normalized"
        The type of laplacian to use.

    Fitted Attributes
    -----------------
    knn_ : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The computed k-nearest-neighbors graph.

    _K : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The kernel matrix.

    dens_dict : dict
        Dictionary containing the density information for each point in the dataset for the computed kernel.

    L : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The laplacian matrix of the kernel graph.

    P : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The diffusion operator of the kernel graph.

    SP : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The shortest-path matrix.

    degree : np.ndarray, shape (n_samples,)
        The degree of each point in the adjacency graph.

    weighted_degree : np.ndarray, shape (n_samples,)
        The weighted degree of each point in the kernel graph.

    """

    def __init__(
        self,
        metric="cosine",
        n_neighbors=10,
        fuzzy=False,
        cknn=False,
        cknn_delta=1.0,
        cknn_candidate_neighbors=None,
        cknn_exact=False,
        pairwise=False,
        sigma=None,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        symmetrize=True,
        backend="hnswlib",
        n_jobs=1,
        laplacian_type="normalized",
        anisotropy=1.0,
        semi_aniso=False,
        n_landmarks=None,
        cache_input=False,
        verbose=False,
        random_state=None,
        use_angular=True,
    ):
        self.n_neighbors = n_neighbors
        self.fuzzy = fuzzy
        self.cknn = cknn
        self.cknn_delta = cknn_delta
        self.cknn_candidate_neighbors = cknn_candidate_neighbors
        self.cknn_exact = cknn_exact
        self.pairwise = pairwise
        self.n_jobs = n_jobs
        self.backend = backend
        self.metric = metric
        self.sigma = sigma
        self.semi_aniso = semi_aniso
        self.adaptive_bw = adaptive_bw
        self.expand_nbr_search = expand_nbr_search
        self.alpha_decaying = alpha_decaying
        self.symmetrize = symmetrize
        self.n_landmarks = n_landmarks
        if cknn and laplacian_type == "normalized":
            warnings.warn(
                "CkNN theory applies to the unweighted graph and its unnormalized Laplacian; "
                "overriding laplacian_type='unnormalized'.",
                UserWarning,
                stacklevel=2,
            )
            self.laplacian_type = "unnormalized"
        else:
            self.laplacian_type = laplacian_type
        self.cache_input = cache_input
        self.verbose = verbose
        self.random_state = random_state
        self.use_angular = use_angular
        self.X = None
        self._K = None
        self.N = None
        self.M = None
        self.dens_dict = None
        self._clusters = None
        self._A = None
        self._Dd = None
        self._degree = None
        self._weighted_degree = None
        self._L = None
        self._SP = None
        self._P = None
        self._laplacian_cache_key = None
        self._diff_op_cache_key = None
        self._shortest_paths_cache_key = None
        self._connected = None
        self.D_inv_sqrt_ = None
        self.components_ = None
        self.components_indices_ = None
        self.sigma_ = None
        self.rho_ = None
        self.umap_sigmas_ = None
        self.umap_rhos_ = None
        self.adaptive_bw_ = None
        self.omega_ = None
        self.expanded_k_neighbor_ = None
        self.adaptive_bw_nbr_expanded_ = None
        self.omega_nbr_expanded_ = None
        self._sample_densities = None
        self.knn_ = None
        self.dens_dict = None
        self.anisotropy = anisotropy

    def __repr__(self, N_CHAR_MAX=700):
        """Return a short summary of the fitted state, sample count, and kernel method."""
        if self._K is not None:
            if self.metric == "precomputed":
                msg = "Kernel() estimator fitted with precomputed distance matrix"
            elif (self.N is not None) and (self.M is not None):
                msg = (
                    "Kernel() estimator fitted with %i samples and %i observations"
                    % (self.N, self.M)
                )
            else:
                msg = "Kernel() estimator without valid fitted data."
        else:
            msg = "Kernel() estimator without any fitted data."
        if self._K is not None:
            if self.fuzzy:
                kernel_msg = " using fuzzy simplicial sets."
            elif self.cknn:
                kernel_msg = " using continuous k-nearest-neighbors."
            else:
                if not self.adaptive_bw:
                    kernel_msg = (
                        " using a kernel with fixed bandwidth sigma = %.2f" % self.sigma
                    )
                else:
                    kernel_msg = " using a kernel with adaptive bandwidth "
                    if self.alpha_decaying:
                        kernel_msg = kernel_msg + "and adaptive decay"
                    if self.expand_nbr_search:
                        kernel_msg = kernel_msg + ", with expanded neighborhood search."
            msg = msg + kernel_msg
        return msg

    def _reset_graph_caches(self):
        """Clear graph-derived caches after changing the kernel matrix."""
        self._A = None
        self._Dd = None
        self._degree = None
        self._weighted_degree = None
        self._L = None
        self._SP = None
        self._P = None
        self._laplacian_cache_key = None
        self._diff_op_cache_key = None
        self._shortest_paths_cache_key = None
        self._connected = None
        self.D_inv_sqrt_ = None
        self.components_ = None
        self.components_indices_ = None
        self._sample_densities = None

    def _parse_backend(self):
        from topo._optional import has

        if self.backend not in {"hnswlib", "sklearn"}:
            raise ValueError(
                f"Invalid backend: {self.backend!r}. Must be 'hnswlib' or 'sklearn'."
            )

        if self.backend == "hnswlib" and not has("hnswlib"):
            warnings.warn(
                "HNSWlib not installed; falling back to scikit-learn. "
                "Install it with: pip install topometry-nosc[ann]",
                stacklevel=2,
            )
            self.backend = "sklearn"

    def fit(self, X, recompute=False, **kwargs):
        """Fit the kernel matrix to the data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data. Accepts NumPy arrays and SciPy CSR sparse matrices.
            If `metric="precomputed"`, this should be a square distance matrix.
        recompute : bool, default=False
            Whether to recompute the kernel if it has already been computed.
        **kwargs
            Additional arguments to be passed to the k-nearest-neighbor backend.

        Returns
        -------
        self : Kernel
            The fitted Kernel instance, populating properties like `K` and `knn`.
        """
        self._parse_backend()
        _check_2d_input(X)

        if self.metric == "precomputed":
            _check_square_matrix(X)

        self.random_state = check_random_state(self.random_state)

        if self.fuzzy:
            self.cknn = False
        if self.cache_input:
            self.X = X
        self.N, self.M = X.shape
        if self._K is None or (self._K is not None and recompute):
            self._K, self.dens_dict = compute_kernel(
                X,
                metric=self.metric,
                fuzzy=self.fuzzy,
                cknn=self.cknn,
                cknn_delta=self.cknn_delta,
                cknn_candidate_neighbors=self.cknn_candidate_neighbors,
                cknn_exact=self.cknn_exact,
                pairwise=self.pairwise,
                n_neighbors=self.n_neighbors,
                sigma=self.sigma,
                adaptive_bw=self.adaptive_bw,
                expand_nbr_search=self.expand_nbr_search,
                alpha_decaying=self.alpha_decaying,
                return_densities=True,
                symmetrize=self.symmetrize,
                backend=self.backend,
                n_jobs=self.n_jobs,
                use_angular=self.use_angular,
                verbose=self.verbose,
                random_state=self.random_state,
                **kwargs,
            )
            self._reset_graph_caches()
        assert self.dens_dict is not None
        dens_dict = cast(dict, self.dens_dict)
        if self.metric != "precomputed" and "knn" in dens_dict:
            self.knn_ = dens_dict["knn"]
        if self.fuzzy:
            self.sigma_ = dens_dict["sigma"]
            self.rho_ = dens_dict["rho"]
            self.umap_sigmas_ = self.sigma_
            self.umap_rhos_ = self.rho_
        elif self.cknn:
            self._A = dens_dict["unweighted_adjacency"]
            self.adaptive_bw_ = dens_dict.get("adaptive_bw")
        else:
            if self.adaptive_bw:
                self.adaptive_bw_ = dens_dict["adaptive_bw"]
                self.omega_ = dens_dict["omega"]
            if self.expand_nbr_search:
                self.expanded_k_neighbor_ = dens_dict["expanded_k_neighbor"]
                self.adaptive_bw_nbr_expanded_ = dens_dict["adaptive_bw_nbr_expanded"]
                self.omega_nbr_expanded_ = dens_dict["omega_nbr_expanded"]
        return self

    def transform(self, X=None):
        """Return the fitted kernel (affinity) matrix.

        Provided for scikit-learn compatibility in pipelines. `X` is ignored
        because out-of-sample extension for the kernel is not supported.
        """
        if self._K is None:
            raise ValueError("No kernel matrix has been fitted yet. Call fit() first.")

        return self._K

    def fit_transform(self, X, y=None, **fit_params):  # type: ignore
        """Fit the kernel to the data and return the affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.
        **fit_params
            Additional parameters passed to `fit`.

        Returns
        -------
        K : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
            The computed kernel (affinity) matrix.
        """
        self.fit(X, **fit_params)
        return self.K

    def adjacency(self):
        """Compute the binary graph adjacency matrix.

        The adjacency matrix defines the unweighted connectivity of the graph.
        It is represented as an N-by-N sparse matrix where entry A_{i,j} is 1
        if the kernel affinity K_{i,j} > 0, and 0 otherwise.
        """
        if self._K is None:
            raise ValueError("No kernel matrix has been fitted yet. Call fit() first.")
        self._A = (self._K > 0).astype(int)  # type: ignore[reportAttributeAccessIssue]
        self._A = self._A.astype(float)  # type: ignore[reportAttributeAccessIssue]
        return self._A

    @property
    def knn(self):
        """The k-nearest-neighbors graph (sparse distance matrix).

        Contains the raw nearest-neighbor distances used to compute the
        bandwidths and affinities. Represented as a CSR sparse matrix.
        """
        if self.knn_ is None:
            raise ValueError(
                "No k-nearest-neighbors graph has been fitted yet or precomputed versions were used!"
            )
        return self.knn_

    @property
    def K(self):
        """The kernel (affinity) matrix.

        Represents the pairwise similarities between samples on the manifold,
        typically decaying from 1 (identical) to 0 (disconnected). Used as
        the weighted adjacency matrix for graph operators.
        """
        if self._K is None:
            raise ValueError("No kernel matrix has been fitted yet. Call fit() first.")
        return self._K

    @property
    def A(self):
        """The binary graph adjacency matrix.

        Unweighted representation of the graph connectivity, generated from
        the non-zero entries of the kernel affinity matrix (K).
        """
        if self._K is None:
            raise ValueError("No kernel matrix has been fitted yet. Call fit() first.")
        if self._A is None:
            self._A = self.adjacency()
        return self._A

    @property
    def degree(self):
        """The node degrees of the binary adjacency matrix.

        A 1-D array containing the number of connected neighbors for each
        sample, ignoring the affinity weights.
        """
        if self._degree is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first."
                )
            self._degree = compute_degree(self.A)
        return self._degree

    @property
    def weighted_degree(self):
        """The weighted node degrees of the kernel affinity matrix.

        A 1-D array containing the sum of affinity weights for each sample.
        Used to normalize graph Laplacians and diffusion operators.
        """
        if self._weighted_degree is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first."
                )
            self._weighted_degree = compute_degree(self._K)
        return self._weighted_degree

    def laplacian(self, laplacian_type=None, recompute=False):
        """Compute the graph Laplacian of this kernel's affinity matrix.

        For a friendly reference, see this material from James Melville:
        https://jlmelville.github.io/smallvis/spectral.html

        Parameters
        ----------
        laplacian_type : str, default=None
            The type of laplacian to use. Can be 'unnormalized', 'normalized', or 'random_walk'.
            If not provided, uses the default `laplacian_type` specified in the constructor.

        Returns
        -------
        L : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
            The computed graph Laplacian matrix.
        """
        if laplacian_type is None:
            laplacian_type = self.laplacian_type
        cache_key = str(laplacian_type)
        if self._L is None or self._laplacian_cache_key != cache_key or recompute:
            self._L, self._Dd = graph_laplacian(
                self.K, laplacian_type=laplacian_type, return_D=True
            )
            self._laplacian_cache_key = cache_key
        return self._L

    @property
    def L(self):
        """The graph Laplacian matrix.

        Use this property for the default cached result. Use the `laplacian` method
        when you need to pass options or recompute with non-default settings.

        Evaluates and caches the
        Laplacian on first access using the `laplacian_type` specified during
        initialization.

        Use this property for the default cached result. Use the `laplacian()`
        method when you need to pass options or recompute with non-default settings.
        """
        if self._L is None:
            return self.laplacian()
        return self._L

    def diff_op(self, anisotropy=1.0, symmetric=True, recompute=False):
        """Compute the [diffusion operator](https://doi.org/10.1016/j.acha.2006.04.006).

        Parameters
        ----------
        anisotropy : float, default=1.0
            Anisotropy to apply. 'Alpha' in the diffusion maps literature.
            Defaults to the anisotropy defined in the constructor.
        symmetric : bool, default=True
            Whether to use a symmetric version of the diffusion operator. This is particularly useful to yield a symmetric operator
            when using anisotropy (alpha > 0), as the diffusion operator P would be asymmetric otherwise, which can be problematic
            during matrix decomposition. Eigenvalues are the same as the asymmetric version, and the right eigenvectors of the original asymmetric
            operator can be recovered by left multiplying by `Kernel.D_inv_sqrt_`.

        Returns
        -------
        P : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
            The graph diffusion operator (Markov transition matrix).

        Notes
        -----
        Populates the `Kernel.D_inv_sqrt_` attribute when `symmetric=True`.
        """
        if anisotropy is None:
            anisotropy = self.anisotropy
        anisotropy = float(anisotropy)
        if anisotropy < 0:
            anisotropy = 0.0
        if anisotropy > 1:
            anisotropy = 1.0
        cache_key = (anisotropy, bool(symmetric))

        if self._P is None or self._diff_op_cache_key != cache_key or recompute:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first."
                )

            if symmetric:
                P, D_inv_sqrt = _safe_diffusion_operator_with_degree(
                    self.K,
                    alpha=anisotropy,
                    semi_aniso=self.semi_aniso,
                )
                self._P = _as_csr_matrix(P)
                self.D_inv_sqrt_ = D_inv_sqrt
                self._P = ((self._P + self._P.T) / 2).tocsr()
            else:
                result = diffusion_operator(
                    self.K,
                    alpha=anisotropy,
                    semi_aniso=self.semi_aniso,
                    symmetric=False,
                )
                self._P = _as_csr_matrix(result)
                self.D_inv_sqrt_ = None
            self._diff_op_cache_key = cache_key

        return self._P

    @property
    def P(self):
        """The graph diffusion operator (Markov transition matrix).

        Use this property for the default cached result. Use the `diff_op` method
        when you need to pass options or recompute with non-default settings.

        Evaluates and caches the
        operator on first access. Describes the probabilities of random walks
        across the graph.

        Use this property for the default cached result. Use the `diff_op()`
        method when you need to pass options or recompute with non-default settings.
        """
        if self._P is None:
            return self.diff_op()
        return self._P

    def shortest_paths(self, landmark=False, indices=None, recompute=False):
        """Compute the shortest paths (geodesic distances) on the graph.

        Notes
        -----
        This method may require O(n_samples^2) memory and heavy computation if
        all-pairs distances are materialized. For large datasets, use landmark mode.

        Parameters
        ----------
        landmark : bool, default=False
            If True, the shortest paths are computed between all pairs of landmarks,
            rather than all sample nodes.
        indices : list of int, default=None
            If None, the shortest paths are computed between all pairs of nodes. Else,
            the shortest paths are computed between all pairs of nodes and nodes with specified indices.

        Returns
        -------
        D : ndarray, shape (n_samples, n_samples)
            The shortest paths matrix. Unreachable nodes evaluate to infinity.
        """
        if landmark:
            raise NotImplementedError("landmark=True is not implemented.")
        if indices is None:
            index_key = None
        elif np.issubdtype(type(indices), np.integer):
            index_key = (int(indices),)
        else:
            index_key = tuple(int(i) for i in np.asarray(indices).ravel())
        cache_key = (index_key,)

        if self._SP is None or self._shortest_paths_cache_key != cache_key or recompute:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first."
                )
            from topo._compat.scipy_graph import graph_shortest_paths

            SP = graph_shortest_paths(
                self._K,
                method="D",
                unweighted=False,
                directed=False,
                indices=indices,
            )
            SP = np.asarray(SP, dtype=float)
            if SP.ndim == 1:
                SP = SP.reshape(1, -1)
            if SP.shape[0] == SP.shape[1]:
                SP = (SP + SP.T) / 2.0
                np.fill_diagonal(SP, 0.0)
            self._SP = SP
            self._shortest_paths_cache_key = cache_key
        return self._SP

    @property
    def SP(self):
        """The shortest-paths (geodesic distance) matrix.

        Use this property for the default cached result. Use the `shortest_paths` method
        when you need to pass options or recompute with non-default settings.

        Evaluates and caches
        the all-pairs shortest paths on the graph on first access.

        Use this property for the default cached result. Use the `shortest_paths()`
        method when you need to pass options or recompute with non-default settings.
        """
        if self._SP is None:
            return self.shortest_paths()
        return self._SP

    def get_indices_distances(self, n_neighbors=None, kernel=True):
        """Return per-row neighbor indices and distances from the kernel or kNN graph.

        Returns
        -------
        indices, distances : tuple of ndarray
            The extracted arrays.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if kernel:
            return get_indices_distances_from_sparse_matrix(self.K, n_neighbors)
        else:
            return get_indices_distances_from_sparse_matrix(self.knn_, n_neighbors)

    def _calculate_imputation_error(self, data, data_prev=None):
        """Compute the difference before and after imputation by diffusion.

        Adapted from [MAGIC](https://github.com/KrishnaswamyLab/MAGIC).

        Parameters
        ----------
        data : array-like
            current data matrix
        data_prev : array-like, optional (default: None)
            previous data matrix. If None, `data` is simply prepared for
            comparison and no error is returned

        Returns
        -------
        error : float
            Procrustes disparity value
        data_curr : array-like
            transformed data to use for the next comparison
        """
        if data_prev is not None:
            _, _, error = procrustes(data_prev, data)
        else:
            error = None
        return error, data

    def impute(self, Y=None, t=None, threshold=0.01, tmax=10):
        """Impute data ``Y`` using the diffusion operator built from ``X``.

        Although the idea behind this is far older, it was first reported in single-cell genomics
        by the Krishnaswamy lab in the MAGIC (Markov Affinity-based Graph Imputation of Cells)
        [manuscript](https://www.cell.com/cell/abstract/S0092-8674(18)30724-4)

        Parameters
        ----------
        Y : np.ndarray, default=None
            The input data to impute. If None, the input data X is imputed (if it was cached by setting
            the `cache_input` parameter to True). Otherwise, you'll have to specify it as Y.

        t : int, default=None
            The number of steps to perform during diffusion. The default `None` iterates until the Procrustes disparity value
            is below the `threshold` parameter.

        threshold : float, default=0.01
            The threshold value for the Procrustes disparity when finding an optimal `t`.

        tmax : int, default=10
            The maximum number of steps to perform during diffusion when estimating an optimal `t`.

        Returns
        -------
        Y_imp : np.ndarray
            The imputed data.
        """
        if self._P is None:
            self._P = self.diff_op()
        if Y is None:
            if self.X is None:
                raise ValueError(
                    "No input data has been fitted yet. Call fit() first with the parameter `cache_input` set to True."
                )
            Y = self.X.copy()
        Y_arr = Y.toarray() if issparse(Y) else np.asarray(Y)
        Y_imp = Y_arr
        if t is None or t < 0:
            P_mat = _ensure_csr_matrix(self._P)
            previous = np.asarray(Y_arr, dtype=float)
            P_power = P_mat.copy()

            for i in range(1, int(tmax) + 1):
                if i > 1:
                    P_power = _ensure_csr_matrix(P_power @ P_mat)
                Y_imp = P_power @ Y_arr
                error, _ = self._calculate_imputation_error(Y_imp, previous)

                if error is not None and error < threshold:
                    logger.info("Optimal t: %s", i)
                    break

                previous = np.asarray(Y_imp, dtype=float)

        else:
            P_mat = _ensure_csr_matrix(self._P)
            t_int = int(t)
            if t_int < 1:
                return Y_arr
            P_power = P_mat.copy()
            for _ in range(1, t_int):
                P_power = _ensure_csr_matrix(P_power @ P_mat)
            Y_imp = P_power @ Y_arr
        return Y_imp

    def _get_landmarks(self, X, n_landmarks=None):
        """Select landmark points (placeholder for future landmark selection)."""
        raise NotImplementedError("Landmark selection is not implemented.")

    def filter(
        self,
        signal,
        replicates=None,
        beta=50,
        target=None,
        filterfunc=None,
        offset=0,
        order=1,
        solver="chebyshev",
        chebyshev_order=100,
    ):
        """Estimate per-sample density over the graph by filtering a signal.

        Inspired by [MELD](https://github.com/KrishnaswamyLab/MELD) for sample-associated density estimation.
        However, you can naturally use this for any signal in your data, not just samples of specific conditions. In practice, this is just
        a simple [PyGSP](https://pygsp.readthedocs.io/en/stable/reference/filters.html#module-pygsp.filters) filter on a graph.
        Indeed, it calls PyGSP, so you'll need it installed to use this function.

        Parameters
        ----------
        signal: array-like
            Signal(s) to filter - usually sample labels.

        replicates: array-like, optional (default None)
            Replicate labels for each sample. If None, no replicates are assumed.

        beta : int, default=50
            Amount of smoothing to apply. Vary this parameter to get good estimates
            - this can vary widely from dataset to dataset.

        target : array-like, default=None
            Similarity matrix to use for filtering. If None, uses the kernel matrix.

        filterfunc : function, default=None
            Function to use for filtering. If None, the default is to use a Laplacian filter.

        offset: float, default=0
            Amount to shift the filter in the eigenvalue spectrum.
            Recommended to use an eigenvalue from the graph based on the
            spectral distribution. Should be in interval [0,1]

        order: int, default=1
            Falloff and smoothness of the filter.
            High order leads to square-like filters.

        solver : string, default='chebyshev'
            Method to solve convex problem. If 'chebyshev', uses a chebyshev polynomial approximation of the corresponding
            filter. Else, if 'exact', uses the eigenvalue solution to the problem

        chebyshev_order : int, default=100
            Order of chebyshev approximation to use.

        Returns
        -------
        densities : DataFrame
            The filtered sample densities.
        """
        if self._sample_densities is None:
            try:
                from pygsp import filters, graphs  # type: ignore
            except ImportError:
                raise ImportError(
                    "pygsp is not installed. Please install it with `pip install pygsp` to use filtering functions."
                )
            import pandas as pd

            # try converting signal labels
            sample_labels = signal.copy()
            samples = np.unique(sample_labels)
            if hasattr(sample_labels, "index"):
                _labels_index = sample_labels.index
            else:
                _labels_index = None
            try:
                labels = sample_labels.values
            except AttributeError:
                labels = sample_labels
            if len(labels.shape) > 1:
                if labels.shape[1] == 1:
                    labels = labels.reshape(-1)
                else:
                    raise ValueError(
                        f"sample_labels must be a single column. Gotshape={labels.shape}"
                    )
            if samples.shape[0] == 2:
                df = pd.DataFrame(
                    [labels == samples[0], labels == samples[1]],
                    columns=_labels_index,
                ).astype(int)
                df.index = samples
                sample_indicators = df.T
            else:
                from sklearn.preprocessing import LabelBinarizer

                _LB = LabelBinarizer()
                _sample_indicators = _LB.fit_transform(sample_labels)

                _sample_indicators_dense = np.asarray(_sample_indicators)

                sample_indicators = pd.DataFrame(
                    _sample_indicators_dense,
                    columns=_LB.classes_,  # type: ignore
                )
            sample_indicators = sample_indicators / sample_indicators.sum(axis=0)
            # convert to pygsp format
            # will need to pad 1's to diagonal for filtering
            if target is None:
                graph = graphs.Graph(self.K)  # type: ignore
            else:
                graph = graphs.Graph(target)  # type: ignore
            graph.estimate_lmax()
            # default to Laplacian filter
            if filterfunc is None:

                def _default_filterfunc(x):
                    return 1 / (1 + (beta * np.abs(x / graph.lmax - offset)) ** order)

                filterfunc = _default_filterfunc

            filt = filters.Filter(graph, filterfunc)  # type: ignore
            densities = filt.filter(
                sample_indicators, method=solver, order=chebyshev_order
            )
            self._sample_densities = pd.DataFrame(
                densities, index=_labels_index, columns=sample_indicators.columns
            )

        if replicates is not None:
            return self._replicate_normalize_densities(replicates)

        return self._sample_densities

    def _replicate_normalize_densities(self, replicates):
        from sklearn.preprocessing import normalize

        replicates = np.unique(replicates)
        assert self._sample_densities is not None
        sample_likelihoods = self._sample_densities.copy()
        for rep in replicates:
            rep_str = str(rep)
            curr_cols = self._sample_densities.columns[
                [str(col).endswith(rep_str) for col in self._sample_densities.columns]
            ]
            if len(curr_cols) == 0:
                continue
            sample_likelihoods[curr_cols] = normalize(
                self._sample_densities[curr_cols], norm="l1"
            )
        return sample_likelihoods

    def is_connected(self):
        """Check if the graph is connected (cached).

        A graph is connected if and only if there exists a (directed) path
        between any two vertices.

        Returns
        -------
        connected : bool
            True if the graph is connected, False otherwise.

        """
        if self._connected is not None:
            return self._connected
        assert self.N is not None
        adjacencies = [self.A]
        for adjacency in adjacencies:
            visited = np.zeros(self.N, dtype=bool)
            stack = set([0])
            while stack:
                vertex = stack.pop()
                if visited[vertex]:
                    continue
                visited[vertex] = True
                neighbors = adjacency[vertex].nonzero()[1]
                stack.update(neighbors)
            if not np.all(visited):
                self._connected = False
                return self._connected
        self._connected = True
        return self._connected

    def _connected_components(self, target=None):
        """Find the connected components of the kernel matrix (or ``target``).

        Other matrices can be specified for use with the `target` parameter.

        Parameters
        ----------
        target : array-like, default=None
            The target matrix to find the connected components of. If None, uses the kernel matrix.

        Returns
        -------
        components : list of np.ndarray
            The connected components of the target matrix.

        labels : list of int
            The labels of the connected components.
        """
        from topo._compat.scipy_graph import graph_connected_components

        if target is None:
            n_components, labels = graph_connected_components(self.K, directed=False)
        else:
            n_components, labels = graph_connected_components(target, directed=False)

        return n_components, labels

    def resistance_distance(self):
        """Compute resistance distances from the cached Laplacian matrix.

        See Klein and Randic [manuscript](https://doi.org/10.1007%2FBF01164627) for details.

        Notes
        -----
        This method computes a dense matrix inverse and requires O(n_samples^2)
        memory. It should not be used on very large graphs.

        Returns
        -------
        rd : sparse matrix
            Resistance distance matrix

        """
        if self.laplacian_type != "unnormalized":
            L = self.laplacian(laplacian_type="unnormalized")
        else:
            L = self.L
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)  # type: ignore
        pseudo = np.linalg.pinv(np.asarray(L_dense, dtype=float))
        diag = np.diag(pseudo)
        rd = diag[:, None] + diag[None, :] - 2.0 * pseudo
        rd = np.maximum(rd, 0.0)
        np.fill_diagonal(rd, 0.0)
        return csr_matrix(rd)

    def sparsify(self, epsilon=0.1, maxiter=10, random_state=None):
        """Sparsify the graph with the Spielman-Srivastava method.

        This originally only called PyGSP but now also has some adaptations.

        Parameters
        ----------
        epsilon : float, default=0.1
            Sparsification parameter, which must be between ``1/sqrt(N)`` and 1.

        maxiter : int, default=10
            Maximum number of iterations.

        random_state : int or RandomState, default=None
            Seed for the random number generator (for reproducible sparsification).

        Notes
        -----
        This depends on `resistance_distance()` and thus inherits its heavy
        computational requirements.

        Returns
        -------
        sparse_graph : sparse matrix
            Sparsified graph affinity matrix.
        """
        assert self.N is not None

        if not 1.0 / np.sqrt(self.N) <= epsilon < 1:
            raise ValueError("Epsilon out of required range!")

        rng = check_random_state(random_state)

        # Not sparse
        rd = self.resistance_distance()
        resistance_distances = (
            rd.toarray() if hasattr(rd, "toarray") else np.asarray(rd)
        )
        W = coo_matrix(self.K)
        W.data[W.data < 1e-10] = 0
        W = W.tocsc()
        W.eliminate_zeros()
        start_nodes, end_nodes, weights = find(tril(W))

        # Calculate the new weights.
        weights = np.maximum(0, weights)  # type: ignore
        Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])  # type: ignore
        Pe = weights * Re
        Pe = np.asarray(Pe, dtype=float)

        total = Pe.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                "Invalid edge sampling probabilities during sparsification."
            )

        Pe = Pe / total

        sparserW = None
        sparserL = None
        for i in range(maxiter):
            # Rudelson, 1996 Random Vectors in the Isotropic Position
            # (too hard to figure out actual C0)
            C0 = 1 / 30.0
            # Rudelson and Vershynin, 2007, Thm. 3.1
            C = 4 * C0
            q = round(self.N * np.log(self.N) * 9 * C**2 / (epsilon**2))

            results = rng.choice(len(Pe), size=int(q), p=Pe)
            values, inv = np.unique(results, return_inverse=True)
            values = values.astype(int)
            freq = np.bincount(inv).astype(int)
            spin_counts = np.array([values, freq]).T
            per_spin_weights = weights / (q * Pe)

            counts = np.zeros(np.shape(weights)[0])
            counts[spin_counts[:, 0]] = spin_counts[:, 1]
            new_weights = counts * per_spin_weights

            sparserW = csc_matrix(
                (new_weights, (start_nodes, end_nodes)), shape=(self.N, self.N)
            )
            sparserW = sparserW + sparserW.T
            degrees = np.asarray(sparserW.sum(axis=1)).ravel()
            sparserL = diags(degrees, 0) - sparserW

            from topo._compat.scipy_graph import graph_connected_components

            n_sparser_components, _ = graph_connected_components(
                sparserW, directed=False
            )
            if n_sparser_components == 1:
                break
            elif i == maxiter - 1:
                logger.warning("Graph is disconnected. Sparsifying anyway...")
            else:
                epsilon -= (epsilon - 1 / np.sqrt(self.N)) / 2.0

        if sparserW is None:
            raise RuntimeError("Sparsification did not run; maxiter must be >= 1.")
        if sparserL is not None:
            sparserW = diags(sparserL.diagonal(), 0) - sparserL  # type: ignore
            sparserW = (sparserW + sparserW.T) / 2.0  # type: ignore

        return sparserW  # type: ignore

    def interpolate(
        self, f_subsampled, keep_inds, target=None, order=100, reg_eps=0.005
    ):
        r"""
        Interpolate a graph signal.

        Parameters
        ----------
        f_subsampled : ndarray
            A graph signal on the graph G.

        keep_inds : ndarray
            List of indices on which the signal is sampled.

        target : array-like, default=None
            Similarity matrix to use for interpolation. If None, uses the kernel matrix.

        order : int
            Degree of the Chebyshev approximation (default = 100).

        reg_eps : float
            The regularized graph Laplacian is $\bar{L}=L+\epsilon I$.
            A smaller epsilon may lead to better regularization,
            but will also require a higher order Chebyshev approximation.

        Returns
        -------
        signal_interpolated : ndarray
            Interpolated graph signal on the full vertex set of G.
        """
        try:
            from pygsp import graphs, reduction  # type: ignore
        except ImportError:
            raise ImportError(
                "pygsp is not installed. Please install it with `pip install pygsp` to use interpolating functions."
            )
        # convert to pygsp format
        if target is None:
            graph = graphs.Graph(self.K)
        else:
            graph = graphs.Graph(target)

        signal_interpolated = reduction.interpolate(
            graph, f_subsampled, keep_inds, order, reg_eps
        )
        return signal_interpolated
