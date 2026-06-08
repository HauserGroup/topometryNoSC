# Standalone analysis functions extracted from TopOGraph.
"""Standalone spectral-analysis functions.

Operator-native diagnostics (spectral selectivity, eigengap analysis and related
scores) that take explicit arguments — operators, scaffolds, eigenvalues — rather
than relying on a :class:`topo.topograph.TopOGraph` instance's internal state.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp

from topo.base.graph_matrix import as_csr_matrix

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_2d_float_array(X, name: str) -> np.ndarray:
    """Return X as a finite 2-D float ndarray."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_1d_float_array(x, name: str) -> np.ndarray:
    """Return x as a finite 1-D float ndarray."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_nonnegative_int(value, name: str) -> int:
    """Validate and return a non-negative integer."""
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{name} must be non-negative.")
    return ivalue


def _validate_positive_int(value, name: str) -> int:
    """Validate and return a positive integer."""
    ivalue = int(value)
    if ivalue < 1:
        raise ValueError(f"{name} must be at least 1.")
    return ivalue


def _as_square_operator(P, n: int, name: str):
    """Validate an n-by-n dense or sparse operator."""
    if P is None:
        raise ValueError(f"{name} must not be None.")

    P_checked = as_csr_matrix(P, name)

    if P_checked.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {P_checked.shape}.")

    return P_checked


def _matvec_or_matmat_power(P, X, t: int):
    """Apply P^t to a vector or matrix."""
    out = X
    for _ in range(t):
        out = P @ out
    return out


# ---------------------------------------------------------------------------
# Spectral selectivity helpers (module-private)
# ---------------------------------------------------------------------------


def _std_cols(A, eps: float = 1e-12) -> np.ndarray:
    """Column-standardize a finite 2-D array."""
    A = _as_2d_float_array(A, "A")
    A = A - np.mean(A, axis=0, keepdims=True)
    sd = np.std(A, axis=0, keepdims=True)
    return A / (sd + eps)


def _spectral_weights(
    evals: np.ndarray | None = None,
    m: int | None = None,
    mode: str = "lambda_over_one_minus_lambda",
    eps: float = 1e-12,
) -> np.ndarray:
    """Return non-negative axis weights for a spectral scaffold.

    ``lambda_over_one_minus_lambda`` mirrors the usual msDM weighting after the
    trivial eigenvalue has been removed. Since this diagnostic computes weighted
    squared coordinate energy, weights are constrained to be non-negative.
    """
    if m is None:
        if evals is None:
            m = 1
        else:
            m = int(np.asarray(evals).size)

    if m < 1:
        raise ValueError("m must be at least 1.")

    if evals is None:
        return np.ones(m, dtype=float)

    ev = _as_1d_float_array(evals, "evals")
    if ev.size != m:
        raise ValueError(
            f"evals must have length matching the scaffold columns ({m}), "
            f"got {ev.size}."
        )

    if mode == "lambda_over_one_minus_lambda":
        # msDM weighting is meaningful for non-trivial eigenvalues in [0, 1).
        # Clip to avoid exploding weights when eigenvalues are numerically near 1.
        ev_clipped = np.clip(ev, 0.0, 1.0 - eps)
        return ev_clipped / (1.0 - ev_clipped + eps)

    if mode == "lambda":
        return np.clip(ev, 0.0, None)

    if mode in {"none", "uniform", None}:
        return np.ones_like(ev, dtype=float)

    raise ValueError(
        "weight_mode must be one of "
        "{'lambda_over_one_minus_lambda', 'lambda', 'none', 'uniform'}."
    )


def _compute_eas(
    Zs: np.ndarray,
    w: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Entropy-based Axis Selectivity.

    EAS is high when a sample's weighted scaffold energy is concentrated in a
    small number of axes. When total weighted energy is numerically zero, EAS is
    set to zero rather than reporting spurious perfect selectivity.
    """
    Zs = _as_2d_float_array(Zs, "Zs")
    n, m = Zs.shape

    if w is None:
        weights = np.ones(m, dtype=float)
    else:
        weights = np.asarray(w, dtype=float).ravel()
        if weights.shape[0] != m:
            raise ValueError(f"w must have length {m}.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("w must contain only finite values.")
        weights = np.clip(weights, 0.0, None)

    E = (Zs**2) * weights[None, :]
    S = np.sum(E, axis=1, keepdims=True)
    nonzero = S[:, 0] > eps

    P = np.zeros_like(E, dtype=float)
    P[nonzero] = E[nonzero] / S[nonzero]

    H = -np.sum(P * np.log(P + eps), axis=1)

    if m == 1:
        EAS = np.ones(n, dtype=float)
    else:
        Hmax = np.log(float(m))
        EAS = 1.0 - (H / (Hmax + eps))

    EAS = np.where(nonzero, EAS, 0.0)
    EAS = np.clip(EAS, 0.0, 1.0)

    kstar = np.argmax(E, axis=1)
    sign_kstar = np.sign(Zs[np.arange(n), kstar])
    radius = np.sqrt(np.square(Zs).sum(axis=1))

    return EAS, kstar, sign_kstar, radius


def _neighbor_indices(
    Zs: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
) -> np.ndarray:
    """Return up to k non-self nearest-neighbor indices for each row."""
    from sklearn.neighbors import NearestNeighbors

    Zs = _as_2d_float_array(Zs, "Zs")
    n = Zs.shape[0]
    k = _validate_positive_int(k, "k")

    if n == 1:
        return np.empty((1, 0), dtype=int)

    # sklearn includes the query point itself when querying the fitted data.
    # Ask for k + 1 neighbors, then drop self.
    n_query_neighbors = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=n_query_neighbors, metric=metric).fit(Zs)
    _, idx = nn.kneighbors(Zs, return_distance=True)

    # Drop each row's self index robustly. Usually it is first, but ties or
    # non-standard metrics can disturb ordering.
    out = []
    for i in range(n):
        row = idx[i]
        row = row[row != i]
        out.append(row[:k])

    max_len = max((row.size for row in out), default=0)
    if max_len == 0:
        return np.empty((n, 0), dtype=int)

    padded = np.full((n, max_len), -1, dtype=int)
    for i, row in enumerate(out):
        padded[i, : row.size] = row

    return padded


def _compute_radiality(
    Zs: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local radial separation relative to neighbors."""
    Zs = _as_2d_float_array(Zs, "Zs")
    n = Zs.shape[0]
    r = np.linalg.norm(Zs, axis=1)

    if n == 1:
        return np.zeros(1, dtype=float), r

    nbr = _neighbor_indices(Zs, k=k, metric=metric)
    z = np.zeros(n, dtype=float)

    for i in range(n):
        row = nbr[i]
        row = row[row >= 0]
        if row.size == 0:
            z[i] = 0.0
            continue

        nbr_r = r[row]
        r_med = np.median(nbr_r)
        q75 = np.percentile(nbr_r, 75)
        q25 = np.percentile(nbr_r, 25)
        iqr = q75 - q25
        z[i] = (r[i] - r_med) / (iqr + eps)

    return z, r


def _compute_lac(
    Zs: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Local Axial Coherence as EVR1 of local PCA."""
    from numpy.linalg import svd

    Zs = _as_2d_float_array(Zs, "Zs")
    n = Zs.shape[0]

    if n == 1:
        return np.zeros(1, dtype=float)

    nbr = _neighbor_indices(Zs, k=k, metric=metric)
    out = np.zeros(n, dtype=float)

    for i in range(n):
        row = nbr[i]
        row = row[row >= 0]
        if row.size < 2:
            out[i] = 0.0
            continue

        Znb = Zs[row]
        Znb = Znb - Znb.mean(axis=0, keepdims=True)
        _, s, _ = svd(Znb, full_matrices=False)
        num = s[0] ** 2
        den = np.sum(s**2) + eps
        out[i] = float(num / den)

    return np.clip(out, 0.0, 1.0)


def _smooth_vector(v: np.ndarray, P, t: int) -> np.ndarray:
    """Apply P^t to a one-dimensional vector."""
    y = np.asarray(v, dtype=float).copy().ravel()
    y = _matvec_or_matmat_power(P, y, t)
    return np.asarray(y, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def spectral_selectivity(
    Z: np.ndarray,
    evals: np.ndarray | None = None,
    *,
    weight_mode: str = "lambda_over_one_minus_lambda",
    standardize: bool = True,
    k_neighbors: int = 30,
    metric: str = "euclidean",
    P: sp.spmatrix | np.ndarray | None = None,
    smooth_t: int = 0,
) -> dict[str, np.ndarray]:
    """Measure how each sample uses the axes of a spectral scaffold.

    These diagnostics summarize per-sample structure in a fitted DM/msDM
    scaffold. They are useful for exploratory interpretation: identifying points
    dominated by one spectral mode, points lying far from the scaffold center,
    and neighborhoods that are locally axis-like. They are not clustering labels
    and should be interpreted together with the embedding and graph metrics.

    Parameters
    ----------
    Z : ndarray, shape (n, m)
        Spectral scaffold coordinates, where rows are samples and columns are
        eigenvector-derived coordinates.
    evals : ndarray or None
        Eigenvalues matching the scaffold columns. If provided, they weight axes
        so smoother or more persistent modes can contribute more strongly. If
        None, all axes are weighted equally.
    weight_mode : {'lambda_over_one_minus_lambda', 'lambda', 'none', 'uniform'}
        How eigenvalues are converted to axis weights. ``'lambda_over_one_minus_lambda'``
        mirrors msDM weighting, ``'lambda'`` uses raw non-negative eigenvalues,
        and ``'none'``/``'uniform'`` gives uniform weights.
    standardize : bool
        If True, center and scale scaffold columns before computing diagnostics.
    k_neighbors : int
        Neighborhood size used for radiality and local axial coherence.
    metric : str
        Metric used for scaffold-space nearest-neighbor searches.
    P : sparse matrix, dense ndarray, or None
        Optional diffusion operator used to smooth scalar diagnostic fields.
    smooth_t : int
        Number of smoothing steps when ``P`` is provided.

    Returns
    -------
    dict
        Dictionary with one value per sample:

        ``EAS``
            Entropy-based axis selectivity; larger values mean the sample's
            scaffold energy is concentrated in fewer axes.
        ``RayScore``
            Axis selectivity modulated by radial separation from neighboring
            samples.
        ``LAC``
            Local axial coherence; larger values mean nearby points are arranged
            more like a one-dimensional local direction in scaffold space.
        ``axis``
            Index of the dominant scaffold axis for each sample.
        ``axis_sign``
            Sign of the dominant axis coordinate, encoded as 0/1.
        ``radius``
            Euclidean distance of each standardized sample from the scaffold
            origin.
    """
    Z_arr = _as_2d_float_array(Z, "Z")
    n, m = Z_arr.shape

    k_neighbors = _validate_positive_int(k_neighbors, "k_neighbors")
    smooth_t = _validate_nonnegative_int(smooth_t, "smooth_t")

    Zs = _std_cols(Z_arr) if standardize else Z_arr.copy()
    w = _spectral_weights(evals, m=m, mode=weight_mode)

    EAS, kstar, sign_k, radius = _compute_eas(Zs, w=w)
    z_rad, _ = _compute_radiality(Zs, k=k_neighbors, metric=metric)
    LAC = _compute_lac(Zs, k=k_neighbors, metric=metric)

    # Sigmoid-transformed radiality emphasizes points that are radially separated
    # from their local neighborhood while preserving bounded scores.
    RayScore = (1.0 / (1.0 + np.exp(-z_rad))) * EAS

    if P is not None and smooth_t > 0:
        P_checked = _as_square_operator(P, n, "P")
        EAS = _smooth_vector(EAS, P_checked, smooth_t)
        RayScore = _smooth_vector(RayScore, P_checked, smooth_t)
        LAC = _smooth_vector(LAC, P_checked, smooth_t)

    return {
        "EAS": np.asarray(EAS, dtype=float).ravel(),
        "RayScore": np.asarray(RayScore, dtype=float).ravel(),
        "LAC": np.asarray(LAC, dtype=float).ravel(),
        "axis": kstar.astype(int),
        "axis_sign": (sign_k > 0).astype(int),
        "radius": np.asarray(radius, dtype=float).ravel(),
    }


def filter_signal(signal, P, t: int = 8) -> np.ndarray:
    """Smooth a one-dimensional graph signal by applying ``P^t``.

    The input is one scalar value per sample. Repeated multiplication by the
    diffusion operator averages that signal over graph neighborhoods, with
    larger ``t`` spreading information farther across the fitted geometry.

    Parameters
    ----------
    signal : array-like, shape (n,)
        Scalar per-sample values to smooth.
    P : sparse matrix or dense ndarray
        Diffusion/operator matrix whose rows and columns correspond to samples.
    t : int
        Number of diffusion steps. ``t=0`` returns the input signal as a float
        array; larger values smooth more strongly.

    Returns
    -------
    ndarray, shape (n,)
        Smoothed signal with one value per sample.
    """
    y = _as_1d_float_array(signal, "signal")
    t = _validate_nonnegative_int(t, "t")
    P_checked = _as_square_operator(P, y.size, "P")

    out = _matvec_or_matmat_power(P_checked, y, t)
    return np.asarray(out, dtype=float).ravel()


def impute(
    X,
    P,
    t: int = 8,
    output: Literal["auto", "sparse", "dense"] = "auto",
    dtype=np.float64,
):
    """Diffuse every column of a data matrix over a graph operator.

    This treats each feature column as a graph signal and applies ``P^t``. The
    result is a geometry-smoothed version of the input matrix. It is useful for
    denoising or exploratory imputation, but the amount of smoothing is entirely
    controlled by the fitted operator and ``t``.

    Parameters
    ----------
    X : array-like or sparse, shape (n, d)
        Data matrix with one row per graph sample and one column per feature or
        signal.
    P : sparse matrix or dense ndarray
        Diffusion/operator matrix with shape ``(n, n)``.
    t : int
        Number of diffusion steps. Larger values smooth more aggressively.
    output : {'auto', 'sparse', 'dense'}
        Output format. ``'auto'`` preserves input sparsity when possible.
    dtype : numpy dtype
        Numeric dtype used for the diffusion computation.

    Returns
    -------
    sparse matrix or ndarray
        Diffused matrix with the same shape as ``X``.
    """
    if output not in {"auto", "sparse", "dense"}:
        raise ValueError("output must be one of {'auto', 'sparse', 'dense'}.")

    t = _validate_nonnegative_int(t, "t")

    if sp.issparse(X):
        X_work = as_csr_matrix(X, name="X", dtype=dtype, copy=True)

        shape = X_work.shape
        if shape is None or len(shape) != 2:
            raise ValueError("X must be a 2-D matrix.")

        n = int(shape[0])
        P_checked = _as_square_operator(P, n, "P")

        X_out = _matvec_or_matmat_power(P_checked, X_work, t)

        if output in {"auto", "sparse"}:
            return as_csr_matrix(X_out, name="output", dtype=dtype, copy=False)
        if sp.issparse(X_out):
            result = X_out.toarray()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            result = np.asarray(X_out)
        return np.asarray(result, dtype=dtype)

    X_work = np.asarray(X, dtype=dtype)
    if X_work.ndim != 2:
        raise ValueError("X must be a 2-D matrix.")
    if X_work.shape[0] == 0 or X_work.shape[1] == 0:
        raise ValueError("X must have at least one row and one column.")
    if not np.all(np.isfinite(X_work)):
        raise ValueError("X must contain only finite values.")

    n = X_work.shape[0]
    P_checked = _as_square_operator(P, n, "P")
    X_out = _matvec_or_matmat_power(P_checked, X_work, t)

    if output in {"auto", "dense"}:
        return np.asarray(X_out, dtype=dtype)
    return sp.csr_matrix(X_out)


def riemann_diagnostics(
    Y: np.ndarray,
    L,
    *,
    center: Literal["median", "mean"] = "median",
    diffusion_t: int = 0,
    diffusion_op=None,
    normalize: str = "symmetric",
    clip_percentile: float = 2.0,
    return_limits: bool = True,
    compute_metric: bool = True,
    compute_scalars: bool = True,
    compute_deformation: bool = True,
) -> dict[str, np.ndarray | tuple[float, float]]:
    """Estimate local distortion of a two-dimensional embedding.

    The function computes a Riemannian metric field for a 2-D layout relative to
    a graph Laplacian/operator and optionally derives scalar deformation maps.
    It is intended for diagnosing where a visualization contracts, expands, or
    distorts local directions.

    Parameters
    ----------
    Y : ndarray, shape (n, 2)
        Two-dimensional embedding to diagnose.
    L : array-like or sparse matrix
        Graph Laplacian or operator defining the reference geometry.
    center : {'median', 'mean'}
        Center used when converting metric tensors into deformation values.
    diffusion_t : int
        Number of diffusion smoothing steps for deformation maps.
    diffusion_op : sparse matrix, dense ndarray, or None
        Operator used for smoothing when ``diffusion_t > 0``.
    normalize : str
        Normalization mode passed to the deformation calculation.
    clip_percentile : float
        Percentile used to clip deformation extremes for stable visualization
        limits.
    return_limits : bool
        If True, include suggested plotting limits for deformation fields.
    compute_metric : bool
        If True, include the local metric tensor ``G``.
    compute_scalars : bool
        If True, include scalar summaries derived from ``G``.
    compute_deformation : bool
        If True, include the deformation scalar from ``calculate_deformation``.

    Returns
    -------
    dict
        Dictionary that may include:

        ``G``
            Local metric tensor for each sample, shape ``(n, 2, 2)``.
        ``anisotropy``
            Log ratio between the largest and smallest local metric eigenvalues.
        ``logdetG``
            Log determinant of the local metric tensor.
        ``deformation``
            Centered/scaled deformation scalar useful for plotting.
        ``limits``
            Suggested robust plotting limits when ``return_limits=True``.
    """
    if center not in {"median", "mean"}:
        raise ValueError("center must be either 'median' or 'mean'.")

    Y_arr = _as_2d_float_array(Y, "Y")
    if Y_arr.shape[1] != 2:
        raise ValueError("Y must have shape (n_samples, 2).")

    n = Y_arr.shape[0]
    diffusion_t = _validate_nonnegative_int(diffusion_t, "diffusion_t")

    _as_square_operator(L, n, "L")
    if diffusion_t > 0 and diffusion_op is not None:
        _as_square_operator(diffusion_op, n, "diffusion_op")

    from topo.eval.rmetric import RiemannMetric, calculate_deformation

    out: dict[str, np.ndarray | tuple[float, float]] = {}
    G = None

    if compute_metric or compute_scalars:
        G = np.asarray(RiemannMetric(Y_arr, L).get_rmetric(), dtype=float)
        if G.shape != (n, 2, 2):
            raise ValueError(
                "RiemannMetric(...).get_rmetric() must return shape "
                f"({n}, 2, 2), got {G.shape}."
            )

        if compute_metric:
            out["G"] = G

    if compute_scalars:
        assert G is not None
        lam = np.linalg.eigvalsh(G)
        lam = np.clip(lam, 1e-12, None)
        out["anisotropy"] = np.log(lam[:, -1] / lam[:, 0])
        out["logdetG"] = np.sum(np.log(lam), axis=1)

    if compute_deformation:
        deform_vals, limits = calculate_deformation(
            Y_arr,
            L,
            center=center,
            diffusion_t=diffusion_t,
            diffusion_op=diffusion_op,
            normalize=normalize,
            clip_percentile=float(clip_percentile),
            return_limits=True,
        )
        out["deformation"] = np.asarray(deform_vals, dtype=float).ravel()
        if return_limits:
            out["limits"] = limits

    return out
