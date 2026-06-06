# Standalone analysis functions extracted from TopOGraph.
"""Standalone spectral-analysis functions.

Operator-native diagnostics (spectral selectivity, eigengap analysis and related
scores) that take explicit arguments — operators, scaffolds, eigenvalues — rather
than relying on a :class:`topo.topograph.TopOGraph` instance's internal state.
"""

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Spectral selectivity helpers (module-private)
# ---------------------------------------------------------------------------


def _std_cols(A, eps=1e-12):
    A = np.asarray(A, float)
    A = A - np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, keepdims=True)
    return A / (sd + eps)


def _spectral_weights(
    evals=None, m=None, mode="lambda_over_one_minus_lambda", eps=1e-12
):
    if evals is None:
        return np.ones(m or 1, float)
    ev = np.asarray(evals, float)
    if mode == "lambda_over_one_minus_lambda":
        return ev / (1.0 - ev + eps)
    elif mode == "lambda":
        return ev
    return np.ones_like(ev)


def _compute_eas(Zs, w=None, eps=1e-12):
    """Entropy-based Axis Selectivity."""
    n, m = Zs.shape
    if w is None:
        w = np.ones(m, float)
    E = (Zs**2) * w[None, :]
    S = np.sum(E, axis=1, keepdims=True) + eps
    P = E / S
    H = -np.sum(P * np.log(P + eps), axis=1)
    Hmax = np.log(m)
    EAS = 1.0 - (H / (Hmax + eps))
    kstar = np.argmax(E, axis=1)
    sign_kstar = np.sign(Zs[np.arange(n), kstar])
    radius = np.sqrt(np.square(Zs).sum(1))
    return EAS, kstar, sign_kstar, radius


def _compute_radiality(Zs, k=30, metric="euclidean", eps=1e-12):
    from sklearn.neighbors import NearestNeighbors

    n = Zs.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k, n - 1), metric=metric).fit(Zs)
    _, idx = nn.kneighbors(Zs, return_distance=True)
    nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
    r = np.linalg.norm(Zs, axis=1)
    r_med = np.median(r[nbr], axis=1)
    q75 = np.percentile(r[nbr], 75, axis=1)
    q25 = np.percentile(r[nbr], 25, axis=1)
    iqr = q75 - q25
    z = (r - r_med) / (iqr + eps)
    return z, r


def _compute_lac(Zs, k=30, metric="euclidean", eps=1e-12):
    """Local Axial Coherence = EVR1 of local PCA."""
    from numpy.linalg import svd
    from sklearn.neighbors import NearestNeighbors

    n = Zs.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k, n - 1), metric=metric).fit(Zs)
    _, idx = nn.kneighbors(Zs, return_distance=True)
    nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
    out = np.zeros(n, float)
    for i in range(n):
        Znb = Zs[nbr[i]]
        Znb = Znb - Znb.mean(0, keepdims=True)
        _, s, _ = svd(Znb, full_matrices=False)
        num = s[0] ** 2
        den = (s**2).sum() + eps
        out[i] = num / den
    return out


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
    P: sp.spmatrix | None = None,
    smooth_t: int = 0,
) -> dict:
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
    weight_mode : {'lambda_over_one_minus_lambda', 'lambda', 'none'}
        How eigenvalues are converted to axis weights. ``'lambda_over_one_minus_lambda'``
        mirrors msDM weighting, ``'lambda'`` uses raw eigenvalues, and any other
        value gives uniform weights.
    standardize : bool
        If True, center and scale scaffold columns before computing diagnostics.
    k_neighbors : int
        Neighborhood size used for radiality and local axial coherence.
    metric : str
        Metric used for scaffold-space nearest-neighbor searches.
    P : sparse matrix or None
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
    Zs = _std_cols(Z) if standardize else np.asarray(Z, float)
    w = _spectral_weights(evals, m=Z.shape[1], mode=weight_mode)

    EAS, kstar, sign_k, r = _compute_eas(Zs, w=w)
    z_rad, _ = _compute_radiality(Zs, k=k_neighbors, metric=metric)
    LAC = _compute_lac(Zs, k=k_neighbors, metric=metric)
    RayScore = (1.0 / (1.0 + np.exp(-z_rad))) * EAS

    if P is not None and int(smooth_t) > 0:

        def _smooth(v):
            u = np.asarray(v, float).copy()
            for _ in range(int(smooth_t)):
                u = P @ u
            return np.asarray(u).ravel()

        EAS = _smooth(EAS)
        RayScore = _smooth(RayScore)
        LAC = _smooth(LAC)

    return dict(
        EAS=EAS,
        RayScore=RayScore,
        LAC=LAC,
        axis=kstar.astype(int),
        axis_sign=(sign_k > 0).astype(int),
        radius=r,
    )


def filter_signal(signal, P, t: int = 8) -> np.ndarray:
    """Smooth a one-dimensional graph signal by applying ``P^t``.

    The input is one scalar value per sample. Repeated multiplication by the
    diffusion operator averages that signal over graph neighborhoods, with
    larger ``t`` spreading information farther across the fitted geometry.

    Parameters
    ----------
    signal : array-like, shape (n,)
        Scalar per-sample values to smooth.
    P : sparse matrix
        Diffusion/operator matrix whose rows and columns correspond to samples.
    t : int
        Number of diffusion steps. ``t=0`` returns the input signal as a float
        array; larger values smooth more strongly.

    Returns
    -------
    ndarray, shape (n,)
        Smoothed signal with one value per sample.
    """
    y = np.asarray(signal, float).copy().ravel()
    for _ in range(int(t)):
        y = P @ y
    return np.asarray(y).ravel()


def impute(X, P, t: int = 8, output: str = "auto", dtype=np.float64):
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
    P : sparse matrix
        Diffusion/operator matrix with shape ``(n, n)``.
    t : int
        Number of diffusion steps. Larger values smooth more aggressively.
    output : {'auto', 'sparse', 'dense'}
        Output format. ``'auto'`` preserves input sparsity when possible.
    dtype : numpy dtype
        Numeric dtype used for the diffusion computation.

    Returns
    -------
    sparse or ndarray
        Diffused matrix with the same shape as ``X``.
    """
    if sp.issparse(X):
        Xc = X.tocsr(copy=True).astype(dtype)
        for _ in range(int(t)):
            Xc = P @ Xc
        if output in ("auto", "sparse"):
            return Xc
        return Xc.toarray()
    else:
        Xd = np.asarray(X, dtype=dtype)
        for _ in range(int(t)):
            Xd = P @ Xd
        if output in ("auto", "dense"):
            return Xd
        return sp.csr_matrix(Xd)


def riemann_diagnostics(
    Y: np.ndarray,
    L,
    *,
    center: str = "median",
    diffusion_t: int = 0,
    diffusion_op=None,
    normalize: str = "symmetric",
    clip_percentile: float = 2.0,
    return_limits: bool = True,
    compute_metric: bool = True,
    compute_scalars: bool = True,
) -> dict:
    """Estimate local distortion of a two-dimensional embedding.

    The function computes a Riemannian metric field for a 2-D layout relative to
    a graph Laplacian/operator and optionally derives scalar deformation maps.
    It is intended for diagnosing where a visualization contracts, expands, or
    distorts local directions.

    Parameters
    ----------
    Y : ndarray, shape (n, 2)
        Two-dimensional embedding to diagnose.
    L : array-like
        Graph Laplacian or operator defining the reference geometry.
    center : {'median', 'mean'}
        Center used when converting metric tensors into deformation values.
    diffusion_t : int
        Number of diffusion smoothing steps for deformation maps.
    diffusion_op : sparse matrix or None
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
    from topo.eval.rmetric import RiemannMetric, calculate_deformation

    out = {}
    G = None
    if compute_metric:
        G = RiemannMetric(Y, L).get_rmetric()
        out["G"] = G

    if compute_scalars:
        if G is None:
            G = RiemannMetric(Y, L).get_rmetric()
            out["G"] = G
        lam = np.linalg.eigvalsh(G)  # type: ignore
        lam = np.clip(lam, 1e-12, None)
        out["anisotropy"] = np.log(lam[:, -1] / lam[:, 0])
        out["logdetG"] = np.sum(np.log(lam), axis=1)

    deform_vals, limits = calculate_deformation(
        Y,
        L,
        center=center,
        diffusion_t=int(max(0, diffusion_t)),
        diffusion_op=diffusion_op,
        normalize=normalize,
        clip_percentile=float(clip_percentile),
        return_limits=True,
    )
    out["deformation"] = deform_vals
    if return_limits:
        out["limits"] = limits

    return out
