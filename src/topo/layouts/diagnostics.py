"""Projection diagnostics and visualization helpers.

These utilities operate on a fitted TopOGraph-like object but are intentionally
kept outside the core layout mixin. They depend on public/canonical fitted state
and projection methods rather than being part of the fitting pipeline.
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def find_ideal_projection(
    tg: Any,
    min_dist_grid: Iterable[float] | None = None,
    spread_grid: Iterable[float] | None = None,
    initial_alpha_grid: Iterable[float] | None = None,
    *,
    multiscale: bool = True,
    num_iters: int = 600,
    save_every: int = 10,
    metric: str = "euclidean",
    n_neighbors: int = 30,
    times: Sequence[int] = (1, 2, 4),
    r: int = 32,
    k_for_pf1: int | None = None,
    symmetric_hint: bool = True,
) -> dict[str, Any]:
    """Grid-search MAP hyperparameters for a fitted TopOGraph-like object.

    The object must expose:
    - ``project(...)``
    - ``P_X_``
    - ``_backend_resolved``
    - ``_n_jobs_effective``
    - ``verbosity``
    - ``msTopoMAP_snapshots`` / ``TopoMAP_snapshots``

    This function does not rerun the best projection automatically. It returns
    the best parameters and score records; the caller can decide whether to run
    the final projection.
    """
    from topo.eval.topo_metrics import get_P, topo_preserve_score

    if min_dist_grid is None:
        min_dist_grid = (0.2, 0.6, 1.0)
    if spread_grid is None:
        spread_grid = (0.8, 1.2, 1.6)
    if initial_alpha_grid is None:
        initial_alpha_grid = (0.4, 1.0, 1.6)

    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if int(num_iters) < 1:
        raise ValueError("num_iters must be >= 1.")
    if int(save_every) < 1:
        raise ValueError("save_every must be >= 1.")

    PX_ref = getattr(tg, "P_X_", None)
    if PX_ref is None:
        raise ValueError(
            "Input-space diffusion operator unavailable. Call fit() first."
        )
    PX_ref = csr_matrix(PX_ref)

    backend = getattr(tg, "_backend_resolved", None)
    if backend not in {"sklearn", "hnswlib"}:
        raise ValueError("Fitted object has invalid `_backend_resolved`.")

    n_jobs = int(getattr(tg, "_n_jobs_effective", 1))
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError("Fitted object has invalid `_n_jobs_effective`.")

    snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"

    best_score = float("-inf")
    best_params: dict[str, float] | None = None
    best_snapshot_scores: list[dict[str, Any]] | None = None
    score_records: list[dict[str, Any]] = []

    for min_dist in min_dist_grid:
        for spread in spread_grid:
            for initial_alpha in initial_alpha_grid:
                params = {
                    "min_dist": float(min_dist),
                    "spread": float(spread),
                    "initial_alpha": float(initial_alpha),
                }

                if getattr(tg, "verbosity", 0) >= 1:
                    logger.info(
                        "[Grid] MAP: min_dist=%s, spread=%s, initial_alpha=%s",
                        params["min_dist"],
                        params["spread"],
                        params["initial_alpha"],
                    )

                tg.project(
                    projection_method="MAP",
                    multiscale=bool(multiscale),
                    num_iters=int(num_iters),
                    save_every=int(save_every),
                    include_init_snapshot=True,
                    **params,
                )

                snapshots = getattr(tg, snap_attr, None) or []
                snapshot_scores: list[dict[str, Any]] = []

                for snap_idx, snap in enumerate(snapshots):
                    if "embedding" not in snap:
                        raise RuntimeError(
                            "MAP snapshot is missing required `embedding` field."
                        )

                    PY = get_P(
                        snap["embedding"],
                        metric=metric,
                        n_neighbors=int(n_neighbors),
                        backend=backend,
                        n_jobs=n_jobs,
                    )
                    PY = csr_matrix(PY)

                    score, parts = topo_preserve_score(
                        PX_ref,
                        PY,
                        times=times,
                        r=int(r),
                        symmetric_hint=bool(symmetric_hint),
                        k_for_pf1=k_for_pf1,
                    )

                    snapshot_scores.append(
                        {
                            "snapshot": snap_idx,
                            "score": float(score),
                            "metrics": {
                                "TP": float(score),
                                "PF1": float(parts.get("PF1", np.nan)),
                                "PJS": float(parts.get("PJS", np.nan)),
                                "SP": float(parts.get("SP", np.nan)),
                            },
                        }
                    )

                final_score = (
                    snapshot_scores[-1]["score"] if snapshot_scores else float("-inf")
                )

                record = {
                    **params,
                    "final_score": float(final_score),
                    "snapshot_scores": snapshot_scores,
                }
                score_records.append(record)

                if final_score > best_score:
                    best_score = float(final_score)
                    best_params = params
                    best_snapshot_scores = snapshot_scores

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "scores": score_records,
        "best_snapshot_scores": best_snapshot_scores,
    }


def run_best_projection(
    tg: Any,
    params: dict[str, float],
    *,
    multiscale: bool = True,
    num_iters: int = 600,
    save_every: int = 10,
) -> np.ndarray:
    """Run MAP once using selected hyperparameters."""
    required = {"min_dist", "spread", "initial_alpha"}
    missing = required.difference(params)
    if missing:
        raise ValueError(f"Missing MAP parameter(s): {sorted(missing)}.")

    Y = tg.project(
        projection_method="MAP",
        multiscale=bool(multiscale),
        num_iters=int(num_iters),
        save_every=int(save_every),
        include_init_snapshot=True,
        min_dist=float(params["min_dist"]),
        spread=float(params["spread"]),
        initial_alpha=float(params["initial_alpha"]),
    )

    return np.asarray(Y)


def visualize_optimization(
    tg: Any,
    num_iters: int = 600,
    save_every: int = 10,
    dpi: int = 120,
    color=None,
    *,
    multiscale: bool = True,
    filename: str | None = None,
    point_size: float = 3.0,
    fps: int = 20,
    include_init_snapshot: bool = True,
    overlay_metrics: bool = False,
):
    """Render an animated MAP optimization GIF for a fitted TopOGraph-like object."""
    from topo.plot import visualize_optimization as _visualize_optimization

    if int(num_iters) < 1:
        raise ValueError("num_iters must be >= 1.")
    if int(save_every) < 1:
        raise ValueError("save_every must be >= 1.")

    snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
    snapshots = getattr(tg, snap_attr, None)

    if not snapshots or len(snapshots) < 2:
        tg.project(
            projection_method="MAP",
            num_iters=max(int(num_iters), int(save_every)),
            save_every=int(save_every),
            include_init_snapshot=bool(include_init_snapshot),
            multiscale=bool(multiscale),
        )
        snapshots = getattr(tg, snap_attr, None)

    if not snapshots:
        raise RuntimeError("No MAP optimization snapshots available.")

    tag = "msTopoMAP" if multiscale else "TopoMAP"
    path = _visualize_optimization(
        snapshots,
        dpi=int(dpi),
        color=color,
        filename=filename,
        point_size=float(point_size),
        fps=int(fps),
        tag=tag,
        overlay_metrics=bool(overlay_metrics),
    )

    if getattr(tg, "verbosity", 0) >= 1:
        logger.info("Wrote %s with %d frames.", path, len(snapshots))

    return path
