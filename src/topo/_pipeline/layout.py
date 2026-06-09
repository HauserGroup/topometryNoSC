"""Layout / projection phase of :class:`topo.topograph.TopOGraph`.

Extracted into a mixin (mirroring :class:`topo.uom.UoMMixin`) to keep the main
orchestrator slim. ``TopOGraph`` inherits ``LayoutBuildMixin``; all methods here
operate on ``self`` and rely on attributes/properties defined on ``TopOGraph``.
"""

import gc
import logging
import time
import warnings
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse

from topo.base.graph_matrix import as_csr_matrix
from topo.layouts.projector import Projector
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.tpgraph.kernels import Kernel

logger = logging.getLogger(__name__)


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    """Return value as a 2-D dense array, rejecting tuples and None."""
    if value is None:
        raise ValueError(f"{name} must not be None.")
    if isinstance(value, tuple):
        raise TypeError(f"{name} must be a 2-D array, not a tuple.")

    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")

    return arr


class LayoutBuildMixin:
    """Projection / layout construction and visualisation methods."""

    # Interface contract — attributes supplied by TopOGraph
    projection_methods: list[str]
    graph_kernel_version: str
    base_kernel_version: str
    ProjectionDict: dict[str, np.ndarray]
    _kernel_msZ: Kernel | None
    _kernel_Z: Kernel | None
    laplacian_type: str
    eigen_tol: float
    runtimes: dict[str, float]
    SpecLayout: np.ndarray | None
    graph_knn: int
    graph_metric: str
    uom_enabled: bool
    msZ_uom: csr_matrix | None
    Z_uom: csr_matrix | None
    EigenbasisDict: dict[str, EigenDecomposition]
    n_jobs: int
    _n_jobs_effective: int
    backend: str
    _backend_resolved: str
    _random_state_resolved: np.random.RandomState
    layout_verbose: bool
    verbosity: int
    msTopoMAP_snapshots: list[dict[str, Any]]
    TopoMAP_snapshots: list[dict[str, Any]]
    uom_eigenvalues_ms_list: list[np.ndarray]
    _uom_active_mode: str
    uom_eigenvalues_dm_list: list[np.ndarray]
    uom_components_: list[np.ndarray] | None
    eigenbasis: EigenDecomposition | None
    base_kernel: Kernel | None
    _random_state_resolved: np.random.RandomState
    P_of_msZ_uom: csr_matrix | None
    P_of_Z_uom: csr_matrix | None

    def _run_projections(self) -> None:
        """Compute requested projections on both DM and msDM scaffold graphs."""
        if not self.projection_methods:
            return

        failures: list[tuple[str, str, Exception]] = []
        successes = 0

        for proj in self.projection_methods:
            for multiscale in (True, False):
                tag = "msZ" if multiscale else "Z/DM"
                try:
                    self.project(projection_method=proj, multiscale=multiscale)
                    successes += 1
                except Exception as exc:
                    failures.append((str(proj), tag, exc))
                    warnings.warn(
                        f"Projection {proj!r} on {tag} failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if successes == 0 and failures:
            msg = "; ".join(f"{proj} on {tag}: {err}" for proj, tag, err in failures)
            raise RuntimeError(f"All requested projections failed: {msg}")

    def _get_projection(self, method: str, multiscale: bool):
        """Look up a stored projection from ProjectionDict."""
        method = str(method)
        tag = "msDM" if multiscale else "DM"

        if method in ("MAP", "Isomap", "IsomorphicMDE", "IsometricMDE"):
            key = (
                f"{method} of {self.graph_kernel_version} from {tag} "
                f"with {self.base_kernel_version}"
            )
        else:
            key = f"{method} of {tag} with {self.base_kernel_version}"

        if key in self.ProjectionDict:
            return self.ProjectionDict[key]

        uom_key = f"{method} of UoM {tag} with {self.base_kernel_version}"
        if uom_key in self.ProjectionDict:
            return self.ProjectionDict[uom_key]

        raise AttributeError(
            f"{method} ({tag}) embedding unavailable. Call .fit() or .project() first."
        )

    # ------------------------------------------------------------------
    # Spectral layout
    # ------------------------------------------------------------------

    def spectral_layout(self, graph=None, n_components: int = 2):
        """Compute a spectral initialization for layout optimization."""
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1.")

        if graph is None:
            if self._kernel_msZ is not None:
                graph = self._kernel_msZ.K
            elif self._kernel_Z is not None:
                graph = self._kernel_Z.K
            else:
                raise ValueError("No graph kernel available. Call .fit() first.")

        shape = getattr(graph, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("graph must be a 2-D square matrix.")
        if int(shape[0]) != int(shape[1]):
            raise ValueError("graph must be square.")

        t0 = time.time()
        rng = self._random_state_resolved

        try:
            spt_result = cast(
                Any,
                spectral_layout(
                    graph,
                    int(n_components),
                    rng,
                    laplacian_type=self.laplacian_type,
                    eigen_tol=self.eigen_tol,
                    return_evals=False,
                ),
            )
            spt = np.asarray(spt_result, dtype=np.float32)

            if spt.ndim != 2 or spt.shape[1] != int(n_components):
                raise RuntimeError("spectral_layout returned an invalid shape.")

            scale = float(np.abs(spt).max()) if spt.size else 0.0
            expansion = 10.0 / scale if np.isfinite(scale) and scale > 0 else 1.0
            noise = rng.normal(
                scale=0.0001,
                size=(int(shape[0]), int(n_components)),
            ).astype(np.float32)
            spt = (spt * expansion).astype(np.float32) + noise

        except Exception:
            graph_csr = as_csr_matrix(graph, "spectral layout fallback graph")
            spt = np.asarray(
                EigenDecomposition(n_components=int(n_components)).fit_transform(
                    graph_csr
                ),
                dtype=np.float32,
            )

        self.runtimes["Spectral"] = time.time() - t0
        self.SpecLayout = spt
        gc.collect()
        return spt

    # ------------------------------------------------------------------
    # project()
    # ------------------------------------------------------------------

    def project(
        self,
        n_components: int = 2,
        init=None,
        projection_method: str | None = None,
        landmarks=None,
        landmark_method: str = "kmeans",
        n_neighbors: int | None = None,
        num_iters: int = 300,
        multiscale: bool = False,
        save_every=None,
        save_limit=None,
        save_callback=None,
        include_init_snapshot: bool = True,
        **kwargs,
    ):
        """Compute a projection and store it in ``ProjectionDict``."""
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1.")

        if n_neighbors is None:
            n_neighbors = self.graph_knn
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")

        if projection_method is None:
            if not self.projection_methods:
                raise ValueError("No projection methods configured.")
            projection_method = self.projection_methods[0]
        projection_method = str(projection_method)

        tag = "msDM" if multiscale else "DM"

        input_mat: np.ndarray | csr_matrix

        if projection_method in ("MAP", "IsomorphicMDE", "IsometricMDE", "Isomap"):
            metric = "precomputed"
            input_mat = self._resolve_projection_operator(multiscale)
            key = (
                f"{self.graph_kernel_version} from {tag} "
                f"with {self.base_kernel_version}"
            )
        else:
            metric = self.graph_metric
            if self.uom_enabled:
                uom_input = self.msZ_uom if multiscale else self.Z_uom
                if uom_input is None:
                    raise AttributeError(
                        f"UoM {tag} scaffold unavailable. Call .fit(X, uom=True)."
                    )
                input_mat = uom_input
            else:
                eig_key = f"{tag} with {self.base_kernel_version}"
                if eig_key not in self.EigenbasisDict:
                    raise AttributeError(f"Eigenbasis {eig_key!r} unavailable.")
                input_mat = _as_2d_array(
                    self.EigenbasisDict[eig_key].transform(X=None),
                    eig_key,
                )
            key = f"{tag} with {self.base_kernel_version}"

        input_shape = getattr(input_mat, "shape", None)
        if input_shape is None or len(input_shape) != 2:
            raise ValueError("Projection input must be a 2-D matrix.")

        if init is not None:
            if isinstance(init, np.ndarray):
                init_Y = init
            elif isinstance(init, str) and init in self.ProjectionDict:
                init_Y = self.ProjectionDict[init]
            else:
                raise ValueError(f"Invalid init: {init}")
        else:
            graph = (
                self._kernel_msZ.K
                if (multiscale and self._kernel_msZ is not None)
                else (self._kernel_Z.K if self._kernel_Z is not None else None)
            )
            if graph is None:
                raise ValueError("No refined kernel for spectral initialization.")
            init_Y = self.spectral_layout(graph=graph, n_components=int(n_components))

        init_Y = np.asarray(init_Y)
        if init_Y.ndim != 2 or init_Y.shape[1] != int(n_components):
            raise ValueError(
                f"init must be a 2-D array with shape (n_samples, {int(n_components)})."
            )
        if init_Y.shape[0] != int(input_shape[0]):
            raise ValueError("init and projection input must have the same row count.")

        projection_key = f"{projection_method} of {key}"
        t0 = time.time()

        proj = Projector(
            n_components=int(n_components),
            projection_method=projection_method,
            metric=metric,
            n_neighbors=n_neighbors,
            n_jobs=self._n_jobs_effective,
            landmarks=landmarks,
            landmark_method=landmark_method,
            num_iters=int(num_iters),
            init=cast(Any, init_Y),
            nbrs_backend=self._backend_resolved,
            keep_estimator=False,
            random_state=self._random_state_resolved,
            verbose=self.layout_verbose,
            save_every=save_every,
            save_limit=save_limit,
            save_callback=save_callback,
            include_init_snapshot=include_init_snapshot,
        )

        Y = proj.fit_transform(input_mat, **kwargs)
        Y = np.asarray(Y)

        if Y.ndim != 2 or Y.shape[1] != int(n_components):
            raise RuntimeError(
                f"{projection_method} returned invalid projection shape {Y.shape}."
            )

        Y_aux = getattr(proj, "aux_", None)

        self.runtimes[projection_key] = time.time() - t0
        if self.verbosity >= 1:
            uom_tag = " [UoM]" if self.uom_enabled else ""
            logger.info(
                "  %s (%s%s) in %.3fs",
                projection_method,
                "msZ" if multiscale else "Z/DM",
                uom_tag,
                self.runtimes[projection_key],
            )

        self.ProjectionDict[projection_key] = Y

        if projection_method == "MAP" and Y_aux and isinstance(Y_aux, dict):
            checkpoints = Y_aux.get("checkpoints")
            if checkpoints:
                if multiscale:
                    self.msTopoMAP_snapshots = checkpoints
                else:
                    self.TopoMAP_snapshots = checkpoints
        return Y

    def _resolve_projection_operator(self, multiscale: bool) -> csr_matrix:
        """Resolve the fitted diffusion operator used as precomputed projection input."""
        if self.uom_enabled:
            P = self.P_of_msZ_uom if multiscale else self.P_of_Z_uom
            if P is None:
                tag = "msDM" if multiscale else "DM"
                raise AttributeError(
                    f"UoM {tag} diffusion operator unavailable. Call .fit(X, uom=True)."
                )
            return as_csr_matrix(P, "UoM projection operator")

        if multiscale:
            if self._kernel_msZ is None:
                raise AttributeError("P_of_msZ unavailable. Call .fit() first.")
            return as_csr_matrix(self._kernel_msZ.P, "P_of_msZ")

        if self._kernel_Z is None:
            raise AttributeError("P_of_Z unavailable. Call .fit() first.")
        return as_csr_matrix(self._kernel_Z.P, "P_of_Z")

    # ------------------------------------------------------------------
    # Eigenspectrum plot
    # ------------------------------------------------------------------

    def eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """Scree plot (calls ``topo.plot.decay_plot``)."""
        from topo._optional import require

        require("matplotlib", purpose="eigenspectrum plotting")
        from topo.plot import decay_plot

        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list:
            mode = getattr(self, "_uom_active_mode", "msDM")
            ev_lists = (
                self.uom_eigenvalues_ms_list
                if mode == "msDM"
                else self.uom_eigenvalues_dm_list
            )
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]
            figs = []
            for j, ev in enumerate(ev_lists):
                figs.append(
                    decay_plot(
                        evals=ev,
                        title=f"Component {j} (n={sizes[j]}) · {mode}",
                        **kwargs,
                    )
                )
            return figs

        eb = (
            self.EigenbasisDict.get(eigenbasis_key)
            if eigenbasis_key
            else self.eigenbasis
        )
        if eb is None:
            raise AttributeError("No eigenbasis available.")
        return decay_plot(evals=eb.eigenvalues, title=eigenbasis_key, **kwargs)

    # ------------------------------------------------------------------
    # find_ideal_projection (grid-search MAP hyperparameters)
    # ------------------------------------------------------------------

    def find_ideal_projection(
        self,
        min_dist_grid=None,
        spread_grid=None,
        initial_alpha_grid=None,
        *,
        multiscale: bool = True,
        num_iters: int = 600,
        save_every: int = 10,
        metric: str = "euclidean",
        n_neighbors: int = 30,
        backend: str | None = None,
        n_jobs: int | None = None,
        times=(1, 2, 4),
        r: int = 32,
        k_for_pf1=None,
        symmetric_hint: bool = True,
        verbosity: int = 1,
    ):
        """Grid-search MAP hyperparameters and select the best projection."""
        from topo.eval.topo_metrics import get_P, topo_preserve_score

        if min_dist_grid is None:
            min_dist_grid = [0.2, 0.6, 1.0]
        if spread_grid is None:
            spread_grid = [0.8, 1.2, 1.6]
        if initial_alpha_grid is None:
            initial_alpha_grid = [0.4, 1.0, 1.6]

        effective_backend = self._backend_resolved if backend is None else str(backend)
        if effective_backend not in {"sklearn", "hnswlib"}:
            raise ValueError("backend must be one of {'sklearn', 'hnswlib'}.")

        effective_n_jobs = self._n_jobs_effective if n_jobs is None else int(n_jobs)
        if effective_n_jobs < -1 or effective_n_jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")

        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1.")

        if self.base_kernel is None:
            raise ValueError("No base kernel available. Call fit() first.")

        PX_ref = self.base_kernel.P
        if not issparse(PX_ref):
            PX_ref = csr_matrix(PX_ref)

        combos = [
            (md, sp_, ia)
            for md in min_dist_grid
            for sp_ in spread_grid
            for ia in initial_alpha_grid
        ]

        best_score = float("-inf")
        best_params: dict[str, float] | None = None
        best_snapshots: list[dict] | None = None
        all_scores = []
        snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"

        for md, sp_, ia in combos:
            if verbosity >= 1:
                logger.info(
                    "[Grid] MAP: min_dist=%s, spread=%s, initial_alpha=%s",
                    md,
                    sp_,
                    ia,
                )

            self.project(
                projection_method="MAP",
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=True,
                min_dist=float(md),
                spread=float(sp_),
                initial_alpha=float(ia),
            )

            snapshots = getattr(self, snap_attr, None) or []
            scores_this = []

            for snap in snapshots:
                Ysnap = snap["embedding"]
                PY = get_P(
                    Ysnap,
                    metric=metric,
                    n_neighbors=int(n_neighbors),
                    backend=effective_backend,
                    n_jobs=effective_n_jobs,
                )
                if not issparse(PY):
                    PY = csr_matrix(PY)

                score, parts = topo_preserve_score(
                    PX_ref,
                    PY,
                    times=times,
                    r=r,
                    symmetric_hint=symmetric_hint,
                    k_for_pf1=k_for_pf1,
                )

                snap["metrics"] = {
                    "TP": float(score),
                    "PF1": float(parts.get("PF1", np.nan)),
                    "PJS": float(parts.get("PJS", np.nan)),
                    "SP": float(parts.get("SP", np.nan)),
                }
                snap["hyperparams"] = {
                    "min_dist": float(md),
                    "spread": float(sp_),
                    "initial_alpha": float(ia),
                }
                scores_this.append(float(score))

            final_score = scores_this[-1] if scores_this else float("-inf")
            all_scores.append(
                {
                    "min_dist": float(md),
                    "spread": float(sp_),
                    "initial_alpha": float(ia),
                    "final_score": final_score,
                }
            )

            if final_score > best_score:
                best_score = final_score
                best_params = {
                    "min_dist": float(md),
                    "spread": float(sp_),
                    "initial_alpha": float(ia),
                }
                best_snapshots = [dict(s) for s in snapshots]

        if best_params is not None:
            self.project(
                projection_method="MAP",
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=True,
                min_dist=best_params["min_dist"],
                spread=best_params["spread"],
                initial_alpha=best_params["initial_alpha"],
            )

        if best_snapshots is not None:
            setattr(self, snap_attr, best_snapshots)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "scores": all_scores,
            "best_snapshots": best_snapshots,
        }

    # ------------------------------------------------------------------
    # Visualization (delegates to topo.plot)
    # ------------------------------------------------------------------

    def visualize_optimization(
        self,
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
        """Produce an animated GIF of MAP training snapshots."""
        from topo.plot import visualize_optimization as _viz

        snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
        snapshots = getattr(self, snap_attr, None)

        if not snapshots or len(snapshots) < 2:
            self.project(
                projection_method="MAP",
                num_iters=max(int(num_iters), int(save_every)),
                save_every=int(save_every),
                include_init_snapshot=bool(include_init_snapshot),
                multiscale=bool(multiscale),
            )
            snapshots = getattr(self, snap_attr, None)

        if not snapshots:
            raise RuntimeError("No snapshots available.")

        tag = "msTopoMAP" if multiscale else "TopoMAP"
        path = _viz(
            snapshots,
            dpi=dpi,
            color=color,
            filename=filename,
            point_size=point_size,
            fps=fps,
            tag=tag,
            overlay_metrics=overlay_metrics,
        )

        if self.verbosity >= 1:
            logger.info("Wrote %s with %d frames.", path, len(snapshots))

        return path
