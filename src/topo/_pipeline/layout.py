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
from sklearn.utils import check_random_state

from topo.layouts.projector import Projector
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.tpgraph.kernels import Kernel

logger = logging.getLogger(__name__)


class LayoutBuildMixin:
    """Projection / layout construction and visualisation methods."""

    # Interface contract — attributes supplied by TopOGraph
    projection_methods: list[str] | None
    graph_kernel_version: str
    base_kernel_version: str
    ProjectionDict: dict[str, np.ndarray]
    _kernel_msZ: Kernel | None
    _kernel_Z: Kernel | None
    random_state: int | np.random.RandomState | None
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
    backend: str
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

    @property
    def P_of_msZ(self) -> csr_matrix:
        P = getattr(self, "_P_of_msZ_mixin", None)
        if P is None:
            raise AttributeError("P_of_msZ unavailable. Call .fit() first.")
        return P if isinstance(P, csr_matrix) else csr_matrix(P)

    @P_of_msZ.setter
    def P_of_msZ(self, value) -> None:
        self._P_of_msZ_mixin = value

    @property
    def P_of_Z(self) -> csr_matrix:
        P = getattr(self, "_P_of_Z_mixin", None)
        if P is None:
            raise AttributeError("P_of_Z unavailable. Call .fit() first.")
        return P if isinstance(P, csr_matrix) else csr_matrix(P)

    @P_of_Z.setter
    def P_of_Z(self, value) -> None:
        self._P_of_Z_mixin = value

    def _run_projections(self):
        """Compute requested 2-D projections on both scaffolds."""
        if self.projection_methods is None:
            return
        for proj in self.projection_methods:
            for ms in (True, False):
                try:
                    self.project(projection_method=proj, multiscale=ms)
                except Exception as e:
                    tag = "msZ" if ms else "Z/DM"
                    warnings.warn(
                        f"Projection '{proj}' on {tag} failed: {e}", RuntimeWarning
                    )

    def _get_projection(self, method, multiscale):
        """Look up a projection from ProjectionDict."""
        tag = "msDM" if multiscale else "DM"
        # Standard key
        if method in ("MAP", "Isomap", "IsomorphicMDE", "IsometricMDE"):
            key = f"{method} of {self.graph_kernel_version} from {tag} with {self.base_kernel_version}"
        else:
            key = f"{method} of {tag} with {self.base_kernel_version}"
        if key in self.ProjectionDict:
            return self.ProjectionDict[key]
        # UoM fallback key
        uom_key = f"{method} of UoM {tag} with {self.base_kernel_version}"
        if uom_key in self.ProjectionDict:
            return self.ProjectionDict[uom_key]
        raise AttributeError(
            f"{method} ({tag}) embedding unavailable. Call .fit() first."
        )

    # ------------------------------------------------------------------
    # Spectral layout
    # ------------------------------------------------------------------

    def spectral_layout(self, graph=None, n_components=2):
        """Compute a spectral initialization for layout optimization."""
        if graph is None:
            if self._kernel_msZ is not None:
                graph = self._kernel_msZ.K
            elif self._kernel_Z is not None:
                graph = self._kernel_Z.K
            else:
                raise ValueError("No graph kernel available. Call .fit() first.")
        t0 = time.time()
        rng = check_random_state(self.random_state)
        try:
            spt_result = cast(
                Any,
                spectral_layout(
                    graph,
                    n_components,
                    rng,
                    laplacian_type=self.laplacian_type,
                    eigen_tol=self.eigen_tol,
                    return_evals=False,
                ),
            )
            spt = np.asarray(spt_result)
            scale = float(np.abs(spt).max())
            expansion = 10.0 / scale if np.isfinite(scale) and scale > 0 else 1.0
            spt = (spt * expansion).astype(np.float32) + rng.normal(
                scale=0.0001, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        except Exception:
            spt = np.asarray(
                EigenDecomposition(n_components=n_components).fit_transform(graph)
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
        """Compute a 2-D projection and store it in ``ProjectionDict``."""
        if n_neighbors is None:
            n_neighbors = self.graph_knn
        if projection_method is None:
            if not self.projection_methods:
                raise ValueError("No projection methods configured.")
            projection_method = self.projection_methods[0]
        projection_method = str(projection_method)

        # Choose refined graph / scaffold
        tag = "msDM" if multiscale else "DM"
        if projection_method in ("MAP", "IsomorphicMDE", "IsometricMDE", "Isomap"):
            metric = "precomputed"
            input_mat = self.P_of_msZ if multiscale else self.P_of_Z
            key = f"{self.graph_kernel_version} from {tag} with {self.base_kernel_version}"
        else:
            metric = self.graph_metric
            if self.uom_enabled:
                input_mat = self.msZ_uom if multiscale else self.Z_uom
            else:
                eig_key = f"{tag} with {self.base_kernel_version}"
                input_mat = self.EigenbasisDict[eig_key].transform(X=None)
            key = f"{tag} with {self.base_kernel_version}"

        # Initialization
        if init is not None:
            if isinstance(init, np.ndarray):
                init_Y = init
            elif isinstance(init, str) and init in self.ProjectionDict:
                init_Y = self.ProjectionDict[init]
            else:
                raise ValueError(f"Invalid init: {init}")
        else:
            g = (
                self._kernel_msZ.K
                if (multiscale and self._kernel_msZ is not None)
                else (self._kernel_Z.K if self._kernel_Z is not None else None)
            )
            if g is None:
                raise ValueError("No refined kernel for spectral initialization.")
            self.SpecLayout = self.spectral_layout(graph=g, n_components=n_components)
            init_Y = self.SpecLayout

        projection_key = f"{projection_method} of {key}"
        t0 = time.time()

        proj = Projector(
            n_components=n_components,
            projection_method=projection_method,
            metric=metric,
            n_neighbors=self.graph_knn,
            n_jobs=self.n_jobs,
            landmarks=landmarks,
            landmark_method=landmark_method,
            num_iters=num_iters,
            init=cast(Any, init_Y),
            nbrs_backend=self.backend,
            keep_estimator=False,
            random_state=self.random_state,
            verbose=self.layout_verbose,
            save_every=save_every,
            save_limit=save_limit,
            save_callback=save_callback,
            include_init_snapshot=include_init_snapshot,
        )

        Y = proj.fit_transform(input_mat, **kwargs)
        Y_aux = getattr(proj, "aux_", None)

        self.runtimes[projection_key] = time.time() - t0
        if self.verbosity >= 1:
            uom_tag = " [UoM]" if self.uom_enabled else ""
            logger.info(
                f"  {projection_method} ({'msZ' if multiscale else 'Z/DM'}{uom_tag}) in {self.runtimes[projection_key]:.3f}s"
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
        backend: str = "hnswlib",
        n_jobs: int = -1,
        times=(1, 2, 4),
        r: int = 32,
        k_for_pf1=None,
        symmetric_hint: bool = True,
        verbosity: int = 1,
    ):
        """Grid-search MAP hyperparameters and select the best 2-D projection."""
        from topo.eval.topo_metrics import get_P, topo_preserve_score

        if min_dist_grid is None:
            min_dist_grid = [0.2, 0.6, 1.0]
        if spread_grid is None:
            spread_grid = [0.8, 1.2, 1.6]
        if initial_alpha_grid is None:
            initial_alpha_grid = [0.4, 1.0, 1.6]

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
                    f"[Grid] MAP: min_dist={md}, spread={sp_}, initial_alpha={ia}"
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
                    n_neighbors=n_neighbors,
                    backend=backend,
                    n_jobs=n_jobs,
                )
                if not issparse(PY):
                    PY = csr_matrix(PY)
                score, parts = topo_preserve_score(
                    PX_ref,
                    PY,
                    times=times,
                    r=r,
                    symmetric_hint=symmetric_hint,
                    k_for_pf1=k_for_pf1,  # type: ignore
                )
                snap["metrics"] = {
                    k: float(v)
                    for k, v in [
                        ("TP", score),
                        ("PF1", parts.get("PF1", np.nan)),
                        ("PJS", parts.get("PJS", np.nan)),
                        ("SP", parts.get("SP", np.nan)),
                    ]
                }
                snap["hyperparams"] = {
                    "min_dist": md,
                    "spread": sp_,
                    "initial_alpha": ia,
                }
                scores_this.append(float(score))

            final_score = scores_this[-1] if scores_this else float("-inf")
            all_scores.append(
                {
                    "min_dist": md,
                    "spread": sp_,
                    "initial_alpha": ia,
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

        if best_snapshots is not None:
            setattr(self, snap_attr, best_snapshots)

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

        if multiscale is None:
            multiscale = bool(self.msTopoMAP_snapshots) or not bool(
                self.TopoMAP_snapshots
            )

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
            logger.info(f"Wrote {path} with {len(snapshots)} frames.")
        return path
