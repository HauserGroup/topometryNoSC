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
from scipy.sparse import csr_matrix

from topo.layouts.projector import Projector
from topo.spectral import LE, EigenDecomposition

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
    ProjectionDict: dict[str, np.ndarray]

    Z_: np.ndarray | csr_matrix | None
    msZ_: np.ndarray | csr_matrix | None
    P_Z_: csr_matrix | None
    P_msZ_: csr_matrix | None
    K_Z_: csr_matrix | None
    K_msZ_: csr_matrix | None

    laplacian_type: str
    eigen_tol: float
    runtimes: dict[str, float]
    SpecLayout: np.ndarray | None

    graph_knn: int
    graph_metric: str
    _n_jobs_effective: int
    _backend_resolved: str
    _random_state_resolved: np.random.RandomState

    layout_verbose: bool
    verbosity: int
    msTopoMAP_snapshots: list[dict[str, Any]]
    TopoMAP_snapshots: list[dict[str, Any]]

    uom_components_: list[np.ndarray] | None
    _uom_active_mode: str
    uom_enabled: bool
    eigenbasis: EigenDecomposition | None

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
        tag = "msDM" if multiscale else "DM"
        key = f"{str(method)} of {tag}"

        if key not in self.ProjectionDict:
            raise AttributeError(
                f"{method} ({tag}) embedding unavailable. Call .fit() or .project() first."
            )

        return self.ProjectionDict[key]

    # ------------------------------------------------------------------
    # Spectral layout
    # ------------------------------------------------------------------

    def _resolve_refined_kernel_graph(self, multiscale: bool) -> csr_matrix:
        """Return the fitted scaffold affinity used for spectral initialization."""
        K = self.K_msZ_ if multiscale else self.K_Z_
        if K is None:
            tag = "msDM" if multiscale else "DM"
            raise AttributeError(
                f"{tag} refined affinity unavailable. Call .fit() first."
            )
        return K

    def spectral_layout(
        self,
        graph=None,
        n_components: int = 2,
        *,
        multiscale: bool = True,
    ):
        """Compute a spectral initialization for layout optimization."""
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1.")

        if graph is None:
            graph = self._resolve_refined_kernel_graph(multiscale)

        shape = getattr(graph, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("graph must be a 2-D square matrix.")
        if int(shape[0]) != int(shape[1]):
            raise ValueError("graph must be square.")

        t0 = time.time()
        rng = self._random_state_resolved

        try:
            spt_result = LE(
                graph,
                n_eigs=int(n_components),
                laplacian_type=self.laplacian_type,
                drop_first=True,
                return_evals=False,
                eigen_tol=self.eigen_tol,
                random_state=rng,
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
            spt = rng.uniform(
                low=-10.0,
                high=10.0,
                size=(int(shape[0]), n_components),
            ).astype(np.float32)

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
        else:
            metric = self.graph_metric
            scaffold = self.msZ_ if multiscale else self.Z_
            if scaffold is None:
                raise AttributeError(f"{tag} scaffold unavailable. Call .fit() first.")
            input_mat = _as_2d_array(scaffold, tag)

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
            graph = self._resolve_refined_kernel_graph(multiscale)
            init_Y = self.spectral_layout(
                graph=graph,
                n_components=int(n_components),
                multiscale=multiscale,
            )

        init_Y = np.asarray(init_Y)
        if init_Y.ndim != 2 or init_Y.shape[1] != int(n_components):
            raise ValueError(
                f"init must be a 2-D array with shape (n_samples, {int(n_components)})."
            )
        if init_Y.shape[0] != int(input_shape[0]):
            raise ValueError("init and projection input must have the same row count.")

        projection_key = f"{projection_method} of {tag}"
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
            logger.info(
                "  %s (%s) in %.3fs",
                projection_method,
                "msZ" if multiscale else "Z/DM",
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
        """Return the fitted scaffold diffusion operator for projection."""
        P = self.P_msZ_ if multiscale else self.P_Z_
        if P is None:
            tag = "msDM" if multiscale else "DM"
            raise AttributeError(
                f"{tag} diffusion operator unavailable. Call .fit() first."
            )
        return P
