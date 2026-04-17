from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "global_score_pca",
    "global_score_laplacian",
    "knn_spearman_r",
    "knn_kendall_tau",
    "geodesic_distance",
    "geodesic_correlation",
    "RiemannMetric",
    "get_eccentricity",
]

_EXPORTS = {
    "global_score_pca": (".global_scores", "global_score_pca"),
    "global_score_laplacian": (".global_scores", "global_score_laplacian"),
    "knn_spearman_r": (".local_scores", "knn_spearman_r"),
    "knn_kendall_tau": (".local_scores", "knn_kendall_tau"),
    "geodesic_distance": (".local_scores", "geodesic_distance"),
    "geodesic_correlation": (".local_scores", "geodesic_correlation"),
    "RiemannMetric": (".rmetric", "RiemannMetric"),
    "get_eccentricity": (".rmetric", "get_eccentricity"),
}

if TYPE_CHECKING:
    from .global_scores import global_score_laplacian as global_score_laplacian
    from .global_scores import global_score_pca as global_score_pca
    from .local_scores import geodesic_correlation as geodesic_correlation
    from .local_scores import geodesic_distance as geodesic_distance
    from .local_scores import knn_kendall_tau as knn_kendall_tau
    from .local_scores import knn_spearman_r as knn_spearman_r
    from .rmetric import RiemannMetric as RiemannMetric
    from .rmetric import get_eccentricity as get_eccentricity


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
