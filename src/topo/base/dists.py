"""Distance gradients used by layout optimization."""

import importlib.util

import numpy as np

_have_numba = importlib.util.find_spec("numba") is not None

if _have_numba:
    from numba import njit  # type: ignore[reportMissingImports]
else:

    def njit(*_args, **_kwargs):  # noqa: D103
        def _decorator(func):
            return func

        return _decorator


@njit(fastmath=True)
def euclidean_grad(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Euclidean distance and gradient with respect to x."""
    diff = x - y
    dist = np.sqrt(np.dot(diff, diff))
    grad = diff / (1e-8 + dist)
    return dist, grad
