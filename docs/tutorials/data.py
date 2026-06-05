"""Data loader for the step-by-step tutorial.

All data loading lives here, on purpose, so the tutorial itself stays focused on
topometry-nosc instead of data wrangling.

The idea: preprocess your dataset **once**, on your own machine, save the result
as a plain ``.npz`` file, and host that file somewhere (a GitHub Release asset
works well). Users then only ever download a ready-made array — no single-cell
packages, no preprocessing, nothing to install beyond numpy.

The only contract the tutorial relies on is the return value of ``load_cells``:

    X      : 2-D float array. One row per item (for example, one cell), one
             column per measurement (for example, one gene).
    labels : 1-D integer array, one group id per row. Used only to colour the
             plots; it is never given to the model.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

# Hosted, already-preprocessed dataset (numpy ``.npz`` with arrays ``X`` and
# ``labels``). Replace this with your own uploaded file, e.g. a GitHub Release
# asset:
#   https://github.com/HauserGroup/topometryNoSC/releases/download/data-v1/example.npz
DATA_URL = (
    "https://github.com/HauserGroup/topometryNoSC/releases/download/data-v1/example.npz"
)

_CACHE = Path.home() / ".cache" / "topometry-nosc"


def load_cells() -> tuple[np.ndarray, np.ndarray]:
    """Return the example dataset as ``(X, labels)``.

    Downloads the hosted ``.npz`` once and caches it locally. If the file is not
    reachable (for example before it has been uploaded), falls back to a small
    dataset that ships with scikit-learn so the tutorial always runs.
    """
    try:
        path = _download_cached(DATA_URL)
        with np.load(path) as npz:
            return npz["X"].astype(np.float32), npz["labels"].astype(int)
    except Exception:
        # Offline fallback: scikit-learn's handwritten digits (1797 x 64, 10
        # groups). Ships with scikit-learn, so no download, no extra packages.
        from sklearn.datasets import load_digits

        data = load_digits()
        return data.data.astype(np.float32), data.target.astype(int)


def _download_cached(url: str) -> Path:
    """Download ``url`` into the local cache once; return the cached path."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        urllib.request.urlretrieve(url, path)  # noqa: S310 (trusted release URL)
    return path
