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

import urllib.request
from pathlib import Path
from shutil import copyfileobj
from typing import Literal

import numpy as np

# Hosted, already-preprocessed dataset (numpy ``.npz`` with arrays ``X`` and
# ``labels``). Replace this with your own uploaded file, e.g. a GitHub Release
# asset:
#   https://github.com/HauserGroup/topometryNoSC/releases/download/data-v1/example.npz
DATA_URL = (
    "https://github.com/HauserGroup/topometryNoSC/releases/download/data-v1/example.npz"
)

_CACHE = Path.home() / ".cache" / "topometry-nosc"


def load_cells(
    source: Literal["auto", "hosted", "builtin"] = "auto",
    return_names: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the example dataset as ``(X, labels)``.

    Parameters
    ----------
    source : {"auto", "hosted", "builtin"}, default "auto"
        Where to get the data:

        * ``"auto"`` — try the hosted ``.npz`` (downloaded once and cached); if
          it is not reachable, fall back to the built-in dataset so the tutorial
          always runs.
        * ``"hosted"`` — use only the hosted ``.npz``; raise if it cannot be
          fetched.
        * ``"builtin"`` — use only scikit-learn's handwritten digits. No
          download, fully offline; pick this if you prefer not to fetch anything.
    return_names : bool, default False
        If ``True``, also return ``label_names``: a 1-D array mapping each
        integer label id to its readable name (e.g. a cell type). For the hosted
        dataset these come from the file; for the built-in one they are just the
        digit strings.

    Returns
    -------
    (X, labels, label_names)
        Always returns a 3-tuple. ``label_names`` is a 1-D array when
        ``return_names=True`` and ``None`` otherwise.
    """
    if source not in ("auto", "hosted", "builtin"):
        raise ValueError(
            f"source must be 'auto', 'hosted' or 'builtin', got {source!r}."
        )

    if source == "builtin":
        X, labels, names = _load_builtin()
    else:
        try:
            path = _download_cached(DATA_URL)
            with np.load(path, allow_pickle=False) as npz:
                X = npz["X"].astype(np.float32)
                labels = npz["labels"].astype(int)
                names = (
                    npz["label_names"]
                    if "label_names" in npz.files
                    else np.array([str(i) for i in np.unique(labels)])
                )
        except Exception:
            if source == "hosted":
                raise
            X, labels, names = _load_builtin()

    return X, labels, (names if return_names else None)


def _load_builtin() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Built-in offline dataset: scikit-learn's handwritten digits.

    1797 rows x 64 measurements, 10 groups. Ships with scikit-learn, so no
    download and no extra packages.
    """
    from sklearn.datasets import load_digits

    raw_X, raw_labels = load_digits(return_X_y=True)
    X = np.asarray(raw_X, dtype=np.float32)
    labels = np.asarray(raw_labels, dtype=int)
    names = np.array([str(i) for i in range(10)])
    return X, labels, names


def _download_cached(url: str, timeout: float = 30.0) -> Path:
    """Download ``url`` into the local cache once; return the cached path."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310
                tmp_path.open("wb") as out,
            ):
                copyfileobj(response, out)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return path
