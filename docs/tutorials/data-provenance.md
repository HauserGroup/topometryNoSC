# Example data: provenance

How the tutorial's `example.npz` was made. A **one-time, offline** step — single-cell
tools live here only, never in the package or the tutorial.

**Source.** PBMC68k (10x Genomics) — ~68,000 human blood cells.

**Preprocessing** (scanpy; mirrors the upstream
[TopOMetry step-by-step](https://topometry.readthedocs.io/en/latest/T2_step_by_step.html)):

| # | Step | Call |
|---|------|------|
| 1 | QC filter — drop cells <200 genes, genes in <3 cells, high mitochondrial % | `sc.pp.filter_cells/filter_genes`, `pct_counts_mt` |
| 2 | Normalize counts per cell → 10⁴ | `sc.pp.normalize_total(target_sum=1e4)` |
| 3 | Log-transform | `sc.pp.log1p` |
| 4 | Keep top 2000 highly-variable genes | `sc.pp.highly_variable_genes` |
| 5 | Z-score per gene, clip at 10 | `sc.pp.scale(max_value=10)` |
| 6 | Cell-type labels | marker-based classifier → `obs['predicted_celltype']` |

Result in memory: `AnnData` 68265 × 2000 (scaled), with `predicted_celltype`.

**Export** to the package's `(X, labels)` contract, sub-sampled for a fast download:

```python
import numpy as np
from scipy.sparse import issparse

# scaled matrix -> dense float32
X = adata.X
X = X.toarray() if issparse(X) else np.asarray(X)
X = X.astype(np.float32)

# cell type -> integer ids (+ readable names)
cats = adata.obs["predicted_celltype"].astype("category")
labels = cats.cat.codes.to_numpy().astype(int)
label_names = np.asarray(cats.cat.categories)        # id -> name

# stratified sub-sample to ~10k cells (keeps every cell type, in proportion)
rng = np.random.default_rng(0)
frac = 10_000 / X.shape[0]
idx = np.concatenate([
    rng.choice(
        np.flatnonzero(labels == c),
        max(1, round((labels == c).sum() * frac)),
        replace=False,
    )
    for c in np.unique(labels)
])
rng.shuffle(idx)
X, labels = X[idx], labels[idx]

np.savez_compressed("example.npz", X=X, labels=labels, label_names=label_names)
```

**Output.** `example.npz` — `X` (cells × 2000, float32), `labels` (int cell-type id),
`label_names` (id → cell type). Upload as a GitHub Release asset; the tutorial's
[`data.py`](data.py) downloads it. Nothing single-cell is needed downstream.

```bash
gh release create data-v1 example.npz --title "Tutorial data" \
  --notes "Preprocessed, sub-sampled PBMC68k for the step-by-step tutorial"
```
