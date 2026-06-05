# Step by step: from a table to a map

This walkthrough takes a plain table of numbers and turns it into a 2-D **map**
where similar items sit close together. It then shows how to read that map and a
few useful things you can do with it.

Everything here works on **any** table. A natural example is single-cell data —
one row per cell, one column per gene — but the steps are identical whether your
rows are cells, images, documents or anything else.

!!! tip "No data wrangling here"
    To keep the tutorial focused, all data loading lives in a small helper,
    [`data.py`](data.py). It returns a ready-to-use table, so we never touch
    cleaning or normalising in the tutorial itself. The recommended pattern is
    to prepare your data **once**, save it, host it, and load it in one line —
    see [Using your own data](#using-your-own-data) at the end.

## Setup

Install the package with plotting support:

```bash
pip install "topometry-nosc[plot]"
```

Then import what we need:

```python
import topo as tp
from data import load_cells   # the helper described above
```

## 1. Get the data

```python
X, labels = load_cells()

print(X.shape)        # (n_rows, n_columns) e.g. (1797, 64)
print(set(labels))    # the groups, used only for colouring later
```

`X` is the table the model learns from: one row per item, one column per
measurement. `labels` are just group names we use to colour the plots — the
model never sees them.

## 2. Build the map

One object does the whole pipeline. Create it and call `fit`:

```python
tg = tp.TopOGraph()
tg.fit(X)

print(tg)
```

In plain terms, `fit` does three things:

1. finds, for each row, the handful of rows most similar to it;
2. uses those similarities to work out the data's underlying shape;
3. gets everything ready to draw a 2-D map.

You don't have to tune anything to start — the defaults are sensible.

## 3. Look at the map

Two ready-made 2-D maps are available after fitting. Draw one and colour it by
group:

```python
tp.plot.scatter(tg.TopoMAP, labels=labels)
```

There is also a "multiscale" version that often separates groups a little more
cleanly:

```python
tp.plot.scatter(tg.msTopoMAP, labels=labels)
```

If similar items end up near each other and the groups form clear regions, the
map is doing its job.

## 4. How many patterns are really in the data?

Your table might have 64 columns (or 2000 genes), but the real structure often
lives in far fewer underlying patterns. The package estimates this for you:

```python
print(tg.global_id)      # a single number: the data's effective complexity
print(tg.intrinsic_dim)  # more detail, including per-region estimates
```

A small number here means the data, despite having many columns, is shaped by
just a few underlying factors.

## 5. How much structure did it find?

The model summarises structure as a list of values, largest first. Plotting them
("the eigenspectrum") shows how much each successive pattern contributes:

```python
tg.eigenspectrum()
```

Where the curve flattens is a rough hint of how many patterns carry real signal;
the long flat tail is mostly noise.

## 6. Try a different layout

`TopoMAP` is one way to draw the map. You can ask for others from the same
fitted model without recomputing the expensive parts:

```python
other = tg.project(projection_method="PaCMAP")
tp.plot.scatter(other, labels=labels)
```

Different layouts trade off local detail versus the big picture; it's worth
trying a couple and keeping whichever tells the clearest story.

## 7. Smooth out noisy measurements

Real measurements are noisy. Because the model knows which rows are similar, it
can average each value over a row's neighbours to reduce noise — a gentle
"smoothing":

```python
X_smoothed = tg.impute(X, t=4)   # larger t = more smoothing
```

Use a small `t`. Too much smoothing erases real differences between groups.

## 8. Save and reload

Fitting can take a while on big tables, so save the result and reload it later:

```python
tg.save("my_map.pkl")

tg2 = tp.load_topograph("my_map.pkl")
tp.plot.scatter(tg2.TopoMAP, labels=labels)
```

## Using your own data

You do **not** change the tutorial to use your own data — you only change the
helper. The recommended workflow keeps every heavy or domain-specific dependency
out of the package entirely:

1. **Prepare once, on your machine.** Do whatever cleaning/normalising your data
   needs in a one-off script (this is the only place tools like single-cell
   packages live). End with two arrays: `X` (rows × columns, float) and
   `labels` (one integer per row).
2. **Save as one plain file.** numpy's `.npz` is ideal — one compressed file,
   loads with numpy alone:
   ```python
   import numpy as np
   np.savez_compressed("example.npz", X=X, labels=labels)
   ```
3. **Host it.** Upload `example.npz` as a GitHub Release asset (free, versioned,
   lives with the repo). Copy its download URL.
4. **Point the helper at it.** Set `DATA_URL` in [`data.py`](data.py) to that
   URL. `load_cells` downloads it once, caches it locally, and returns
   `(X, labels)` — no single-cell packages, no preprocessing for anyone who runs
   the tutorial.

That's the whole point: the tutorial, and your users, only ever see a ready-made
table.
