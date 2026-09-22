"""Reading a point set from disk, without knowing what the points are.

The library's contract is five columns::

    x, y, id, category, importance

Anything else in the file is ignored. Column names are configurable so the
caller does not have to rename their data, and the defaults are generic --
`category`, not `cluster_id`; `importance`, not any particular
measure of it.

Supported: parquet, csv, npz. Parquet is the documented entry point because it
is what the acceptance check uses, but nothing depends on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .format import PointSet

DEFAULT_COLUMNS = {
    "x": "x",
    "y": "y",
    "id": "id",
    "category": "category",
    "importance": "importance",
}


def _normalise_importance(v: np.ndarray) -> np.ndarray:
    """Map an arbitrary positive scalar into uint16, preserving order.

    Importance is opaque: it may be a count in the millions or a float in
    0..1. Ranks are what selection actually uses, so the values are rank-
    transformed into 0..65535. That keeps every distinction the ordering makes
    while fitting the format, and makes the field comparable across datasets.
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return np.zeros(0, dtype=np.uint16)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    if len(v) == 1:
        return np.asarray([65535], dtype=np.uint16)
    return np.round(ranks / (len(v) - 1) * 65535).astype(np.uint16)


def load_points(path: str | Path, columns: dict[str, str] | None = None,
                rank_importance: bool = True) -> PointSet:
    path = Path(path)
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    suffix = path.suffix.lower()

    if suffix == ".npz":
        d = np.load(path, allow_pickle=False)
        table = {k: d[k] for k in d.files}
    elif suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        t = pq.read_table(path)
        table = {name: t.column(name).to_numpy(zero_copy_only=False)
                 for name in t.column_names}
    elif suffix in (".csv", ".tsv"):
        import csv

        sep = "\t" if suffix == ".tsv" else ","
        rows: dict[str, list] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=sep):
                for k, v in row.items():
                    rows.setdefault(k, []).append(v)
        table = {k: np.asarray(v) for k, v in rows.items()}
    else:
        raise ValueError(f"unsupported input {suffix!r}; use parquet, csv or npz")

    missing = [src for src in cols.values() if src not in table]
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s) {missing}. "
            f"Present: {sorted(table)[:12]}. "
            f"Use --columns to map your names onto x,y,id,category,importance."
        )

    imp_raw = np.asarray(table[cols["importance"]], dtype=np.float64)
    return PointSet(
        x=np.asarray(table[cols["x"]], dtype=np.float32),
        y=np.asarray(table[cols["y"]], dtype=np.float32),
        ids=np.asarray(table[cols["id"]], dtype=np.uint32),
        category=np.asarray(table[cols["category"]], dtype=np.uint32),
        importance=(_normalise_importance(imp_raw) if rank_importance
                    else np.asarray(imp_raw, dtype=np.uint16)),
    )
