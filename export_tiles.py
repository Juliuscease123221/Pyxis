"""Adapter: Overworld data -> the generic point/manifest schema `tiles` consumes.

This script lives **outside** the tiles package on purpose. `tiles` ships as
its own project and must not know what a game is, so everything Steam-specific
happens here: reading the catalog, the cluster tree and the layout, and
flattening them into five anonymous columns plus a category manifest.

Anything in this file is Overworld's problem. Anything in `tiles/` would have to
work unchanged for a different dataset.

Usage:
    python export_tiles.py --out data/tiles_input
    python -m tiles build data/tiles_input/points.parquet data/tiles_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embed.common import load_games

ROOT = Path(__file__).resolve().parent
CLUSTER = ROOT / "data" / "cluster"
LAYOUT = ROOT / "data" / "layout"


def leaf_owner(nodes: dict) -> dict[int, int]:
    """game row index -> the deepest node that directly holds it."""
    owner: dict[int, int] = {}
    for nid, rec in nodes.items():
        for m in rec["members"]:
            owner[int(m)] = int(nid)
    return owner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="tree_rleiden_labelled")
    ap.add_argument("--coords", default="coords.npz")
    ap.add_argument("--out", default="data/tiles_input")
    args = ap.parse_args()

    g = load_games()
    tree = json.loads((CLUSTER / f"{args.tree}.json").read_text(encoding="utf-8"))
    nodes = tree["nodes"]

    d = np.load(LAYOUT / args.coords, allow_pickle=False)
    Y = d["Y"].astype(np.float32)
    if not np.array_equal(d["appids"], g.appids):
        print("appid order mismatch between layout and catalog")
        return 1

    cons_path = LAYOUT / "consistency.json"
    cons = (json.loads(cons_path.read_text(encoding="utf-8"))["per_node"]
            if cons_path.exists() else {})

    owner = leaf_owner(nodes)
    missing = [i for i in range(len(g)) if i not in owner]
    if missing:
        print(f"{len(missing):,} games have no cluster; aborting")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- points: five anonymous columns ----------------------------------
    category = np.asarray([owner[i] for i in range(len(g))], dtype=np.uint32)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "x": Y[:, 0],
        "y": Y[:, 1],
        "id": g.appids.astype(np.uint32),
        "category": category,
        "importance": g.review_count.astype(np.float64),
    })
    pq.write_table(table, out / "points.parquet")
    print(f"points.parquet: {len(g):,} rows, columns {table.column_names}")

    # ---- manifest --------------------------------------------------------
    cats = []
    for nid, rec in nodes.items():
        c = cons.get(str(nid), {})
        bbox = c.get("bbox")
        centroid = c.get("centroid")
        if bbox is None or centroid is None:
            idx = np.asarray(rec["members"], dtype=np.int64)
            if len(idx) == 0:
                idx = np.asarray([0], dtype=np.int64)
            pts = Y[idx]
            bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())]
            centroid = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        cats.append({
            "id": int(nid),
            "parent": None if rec["parent"] is None else int(rec["parent"]),
            "label": rec.get("label", ""),
            "bbox": [float(v) for v in bbox],
            "centroid": [float(v) for v in centroid],
            "size": int(rec.get("size", 0)),
            "depth": int(rec.get("depth", 0)),
            "confidence": rec.get("confidence"),
            "purity": c.get("purity"),
            "containment": c.get("containment"),
        })

    from tiles.manifest import write as write_manifest

    info = write_manifest(cats, out / "manifest.json", strict=True)
    print(f"manifest.json: {info['n']:,} categories, "
          f"{info['bytes'] / 1e6:.2f} MB, {len(info['problems'])} problems")

    have = lambda k: sum(1 for c in cats if c.get(k) is not None)  # noqa: E731
    print(f"  with confidence  {have('confidence'):,}")
    print(f"  with purity      {have('purity'):,}")
    print(f"  with containment {have('containment'):,}")
    print(f"\nnext: python -m tiles build {out / 'points.parquet'} data/tiles_out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
