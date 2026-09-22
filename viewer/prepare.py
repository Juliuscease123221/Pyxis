"""Stage viewer assets: one blob of every point, the manifest, and neighbours.

Step one of Phase 6 is getting points on screen and *looking* at them. Tile
streaming is the right architecture and is what `tiles/` exists for, but it is
not the shortest path to a first picture, and a first picture is what tells you
whether anything upstream is wrong in a way the metrics missed.

So this emits:

  all.bin        every point in the tiles binary format, one blob
  manifest.json  the category tree (parent, label, confidence, bbox, ...)
  meta.json      names for tooltips, keyed by point index
  knn.bin        each game's true high-dimensional nearest neighbours

`knn.bin` is the one that is not an expedient. Screen proximity is not
similarity here -- only 10-15% of a game's real neighbours are among its
nearest on screen -- so hover must read from the embedding, never from the
layout. Precomputing k=10 per game costs 56,129 x 10 x 4 bytes = 2.2 MB and
removes any temptation to query by distance in the viewer.

Usage:
    python -m viewer.prepare
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors
from tiles.format import PointSet, encode

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "viewer" / "public"
CLUSTER = ROOT / "data" / "cluster"
LAYOUT = ROOT / "data" / "layout"


def knn_lists(X: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """True k nearest neighbours in the embedding, by cosine."""
    n = len(X)
    out = np.empty((n, k), dtype=np.uint32)
    for s in range(0, n, block):
        sims = X[s:s + block] @ X.T
        for r in range(len(sims)):
            sims[r, s + r] = -np.inf
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), axis=1)
        out[s:s + len(sims)] = ordered.astype(np.uint32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="tree_rleiden_labelled")
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    PUBLIC.mkdir(parents=True, exist_ok=True)
    g = load_games()
    tree = json.loads((CLUSTER / f"{args.tree}.json").read_text(encoding="utf-8"))
    nodes = tree["nodes"]

    d = np.load(LAYOUT / "coords.npz", allow_pickle=False)
    Y = d["Y"].astype(np.float32)

    owner: dict[int, int] = {}
    for nid, rec in nodes.items():
        for m in rec["members"]:
            owner[int(m)] = int(nid)

    cat = np.asarray([owner[i] for i in range(len(g))], dtype=np.uint32)
    rev = g.review_count.astype(np.float64)
    order = np.argsort(rev, kind="stable")
    ranks = np.empty(len(rev))
    ranks[order] = np.arange(len(rev))
    imp = np.round(ranks / max(len(rev) - 1, 1) * 65535).astype(np.uint16)

    ps = PointSet(Y[:, 0].copy(), Y[:, 1].copy(),
                  np.arange(len(g), dtype=np.uint32), cat, imp)
    (PUBLIC / "all.bin").write_bytes(encode(ps))
    print(f"all.bin        {len(g):,} points, "
          f"{(PUBLIC / 'all.bin').stat().st_size / 1e6:.2f} MB")

    # manifest, with 2D geometry attached
    cons_path = LAYOUT / "consistency.json"
    cons = (json.loads(cons_path.read_text(encoding="utf-8"))["per_node"]
            if cons_path.exists() else {})
    def subtree_members(nid: str) -> np.ndarray:
        """Every point under `nid`. An internal node usually holds none itself.

        The first version of this fell back to point 0 when `members` was
        empty, which is true of the root and of every internal Leiden node.
        That produced a zero-area bounding box, and since footprint LOD divides
        by bbox area, every visibility test came out NaN. A parent's box must
        cover its whole subtree.
        """
        out: list[int] = []
        stack = [nid]
        while stack:
            n = stack.pop()
            r = nodes[str(n)]
            out.extend(r["members"])
            stack.extend(r["children"])
        return np.asarray(out, dtype=np.int64)

    cats = []
    for nid, rec in nodes.items():
        c = cons.get(str(nid), {})
        bbox, centroid = c.get("bbox"), c.get("centroid")
        if bbox is None:
            idx = subtree_members(nid)
            pts = Y[idx] if len(idx) else Y
            bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())]
            centroid = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        # Robust extent for level-of-detail.
        #
        # A raw bounding box is set by a category's two most extreme members,
        # so one stray game stretches it across the map: zooming to the
        # Platformer territory's bbox gave 1.35x magnification, because a
        # handful of its 5,630 members sit nowhere near the rest. Footprint LOD
        # divides by this area, so outlier-inflated boxes make large
        # territories look permanently too big to label.
        #
        # `bbox_core` is the 5th-95th percentile box: where the category
        # actually is. The full bbox is kept for fit-to-view and hit testing,
        # which do want the true extent.
        idx = subtree_members(nid)
        if len(idx):
            pts = Y[idx]
            lo = np.percentile(pts, 5, axis=0)
            hi = np.percentile(pts, 95, axis=0)
            core = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        else:
            core = [float(v) for v in bbox]

        cats.append({
            "id": int(nid),
            "parent": None if rec["parent"] is None else int(rec["parent"]),
            "label": rec.get("label", ""),
            "bbox": [float(v) for v in bbox],
            "bbox_core": core,
            "centroid": [float(v) for v in centroid],
            "size": int(rec.get("size", 0)),
            "depth": int(rec.get("depth", 0)),
            "confidence": rec.get("confidence") or "none",
            "purity": c.get("purity"),
        })
    (PUBLIC / "manifest.json").write_text(
        json.dumps({"version": 1, "categories": cats}), encoding="utf-8")
    print(f"manifest.json  {len(cats):,} categories")

    # names for tooltips
    (PUBLIC / "meta.json").write_text(json.dumps({
        "names": g.names,
        "appids": [int(a) for a in g.appids],
        "tags": [t[:5] for t in g.tags],
        "reviews": [int(r) for r in g.review_count],
    }), encoding="utf-8")
    print(f"meta.json      {(PUBLIC / 'meta.json').stat().st_size / 1e6:.2f} MB")

    # true high-dimensional neighbours
    X, appids, _ = load_vectors(args.variant)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    nn = knn_lists(X.astype(np.float32), args.k)
    blob = struct.pack("<4sHHI", b"AKNN", 1, args.k, len(nn)) + nn.tobytes()
    (PUBLIC / "knn.bin").write_bytes(blob)
    print(f"knn.bin        k={args.k}, "
          f"{(PUBLIC / 'knn.bin').stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
