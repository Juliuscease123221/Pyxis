"""Stage viewer assets: a tile pyramid, a manifest, sharded metadata, neighbours.

Everything here is Atlas-specific. The tile pyramid is built by the `tiles`
library, which is handed five anonymous columns and never learns what they
mean.

### What is eager and what is lazy

Cold load fetches only `manifest.json` and the tiles covering the opening
viewport. Everything else arrives when something needs it:

  tiles/z/x/y.bin   fetched per viewport, LRU-cached in the browser
  meta/NNNN.json    names and tags, sharded by point-index block, fetched when
                    a point in that block is first hovered
  knn.bin           true high-dimensional neighbours, fetched on first hover

The first version of this shipped one 1.01 MB blob of every point plus a
5.62 MB `meta.json` loaded eagerly, which made the tile engine dead code and
the app's scaling story theoretical. Sharding metadata by point-index block
rather than by tile is deliberate: a point appears in several tiles (that
redundancy is what makes tiles independently renderable), so tile-keyed
metadata would duplicate it at every zoom level.

Usage:
    python -m viewer.prepare
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors
from ingest.db import connect
from tiles.format import PointSet, encode
from tiles.quadtree import build as build_tiles

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "viewer" / "public"
CLUSTER = ROOT / "data" / "cluster"
LAYOUT = ROOT / "data" / "layout"

META_SHARD = 4096


def knn_lists(X: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
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
    ap.add_argument("--max-zoom", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=1200)
    ap.add_argument("--gzip", action="store_true", default=True)
    args = ap.parse_args()

    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True)

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

    # ---- tile pyramid, via the standalone library ------------------------
    ts = build_tiles(ps, max_zoom=args.max_zoom, max_points=args.max_points,
                     strategy="stratified")
    tdir = PUBLIC / "tiles"
    total = 0
    for t in ts.tiles:
        blob = encode(t.points, bbox=t.bounds)
        if args.gzip:
            blob = gzip.compress(blob, 6)
        p = tdir / str(t.tid.z) / str(t.tid.x)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{t.tid.y}.bin").write_bytes(blob)
        total += len(blob)
    st = ts.stats()
    print(f"tiles/         {st['n_tiles']:,} tiles, {total / 1e6:.2f} MB"
          f"  ({'gzipped' if args.gzip else 'raw'})")
    print(f"               per zoom {st['tiles_per_zoom']}")

    # ---- manifest ---------------------------------------------------------
    cons_path = LAYOUT / "consistency.json"
    cons = (json.loads(cons_path.read_text(encoding="utf-8"))["per_node"]
            if cons_path.exists() else {})

    def subtree(nid: str) -> np.ndarray:
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
        idx = subtree(nid)
        pts = Y[idx] if len(idx) else Y
        bbox = c.get("bbox") or [float(pts[:, 0].min()), float(pts[:, 1].min()),
                                 float(pts[:, 0].max()), float(pts[:, 1].max())]
        centroid = c.get("centroid") or [float(pts[:, 0].mean()),
                                         float(pts[:, 1].mean())]
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        cats.append({
            "id": int(nid),
            "parent": None if rec["parent"] is None else int(rec["parent"]),
            "label": rec.get("label", ""),
            "terms": [t for t, _ in (rec.get("terms") or [])][:4],
            "bbox": [float(v) for v in bbox],
            "bbox_core": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
            "centroid": [float(v) for v in centroid],
            "size": int(rec.get("size", 0)),
            "depth": int(rec.get("depth", 0)),
            "confidence": rec.get("confidence") or "none",
            "purity": c.get("purity"),
        })

    xs, ys = np.sort(Y[:, 0]), np.sort(Y[:, 1])
    q = lambda a, p: float(a[int((len(a) - 1) * p)])  # noqa: E731
    manifest = {
        "version": 2,
        "bounds": list(ts.bounds),
        "core_extent": [q(xs, 0.005), q(ys, 0.005), q(xs, 0.995), q(ys, 0.995)],
        "max_zoom": args.max_zoom,
        "n_points": int(len(g)),
        "meta_shard": META_SHARD,
        "gzip_tiles": bool(args.gzip),
        "categories": cats,
    }
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"manifest.json  {len(cats):,} categories, "
          f"{(PUBLIC / 'manifest.json').stat().st_size / 1e6:.2f} MB")

    # ---- metadata, sharded by point-index block --------------------------
    #
    # The hover card needs score, release year and price, which are in the
    # catalog but not in the Games dataclass. Pulled here in one pass, keyed by
    # appid, then indexed in the same row order as everything else.
    #
    # Note the point record's `id` is a ROW INDEX, not an appid: the tile
    # format carries an opaque u32 and the library never learns what it means.
    # The appid lives here, which is what the card and the store link use.
    conn = connect()
    extra = {int(r["appid"]): (r["review_score"], r["release_date"], r["price"])
             for r in conn.execute(
                 "SELECT appid, review_score, release_date, price FROM games")}
    conn.close()

    def year_of(s: str | None) -> int | None:
        if not s:
            return None
        for tok in str(s).replace(",", " ").split():
            if len(tok) == 4 and tok.isdigit():
                return int(tok)
        return None

    mdir = PUBLIC / "meta"
    mdir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(g) + META_SHARD - 1) // META_SHARD
    biggest = 0
    for s in range(n_shards):
        lo_i, hi_i = s * META_SHARD, min((s + 1) * META_SHARD, len(g))
        block = [int(a) for a in g.appids[lo_i:hi_i]]
        payload = {
            "from": lo_i,
            "names": g.names[lo_i:hi_i],
            "appids": block,
            "tags": [t[:5] for t in g.tags[lo_i:hi_i]],
            "reviews": [int(r) for r in g.review_count[lo_i:hi_i]],
            "scores": [round(extra.get(a, (None,))[0] or 0, 3) for a in block],
            "years": [year_of(extra.get(a, (None, None))[1]) for a in block],
            "prices": [extra.get(a, (None, None, None))[2] for a in block],
        }
        f = mdir / f"{s:04d}.json"
        f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        biggest = max(biggest, f.stat().st_size)
    meta_total = sum(p.stat().st_size for p in mdir.glob("*.json"))
    print(f"meta/          {n_shards} shards, {meta_total / 1e6:.2f} MB total, "
          f"{biggest / 1e6:.2f} MB largest  (lazy)")

    # ---- search index ----------------------------------------------------
    #
    # A separate lazy asset, fetched when the search box is first focused --
    # metadata is sharded by point-index block and loaded on hover, so there is
    # no client-side name list until something asks for one.
    #
    # Kept minimal and column-oriented: parallel arrays compress far better
    # than an array of objects, since each column is homogeneous. The point
    # index is deliberately absent -- it is the array position, so storing it
    # would be 56,129 integers of pure redundancy.
    #
    # Cluster labels are NOT duplicated here: the manifest already carries
    # every category with its label and bounding box and is loaded at startup,
    # so label search runs against that.
    search = {
        "n": int(len(g)),
        "names": g.names,
        "appids": [int(a) for a in g.appids],
        "x": [round(float(v), 3) for v in Y[:, 0]],
        "y": [round(float(v), 3) for v in Y[:, 1]],
        "reviews": [int(r) for r in g.review_count],
        "years": [year_of(extra.get(int(a), (None, None))[1]) for a in g.appids],
    }
    sp = PUBLIC / "search.json"
    sp.write_text(json.dumps(search, ensure_ascii=False, separators=(",", ":")),
                  encoding="utf-8")
    raw = sp.stat().st_size
    gzsize = len(gzip.compress(sp.read_bytes(), 6))
    print(f"search.json    {raw / 1e6:.2f} MB raw, {gzsize / 1e6:.2f} MB gzipped"
          f"  (lazy, on first focus of the search box)")

    # ---- true neighbours --------------------------------------------------
    X, appids, _ = load_vectors(args.variant)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    nn = knn_lists(X.astype(np.float32), args.k)
    blob = struct.pack("<4sHHI", b"AKNN", 1, args.k, len(nn)) + nn.tobytes()
    (PUBLIC / "knn.bin").write_bytes(blob)
    print(f"knn.bin        k={args.k}, "
          f"{(PUBLIC / 'knn.bin').stat().st_size / 1e6:.2f} MB  (lazy)")

    eager = ((PUBLIC / "manifest.json").stat().st_size
             + sum(p.stat().st_size for p in (tdir / "0").rglob("*.bin")))
    print(f"\ncold load (manifest + z0 tile): {eager / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
