"""Phase 4: the 2D display layout, and the hierarchy-consistency measurement.

This UMAP is a **different object** from the one in `cluster/umap_reduce.py`.
That one reduces to 5 dims with `min_dist=0.0`, is tuned for density
clustering and is never rendered. This one produces the coordinates people
look at: 2 dims, `min_dist=0.05` so clusters stay tight at 2-pixel dots, and
it feeds the quadtree. Neither is derived from the other; they are parallel
branches off the same vectors, exactly as SPEC.md's architecture requires.

### One deviation, for a measured reason

SPEC.md says to run UMAP from the 50-dim PCA. Phase 3 established that
PCA-50 of variant C is **85.8% tag loading** -- the tag half packs equal
variance into 128 dims against the text half's 384, so PCA takes tag
directions first and silently discards the alpha=0.5 blend chosen in Phase 2.
Laying out from that would put the map in a different space from the tree.
This runs UMAP on the full 512-d vectors with `metric='cosine'`, matching both
the embedding and the clustering input.

### What is measured, before any fix is attempted

**purity(node)** -- of the points falling inside a node's 2D region, what
fraction actually belong to that node. Low purity means the region is
contaminated by games from elsewhere, so a label drawn over it is a lie about
some of what it covers.

**containment(node)** -- what fraction of a node's members fall inside its
*parent's* 2D region. This is the hierarchy-consistency number: a child whose
members scatter outside its parent's territory cannot be drawn as nested, and
the map shows a child spilling across a boundary.

Note that "fraction of members inside the cluster's own hull" is 1.0 by
construction when the hull is the convex hull of those members, so it measures
nothing. The two quantities above are the informative pair.

Regions use a **trimmed** convex hull -- the hull of the points within a
quantile of the centroid -- because one stray member otherwise inflates a
region across half the map and makes every purity number meaningless.

Usage:
    python -m layout.layout --fit                 # run UMAP, save coords
    python -m layout.layout --measure             # containment/purity per depth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "layout"
CLUSTER_DATA = ROOT / "data" / "cluster"


def fit_layout(X: np.ndarray, n_neighbors: int, min_dist: float,
               seed: int) -> tuple[np.ndarray, float]:
    import umap

    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    Y = reducer.fit_transform(X)
    return np.ascontiguousarray(Y, dtype=np.float32), time.perf_counter() - t0


def trimmed_hull(P: np.ndarray, keep: float = 0.95):
    """Convex hull of the central `keep` fraction of points, by distance to centroid.

    An untrimmed hull is hostage to its worst outlier: one member landing
    across the map stretches the region over unrelated territory and drives
    purity to near zero for reasons that have nothing to do with the layout.
    """
    from scipy.spatial import ConvexHull

    if len(P) < 3:
        return None
    c = P.mean(axis=0)
    d = np.linalg.norm(P - c, axis=1)
    cut = np.quantile(d, keep)
    core = P[d <= cut]
    if len(core) < 3:
        core = P
    try:
        return ConvexHull(core)
    except Exception:  # degenerate / collinear
        return None


def inside(hull, pts: np.ndarray) -> np.ndarray:
    """Boolean mask of which points lie inside a ConvexHull (half-space test)."""
    if hull is None:
        return np.zeros(len(pts), dtype=bool)
    eq = hull.equations                      # (f, 3): a*x + b*y + c <= 0 inside
    return np.all(pts @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)


def measure(tree: dict, Y: np.ndarray, keep: float, sample_outside: int,
            seed: int = 0) -> dict:
    """Containment and purity per node, aggregated per depth."""
    nodes = tree["nodes"]
    root = str(tree["root"])
    rng = np.random.default_rng(seed)

    def subtree(nid: str) -> np.ndarray:
        out: list[int] = []
        stack = [nid]
        while stack:
            n = stack.pop()
            rec = nodes[str(n)]
            out.extend(rec["members"])
            stack.extend(rec["children"])
        return np.asarray(out, dtype=np.int64)

    members = {nid: subtree(nid) for nid in nodes}
    hulls = {}
    for nid, idx in members.items():
        hulls[nid] = trimmed_hull(Y[idx], keep) if len(idx) >= 3 else None

    # a sample of the whole catalog, for the purity denominator
    pool = rng.choice(len(Y), min(sample_outside, len(Y)), replace=False)

    per_depth: dict[int, dict] = defaultdict(
        lambda: {"purity": [], "containment": [], "n": 0, "bbox": 0})
    per_node: dict[str, dict] = {}

    for nid, rec in nodes.items():
        if nid == root:
            continue
        depth = rec["depth"]
        idx = members[nid]
        h = hulls[nid]
        per_depth[depth]["n"] += 1

        # purity: of catalog points inside this region, how many are members
        if h is not None:
            mask = inside(h, Y[pool])
            n_in = int(mask.sum())
            if n_in:
                member_set = set(idx.tolist())
                hits = sum(1 for p in pool[mask] if int(p) in member_set)
                pur = hits / n_in
            else:
                pur = float("nan")
        else:
            pur = float("nan")

        # containment: fraction of my members inside my PARENT's region
        par = rec["parent"]
        if par is None:
            cont = float("nan")
        else:
            ph = hulls[str(par)]
            cont = float(inside(ph, Y[idx]).mean()) if ph is not None else float("nan")

        if not np.isnan(pur):
            per_depth[depth]["purity"].append(pur)
        if not np.isnan(cont):
            per_depth[depth]["containment"].append(cont)

        # 2D bbox + centroid, written back for Phase 6
        pts = Y[idx]
        per_node[nid] = {
            "bbox": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                     float(pts[:, 0].max()), float(pts[:, 1].max())],
            "centroid": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
            "purity": None if np.isnan(pur) else round(pur, 4),
            "containment": None if np.isnan(cont) else round(cont, 4),
        }

    return {"per_depth": dict(per_depth), "per_node": per_node}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--tree", default="tree_rleiden_labelled")
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--keep", type=float, default=0.95,
                    help="trimmed-hull quantile")
    ap.add_argument("--sample", type=int, default=20000,
                    help="catalog sample for the purity denominator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    coords_path = DATA / "coords.npz"

    g = load_games()

    if args.fit or not coords_path.exists():
        X, appids, _ = load_vectors(args.variant)
        if not np.array_equal(appids, g.appids):
            print("appid order mismatch")
            return 1
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        print(f"fitting display UMAP: {X.shape} -> 2D  "
              f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, cosine)")
        Y, secs = fit_layout(X, args.n_neighbors, args.min_dist, args.seed)
        np.savez_compressed(coords_path, Y=Y, appids=appids,
                            meta=json.dumps({
                                "source_variant": args.variant,
                                "n_neighbors": args.n_neighbors,
                                "min_dist": args.min_dist, "metric": "cosine",
                                "seconds": round(secs, 1),
                            }))
        print(f"  fitted in {secs:.0f}s -> {coords_path}")
    else:
        print(f"using existing {coords_path}")

    d = np.load(coords_path, allow_pickle=False)
    Y = d["Y"].astype(np.float64)

    if not args.measure:
        return 0

    tree = json.loads((CLUSTER_DATA / f"{args.tree}.json").read_text(encoding="utf-8"))
    print(f"\nmeasuring against {args.tree} "
          f"({len(tree['nodes']):,} nodes, trimmed hull keep={args.keep})")
    res = measure(tree, Y, args.keep, args.sample, args.seed)

    print("\n" + "=" * 72)
    print("HIERARCHY CONSISTENCY IN 2D  (measured before any layout fix)")
    print("=" * 72)
    print(f"{'depth':>6}{'nodes':>8}{'containment':>28}{'purity':>26}")
    print(f"{'':>6}{'':>8}{'mean':>10}{'median':>9}{'<0.9':>9}"
          f"{'mean':>10}{'median':>9}{'<0.5':>7}")
    print("-" * 72)
    for depth in sorted(res["per_depth"]):
        s = res["per_depth"][depth]
        c = np.asarray(s["containment"]) if s["containment"] else np.asarray([np.nan])
        p = np.asarray(s["purity"]) if s["purity"] else np.asarray([np.nan])
        print(f"{depth:>6}{s['n']:>8,}"
              f"{np.nanmean(c):>10.3f}{np.nanmedian(c):>9.3f}"
              f"{np.mean(c < 0.9) * 100:>8.0f}%"
              f"{np.nanmean(p):>10.3f}{np.nanmedian(p):>9.3f}"
              f"{np.mean(p < 0.5) * 100:>6.0f}%")

    out = DATA / "consistency.json"
    out.write_text(json.dumps({
        "tree": args.tree,
        "keep": args.keep,
        "per_depth": {
            str(k): {
                "n": v["n"],
                "containment_mean": float(np.nanmean(v["containment"]))
                if v["containment"] else None,
                "containment_median": float(np.nanmedian(v["containment"]))
                if v["containment"] else None,
                "purity_mean": float(np.nanmean(v["purity"]))
                if v["purity"] else None,
                "purity_median": float(np.nanmedian(v["purity"]))
                if v["purity"] else None,
            } for k, v in res["per_depth"].items()
        },
        "per_node": res["per_node"],
    }), encoding="utf-8")
    print(f"\nwrote {out}  (bbox + centroid per node, for Phase 6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
