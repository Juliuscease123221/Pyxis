"""Recursive spherical k-means taxonomy -- SPEC.md's documented fallback.

### Why this exists

HDBSCAN's condensed tree on this catalog is a caterpillar: one spine that sheds
a small cluster at each of 143 levels while the bulk of the games continue
down. Collapsing the spine (see `tree.collapse`) turns it into a real n-ary
tree, but no collapse threshold escapes the underlying trade-off:

    spine_ratio 0.3  ->  28.8% in the biggest node, but 63.7% unclustered
    spine_ratio 0.9  ->  20.6% unclustered, but 79.4% in the biggest node

Both ends describe the same fact: half the catalog is one undifferentiated
mass with no internal density structure, and the only choice is whether to call
it "noise" or "one enormous cluster". Reading the labels confirms it -- the
49%-of-catalog node came out as "Visual Novel / Action Roguelike / Turn-Based
Strategy", and Hollow Knight's path ended at "Golf / Sokoban / Pinball".

That is not a labelling bug. Steam is a continuum: there is no density valley
between action-roguelikes and action-platformers, so a density-based method
correctly reports one connected region. HDBSCAN is answering honestly; the
question does not suit it.

SPEC.md anticipated exactly this ("Condensed tree is a mess after collapsing
-> Recursive k-means, fixed branching -- less principled, always balanced").

### What this gives up, and what it keeps

**Gives up:** the number of clusters is imposed rather than discovered, and
every split is forced whether or not a real boundary exists. A k-means boundary
through a continuum is arbitrary at the margin -- two similar games either side
of it get different labels. HDBSCAN would have declined to split there.

**Keeps:** the differentiator is untouched. SPEC.md's claim is that labels
come from a *semantic cluster hierarchy* rather than from spatial tiles, and
this is still a hierarchy built in high-dimensional embedding space, never on
2D coordinates. The WizMap contrast holds.

**Gains:** every game lands in a leaf (0% unclustered, no soft assignment), the
branching factor is uniform so no level is empty, and every node has enough
members for c-TF-IDF to say something.

### Spherical k-means

Vectors are L2-normalised and re-normalised after each centroid update, so
Euclidean k-means on the sphere is equivalent to maximising cosine similarity
-- the metric everything in Phase 2 was built and evaluated on.

Runs on the full 512-d variant C rather than the PCA output, deliberately. The
PCA step exists to rescue HDBSCAN's density estimate; k-means does not need it,
and skipping it avoids the modality imbalance PCA introduces (at 50 dims the
retained subspace was 85.8% tag loading, which would have quietly undone the
alpha=0.5 blend).

Usage:
    python -m cluster.kmeans_tree --k 8 --max-depth 5
    python -m cluster.kmeans_tree --variant variant_c_a05 --min-leaf 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from embed.common import load_games, load_vectors

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def spherical_kmeans(X: np.ndarray, k: int, seed: int, minibatch: bool):
    """k-means on L2-normalised vectors == cosine clustering."""
    cls = MiniBatchKMeans if minibatch else KMeans
    kw = dict(n_clusters=k, random_state=seed, n_init=3)
    if minibatch:
        kw.update(batch_size=4096, max_iter=200, n_init=3)
    km = cls(**kw).fit(X)
    return km.labels_


def build(X: np.ndarray, k: int, max_depth: int, min_leaf: int,
          min_split: int, seed: int, minibatch_over: int) -> dict:
    """Recursively split the catalog into a balanced n-ary tree.

    A node splits when it holds at least `min_split` games and is above the
    depth cap. `k` is reduced for small nodes so a split never produces a child
    below `min_leaf`, which keeps every leaf big enough to label.
    """
    nodes: dict[int, dict] = {}
    counter = [0]

    def new_node(parent, depth, members) -> int:
        nid = counter[0]
        counter[0] += 1
        nodes[nid] = {"parent": parent, "children": [], "depth": depth,
                      "members": [], "size": len(members)}
        return nid

    def recurse(members: np.ndarray, parent: int | None, depth: int) -> int:
        nid = new_node(parent, depth, members)

        too_small = len(members) < min_split
        too_deep = depth >= max_depth
        if too_small or too_deep:
            nodes[nid]["members"] = members.tolist()
            return nid

        kk = min(k, max(2, len(members) // min_leaf))
        if kk < 2:
            nodes[nid]["members"] = members.tolist()
            return nid

        sub = X[members]
        labels = spherical_kmeans(sub, kk, seed + depth,
                                  minibatch=len(members) > minibatch_over)

        groups = [members[labels == c] for c in range(kk)]
        groups = [gp for gp in groups if len(gp) > 0]
        if len(groups) < 2:
            nodes[nid]["members"] = members.tolist()
            return nid

        # children too small to stand alone stay with the parent
        keep, residue = [], []
        for gp in groups:
            (keep if len(gp) >= min_leaf else residue).append(gp)
        if len(keep) < 2:
            nodes[nid]["members"] = members.tolist()
            return nid
        if residue:
            nodes[nid]["members"] = np.concatenate(residue).tolist()

        for gp in keep:
            child = recurse(gp, nid, depth + 1)
            nodes[nid]["children"].append(child)
        return nid

    root = recurse(np.arange(len(X)), None, 0)
    return {"root": root, "nodes": nodes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-leaf", type=int, default=30)
    ap.add_argument("--min-split", type=int, default=120)
    ap.add_argument("--minibatch-over", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tree_kmeans")
    args = ap.parse_args()

    g = load_games()
    X, appids, meta = load_vectors(args.variant)
    if not np.array_equal(appids, g.appids):
        print("appid order mismatch")
        return 1
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    X = np.ascontiguousarray(X, dtype=np.float32)

    print(f"input   {args.variant}  {X.shape}  (spherical k-means, no PCA)")
    print(f"params  k={args.k}  max_depth={args.max_depth}  "
          f"min_leaf={args.min_leaf}  min_split={args.min_split}")

    t0 = time.perf_counter()
    res = build(X, args.k, args.max_depth, args.min_leaf, args.min_split,
                args.seed, args.minibatch_over)
    secs = time.perf_counter() - t0
    nodes = res["nodes"]

    # roll sizes up
    sizes: dict[int, int] = {}
    for nid in sorted(nodes, key=lambda n: -nodes[n]["depth"]):
        sizes[nid] = len(nodes[nid]["members"]) + sum(
            sizes.get(c, 0) for c in nodes[nid]["children"])
        nodes[nid]["size"] = sizes[nid]

    depth_hist = Counter(n["depth"] for n in nodes.values())
    branch_hist = Counter(len(n["children"]) for n in nodes.values())
    internal = [len(n["children"]) for n in nodes.values() if n["children"]]
    leaves = [nid for nid in nodes if not nodes[nid]["children"]]
    placed = sum(len(n["members"]) for n in nodes.values())
    root = res["root"]

    print(f"\nbuilt in {secs:.1f}s")
    print(f"  nodes                 {len(nodes):,}")
    print(f"  leaves                {len(leaves):,}")
    print(f"  max depth             {max(depth_hist)}")
    print(f"  mean fan-out          {np.mean(internal):.2f}")
    print(f"  games placed          {placed:,} / {len(X):,}")
    print(f"  held at root          {len(nodes[root]['members']):,} "
          f"({len(nodes[root]['members']) / len(X) * 100:.1f}%)")

    print("\n  nodes per depth:")
    for d in sorted(depth_hist):
        print(f"    depth {d}  {depth_hist[d]:>6,}")
    print(f"\n  branching factor: {dict(sorted(branch_hist.items()))}")

    non_root = [sizes[n] for n in nodes if n != root]
    leaf_sizes = sorted(len(nodes[n]["members"]) for n in leaves)
    print(f"\n  cluster sizes:  min={min(non_root)}  "
          f"median={np.median(non_root):.0f}  p95={np.percentile(non_root, 95):.0f}"
          f"  max={max(non_root):,}  (biggest = {max(non_root) / len(X) * 100:.1f}%)")
    print(f"  leaf sizes:     min={min(leaf_sizes)}  "
          f"median={np.median(leaf_sizes):.0f}  max={max(leaf_sizes):,}")

    payload = {
        "meta": {
            "method": "recursive spherical k-means",
            "source_variant": args.variant,
            "k": args.k, "max_depth": args.max_depth,
            "min_leaf": args.min_leaf, "min_split": args.min_split,
            "seconds": round(secs, 1),
            "n_games": int(len(X)),
            "unclustered_fraction": len(nodes[root]["members"]) / len(X),
        },
        "root": root,
        "nodes": {str(n): {**nodes[n], "children": nodes[n]["children"]}
                  for n in nodes},
    }
    out = DATA / f"{args.out}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
