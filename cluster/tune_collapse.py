"""Tune the collapse, reusing one HDBSCAN fit per (min_cluster_size, min_samples).

The HDBSCAN fit is the expensive part and depends only on min_cluster_size and
min_samples; `spine_ratio` and `min_branch` are pure post-processing of the
condensed tree. Refitting for each combination wastes minutes per cell, so the
fit is cached and only the collapse is re-run.

Reported per configuration:

  unclust%   games landing on the root with no cluster of their own -- the
             number that actually matters, since it is the fraction of the map
             that will be soft-assigned rather than genuinely placed
  nodes      surviving cluster nodes
  fan        mean fan-out at internal nodes (2.0 means it is still binary)
  biggest    largest single cluster as a share of the catalog; a depth-1 node
             holding half the games is a failed taxonomy however good the rest
             of the numbers look

Usage:
    python -m cluster.tune_collapse --input pca --dims 20 --normalize
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from .tree import CondensedTree, assign_members, collapse, load_input


def evaluate(ct: CondensedTree, n: int, min_branch: int, max_depth: int,
             spine_ratio: float) -> dict:
    res = collapse(ct, min_branch, max_depth, spine_ratio)
    tree = res["tree"]
    members = assign_members(ct, tree)

    sizes: dict[int, int] = {}
    for node in sorted(tree, key=lambda x: -tree[x]["depth"]):
        sizes[node] = len(members.get(node, ())) + sum(
            sizes.get(c, 0) for c in tree[node]["children"])

    root = min(tree, key=lambda x: tree[x]["depth"])
    unclustered = len(members.get(root, ()))
    internal = [len(tree[x]["children"]) for x in tree if tree[x]["children"]]
    non_root = [sizes[x] for x in tree if x != root]

    return {
        "unclustered": unclustered / n,
        "nodes": len(tree) - 1,
        "fan": float(np.mean(internal)) if internal else 0.0,
        "depth": max(tree[x]["depth"] for x in tree),
        "biggest": (max(non_root) / n) if non_root else 0.0,
        "leaves": sum(1 for x in tree if not tree[x]["children"]),
        "median_size": float(np.median(non_root)) if non_root else 0.0,
        "spine_merges": res["spine_merges"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="pca")
    ap.add_argument("--dims", type=int, default=20)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--min-cluster-size", type=int, nargs="+", default=[15, 25, 40])
    ap.add_argument("--min-samples", type=int, nargs="+", default=[1])
    ap.add_argument("--spine-ratio", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    import hdbscan

    X, _, meta = load_input(args.input)
    if args.dims:
        X = np.ascontiguousarray(X[:, :args.dims])
    if args.normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    n = len(X)
    print(f"input {args.input} {X.shape} normalize={args.normalize}\n")

    print(f"{'mcs':>5}{'ms':>4}{'spine':>7}{'unclust':>9}{'nodes':>7}"
          f"{'leaves':>8}{'fan':>6}{'depth':>7}{'biggest':>9}{'medsz':>8}")
    print("-" * 70)

    for mcs in args.min_cluster_size:
        for ms in args.min_samples:
            c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                core_dist_n_jobs=-1).fit(X)
            ct = CondensedTree(c.condensed_tree_.to_pandas(), n)
            for sr in args.spine_ratio:
                r = evaluate(ct, n, mcs, args.max_depth, sr)
                print(f"{mcs:>5}{ms:>4}{sr:>7.1f}"
                      f"{r['unclustered'] * 100:8.1f}%{r['nodes']:>7,}"
                      f"{r['leaves']:>8,}{r['fan']:>6.1f}{r['depth']:>7}"
                      f"{r['biggest'] * 100:8.1f}%{r['median_size']:>8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
