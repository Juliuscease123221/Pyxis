"""Parameter sweep for HDBSCAN on the Overworld embedding.

The first run at SPEC.md's suggested settings (50 dims, min_cluster_size=25,
min_samples=5) put **84.2% of the catalog in noise** and produced a condensed
tree 53 levels deep with a branching factor of 2 at every level -- a single
cluster shedding small groups over and over. That is chaining, not a taxonomy.

Three knobs plausibly cause it, so they are swept rather than guessed:

* **dims.** 50-dim PCA retains only 54.6% variance here and distances are badly
  concentrated (CV of pairwise distance 0.12). Density estimation needs
  contrast; fewer dims usually restores it.
* **min_samples.** This sets how conservative the core-distance estimate is.
  Small values let a thin bridge of points connect two regions, which is
  exactly the mechanism behind chaining.
* **cluster_selection_method.** EOM picks a few large stable clusters and calls
  everything else noise; `leaf` takes the fine-grained leaves instead.

Reported per configuration: noise fraction, flat cluster count, condensed-tree
size and depth, and the branching profile. Noise fraction is the headline --
every game needs a position on the map.

Usage:
    python -m cluster.sweep --sample 20000
    python -m cluster.sweep --full
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter

import numpy as np

from .tree import CondensedTree, load_input


def run_one(X: np.ndarray, mcs: int, ms: int, method: str) -> dict:
    import hdbscan

    t0 = time.perf_counter()
    c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                        cluster_selection_method=method, core_dist_n_jobs=-1)
    c.fit(X)
    secs = time.perf_counter() - t0

    labels = c.labels_
    df = c.condensed_tree_.to_pandas()
    ct = CondensedTree(df, len(X))
    st = ct.stats()
    branch = st["branching"]
    n_forks = sum(v for k, v in branch.items() if k >= 2)

    return {
        "noise": float((labels < 0).mean()),
        "flat": int(labels.max()) + 1,
        "nodes": st["n_cluster_nodes"],
        "max_depth": st["max_depth"],
        "forks": n_forks,
        "median_size": float(np.median(st["sizes"])) if st["sizes"] else 0.0,
        "secs": secs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="pca")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--min-cluster-size", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--min-samples", type=int, nargs="+", default=[5, 15])
    ap.add_argument("--methods", nargs="+", default=["eom", "leaf"])
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise after truncating dims (cosine-like)")
    args = ap.parse_args()

    X0, _, meta = load_input(args.input)
    if not args.full:
        rng = np.random.default_rng(0)
        X0 = X0[rng.choice(len(X0), min(args.sample, len(X0)), replace=False)]
    print(f"input {args.input}  {X0.shape}  normalize={args.normalize}\n")

    print(f"{'dims':>5}{'mcs':>5}{'ms':>4}{'method':>7}"
          f"{'noise':>9}{'flat':>7}{'nodes':>8}{'depth':>7}{'forks':>7}"
          f"{'medsz':>8}{'secs':>7}")
    print("-" * 78)

    best = None
    for dims, mcs, ms, method in itertools.product(
            args.dims, args.min_cluster_size, args.min_samples, args.methods):
        X = np.ascontiguousarray(X0[:, :dims])
        if args.normalize:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        try:
            r = run_one(X, mcs, ms, method)
        except Exception as e:  # noqa: BLE001
            print(f"{dims:>5}{mcs:>5}{ms:>4}{method:>7}   failed: {type(e).__name__}")
            continue
        print(f"{dims:>5}{mcs:>5}{ms:>4}{method:>7}"
              f"{r['noise'] * 100:8.1f}%{r['flat']:>7,}{r['nodes']:>8,}"
              f"{r['max_depth']:>7}{r['forks']:>7,}{r['median_size']:>8.0f}"
              f"{r['secs']:>7.1f}")
        score = (r["noise"], -r["forks"])
        if best is None or score < best[0]:
            best = (score, (dims, mcs, ms, method), r)

    if best:
        (dims, mcs, ms, method), r = best[1], best[2]
        print("\nlowest noise: "
              f"dims={dims} min_cluster_size={mcs} min_samples={ms} method={method}"
              f"  -> {r['noise'] * 100:.1f}% noise, {r['forks']:,} forking nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
