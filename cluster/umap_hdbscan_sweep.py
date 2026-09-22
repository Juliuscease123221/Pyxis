"""Sweep HDBSCAN across every UMAP clustering reduction.

Reports the numbers that actually decide this, which are not the ones HDBSCAN
prints by default:

  root_out   games falling out at the *root* of the condensed tree -- the
             honest unclustered fraction. HDBSCAN's `labels_` noise is EOM
             flat-label selection, which this pipeline never uses, and on the
             512-d space it read 84% while the true figure was 20-40%.
  forks      condensed-tree nodes with >=2 cluster children. A caterpillar has
             almost none relative to its depth; a real hierarchy has many.
  depth      raw condensed-tree depth. High depth with few forks is the
             caterpillar signature.

Usage:
    python -m cluster.umap_hdbscan_sweep
    python -m cluster.umap_hdbscan_sweep --min-cluster-size 25 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from .tree import CondensedTree, load_input

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="default: every data/cluster/umap_*.npz")
    ap.add_argument("--min-cluster-size", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--min-samples", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    import hdbscan

    names = args.inputs or sorted(p.stem for p in DATA.glob("umap_*.npz"))
    if not names:
        print("no umap reductions found")
        return 1

    print(f"{'reduction':<16}{'mcs':>5}{'ms':>4}{'flat_noise':>12}{'flat_k':>8}"
          f"{'nodes':>7}{'depth':>7}{'forks':>7}{'root_out':>10}{'medsz':>8}")
    print("-" * 84)

    rows = []
    for name in names:
        X, _, meta = load_input(name)
        X = np.ascontiguousarray(X, dtype=np.float64)
        for mcs, ms in itertools.product(args.min_cluster_size, args.min_samples):
            c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                core_dist_n_jobs=-1).fit(X)
            ct = CondensedTree(c.condensed_tree_.to_pandas(), len(X))
            st = ct.stats()
            forks = sum(v for k, v in st["branching"].items() if k >= 2)
            root_out = len(ct.points_falling_out.get(ct.root, [])) / len(X)
            med = float(np.median(st["sizes"])) if st["sizes"] else 0.0
            rows.append({
                "name": name, "mcs": mcs, "ms": ms, "forks": forks,
                "root_out": root_out, "nodes": st["n_cluster_nodes"],
                "depth": st["max_depth"],
                "flat_noise": float((c.labels_ < 0).mean()),
            })
            print(f"{name:<16}{mcs:>5}{ms:>4}{(c.labels_ < 0).mean() * 100:11.1f}%"
                  f"{c.labels_.max() + 1:>8}{st['n_cluster_nodes']:>7}"
                  f"{st['max_depth']:>7}{forks:>7}{root_out * 100:9.1f}%{med:>8.0f}")

    print("\nreference, 512-d cosine (the rejected space):")
    print("  pca              25   5       82.4%      41    220     57     110     39.8%")

    best = max(rows, key=lambda r: (r["forks"], -r["root_out"]))
    print(f"\nmost forking structure: {best['name']} mcs={best['mcs']} ms={best['ms']}"
          f"  -> {best['forks']:,} forks, {best['root_out'] * 100:.1f}% at root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
