"""Tree statistics report: shape, balance, label health, noise accounting.

Everything here is structural. None of it proves the labels are good -- that is
`cluster.paths`, read by a person -- but a tree that fails these is not worth
labelling.

Usage:
    python -m cluster.stats --tree tree_k12_labelled
    python -m cluster.stats --tree tree_k12_labelled --compare tree_main_labelled
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def summarise(payload: dict) -> dict:
    nodes = payload["nodes"]
    root = str(payload["root"])
    n_games = payload["meta"].get("n_games") or sum(
        len(r["members"]) for r in nodes.values())

    depth_hist = Counter(r["depth"] for r in nodes.values())
    branch_hist = Counter(len(r["children"]) for r in nodes.values())
    internal = [len(r["children"]) for r in nodes.values() if r["children"]]
    leaves = [nid for nid, r in nodes.items() if not r["children"]]
    sizes = [r["size"] for nid, r in nodes.items() if nid != root]
    leaf_sizes = [nodes[nid]["size"] for nid in leaves]

    # where games actually live
    land = defaultdict(int)
    for nid, r in nodes.items():
        land[r["depth"]] += len(r["members"])

    # label health
    repeats = checked = 0
    for nid, r in nodes.items():
        p = r["parent"]
        if p is None:
            continue
        pt = {t for t, _ in nodes[str(p)].get("terms", [])[:3]}
        ct = {t for t, _ in r.get("terms", [])[:3]}
        if not pt:
            continue
        checked += 1
        if pt & ct:
            repeats += 1

    labels = [r.get("label", "") for r in nodes.values() if r.get("label")]
    unnamed = sum(1 for x in labels if x == "unnamed")

    return {
        "n_games": n_games,
        "nodes": len(nodes) - 1,
        "leaves": len(leaves),
        "max_depth": max(depth_hist),
        "depth_hist": dict(sorted(depth_hist.items())),
        "branch_hist": dict(sorted(branch_hist.items())),
        "mean_fanout": float(np.mean(internal)) if internal else 0.0,
        "sizes": sizes,
        "leaf_sizes": leaf_sizes,
        "biggest_share": max(sizes) / n_games if sizes else 0.0,
        "land": dict(sorted(land.items())),
        "root_held": len(nodes[root]["members"]),
        "repeat_rate": repeats / checked if checked else float("nan"),
        "unnamed": unnamed,
        "meta": payload["meta"],
    }


def show(name: str, s: dict) -> None:
    n = s["n_games"]
    print("=" * 72)
    print(f"{name}   ({s['meta'].get('method', 'unknown method')})")
    print("=" * 72)
    print(f"  games                    {n:,}")
    print(f"  cluster nodes            {s['nodes']:,}")
    print(f"  leaves                   {s['leaves']:,}")
    print(f"  max depth                {s['max_depth']}")
    print(f"  mean fan-out             {s['mean_fanout']:.2f}")

    print("\n  CLUSTERS PER DEPTH")
    for d, c in s["depth_hist"].items():
        print(f"    depth {d}   {c:>6,} clusters")

    print("\n  BRANCHING FACTOR")
    for b, c in s["branch_hist"].items():
        kind = "leaves" if b == 0 else f"{b}-way"
        print(f"    {kind:<10} {c:>6,} nodes")

    sz = s["sizes"]
    ls = s["leaf_sizes"]
    print("\n  CLUSTER SIZE DISTRIBUTION")
    print(f"    all nodes:  min={min(sz)}  p25={np.percentile(sz, 25):.0f}  "
          f"median={np.median(sz):.0f}  p75={np.percentile(sz, 75):.0f}  "
          f"p95={np.percentile(sz, 95):.0f}  max={max(sz):,}")
    print(f"    leaves:     min={min(ls)}  median={np.median(ls):.0f}  "
          f"max={max(ls):,}")
    print(f"    largest cluster is {s['biggest_share'] * 100:.1f}% of the catalog")

    print("\n  WHERE GAMES LAND")
    for d, c in s["land"].items():
        if c:
            print(f"    depth {d}   {c:>7,}  ({c / n * 100:5.1f}%)")

    print("\n  NOISE / COVERAGE")
    unc = s["meta"].get("unclustered_fraction")
    if unc is not None:
        print(f"    unclustered (held at root)  {s['root_held']:,}"
              f"  ({unc * 100:.1f}%)")
    if "n_flat_noise" in s["meta"]:
        print(f"    HDBSCAN flat labels_ noise  {s['meta']['n_flat_noise']:,}"
              f"  ({s['meta']['flat_noise_fraction'] * 100:.1f}%)  [EOM, not used]")
    if s["meta"].get("strategy"):
        print(f"    strategy                    {s['meta']['strategy']}")
    placed = sum(s["land"].values())
    print(f"    games placed                {placed:,} / {n:,}"
          f"  ({placed / n * 100:.1f}%)")

    print("\n  LABEL HEALTH")
    print(f"    children reusing a parent top-3 term  "
          f"{s['repeat_rate'] * 100:.1f}%")
    print(f"    unnamed nodes                         {s['unnamed']:,}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="tree_k12_labelled")
    ap.add_argument("--compare", nargs="*", default=[])
    args = ap.parse_args()

    for name in [args.tree, *args.compare]:
        try:
            show(name, summarise(load(name)))
        except FileNotFoundError:
            print(f"(missing {name})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
