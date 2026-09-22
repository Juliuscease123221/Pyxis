"""Phase 3 validation: print root-to-leaf label paths and read them.

The acceptance test for this phase is not a score. It is whether walking the
tree from root to leaf produces labels a person would say out loud. A path
reading `Indie -> Indie -> Action Indie -> Indie Action` is a failed tree
however good its NMI is, and no metric in this repo would catch it.

So: 20 randomly sampled games, and a fixed set of well-known ones where the
right answer is obvious to anyone who plays games.

Usage:
    python -m cluster.paths --tree tree_main_labelled
    python -m cluster.paths --random 20 --seed 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from embed.common import load_games

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"

KNOWN = {
    367520: "Hollow Knight",
    413150: "Stardew Valley",
    1145360: "Hades",
    236850: "Europa Universalis IV",
    570: "Dota 2",
    252490: "Rust",
    620: "Portal 2",
    105600: "Terraria",
    377160: "Fallout 4",
    892970: "Valheim",
}


def build_index(nodes: dict) -> dict[int, str]:
    """game row index -> the deepest node that directly holds it."""
    owner: dict[int, str] = {}
    for nid, rec in nodes.items():
        for m in rec["members"]:
            owner[m] = nid
    return owner


def path_of(nodes: dict, nid: str) -> list[str]:
    out = []
    cur: str | None = nid
    while cur is not None:
        rec = nodes[str(cur)]
        out.append(rec.get("label", f"node {cur}"))
        cur = rec["parent"]
        if cur is not None:
            cur = str(cur)
    return list(reversed(out))


def render(g, nodes, owner, i: int, show_tags: int = 5) -> str:
    nid = owner.get(i)
    if nid is None:
        return f"  {g.names[i][:40]:<40}  (no cluster)"
    p = path_of(nodes, nid)
    size = nodes[nid]["size"]
    tags = ", ".join(g.tags[i][:show_tags])
    head = f"  {g.names[i][:38]:<38}"
    body = "  →  ".join(p[1:]) if len(p) > 1 else "(root only)"
    return f"{head}  {body}\n{'':<40}  [n={size}]  tags: {tags}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="tree_main_labelled")
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = load_games()
    payload = json.loads((DATA / f"{args.tree}.json").read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    owner = build_index(nodes)

    print("=" * 78)
    print("TOP-LEVEL TERRITORIES (depth 1)")
    print("=" * 78)
    root = str(payload["root"])
    tops = sorted(nodes[root]["children"], key=lambda c: -nodes[str(c)]["size"])
    for c in tops:
        rec = nodes[str(c)]
        kids = len(rec["children"])
        print(f"  {rec['size']:>7,}  {rec.get('label', '?')[:56]:<56} "
              f"({kids} children)")

    print("\n" + "=" * 78)
    print("KNOWN GAMES")
    print("=" * 78)
    for appid, name in KNOWN.items():
        i = g.index_of(appid)
        if i is None:
            print(f"  {name}: not in catalog")
            continue
        print(render(g, nodes, owner, i))

    print("\n" + "=" * 78)
    print(f"{args.random} RANDOM GAMES (seed {args.seed})")
    print("=" * 78)
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(len(g), args.random, replace=False):
        print(render(g, nodes, owner, int(i)))

    # depth distribution of where games actually land
    depths = defaultdict(int)
    for i, nid in owner.items():
        depths[nodes[nid]["depth"]] += 1
    print("\n" + "=" * 78)
    print("WHERE GAMES LAND")
    print("=" * 78)
    tot = sum(depths.values())
    for d in sorted(depths):
        print(f"  depth {d}: {depths[d]:>7,}  ({depths[d] / tot * 100:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
