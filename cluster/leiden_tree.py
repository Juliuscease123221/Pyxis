"""Multi-resolution Leiden: a hierarchy from graph community detection.

### Why a graph method suits this data

Centroid methods impose a shape (roughly spherical, roughly equal-variance) and
density methods need a density valley to cut on. Steam's embedding has neither
-- it is a connected continuum with varying local density, which is why
HDBSCAN returned a depth-143 caterpillar and why plain k-means had to force
every split.

Community detection asks a different question: not "where are the gaps" but
"which games are more connected to each other than to the rest of the graph".
A k-NN graph over cosine similarity turns a continuum into exactly the kind of
object modularity optimisation handles well, and communities come out at
naturally varying sizes rather than at a size the method assumed.

### Building the hierarchy

Leiden is flat -- one resolution, one partition. The hierarchy comes from
running several resolutions and nesting them by **membership containment**:

    for each community C at resolution r+1:
        parent(C) = the resolution-r community holding most of C's members

Rising resolution yields finer communities, so a fine community is almost
always contained in a coarse one. Where it is not -- where a fine community
straddles two coarse ones -- majority containment picks the larger overlap and
the straddle is recorded as `purity`, so an imperfect nesting is visible rather
than silent. A level that adds nothing (a child holding essentially all of its
parent) is collapsed, the same pass-through rule as the HDBSCAN path.

Usage:
    python -m cluster.leiden_tree --resolutions 0.2 0.5 1.0 2.0 4.0
    python -m cluster.leiden_tree --knn 20 --out tree_leiden
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def knn_graph(X: np.ndarray, k: int, block: int = 2048):
    """Mutual-ish cosine k-NN graph as (edges, weights).

    X must be L2-normalised, so a dot product is the cosine. Edges are
    deduplicated by ordering each pair, which symmetrises the graph: if A lists
    B but B does not list A, the edge still exists. That matters on a continuum
    where degree varies a lot.
    """
    n = len(X)
    rows, cols, vals = [], [], []
    for s in range(0, n, block):
        sims = X[s:s + block] @ X.T
        for r in range(len(sims)):
            sims[r, s + r] = -np.inf
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(len(sims)):
            src = s + r
            for c in idx[r]:
                w = float(sims[r, c])
                if w <= 0:
                    continue
                a, b = (src, int(c)) if src < c else (int(c), src)
                rows.append(a)
                cols.append(b)
                vals.append(w)
    seen: dict[tuple[int, int], float] = {}
    for a, b, w in zip(rows, cols, vals):
        key = (a, b)
        if w > seen.get(key, -1.0):
            seen[key] = w
    edges = list(seen.keys())
    weights = [seen[e] for e in edges]
    return edges, weights


def run_leiden(n: int, edges, weights, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg as la

    gph = ig.Graph(n=n, edges=edges, directed=False)
    gph.es["weight"] = weights
    part = la.find_partition(
        gph, la.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--resolutions", type=float, nargs="+",
                    default=[0.2, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tree_leiden")
    args = ap.parse_args()

    g = load_games()
    X, appids, _ = load_vectors(args.variant)
    if not np.array_equal(appids, g.appids):
        print("appid order mismatch")
        return 1
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X)

    print(f"input {args.variant} {X.shape}")
    print(f"building cosine {args.knn}-NN graph ...")
    t0 = time.perf_counter()
    edges, weights = knn_graph(X, args.knn)
    print(f"  {len(edges):,} edges in {time.perf_counter() - t0:.0f}s")

    levels: list[np.ndarray] = []
    print(f"\n{'resolution':>11}{'communities':>14}{'>=min_size':>12}{'secs':>8}")
    print("-" * 46)
    for r in sorted(args.resolutions):
        t1 = time.perf_counter()
        memb = run_leiden(n, edges, weights, r, args.seed)
        sizes = Counter(memb.tolist())
        big = sum(1 for v in sizes.values() if v >= args.min_size)
        levels.append(memb)
        print(f"{r:>11.2f}{len(sizes):>14,}{big:>12,}"
              f"{time.perf_counter() - t1:>8.0f}")

    # nest by containment
    nodes: dict[int, dict] = {}
    counter = [0]

    def new(parent, depth, purity=1.0):
        nid = counter[0]
        counter[0] += 1
        nodes[nid] = {"parent": parent, "children": [], "depth": depth,
                      "members": [], "size": 0, "purity": round(float(purity), 3)}
        return nid

    root = new(None, 0)
    prev_nodes: dict[int, int] = {}
    prev_mem: dict[int, np.ndarray] = {}
    owner = np.full(n, root, dtype=np.int64)
    straddles = 0

    for memb in levels:
        cur_nodes: dict[int, int] = {}
        cur_mem: dict[int, np.ndarray] = {}
        for cid in np.unique(memb):
            idx = np.flatnonzero(memb == cid)
            if len(idx) < args.min_size:
                continue
            if not prev_mem:
                parent, purity = root, 1.0
            else:
                counts = {pid: np.intersect1d(idx, pm, assume_unique=False).size
                          for pid, pm in prev_mem.items()}
                best = max(counts, key=counts.get)
                hit = counts[best]
                if hit == 0:
                    parent, purity = root, 0.0
                else:
                    parent, purity = prev_nodes[best], hit / len(idx)
                if purity < 0.9:
                    straddles += 1
            nid = new(parent, nodes[parent]["depth"] + 1, purity)
            nodes[parent]["children"].append(nid)
            cur_nodes[int(cid)] = nid
            cur_mem[int(cid)] = idx
            owner[idx] = nid
        if cur_nodes:
            prev_nodes, prev_mem = cur_nodes, cur_mem

    for i in range(n):
        nodes[int(owner[i])]["members"].append(i)
    for nid in sorted(nodes, key=lambda x: -nodes[x]["depth"]):
        nodes[nid]["size"] = len(nodes[nid]["members"]) + sum(
            nodes[c]["size"] for c in nodes[nid]["children"])

    leaves = [x for x in nodes if not nodes[x]["children"]]
    internal = [len(nodes[x]["children"]) for x in nodes if nodes[x]["children"]]
    depth_hist = Counter(nodes[x]["depth"] for x in nodes)
    non_root = [nodes[x]["size"] for x in nodes if x != root]

    print(f"\n  nodes              {len(nodes):,}")
    print(f"  leaves             {len(leaves):,}")
    print(f"  max depth          {max(depth_hist)}")
    print(f"  mean fan-out       {np.mean(internal):.2f}")
    print(f"  straddling nodes   {straddles:,}  (containment purity < 0.9)")
    print(f"  clusters per depth {dict(sorted(depth_hist.items()))}")
    print(f"  cluster sizes      min={min(non_root)}  "
          f"median={np.median(non_root):.0f}  max={max(non_root):,}"
          f"  (biggest {max(non_root) / n * 100:.1f}%)")
    print(f"  held at root       {len(nodes[root]['members']):,}"
          f"  ({len(nodes[root]['members']) / n * 100:.1f}%)")

    payload = {
        "meta": {
            "method": "multi-resolution Leiden on cosine kNN graph",
            "source": args.variant, "knn": args.knn,
            "resolutions": sorted(args.resolutions), "min_size": args.min_size,
            "n_games": n, "straddling_nodes": straddles,
            "unclustered_fraction": len(nodes[root]["members"]) / n,
        },
        "root": root,
        "nodes": {str(x): nodes[x] for x in nodes},
    }
    out = DATA / f"{args.out}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
