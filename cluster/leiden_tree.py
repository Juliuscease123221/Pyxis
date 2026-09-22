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



def build_recursive(X: np.ndarray, knn: int, resolution: float, max_depth: int,
                    min_split: int, min_size: int, seed: int) -> tuple[dict, int, dict]:
    """Recursive Leiden: exact nesting by construction.

    Multi-resolution Leiden runs each resolution over the *whole* graph and
    then infers parenthood by majority containment. Nothing forces a fine
    community to sit inside one coarse community, and on this catalog 709 of
    them straddled two -- median containment purity 0.90, half the nodes below
    it.

    That is fatal for Phase 4 rather than untidy. Phase 4's job is keeping
    children inside their parent's 2D region; a child whose members genuinely
    belong to two parents gets torn across two territories, and on screen that
    reads as a rendering bug.

    Recursive Leiden removes the failure mode instead of measuring it. Each
    community is re-clustered on a k-NN graph rebuilt from *only its own
    members*, so a child is computed from its parent's membership and cannot
    straddle. Purity is 1.0 everywhere, by construction, not by luck.

    The subgraph rebuild also changes what the algorithm sees: neighbours are
    recomputed within the territory, so "nearest" means nearest *among
    strategy games* rather than nearest in the whole catalog. That is a better
    question at depth, and it attacks the arbitrary-merge-order complaint from
    the other side.

    Cost is trivial here -- the top-level graph dominates and every subgraph is
    a fraction of it.
    """
    nodes: dict[int, dict] = {}
    counter = [0]
    stats = {"leiden_runs": 0, "refused_no_split": 0, "by_depth": Counter()}

    def new(parent, depth, n_members):
        nid = counter[0]
        counter[0] += 1
        nodes[nid] = {"parent": parent, "children": [], "depth": depth,
                      "members": [], "size": n_members, "purity": 1.0}
        return nid

    def recurse(idx: np.ndarray, parent: int | None, depth: int) -> int:
        nid = new(parent, depth, len(idx))
        stats["by_depth"][depth] += 1

        if len(idx) < min_split or depth >= max_depth:
            nodes[nid]["members"] = idx.tolist()
            return nid

        sub = np.ascontiguousarray(X[idx])
        edges, weights = knn_graph(sub, min(knn, max(2, len(sub) - 1)))
        if not edges:
            nodes[nid]["members"] = idx.tolist()
            return nid

        memb = run_leiden(len(sub), edges, weights, resolution, seed)
        stats["leiden_runs"] += 1

        groups = [idx[memb == c] for c in np.unique(memb)]
        keep = [gp for gp in groups if len(gp) >= min_size]
        residue = [gp for gp in groups if len(gp) < min_size]

        # a single surviving community is not a split: it re-describes the
        # parent, so stop rather than add a pass-through level
        if len(keep) < 2:
            stats["refused_no_split"] += 1
            nodes[nid]["members"] = idx.tolist()
            return nid

        if residue:
            nodes[nid]["members"] = np.concatenate(residue).tolist()
        for gp in keep:
            nodes[nid]["children"].append(recurse(gp, nid, depth + 1))
        return nid

    root = recurse(np.arange(len(X)), None, 0)
    return nodes, root, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--resolutions", type=float, nargs="+",
                    default=[0.2, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["multires", "recursive"],
                    default="recursive",
                    help="recursive rebuilds a subgraph per community, so nesting is exact; multires infers parents by majority containment")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="recursive mode: resolution used at every level")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-split", type=int, default=120)
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

    if args.mode == "recursive":
        print(f"recursive Leiden: resolution {args.resolution} at every level,"
              f" knn {args.knn}, max_depth {args.max_depth}")
        print("(a subgraph is rebuilt per community, so nesting is exact)\n")
        t0 = time.perf_counter()
        nodes, root, st = build_recursive(
            X, args.knn, args.resolution, args.max_depth,
            args.min_split, args.min_size, args.seed)
        secs = time.perf_counter() - t0

        for nid in sorted(nodes, key=lambda x: -nodes[x]["depth"]):
            nodes[nid]["size"] = len(nodes[nid]["members"]) + sum(
                nodes[c]["size"] for c in nodes[nid]["children"])

        leaves = [x for x in nodes if not nodes[x]["children"]]
        internal = [len(nodes[x]["children"]) for x in nodes if nodes[x]["children"]]
        depth_hist = Counter(nodes[x]["depth"] for x in nodes)
        non_root = [nodes[x]["size"] for x in nodes if x != root]
        placed = sum(len(nodes[x]["members"]) for x in nodes)

        print(f"built in {secs:.0f}s  ({st['leiden_runs']:,} Leiden runs,"
              f" {st['refused_no_split']:,} nodes left whole for lack of a split)")
        print(f"  nodes              {len(nodes):,}")
        print(f"  leaves             {len(leaves):,}")
        print(f"  max depth          {max(depth_hist)}")
        print(f"  mean fan-out       {np.mean(internal):.2f}")
        print(f"  containment purity 1.000 by construction (no straddling possible)")
        print(f"  clusters per depth {dict(sorted(depth_hist.items()))}")
        print(f"  cluster sizes      min={min(non_root)}  "
              f"median={np.median(non_root):.0f}  max={max(non_root):,}"
              f"  (biggest {max(non_root) / n * 100:.1f}%)")
        print(f"  held at root       {len(nodes[root]['members']):,}"
              f"  ({len(nodes[root]['members']) / n * 100:.1f}%)")
        print(f"  games placed       {placed:,} / {n:,}")

        payload = {
            "meta": {
                "method": "recursive Leiden on per-community cosine kNN subgraphs",
                "source": args.variant, "knn": args.knn,
                "resolution": args.resolution, "max_depth": args.max_depth,
                "min_size": args.min_size, "min_split": args.min_split,
                "n_games": n, "seconds": round(secs, 1),
                "leiden_runs": st["leiden_runs"],
                "straddling_nodes": 0,
                "exact_nesting": True,
                "unclustered_fraction": len(nodes[root]["members"]) / n,
            },
            "root": root,
            "nodes": {str(x): nodes[x] for x in nodes},
        }
        out = DATA / f"{args.out}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

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
