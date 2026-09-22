"""Phase 3 steps 2-3: HDBSCAN condensed tree -> clean n-ary taxonomy.

SPEC.md flags step 3 as the one that actually bites, and it is right, but not
quite for the stated reason. The *raw* single-linkage tree is full of splits
where a cluster divides into itself plus four points; the condensed tree has
already absorbed those -- that is what condensing means. Points that drop out
appear as point-children with a lambda value, not as sibling branches.

What the condensed tree leaves behind is a different degeneracy: it is strictly
**binary**. Every cluster splits into exactly two children, so a genuine 6-way
distinction inside "Strategy" arrives as a five-deep cascade of binary splits,
each one shaving off one real group and passing the rest down. Walking that
root-to-leaf gives paths like Strategy -> Strategy -> Strategy -> Strategy ->
4X, which is exactly the unreadable output SPEC.md warns about, arrived at by
a different route.

So the collapse has two jobs:

1. **Dissolve insignificant nodes.** A cluster below `min_branch` is not a real
   branch; its members belong to its parent.
2. **Make transparent any node that is not a real fork.** A kept node earns its
   place only if at least two of its children survive. A node with one surviving
   child is a pass-through: it re-describes its parent and adds a level to every
   path beneath it. Lifting its child into its grandparent is what turns the
   binary cascade into the n-ary taxonomy a person would draw.

Both are reported as before/after counts.

Noise is not dropped. See `assign_noise`.

Usage:
    python -m cluster.tree --stats-only          # fit, report, write nothing
    python -m cluster.tree --input pca_balanced
    python -m cluster.tree --min-cluster-size 25 --min-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cluster"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_input(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = DATA / f"{name}.npz"
    d = np.load(path, allow_pickle=False)
    return d["X"].astype(np.float64), d["appids"], json.loads(str(d["meta"]))


# --------------------------------------------------------------------------
# raw condensed-tree structure
# --------------------------------------------------------------------------

class CondensedTree:
    """Cluster-node view of hdbscan's condensed tree.

    The condensed tree mixes two kinds of row. A row whose `child` is below
    `n_points` is an individual game falling out of a cluster at `lambda_val`;
    a row whose `child` is at or above `n_points` is a child *cluster*. Keeping
    them apart is the whole of the parsing work.
    """

    def __init__(self, df, n_points: int):
        self.n_points = n_points
        self.parent_of: dict[int, int] = {}
        self.children: dict[int, list[int]] = defaultdict(list)
        self.size: dict[int, int] = {}
        self.points_falling_out: dict[int, list[int]] = defaultdict(list)
        self.lambda_of: dict[int, float] = {}

        for parent, child, lam, csize in df.itertuples(index=False):
            parent, child, csize = int(parent), int(child), int(csize)
            if child >= n_points:
                self.parent_of[child] = parent
                self.children[parent].append(child)
                self.size[child] = csize
                self.lambda_of[child] = float(lam)
            else:
                self.points_falling_out[parent].append(child)

        self.root = min(self.children) if self.children else n_points
        self.size[self.root] = n_points

    def descendant_points(self, node: int) -> list[int]:
        """Every game under `node`, including those that fell out below it."""
        out: list[int] = []
        stack = [node]
        while stack:
            n = stack.pop()
            out.extend(self.points_falling_out.get(n, ()))
            stack.extend(self.children.get(n, ()))
        return out

    def depth_of(self, node: int) -> int:
        d = 0
        while node in self.parent_of:
            node = self.parent_of[node]
            d += 1
        return d

    def stats(self) -> dict:
        nodes = list(self.size)
        branching = Counter(len(self.children.get(n, ())) for n in nodes)
        depths = Counter(self.depth_of(n) for n in nodes)
        sizes = sorted(self.size[n] for n in nodes if n != self.root)
        return {
            "n_cluster_nodes": len(nodes) - 1,
            "branching": dict(sorted(branching.items())),
            "depth_hist": dict(sorted(depths.items())),
            "max_depth": max(depths) if depths else 0,
            "sizes": sizes,
        }


# --------------------------------------------------------------------------
# collapse
# --------------------------------------------------------------------------

def collapse(ct: CondensedTree, min_branch: int, max_depth: int,
             spine_ratio: float = 0.5) -> dict:
    """Condensed tree -> clean n-ary taxonomy.

    ### What the tree actually looks like

    HDBSCAN's condensed tree here is a **caterpillar**, not a balanced
    hierarchy: 320 cluster nodes, depth 91, branching factor 2 at nearly every
    level. One long spine runs from the root, and at each step it sheds a small
    real cluster while the bulk of the catalog continues down. Read literally,
    that gives paths 91 levels deep in which 90 of the levels are the same
    undifferentiated mass of games.

    This is SPEC.md's "cluster divides into itself plus a handful of points",
    arrived at through the condensed tree rather than the raw linkage tree. The
    points do not appear as a sibling branch -- condensing already absorbed
    those -- but the *spine* does, and it is the same pathology one level up.

    ### The three rules

    1. **Spine collapse.** If a child holds at least `spine_ratio` of its
       parent's points, it is not a subdivision of the parent, it *is* the
       parent, minus whatever was shed. Merge it upward and promote its
       children. Applied repeatedly, this walks the whole spine and gathers
       every shed cluster into one wide fan-out -- which is precisely the n-ary
       taxonomy a person would draw, and the single most important step here.
    2. **Dissolve small.** A child below `min_branch` is not a real branch; its
       members belong to its parent.
    3. **Pass-through.** A node left with one surviving child re-describes its
       parent and costs every descendant a level. Make it transparent.

    Then cap depth.
    """
    tree: dict[int, dict] = {}
    stats = {"spine_merges": 0, "dissolved_small": 0,
             "dissolved_passthrough": 0, "truncated_at_depth_cap": 0}

    def effective_children(node: int) -> list[int]:
        """Children after walking down any spine continuing from `node`.

        A child holding >= spine_ratio of this node's points is expanded in
        place: its own children take its position. Repeats until every
        remaining child is genuinely a subdivision.
        """
        parent_size = max(ct.size.get(node, 0), 1)
        frontier = list(ct.children.get(node, ()))
        out: list[int] = []
        guard = 0
        while frontier and guard < 10_000:
            guard += 1
            c = frontier.pop()
            if ct.size.get(c, 0) >= spine_ratio * parent_size:
                kids = ct.children.get(c, ())
                if kids:
                    stats["spine_merges"] += 1
                    frontier.extend(kids)
                    continue
                # a spine leaf: it is the parent's own residue, not a branch
                stats["spine_merges"] += 1
                continue
            out.append(c)
        return out

    def build(node: int, parent: int | None, depth: int) -> None:
        tree[node] = {"parent": parent, "children": [], "depth": depth}
        if depth >= max_depth:
            stats["truncated_at_depth_cap"] += len(ct.children.get(node, ()))
            return

        kids = effective_children(node)
        keep = []
        for c in kids:
            if ct.size.get(c, 0) < min_branch:
                stats["dissolved_small"] += 1
                continue
            keep.append(c)

        # a single surviving child is a pass-through: skip the level entirely
        while len(keep) == 1:
            only = keep[0]
            nxt = [c for c in effective_children(only)
                   if ct.size.get(c, 0) >= min_branch]
            if not nxt:
                break                      # it is a real leaf, keep it
            stats["dissolved_passthrough"] += 1
            keep = nxt

        for c in keep:
            tree[node]["children"].append(c)
            build(c, node, depth + 1)

    sys.setrecursionlimit(10_000)
    build(ct.root, None, 0)
    return {"tree": tree, **stats}


# --------------------------------------------------------------------------
# membership and noise
# --------------------------------------------------------------------------

def assign_members(ct: CondensedTree, tree: dict[int, dict]) -> dict[int, list[int]]:
    """Every game to the deepest surviving cluster that contains it.

    A game that fell out of a dissolved node lands on the nearest surviving
    ancestor rather than vanishing, so membership over the kept tree is a
    partition of all 56,129 games.
    """
    members: dict[int, list[int]] = defaultdict(list)
    for node in tree:
        for p in ct.points_falling_out.get(node, ()):
            members[node].append(p)

    # points that fell out of dissolved nodes: walk up to the nearest kept one
    kept = set(tree)
    for node, pts in ct.points_falling_out.items():
        if node in kept:
            continue
        anc = node
        while anc not in kept and anc in ct.parent_of:
            anc = ct.parent_of[anc]
        if anc in kept:
            members[anc].extend(pts)
    return members


def assign_noise(X: np.ndarray, labels: np.ndarray, members: dict[int, list[int]],
                 tree: dict[int, dict], strategy: str) -> tuple[dict[int, list[int]], dict]:
    """Decide what happens to games HDBSCAN called noise.

    Every game needs a position on the map, so noise cannot silently disappear.
    Two honest options:

      soft     assign each noise point to the nearest kept cluster by centroid
               distance. Every game gets a label. The cost is that a genuinely
               isolated game is given a cluster it does not really belong to,
               and the label it inherits is a small lie.
      keep     leave noise attached to the root as explicit unclustered
               territory. Honest, and the map can render it differently (dimmer,
               unlabelled), but it leaves a large region with no semantics.

    `soft` is the default because the deliverable is a *map*: a game with no
    position is worse than a game with an approximate one, and the alternative
    puts a double-digit fraction of the catalog in a grey zone. The assignment
    is recorded per game so the viewer can still render soft-assigned points
    differently from core members.
    """
    # The honest "unclustered" number is games that land on the ROOT of the
    # collapsed tree -- they were never inside any surviving cluster. HDBSCAN's
    # own labels_ call ~84% of the catalog noise, but that is EOM *flat-label
    # selection*, which this pipeline does not use: the condensed tree assigns
    # 100% of games to some node. Reporting the flat-label figure here would
    # badly overstate the problem.
    root = min(tree, key=lambda n: tree[n]["depth"])
    root_only = set(members.get(root, ()))
    stats = {
        "n_flat_noise": int((labels < 0).sum()),
        "flat_noise_fraction": float((labels < 0).mean()),
        "n_unclustered": len(root_only),
        "unclustered_fraction": float(len(root_only) / len(labels)),
        "strategy": strategy,
    }
    if not root_only or strategy == "keep":
        return members, stats

    leaves = [n for n in tree if not tree[n]["children"] and members.get(n)]
    if not leaves:
        return members, stats

    # Reassign, do not duplicate: these games are currently held by the root.
    noise_idx = np.asarray(sorted(root_only), dtype=np.int64)
    members[root] = []

    cent = np.vstack([X[members[n]].mean(axis=0) for n in leaves])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)

    Q = X[noise_idx]
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)

    assigned = np.empty(len(noise_idx), dtype=np.int64)
    for s in range(0, len(Q), 4096):
        sims = Q[s:s + 4096] @ cent.T
        assigned[s:s + 4096] = sims.argmax(axis=1)

    soft: dict[int, list[int]] = defaultdict(list)
    for pos, li in zip(noise_idx, assigned):
        soft[leaves[int(li)]].append(int(pos))
    for n, pts in soft.items():
        members[n].extend(pts)

    stats["soft_assigned"] = int(len(noise_idx))
    stats["receiving_clusters"] = int(len(soft))
    return members, stats


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def pct(xs: list[int], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def report_sizes(sizes: list[int], label: str) -> None:
    if not sizes:
        print(f"  {label}: none")
        return
    print(f"  {label}: n={len(sizes):,}  min={min(sizes)}  "
          f"p25={pct(sizes, 25):.0f}  median={pct(sizes, 50):.0f}  "
          f"p75={pct(sizes, 75):.0f}  p95={pct(sizes, 95):.0f}  max={max(sizes):,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="pca")
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-branch", type=int, default=None,
                    help="dissolve clusters smaller than this (default 2x min-cluster-size)")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--dims", type=int, default=None,
                    help="truncate the PCA input to this many dims")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise after truncation (cosine geometry, matching how the embeddings were built and evaluated)")
    ap.add_argument("--spine-ratio", type=float, default=0.5,
                    help="a child holding this fraction of its parent is the parent continuing, not a subdivision")
    ap.add_argument("--noise", choices=["soft", "keep"], default="soft")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import hdbscan

    X, appids, meta = load_input(args.input)
    if args.dims:
        X = np.ascontiguousarray(X[:, :args.dims])
    if args.normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    min_branch = args.min_branch or args.min_cluster_size * 2
    print(f"input            {args.input}  {X.shape}  "
          f"(mode={meta.get('mode')}, from {meta.get('source_variant')})")
    print(f"min_cluster_size {args.min_cluster_size}   min_samples {args.min_samples}"
          f"   min_branch {min_branch}   max_depth {args.max_depth}")

    t0 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X)
    fit_s = time.perf_counter() - t0
    print(f"\nHDBSCAN fit in {fit_s / 60:.1f} min")

    labels = clusterer.labels_
    n_noise = int((labels < 0).sum())
    flat = int(labels.max()) + 1
    print(f"  flat clusters returned      {flat:,}")
    print(f"  games assigned to NOISE     {n_noise:,}  "
          f"({n_noise / len(labels) * 100:.1f}%)")

    df = clusterer.condensed_tree_.to_pandas()
    ct = CondensedTree(df, len(X))
    raw = ct.stats()

    print("\n" + "=" * 70)
    print("RAW CONDENSED TREE (before any collapse)")
    print("=" * 70)
    print(f"  cluster nodes               {raw['n_cluster_nodes']:,}")
    print(f"  max depth                   {raw['max_depth']}")
    print(f"  branching factor histogram  {raw['branching']}")
    report_sizes(raw["sizes"], "cluster sizes")
    print("\n  nodes per depth:")
    for d, n in list(raw["depth_hist"].items())[:15]:
        print(f"    depth {d:>3}  {n:>6,}")
    if len(raw["depth_hist"]) > 15:
        print(f"    ... {len(raw['depth_hist']) - 15} deeper levels")

    res = collapse(ct, min_branch, args.max_depth, args.spine_ratio)
    tree = res["tree"]
    members = assign_members(ct, tree)

    # roll member counts up the tree
    sizes: dict[int, int] = {}
    for node in sorted(tree, key=lambda n: -tree[n]["depth"]):
        sizes[node] = len(members.get(node, ())) + sum(
            sizes.get(c, 0) for c in tree[node]["children"])

    print("\n" + "=" * 70)
    print("COLLAPSE")
    print("=" * 70)
    print(f"  before                      {raw['n_cluster_nodes']:,} cluster nodes,"
          f" max depth {raw['max_depth']}")
    print(f"  spine merges                {res['spine_merges']:,}"
          f"   (child held >={args.spine_ratio:.0%} of parent: same cluster continuing)")
    print(f"  dissolved: below min_branch {res['dissolved_small']:,}")
    print(f"  dissolved: pass-through     {res['dissolved_passthrough']:,}"
          f"   (single-child, re-describes parent)")
    print(f"  cut at depth cap            {res['truncated_at_depth_cap']:,}")
    print(f"  after                       {len(tree) - 1:,} cluster nodes,"
          f" max depth {max(t['depth'] for t in tree.values())}")

    depth_hist = Counter(t["depth"] for t in tree.values())
    branch_hist = Counter(len(t["children"]) for t in tree.values())
    print("\n  nodes per depth:")
    for d in sorted(depth_hist):
        print(f"    depth {d}  {depth_hist[d]:>6,}")
    print(f"\n  branching factor: {dict(sorted(branch_hist.items()))}")
    internal = [len(t["children"]) for t in tree.values() if t["children"]]
    if internal:
        print(f"    mean fan-out at internal nodes: {np.mean(internal):.2f}")
    report_sizes(sorted(sizes[n] for n in tree if n != ct.root), "cluster sizes")

    members, nstats = assign_noise(X, labels, members, tree, args.noise)
    print("\n" + "=" * 70)
    print("NOISE")
    print("=" * 70)
    print(f"  HDBSCAN flat labels_ noise  {nstats['n_flat_noise']:,}"
          f"  ({nstats['flat_noise_fraction'] * 100:.1f}%)  <- EOM selection, NOT used")
    print(f"  truly unclustered (at root) {nstats['n_unclustered']:,}"
          f"  ({nstats['unclustered_fraction'] * 100:.1f}% of the catalog)")
    print(f"  strategy                    {nstats['strategy']}")
    if nstats.get("soft_assigned"):
        print(f"  soft-assigned to nearest    {nstats['soft_assigned']:,}"
              f"  across {nstats['receiving_clusters']:,} leaf clusters")
    covered = sum(len(v) for v in members.values())
    distinct = len({p for v in members.values() for p in v})
    print(f"  games placed                 {covered:,} rows, {distinct:,} distinct"
          f" / {len(X):,}  ({distinct / len(X) * 100:.1f}%)")
    if covered != distinct:
        print(f"  !! {covered - distinct:,} duplicate placements")

    if args.stats_only:
        print("\n(--stats-only: nothing written)")
        return 0

    out = Path(args.out) if args.out else DATA / f"tree_{args.input}.json"
    payload = {
        "meta": {
            "input": args.input,
            "source_variant": meta.get("source_variant"),
            "pca_mode": meta.get("mode"),
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "min_branch": min_branch,
            "max_depth": args.max_depth,
            "noise_strategy": args.noise,
            "fit_seconds": round(fit_s, 1),
            **{k: v for k, v in res.items() if k != "tree"},
            **nstats,
        },
        "root": ct.root,
        "nodes": {
            str(n): {
                "parent": tree[n]["parent"],
                "children": tree[n]["children"],
                "depth": tree[n]["depth"],
                "size": sizes.get(n, 0),
                "members": sorted(members.get(n, ())),
            }
            for n in tree
        },
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    np.save(DATA / f"labels_{args.input}.npy", labels)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
