"""Phase 4 layout variants, and a size-normalised purity measure.

### The constraint that rules out the obvious fix

SPEC.md offers recursive layout as fix #2: lay out top-level clusters as
points, allocate each a region, run UMAP inside it, affine-transform into
place. It guarantees containment and would fix purity outright.

**It would also break Phase 7.** Frontier recommendations are defined as
territory "adjacent to your hot zones" -- the whole feature is that a cluster
near your playtime centroid is a good recommendation, and that the user can
*see* why. Recursive layout allocates each territory an arbitrary region and
lays it out independently, so the distance between two territories becomes a
property of the packing algorithm, not of the data. SPEC.md says as much in
one line ("cross-region distances become meaningless") without connecting it
to Phase 7, but that is the connection: fixing purity this way silently
destroys the thing the personal layer is built on.

So the cheaper fixes are tried first, and both preserve data-driven
cross-territory distance.

### The likely diagnosis

Clustering runs in UMAP-5d; the baseline layout is a *separate* UMAP-2d fit
from the 512-d vectors. Two independent stochastic reductions of the same data
have no reason to agree, so a community that is contiguous in 5-d can be
scattered in 2-d. This is disagreement between spaces, not UMAP damage --
which the purity spread supports: the low-purity territories (Puzzle 0.13,
RPG/JRPG 0.08, Shooter/FPS 0.38) are cross-cutting attributes, while the
high-purity ones (Visual Novel 0.93, Action Roguelike 0.91, Platformer 0.88)
are genuine neighbourhoods.

Two fixes, in order of effort:

1. **supervised** -- pass the depth-1 territory as `y` to the layout UMAP,
   sweeping `target_weight`. Pulls territories toward contiguity while leaving
   cross-territory distance data-driven.
2. **chained** -- fit the 2-d layout from the same UMAP-5d space the
   clustering used (512 -> 5 -> 2), so the display space is a further
   reduction of the space the communities were found in.

### Purity lift

Raw purity is size-correlated (0.169 for clusters under 50 games, 0.443 for
1,000+), because a small cluster's hull in a dense map contains non-members
however good the layout is. Cross-level comparison is therefore meaningless as
it stands.

`purity_lift = observed / expected`, where expected comes from a **size-matched
random baseline**: clusters of the same size drawn at random from the catalog,
hulls computed the same way. Same lift construction as the label scoring, so
the codebase stays consistent. Lift 1.0 means no better than a random set of
that size; higher means genuinely spatially coherent.

Usage:
    python -m layout.variants --fit supervised --target-weight 0.1 0.3 0.5
    python -m layout.variants --fit chained
    python -m layout.variants --compare
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
from .layout import CLUSTER_DATA, DATA, inside, trimmed_hull


def depth1_labels(tree: dict, n: int) -> np.ndarray:
    """Territory id per game; -1 for anything held above depth 1."""
    nodes = tree["nodes"]
    root = str(tree["root"])
    y = np.full(n, -1, dtype=np.int64)
    for t, child in enumerate(nodes[root]["children"]):
        stack = [child]
        while stack:
            nid = stack.pop()
            rec = nodes[str(nid)]
            for m in rec["members"]:
                y[m] = t
            stack.extend(rec["children"])
    return y


def fit_variant(kind: str, X512: np.ndarray, X5: np.ndarray | None,
                y: np.ndarray | None, target_weight: float,
                n_neighbors: int, min_dist: float, seed: int):
    import umap

    t0 = time.perf_counter()
    kw = dict(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
              random_state=seed, verbose=False)
    if kind == "chained":
        # the 5-d clustering space is already euclidean-ish; cosine on it is
        # not meaningful, so euclidean is correct here
        reducer = umap.UMAP(metric="euclidean", **kw)
        src = X5
    elif kind == "supervised":
        reducer = umap.UMAP(metric="cosine", target_weight=target_weight, **kw)
        src = X512
    else:
        reducer = umap.UMAP(metric="cosine", **kw)
        src = X512

    Y = reducer.fit_transform(src, y=y if kind == "supervised" else None)
    return np.ascontiguousarray(Y, dtype=np.float32), time.perf_counter() - t0


def purity_baseline(Y: np.ndarray, sizes: list[int], keep: float,
                    pool: np.ndarray, reps: int, seed: int) -> dict[int, float]:
    """Expected purity for a random cluster of each size, measured not assumed."""
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    n = len(Y)
    for s in sizes:
        vals = []
        for _ in range(reps):
            idx = rng.choice(n, min(s, n), replace=False)
            h = trimmed_hull(Y[idx], keep)
            if h is None:
                continue
            mask = inside(h, Y[pool])
            m = int(mask.sum())
            if not m:
                continue
            member = set(idx.tolist())
            hits = sum(1 for p in pool[mask] if int(p) in member)
            vals.append(hits / m)
        out[s] = float(np.mean(vals)) if vals else float("nan")
    return out


def size_bin(k: int) -> int:
    """Log-ish bins, so the baseline is computed a handful of times not 1,100."""
    for b in (30, 50, 80, 130, 210, 340, 550, 900, 1500, 2500, 4000, 6500):
        if k <= b:
            return b
    return 10000


def evaluate(Y: np.ndarray, tree: dict, keep: float, sample: int,
             reps: int, seed: int) -> dict:
    nodes = tree["nodes"]
    root = str(tree["root"])
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(Y), min(sample, len(Y)), replace=False)

    def subtree(nid: str) -> np.ndarray:
        out: list[int] = []
        stack = [nid]
        while stack:
            nn = stack.pop()
            rec = nodes[str(nn)]
            out.extend(rec["members"])
            stack.extend(rec["children"])
        return np.asarray(out, dtype=np.int64)

    members = {nid: subtree(nid) for nid in nodes}
    hulls = {nid: (trimmed_hull(Y[idx], keep) if len(idx) >= 3 else None)
             for nid, idx in members.items()}

    bins = sorted({size_bin(len(members[nid])) for nid in nodes if nid != root})
    base = purity_baseline(Y, bins, keep, pool, reps, seed)

    per_depth = defaultdict(lambda: {"cont": [], "pur": [], "lift": [], "n": 0})
    per_node = {}
    for nid, rec in nodes.items():
        if nid == root:
            continue
        idx = members[nid]
        h = hulls[nid]
        d = rec["depth"]
        per_depth[d]["n"] += 1

        pur = float("nan")
        if h is not None:
            mask = inside(h, Y[pool])
            m = int(mask.sum())
            if m:
                ms = set(idx.tolist())
                pur = sum(1 for p in pool[mask] if int(p) in ms) / m

        par = rec["parent"]
        cont = float("nan")
        if par is not None and hulls[str(par)] is not None:
            cont = float(inside(hulls[str(par)], Y[idx]).mean())

        exp = base.get(size_bin(len(idx)), float("nan"))
        lift = (pur / exp) if (exp and exp > 1e-9 and not np.isnan(pur)) else float("nan")

        if not np.isnan(cont):
            per_depth[d]["cont"].append(cont)
        if not np.isnan(pur):
            per_depth[d]["pur"].append(pur)
        if not np.isnan(lift):
            per_depth[d]["lift"].append(lift)
        per_node[nid] = {"purity": pur, "lift": lift, "containment": cont,
                         "size": len(idx)}

    return {"per_depth": dict(per_depth), "per_node": per_node,
            "baseline": base}


def report(name: str, res: dict) -> None:
    print(f"\n{name}")
    print(f"{'depth':>6}{'nodes':>7}{'contain':>10}{'purity':>9}{'lift':>8}")
    print("-" * 40)
    for d in sorted(res["per_depth"]):
        s = res["per_depth"][d]
        c = np.nanmean(s["cont"]) if s["cont"] else float("nan")
        p = np.nanmean(s["pur"]) if s["pur"] else float("nan")
        l = np.nanmedian(s["lift"]) if s["lift"] else float("nan")
        print(f"{d:>6}{s['n']:>7,}{c:>10.3f}{p:>9.3f}{l:>8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--tree", default="tree_rleiden_labelled")
    ap.add_argument("--fit", nargs="*", default=[],
                    choices=["baseline", "supervised", "chained"])
    ap.add_argument("--target-weight", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5])
    ap.add_argument("--cluster-space", default="umap_c5_n15")
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--keep", type=float, default=0.95)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    g = load_games()
    tree = json.loads((CLUSTER_DATA / f"{args.tree}.json").read_text(encoding="utf-8"))
    DATA.mkdir(parents=True, exist_ok=True)

    X512, appids, _ = load_vectors(args.variant)
    X512 = X512 / np.maximum(np.linalg.norm(X512, axis=1, keepdims=True), 1e-9)
    d5 = np.load(CLUSTER_DATA / f"{args.cluster_space}.npz", allow_pickle=False)
    X5 = d5["X"].astype(np.float32)
    y = depth1_labels(tree, len(g))
    print(f"territories: {len(set(y.tolist())) - (1 if -1 in y else 0)}  "
          f"unassigned: {(y < 0).sum():,}")

    for kind in args.fit:
        weights = args.target_weight if kind == "supervised" else [0.0]
        for w in weights:
            tag = f"{kind}" + (f"_tw{str(w).replace('.', '')}"
                               if kind == "supervised" else "")
            Y, secs = fit_variant(kind, X512, X5, y, w, args.n_neighbors,
                                  args.min_dist, args.seed)
            np.savez_compressed(DATA / f"coords_{tag}.npz", Y=Y, appids=appids,
                                meta=json.dumps({"kind": kind, "target_weight": w,
                                                 "seconds": round(secs, 1)}))
            print(f"  fitted {tag} in {secs:.0f}s")

    if not args.compare:
        return 0

    print("\n" + "=" * 60)
    print("LAYOUT VARIANTS  (purity lift = observed / size-matched random)")
    print("=" * 60)

    found = [("baseline", DATA / "coords.npz")]
    found += [(p.stem.replace("coords_", ""), p)
              for p in sorted(DATA.glob("coords_*.npz"))]

    summary = {}
    for name, path in found:
        if not path.exists():
            continue
        Y = np.load(path, allow_pickle=False)["Y"].astype(np.float64)
        res = evaluate(Y, tree, args.keep, args.sample, args.reps, args.seed)
        report(name, res)
        summary[name] = {
            str(d): {
                "n": v["n"],
                "containment": float(np.nanmean(v["cont"])) if v["cont"] else None,
                "purity": float(np.nanmean(v["pur"])) if v["pur"] else None,
                "purity_lift": float(np.nanmedian(v["lift"])) if v["lift"] else None,
            } for d, v in res["per_depth"].items()
        }
        # qualitative: did the known-bad territories become contiguous?
        root = str(tree["root"])
        tops = sorted(tree["nodes"][root]["children"],
                      key=lambda c: -tree["nodes"][str(c)]["size"])
        print("   depth-1 purity:", ", ".join(
            f"{tree['nodes'][str(c)]['label'].split(' / ')[0][:14]}"
            f" {res['per_node'][str(c)]['purity']:.2f}" for c in tops[:8]))

    (DATA / "variants.json").write_text(json.dumps(summary), encoding="utf-8")
    print(f"\nwrote {DATA / 'variants.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
