"""Bisecting k-means with *earned* splits: a cluster only divides if the split
is justified, otherwise it stays a leaf.

### The problem this fixes

Plain recursive k-means forces every split. Given a region with no internal
structure it still returns k parts, drawing arbitrary boundaries through a
continuum: two near-identical games either side of a cut get different labels,
and the method has no way to say "there is nothing here to divide". That
produced the residue buckets in `tree_k12` -- most visibly a 5,522-game node
labelled "Indie / Casual / VR", which is 10% of the catalog presented as
though it were a genre.

### The gate

At each node, candidate splits for every k in a searched range are scored, and
the best is accepted only if it clears a threshold. Two criteria:

**silhouette** (default). Mean silhouette of the proposed partition, on a
sample for tractability. Ranges -1..1; positive means points sit nearer their
own cluster than the next. A threshold around 0.02-0.05 is a weak but real
demand -- on a true continuum the silhouette of any partition sits near zero,
which is precisely the signal we want.

**bic** (X-means style). Compares the BIC of the k-way model against the
1-cluster model under a spherical-Gaussian assumption, penalising parameters.
More principled, but the Gaussian assumption is poor for L2-normalised
embeddings on a sphere, so it is offered rather than defaulted.

A node that fails the gate becomes a leaf regardless of size. Depth therefore
varies by region -- dense, well-structured territories subdivide further than
smooth ones -- which is the behaviour HDBSCAN would have given if the space
had density contrast.

### k search

`--k-min`..`--k-max` are searched at every node and the best-scoring k is
taken, rather than fixing k globally. A node with three genuine sub-genres gets
3, not 12.

Usage:
    python -m cluster.gated_tree --criterion silhouette --threshold 0.03
    python -m cluster.gated_tree --k-min 2 --k-max 12 --max-depth 6
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
from sklearn.metrics import silhouette_score

from embed.common import load_games, load_vectors

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def fit_kmeans(X: np.ndarray, k: int, seed: int, minibatch: bool):
    cls = MiniBatchKMeans if minibatch else KMeans
    kw: dict = dict(n_clusters=k, random_state=seed, n_init=3)
    if minibatch:
        kw.update(batch_size=4096, max_iter=200)
    km = cls(**kw).fit(X)
    return km.labels_, km.cluster_centers_


def bic_gain(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """BIC of the k-cluster model minus BIC of the 1-cluster model.

    Spherical-Gaussian likelihood with a shared variance, the X-means form.
    Positive means the split pays for its extra parameters.
    """
    n, d = X.shape
    k = len(centers)
    if k < 2 or n <= k:
        return -np.inf

    def _bic(lbl, cen):
        kk = len(cen)
        rss = float(((X - cen[lbl]) ** 2).sum())
        var = rss / max(n - kk, 1) / max(d, 1)
        if var <= 0:
            return -np.inf
        ll = 0.0
        for c in range(kk):
            nc = int((lbl == c).sum())
            if nc == 0:
                continue
            ll += (nc * np.log(max(nc, 1))
                   - nc * np.log(n)
                   - nc * d / 2 * np.log(2 * np.pi * var)
                   - (nc - 1) * d / 2)
        params = kk * (d + 1)
        return ll - params / 2 * np.log(n)

    one = np.zeros(n, dtype=int)
    return _bic(labels, centers) - _bic(one, X.mean(axis=0, keepdims=True))


def score_split(X: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                criterion: str, sil_sample: int, seed: int) -> float:
    if len(set(labels.tolist())) < 2:
        return -np.inf
    if criterion == "bic":
        return bic_gain(X, labels, centers)
    n = len(X)
    if n > sil_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sil_sample, replace=False)
        return float(silhouette_score(X[idx], labels[idx], metric="euclidean"))
    return float(silhouette_score(X, labels, metric="euclidean"))


def build(X: np.ndarray, k_min: int, k_max: int, max_depth: int, min_leaf: int,
          min_split: int, criterion: str, threshold: float, sil_sample: int,
          seed: int, minibatch_over: int, k_tolerance: float = 0.85
          ) -> tuple[dict, dict]:
    nodes: dict[int, dict] = {}
    counter = [0]
    stats = {"gate_pass": 0, "gate_fail": 0, "k_chosen": Counter(),
             "fail_by_depth": Counter(), "scores": []}

    def recurse(members: np.ndarray, parent: int | None, depth: int) -> int:
        nid = counter[0]
        counter[0] += 1
        nodes[nid] = {"parent": parent, "children": [], "depth": depth,
                      "members": [], "size": len(members), "split_score": None}

        if len(members) < min_split or depth >= max_depth:
            nodes[nid]["members"] = members.tolist()
            return nid

        sub = X[members]
        mb = len(members) > minibatch_over
        upper = min(k_max, max(2, len(members) // min_leaf))
        best = None
        candidates: list[tuple[float, int, np.ndarray]] = []
        for k in range(k_min, upper + 1):
            lbl, cen = fit_kmeans(sub, k, seed + depth, mb)
            sizes = np.bincount(lbl, minlength=k)
            if (sizes < min_leaf).any():
                continue
            s = score_split(sub, lbl, cen, criterion, sil_sample, seed)
            candidates.append((s, k, lbl))
            if best is None or s > best[0]:
                best = (s, k, lbl)

        # Silhouette rises monotonically as k falls -- a 2-way split almost
        # always scores above a 6-way one even when six genuine sub-genres are
        # present. Left alone it chose k=2 for 201 of 317 splits, producing a
        # deep binary cascade whose paths read "Platformer -> Platformer ->
        # Platformer" (37.5% parent-term repetition), which is the failure this
        # whole phase is trying to avoid.
        #
        # So the best score sets a bar and the *widest* k clearing that bar
        # wins. The criterion still decides whether a split happens at all; it
        # just no longer also decides that every split is binary.
        if best is not None and candidates:
            bar = best[0] * k_tolerance if best[0] > 0 else best[0] / max(k_tolerance, 1e-9)
            widest = max((c for c in candidates if c[0] >= bar),
                         key=lambda c: c[1], default=None)
            if widest is not None:
                best = widest

        if best is None or best[0] < threshold:
            stats["gate_fail"] += 1
            stats["fail_by_depth"][depth] += 1
            if best is not None:
                stats["scores"].append(best[0])
            nodes[nid]["members"] = members.tolist()
            nodes[nid]["split_score"] = None if best is None else round(best[0], 4)
            return nid

        score, k, lbl = best
        stats["gate_pass"] += 1
        stats["k_chosen"][k] += 1
        stats["scores"].append(score)
        nodes[nid]["split_score"] = round(score, 4)

        for c in range(k):
            grp = members[lbl == c]
            if len(grp) == 0:
                continue
            nodes[nid]["children"].append(recurse(grp, nid, depth + 1))
        return nid

    root = recurse(np.arange(len(X)), None, 0)
    return {"root": root, "nodes": nodes}, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--input", default=None,
                    help="use a data/cluster/<name>.npz reduction instead")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-leaf", type=int, default=30)
    ap.add_argument("--min-split", type=int, default=120)
    ap.add_argument("--criterion", choices=["silhouette", "bic"],
                    default="silhouette")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--sil-sample", type=int, default=3000)
    ap.add_argument("--k-tolerance", type=float, default=0.85,
                    help="accept the widest k scoring at least this fraction of the best, countering silhouette's bias toward k=2")
    ap.add_argument("--minibatch-over", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tree_gated")
    args = ap.parse_args()

    g = load_games()
    if args.input:
        d = np.load(DATA / f"{args.input}.npz", allow_pickle=False)
        X, appids = d["X"].astype(np.float32), d["appids"]
        src = args.input
    else:
        X, appids, _ = load_vectors(args.variant)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        src = args.variant
    if not np.array_equal(appids, g.appids):
        print("appid order mismatch")
        return 1
    X = np.ascontiguousarray(X, dtype=np.float32)

    print(f"input     {src}  {X.shape}")
    print(f"gate      {args.criterion} >= {args.threshold}")
    print(f"k search  {args.k_min}..{args.k_max} per node   max_depth {args.max_depth}")
    print(f"k pick    widest k scoring >= {args.k_tolerance:.0%} of the best")

    t0 = time.perf_counter()
    res, st = build(X, args.k_min, args.k_max, args.max_depth, args.min_leaf,
                    args.min_split, args.criterion, args.threshold,
                    args.sil_sample, args.seed, args.minibatch_over,
                    args.k_tolerance)
    secs = time.perf_counter() - t0
    nodes = res["nodes"]
    root = res["root"]

    sizes: dict[int, int] = {}
    for nid in sorted(nodes, key=lambda n: -nodes[n]["depth"]):
        sizes[nid] = len(nodes[nid]["members"]) + sum(
            sizes.get(c, 0) for c in nodes[nid]["children"])
        nodes[nid]["size"] = sizes[nid]

    leaves = [n for n in nodes if not nodes[n]["children"]]
    internal = [len(nodes[n]["children"]) for n in nodes if nodes[n]["children"]]
    depth_hist = Counter(nodes[n]["depth"] for n in nodes)
    non_root = [sizes[n] for n in nodes if n != root]

    print(f"\nbuilt in {secs / 60:.1f} min")
    print(f"  nodes              {len(nodes):,}")
    print(f"  leaves             {len(leaves):,}")
    print(f"  max depth          {max(depth_hist)}")
    print(f"  mean fan-out       {np.mean(internal):.2f}" if internal else "")
    print(f"\n  SPLIT GATE ({args.criterion} >= {args.threshold})")
    print(f"    splits accepted  {st['gate_pass']:,}")
    print(f"    splits REFUSED   {st['gate_fail']:,}"
          f"   <- nodes kept as leaves because there was no structure")
    if st["scores"]:
        sc = np.asarray(st["scores"])
        print(f"    score range      {sc.min():.3f} .. {sc.max():.3f}"
              f"  (median {np.median(sc):.3f})")
    print(f"    refusals by depth {dict(sorted(st['fail_by_depth'].items()))}")
    print(f"    k chosen          {dict(sorted(st['k_chosen'].items()))}")

    print(f"\n  clusters per depth {dict(sorted(depth_hist.items()))}")
    print(f"  cluster sizes      min={min(non_root)}  median={np.median(non_root):.0f}"
          f"  p95={np.percentile(non_root, 95):.0f}  max={max(non_root):,}"
          f"  (biggest {max(non_root) / len(X) * 100:.1f}%)")
    leaf_sizes = [len(nodes[n]["members"]) for n in leaves]
    print(f"  leaf sizes         min={min(leaf_sizes)}  "
          f"median={np.median(leaf_sizes):.0f}  max={max(leaf_sizes):,}")

    payload = {
        "meta": {
            "method": f"bisecting k-means, {args.criterion}-gated splits",
            "source": src, "criterion": args.criterion,
            "threshold": args.threshold, "k_min": args.k_min, "k_max": args.k_max,
            "max_depth": args.max_depth, "min_leaf": args.min_leaf,
            "min_split": args.min_split, "seconds": round(secs, 1),
            "n_games": int(len(X)),
            "splits_accepted": st["gate_pass"], "splits_refused": st["gate_fail"],
            "unclustered_fraction": len(nodes[root]["members"]) / len(X),
        },
        "root": root,
        "nodes": {str(n): nodes[n] for n in nodes},
    }
    out = DATA / f"{args.out}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
