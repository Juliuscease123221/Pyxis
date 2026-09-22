"""Does a layout still carry the data's geometry, or has it imposed one?

Purity and containment reward separation. A layout that shatters the catalog
into 27 disjoint blobs scores perfectly on both and is useless: Phase 7's
frontier recommendations are defined as territory *adjacent to* the user's hot
zones, and Phase 6's hover-and-inspect assumes nearby points on screen are
similar games. Neither survives a layout that invented its own geometry.

Supervised UMAP returned purity 1.000 at depth 1 for every territory. That is
the number that should trigger this check rather than end the search, because
it is also what recursive layout — the fix ruled out for breaking Phase 7 —
would produce.

Two measures, aimed at the two things that would break:

**global fidelity** (Phase 7). Spearman correlation between territory-centroid
distances in the original 512-d cosine space and in the 2-d layout. If
territories keep their relative arrangement, "adjacent" still means something
and frontier scoring is safe. If the correlation collapses, adjacency has
become an artifact of the layout and the personal layer loses its basis.

**local fidelity** (Phase 6). Overlap between a game's 10 nearest neighbours in
512-d and its 10 nearest on screen. If hovering a point shows unrelated games,
the map lies at the level a viewer can actually check.

Usage:
    python -m layout.fidelity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors
from .layout import CLUSTER_DATA, DATA
from .variants import depth1_labels


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / d) if d else float("nan")


def territory_fidelity(X: np.ndarray, Y: np.ndarray, y: np.ndarray) -> dict:
    """Do territories keep their relative arrangement?"""
    tids = sorted(set(y.tolist()) - {-1})
    cx = np.vstack([X[y == t].mean(axis=0) for t in tids])
    cy = np.vstack([Y[y == t].mean(axis=0) for t in tids])
    cx /= np.maximum(np.linalg.norm(cx, axis=1, keepdims=True), 1e-9)

    iu = np.triu_indices(len(tids), 1)
    d_hi = (1.0 - cx @ cx.T)[iu]                      # cosine distance
    d_lo = np.linalg.norm(cy[:, None, :] - cy[None, :, :], axis=-1)[iu]
    return {"spearman": spearman(d_hi, d_lo), "n_territories": len(tids)}


def neighbour_fidelity(X: np.ndarray, Y: np.ndarray, k: int = 10,
                       sample: int = 3000, seed: int = 0) -> float:
    """Fraction of each game's high-dim k-NN that are still k-NN on screen."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    keep = 0.0
    for s in range(0, len(idx), 512):
        chunk = idx[s:s + 512]
        sims = X[chunk] @ X.T
        sims[np.arange(len(chunk)), chunk] = -np.inf
        hi = np.argpartition(-sims, k, axis=1)[:, :k]

        d = ((Y[chunk][:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        d[np.arange(len(chunk)), chunk] = np.inf
        lo = np.argpartition(d, k, axis=1)[:, :k]

        for r in range(len(chunk)):
            keep += len(set(hi[r].tolist()) & set(lo[r].tolist())) / k
    return keep / len(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--tree", default="tree_rleiden_labelled")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    g = load_games()
    X, appids, _ = load_vectors(args.variant)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    tree = json.loads((CLUSTER_DATA / f"{args.tree}.json").read_text(encoding="utf-8"))
    y = depth1_labels(tree, len(g))

    found = [("baseline", DATA / "coords.npz")]
    found += [(p.stem.replace("coords_", ""), p)
              for p in sorted(DATA.glob("coords_*.npz"))]

    print("=" * 70)
    print("LAYOUT FIDELITY  (does the layout still carry the data's geometry?)")
    print("=" * 70)
    print(f"{'layout':<20}{'territory arrangement':>24}{'neighbour overlap':>22}")
    print(f"{'':<20}{'(Spearman, Phase 7)':>24}{'(10-NN, Phase 6)':>22}")
    print("-" * 70)

    out = {}
    for name, path in found:
        if not path.exists():
            continue
        Y = np.load(path, allow_pickle=False)["Y"].astype(np.float64)
        tf = territory_fidelity(X, Y, y)
        nf = neighbour_fidelity(X, Y, args.k, args.sample)
        out[name] = {"territory_spearman": tf["spearman"], "knn_overlap": nf}
        print(f"{name:<20}{tf['spearman']:>24.3f}{nf * 100:>21.1f}%")

    (DATA / "fidelity.json").write_text(json.dumps(out), encoding="utf-8")
    print("\nA layout that invented its own geometry scores well on purity and")
    print("badly here. Both numbers must hold up for the map to be honest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
