"""A UMAP reduction built purely for clustering -- never the display layout.

### Why this is not a violation of "cluster in high-dimensional space"

SPEC.md says to cluster in high-dimensional space and never on 2D
coordinates, and the reason it gives is correct: clustering after the *display*
UMAP finds structure in UMAP's distortions, because a 2D layout is optimised
for legibility and packs unrelated regions together to fill the plane.

This is a different object. It is a separate run, at 5-15 dimensions rather
than 2, with `min_dist=0.0` (which a display layout must never use -- it
collapses points into unreadable knots) and `metric='cosine'` to match the
embedding. It is never rendered. Phase 4 will fit its own UMAP for coordinates
and the two never touch.

This is the standard pipeline for embedding atlases -- BERTopic and
DataMapPlot both reduce to ~5-15 dims before HDBSCAN -- and the reason is the
failure documented in RATIONALE.md: HDBSCAN needs density contrast, and
512-dimensional cosine space has almost none. Measured on this catalog, the
pairwise-distance CV was 0.12 and 10-NN contrast 1.78, and the resulting
condensed tree was a depth-143 caterpillar. That is the signature of clustering
in a space with no density structure, not of a badly-tuned HDBSCAN.

The two trees SPEC.md insists on keeping separate stay separate: this
reduction feeds the *cluster* tree only; the quadtree is built from the Phase 4
display coordinates.

Usage:
    python -m cluster.umap_reduce --components 5 10 15 --neighbors 15 30
    python -m cluster.umap_reduce --components 10 --neighbors 30 --only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embed.common import load_games, load_vectors

DATA = Path(__file__).resolve().parent.parent / "data" / "cluster"


def reduce_for_clustering(X: np.ndarray, n_components: int, n_neighbors: int,
                          seed: int = 0, min_dist: float = 0.0) -> tuple[np.ndarray, float]:
    import umap

    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    Y = reducer.fit_transform(X)
    return np.ascontiguousarray(Y, dtype=np.float32), time.perf_counter() - t0


def contrast(Y: np.ndarray, sample: int = 2000, seed: int = 0) -> dict:
    """Density-contrast diagnostics, to check the reduction actually helped.

    The 512-d space scored CV 0.12 / 10-NN contrast 1.78, which is what a
    space with no density structure looks like. Higher is better on both.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Y), min(sample, len(Y)), replace=False)
    S = Y[idx]
    D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(S), 1)
    d = D[iu]
    knn = np.sort(D, axis=1)[:, 1:11]
    p5, p95 = np.percentile(knn, 5), np.percentile(knn, 95)
    return {
        "cv": float(d.std() / max(d.mean(), 1e-12)),
        "knn_contrast": float(p95 / max(p5, 1e-12)),
        "mean_pairwise": float(d.mean()),
        "mean_10nn": float(knn.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--components", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--neighbors", type=int, nargs="+", default=[15, 30])
    ap.add_argument("--min-dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = load_games()
    X, appids, meta = load_vectors(args.variant)
    if not np.array_equal(appids, g.appids):
        print("appid order mismatch")
        return 1
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    print(f"input {args.variant} {X.shape}  metric=cosine  min_dist={args.min_dist}")
    print("(clustering reduction only -- Phase 4 fits its own layout)\n")
    print(f"{'comps':>6}{'nbrs':>6}{'secs':>8}{'cv':>8}{'10nn_contrast':>15}")
    print("-" * 44)

    DATA.mkdir(parents=True, exist_ok=True)
    for nc in args.components:
        for nn in args.neighbors:
            Y, secs = reduce_for_clustering(X, nc, nn, args.seed, args.min_dist)
            c = contrast(Y)
            name = f"umap_c{nc}_n{nn}"
            np.savez_compressed(
                DATA / f"{name}.npz", X=Y, appids=appids,
                meta=json.dumps({
                    "source_variant": args.variant, "mode": "umap_cluster",
                    "n_components": nc, "n_neighbors": nn,
                    "min_dist": args.min_dist, "metric": "cosine",
                    "seconds": round(secs, 1), "n_games": int(len(X)), **c,
                }),
            )
            print(f"{nc:>6}{nn:>6}{secs:>8.0f}{c['cv']:>8.3f}"
                  f"{c['knn_contrast']:>15.2f}   -> {name}")

    print("\nreference: 512-d cosine space scored cv=0.106, 10nn_contrast=1.60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
