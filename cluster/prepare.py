"""Phase 3 step 1: PCA the chosen embedding down to ~50 dims.

HDBSCAN degrades in high dimensions -- mutual reachability distances become
concentrated and the density estimate loses contrast. SPEC.md asks for ~50
dims keeping ~90% variance.

A diagnostic worth having on a hybrid input: variant C is
`[0.5 * A_384d, 0.5 * B_128d]`. Both halves carry the same total variance by
construction (each is unit-norm before the 0.5 scaling, so each contributes
0.25 of squared norm per row), but they spread it very differently -- the tag
half packs it into 128 dims, the text half into 384. PCA takes directions in
order of variance, so it will preferentially pick up the *denser* half first.
`half_mass()` reports how the retained components actually split, so the
clustering input is not silently 90% one modality.

Usage:
    python -m cluster.prepare                        # variant_c_a05 -> 50 dims
    python -m cluster.prepare --target-variance 0.90 # choose dims by variance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from embed.common import load_games, load_vectors

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "cluster"

# Dimensions of variant C's two halves, for the split diagnostic.
TEXT_DIMS = 384
TAG_DIMS = 128


def half_mass(components: np.ndarray, text_dims: int = TEXT_DIMS) -> tuple[float, float]:
    """Fraction of retained component loading coming from each half of a hybrid.

    Each PCA component is a unit vector over the 512 input dims. Summing the
    squared loadings over the text block vs the tag block says which modality
    the retained subspace is actually built from.
    """
    if components.shape[1] <= text_dims:
        return 1.0, 0.0
    text = float((components[:, :text_dims] ** 2).sum())
    tags = float((components[:, text_dims:] ** 2).sum())
    total = text + tags
    return text / total, tags / total


def prepare_balanced(X: np.ndarray, dims: int, seed: int = 0
                     ) -> tuple[np.ndarray, dict]:
    """PCA each half of the hybrid separately, to dims/2 each, then rejoin.

    Plain PCA over the concatenated vector does not preserve the alpha=0.5
    blend. Both halves carry equal total variance, but the tag half packs it
    into 128 dims and the text half spreads it over 384, so the leading
    components are overwhelmingly tag directions: at 50 dims the retained
    subspace is 85.8% tag loading. Clustering that is clustering variant B,
    whatever alpha said.

    Reducing each half on its own budget and L2-normalising both before
    rejoining keeps the two modalities at parity in the clustering input, which
    is what alpha=0.5 was chosen to mean.
    """
    half = dims // 2
    out, meta = [], {}
    for label, block, lo, hi in (("text", "A", 0, TEXT_DIMS),
                                 ("tag", "B", TEXT_DIMS, TEXT_DIMS + TAG_DIMS)):
        sub = X[:, lo:hi]
        p = PCA(n_components=min(half, sub.shape[1] - 1), random_state=seed,
                svd_solver="randomized")
        Y = p.fit_transform(sub)
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
        out.append(Y)
        meta[f"{label}_variance"] = float(p.explained_variance_ratio_.sum())
        meta[f"{label}_dims"] = int(Y.shape[1])
    return np.ascontiguousarray(np.hstack(out), dtype=np.float32), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="variant_c_a05")
    ap.add_argument("--dims", type=int, default=50)
    ap.add_argument("--target-variance", type=float, default=None,
                    help="if set, pick the smallest dims reaching this variance")
    ap.add_argument("--mode", choices=["joint", "balanced"], default="joint",
                    help="joint = PCA the concatenated vector (lets the denser"
                         " half dominate); balanced = PCA each half separately")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = load_games()
    X, appids, meta = load_vectors(args.variant)
    if not np.array_equal(appids, g.appids):
        print("appid order mismatch between vectors and catalog")
        return 1

    print(f"input   {args.variant}  {X.shape}  {meta.get('method', '')}")
    print(f"mode    {args.mode}")

    if args.mode == "balanced":
        Xp, bmeta = prepare_balanced(X, args.dims, args.seed)
        print(f"\n  text half -> {bmeta['text_dims']} dims"
              f"  ({bmeta['text_variance'] * 100:.1f}% of its own variance)")
        print(f"  tag half  -> {bmeta['tag_dims']} dims"
              f"  ({bmeta['tag_variance'] * 100:.1f}% of its own variance)")
        print("  each half L2-normalised before rejoining, so the two modalities")
        print("  enter the clustering input at parity")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else OUT_DIR / "pca_balanced.npz"
        np.savez_compressed(
            out, X=Xp, appids=appids,
            meta=json.dumps({
                "source_variant": args.variant, "mode": "balanced",
                "dims": int(Xp.shape[1]), "n_games": int(X.shape[0]), **bmeta,
            }),
        )
        print(f"\nsaved   {out}   {Xp.shape}")
        return 0

    # Centre before PCA. sklearn does this internally, but we need the mean
    # later to project anything new into the same space.
    t0 = time.perf_counter()
    probe_dims = max(args.dims, 128)
    pca = PCA(n_components=probe_dims, random_state=args.seed, svd_solver="randomized")
    Xp_full = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    elapsed = time.perf_counter() - t0

    if args.target_variance:
        k = int(np.searchsorted(cum, args.target_variance) + 1)
        k = min(k, probe_dims)
    else:
        k = args.dims

    Xp = np.ascontiguousarray(Xp_full[:, :k], dtype=np.float32)

    print(f"pca     fit {probe_dims} comps in {elapsed:.1f}s")
    print()
    print("  variance retained by dimension count:")
    for d in (10, 25, 50, 75, 100, probe_dims):
        if d <= probe_dims:
            mark = "  <- chosen" if d == k else ""
            print(f"    {d:>4} dims   {cum[d - 1] * 100:5.1f}%{mark}")
    if k not in (10, 25, 50, 75, 100, probe_dims):
        print(f"    {k:>4} dims   {cum[k - 1] * 100:5.1f}%  <- chosen")

    tshare, gshare = half_mass(pca.components_[:k])
    print()
    print(f"  retained subspace composition:")
    print(f"    text half (A, {TEXT_DIMS}d)   {tshare * 100:5.1f}% of loading mass")
    print(f"    tag half  (B, {TAG_DIMS}d)    {gshare * 100:5.1f}% of loading mass")
    if gshare > 0.75 or tshare > 0.75:
        heavier = "tag" if gshare > tshare else "text"
        print(f"    NOTE: the retained subspace is {heavier}-dominated despite the")
        print(f"          two halves carrying equal variance at alpha=0.5.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "pca.npz"
    np.savez_compressed(
        out,
        X=Xp,
        appids=appids,
        meta=json.dumps({
            "source_variant": args.variant,
            "mode": "joint",
            "dims": int(k),
            "explained_variance": float(cum[k - 1]),
            "text_loading_share": tshare,
            "tag_loading_share": gshare,
            "n_games": int(X.shape[0]),
            "seconds": round(elapsed, 2),
        }),
    )
    print(f"\nsaved   {out}   {Xp.shape}  {cum[k - 1] * 100:.1f}% variance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
