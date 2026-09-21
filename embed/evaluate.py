"""Phase 2 evaluation: spot checks, genre agreement, tag prediction.

Run this before building anything downstream. Bad embeddings produce a
beautiful meaningless map, and the failure is not visible in the map.

Three metrics, in order of how much they should be trusted:

1. **Spot checks.** 10 nearest neighbours for 15 well-known games, read by a
   human. Crude, but it is the only metric that catches the specific failure
   SPEC.md warns about -- an embedding that has learned marketing register
   rather than gameplay. Two games can share a genre and a dozen tags and still
   be nothing alike.

2. **Genre agreement.** Fraction of each game's 10-NN sharing >=1 Steam genre,
   against a random-pairs baseline. Read the *lift over baseline*, not the
   absolute number: with "Indie" on 59% of the catalog, random pairs already
   agree often.

3. **Tag prediction.** Mask one tag per game, check whether the neighbours'
   tags recover it, precision@5.

### Leakage, stated plainly

Both automated metrics leak, in different directions, and the numbers are not
comparable across variants unless that is handled:

- Variant B is built *from tags*. Scoring it on tag prediction with the masked
  tag still present in its input is circular. `--holdout` rebuilds B with the
  masked tags removed at build time, which makes the comparison fair.
- Variant A's text template *contains the genre list*, so genre agreement is
  partly circular for A.
- Steam's genre names ("Indie", "Action", "RPG", "Strategy") are also tag
  names, so genre information reaches B through the tag vocabulary regardless.
  `--strip-genre-tags` removes the overlapping tags from B's input to measure
  how much of its genre agreement is real structure rather than copying.

Usage:
    python -m embed.evaluate --variant variant_b
    python -m embed.evaluate --variant variant_b --spot-only
    python -m embed.evaluate --variant variant_b variant_a variant_c
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .common import Games, choose_masked_tags, load_games, load_vectors

# Well-known games with unambiguous neighbourhoods a person can check by eye.
# Chosen to span genres, so a variant that collapses everything into one blob
# is visible immediately.
PROBES: dict[int, str] = {
    367520: "Hollow Knight",          # expect Ori, Blasphemous, Dead Cells
    413150: "Stardew Valley",         # expect farming / life sim
    1145360: "Hades",                 # expect roguelites
    236850: "Europa Universalis IV",  # expect grand strategy
    570: "Dota 2",                    # expect MOBA
    252490: "Rust",                   # expect survival
    620: "Portal 2",                  # expect first-person puzzle
    391540: "Undertale",              # expect story-rich RPG
    105600: "Terraria",               # expect sandbox / crafting
    268910: "Cuphead",                # expect hard run-and-gun platformers
    322330: "Don't Starve Together",  # expect co-op survival
    546560: "Half-Life: Alyx",        # expect VR shooters
    1174180: "Red Dead Redemption 2", # expect open-world action
    377160: "Fallout 4",              # expect open-world RPG
    892970: "Valheim",                # expect survival craft
}


def knn(X: np.ndarray, idx: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """Top-k cosine neighbours for the rows in `idx`, excluding self.

    X must be L2-normalised, so a dot product is the cosine. Blocked to keep
    the similarity matrix off the heap: 56k x 56k float32 would be 12.6 GB.
    """
    out = np.empty((len(idx), k), dtype=np.int32)
    for s in range(0, len(idx), block):
        chunk = idx[s:s + block]
        sims = X[chunk] @ X.T                       # (b, n)
        sims[np.arange(len(chunk)), chunk] = -np.inf  # drop self
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), axis=1
        )
        out[s:s + len(chunk)] = ordered
    return out


def spot_check(g: Games, X: np.ndarray, k: int = 10) -> int:
    """Print k nearest neighbours for each probe game. Returns probes found."""
    found = [(a, n) for a, n in PROBES.items() if g.index_of(a) is not None]
    if not found:
        print("  no probe games present in the catalog")
        return 0

    idx = np.asarray([g.index_of(a) for a, _ in found], dtype=np.int64)
    nn = knn(X, idx, k)

    for row, (appid, label) in enumerate(found):
        i = idx[row]
        print(f"\n  {g.names[i]}")
        print(f"    tags: {', '.join(g.tags[i][:6])}")
        for rank, j in enumerate(nn[row], 1):
            sim = float(X[i] @ X[j])
            print(f"    {rank:>2}. {sim:.3f}  {g.names[j][:48]:<48}"
                  f"  {', '.join(g.tags[j][:3])}")
    return len(found)


def genre_agreement(g: Games, X: np.ndarray, k: int = 10, sample: int = 4000,
                    seed: int = 0) -> dict:
    """Fraction of k-NN sharing >=1 genre, vs a random-pairs baseline."""
    rng = np.random.default_rng(seed)
    have = np.asarray([i for i in range(len(g)) if g.genres[i]], dtype=np.int64)
    if len(have) == 0:
        return {"agreement": float("nan"), "baseline": float("nan"), "lift": float("nan")}
    idx = rng.choice(have, size=min(sample, len(have)), replace=False)
    nn = knn(X, idx, k)

    gsets = [set(x) for x in g.genres]
    hits = tot = 0
    for row, i in enumerate(idx):
        gi = gsets[i]
        for j in nn[row]:
            if gsets[j]:
                tot += 1
                if gi & gsets[j]:
                    hits += 1
    agreement = hits / tot if tot else float("nan")

    # baseline: random pairs drawn from the same pool
    b_hits = b_tot = 0
    a = rng.choice(have, size=sample * k)
    b = rng.choice(have, size=sample * k)
    for i, j in zip(a, b):
        if i == j:
            continue
        b_tot += 1
        if gsets[i] & gsets[j]:
            b_hits += 1
    baseline = b_hits / b_tot if b_tot else float("nan")

    return {
        "agreement": agreement,
        "baseline": baseline,
        "lift": agreement / baseline if baseline else float("nan"),
        "n_sampled": int(len(idx)),
    }


def tag_prediction(g: Games, X: np.ndarray, masked: dict[int, str], k: int = 10,
                   top: int = 5, sample: int = 4000, seed: int = 0) -> dict:
    """Precision@`top`: does a game's masked tag appear in its neighbours' tags?

    `masked` maps row index -> the held-out tag. For a fair score the vectors
    should have been built without those tags (see --holdout); otherwise the
    number is inflated for tag-derived variants and this is reported as such.
    """
    rng = np.random.default_rng(seed)
    candidates = np.asarray(sorted(masked.keys()), dtype=np.int64)
    if len(candidates) == 0:
        return {"precision_at_k": float("nan"), "n_sampled": 0}
    idx = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
    nn = knn(X, idx, k)

    hits = 0
    for row, i in enumerate(idx):
        target = masked[int(i)]
        known = set(g.tags[i]) - {target}
        votes: dict[str, float] = {}
        for rank, j in enumerate(nn[row]):
            w = 1.0 / (1 + rank)
            for t in g.tags[j]:
                if t in known:
                    continue  # already known; not a prediction
                votes[t] = votes.get(t, 0.0) + w
        pred = [t for t, _ in sorted(votes.items(), key=lambda kv: -kv[1])[:top]]
        if target in pred:
            hits += 1

    return {
        "precision_at_k": hits / len(idx),
        "n_sampled": int(len(idx)),
        "top": top,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", nargs="+", default=["variant_b"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--spot-only", action="store_true")
    args = ap.parse_args()

    g = load_games()
    print(f"catalog: {len(g):,} games\n")

    for name in args.variant:
        X, appids, meta = load_vectors(name)
        if not np.array_equal(appids, g.appids):
            print(f"!! {name}: appid order does not match the catalog; skipping")
            continue
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

        print("=" * 78)
        print(f"{name}   dims={X.shape[1]}   {meta.get('method', '')}")
        print("=" * 78)

        print("\nSPOT CHECKS (10 nearest neighbours)")
        spot_check(g, X, k=args.k)

        if args.spot_only:
            print()
            continue

        print("\n\nGENRE AGREEMENT")
        ga = genre_agreement(g, X, k=args.k, sample=args.sample)
        print(f"  {ga['agreement'] * 100:.1f}% of {args.k}-NN share >=1 genre")
        print(f"  {ga['baseline'] * 100:.1f}% baseline (random pairs)")
        print(f"  lift {ga['lift']:.2f}x")

        print("\nTAG PREDICTION")
        masked = choose_masked_tags(g)
        tp = tag_prediction(g, X, masked, k=args.k, sample=args.sample)
        print(f"  precision@{tp['top']} = {tp['precision_at_k'] * 100:.1f}%"
              f"  (n={tp['n_sampled']:,})")
        if meta.get("variant") == "B" and not meta.get("holdout"):
            print("  NOTE: variant B was built from tags including the masked one;")
            print("        this number is circular. Rebuild with --holdout to compare.")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
