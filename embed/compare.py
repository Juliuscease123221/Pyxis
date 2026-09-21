"""Score every built variant on the same metrics and emit embed/variants.md.

The Phase 2 acceptance is a written comparison, not a winner. This produces the
table; the reasoning is written underneath it by hand.

Fairness rules enforced here:

* Every variant is scored against the *same* held-out tag set
  (`choose_masked_tags`, seeded), so tag-prediction numbers are comparable.
* A variant is only allowed into the tag-prediction column if it was built
  without the masked tags. Variants that saw them are reported separately as
  "circular", never in the same column, because the inflation is large: for
  variant B it was 80.5% vs 52.9%, a 27.6-point gap.
* Genre agreement is reported as lift over a random-pairs baseline. The
  absolute number is close to meaningless -- "Indie" alone is on 59% of the
  catalog, so random pairs agree 75.4% of the time.

Usage:
    python -m embed.compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .common import VEC_DIR, choose_masked_tags, load_games, load_vectors
from .evaluate import PROBES, genre_agreement, knn, tag_prediction

# Variants built with the masked tags removed, so their tag-prediction score is
# honest. Anything else is reported as circular.
HOLDOUT_SUFFIX = "_holdout"


def probe_neighbours(g, X, appid: int, k: int = 5) -> list[str]:
    i = g.index_of(appid)
    if i is None:
        return []
    nn = knn(X, np.asarray([i]), k)[0]
    return [g.names[j] for j in nn]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "variants.md"))
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()

    g = load_games()
    masked = choose_masked_tags(g)
    names = sorted(p.stem for p in VEC_DIR.glob("*.npz"))
    if not names:
        print("no vectors built yet")
        return 1

    rows = []
    for name in names:
        X, appids, meta = load_vectors(name)
        if not np.array_equal(appids, g.appids):
            print(f"  skip {name}: appid order mismatch")
            continue
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

        ga = genre_agreement(g, X, k=10, sample=args.sample)
        tp = tag_prediction(g, X, masked, k=10, sample=args.sample)
        clean = bool(meta.get("holdout")) or name.endswith(HOLDOUT_SUFFIX)

        rows.append({
            "name": name,
            "variant": meta.get("variant", "?"),
            "dims": X.shape[1],
            "method": meta.get("method", ""),
            "genre_lift": ga["lift"],
            "genre_raw": ga["agreement"],
            "tagp": tp["precision_at_k"],
            "clean": clean,
            "gps": meta.get("games_per_sec"),
            "hk": probe_neighbours(g, X, 367520, 5),
        })
        flag = "" if clean else "  (circular)"
        print(f"  {name:<28} lift {ga['lift']:.2f}x   "
              f"tag@5 {tp['precision_at_k'] * 100:5.1f}%{flag}")

    baseline = genre_agreement(g, load_and_unit(names[0], g), sample=args.sample)["baseline"]
    write_report(args.out, g, rows, baseline)
    print(f"\nwrote {args.out}")
    return 0


def load_and_unit(name, g):
    X, _, _ = load_vectors(name)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def write_report(path, g, rows, baseline) -> None:
    clean = [r for r in rows if r["clean"]]
    dirty = [r for r in rows if not r["clean"]]

    L = []
    L.append("# Phase 2 — embedding variants\n")
    L.append(f"Catalog: **{len(g):,} games**. All variants share the same row "
             "order (appid ascending) and are scored against the same seeded "
             "held-out tag set.\n")

    L.append("\n## How to read these numbers\n")
    L.append("**Genre agreement** is the fraction of each game's 10 nearest "
             "neighbours sharing at least one Steam genre. The absolute value is "
             "nearly meaningless: `Indie` alone is on 59% of the catalog, so "
             f"random pairs already agree **{baseline * 100:.1f}%** of the time. "
             "Read the lift column.\n")
    L.append("\n**Tag prediction** masks one tag per game (drawn from the middle "
             "of the vote ranking) and asks whether the 10 nearest neighbours' "
             "tags recover it, precision@5. It is only meaningful for variants "
             "built *without* those tags — see the circularity note below.\n")

    L.append("\n## Results\n")
    L.append("| variant | dims | genre lift | tag pred @5 | method |")
    L.append("| --- | ---: | ---: | ---: | --- |")
    for r in sorted(clean, key=lambda r: -r["tagp"]):
        L.append(f"| `{r['name']}` | {r['dims']} | {r['genre_lift']:.2f}x | "
                 f"{r['tagp'] * 100:.1f}% | {r['method']} |")

    if dirty:
        L.append("\n### Scored, but circular on tag prediction\n")
        L.append("These variants were built from inputs containing the masked "
                 "tag, so their tag-prediction score is inflated and is **not** "
                 "comparable with the table above. Listed for completeness.\n")
        L.append("| variant | dims | genre lift | tag pred @5 (inflated) | method |")
        L.append("| --- | ---: | ---: | ---: | --- |")
        for r in sorted(dirty, key=lambda r: -r["tagp"]):
            L.append(f"| `{r['name']}` | {r['dims']} | {r['genre_lift']:.2f}x | "
                     f"{r['tagp'] * 100:.1f}% | {r['method']} |")

    L.append("\n## Spot check: Hollow Knight's 5 nearest neighbours\n")
    L.append("The metric that catches an embedding which has learned marketing "
             "register rather than gameplay.\n")
    L.append("| variant | neighbours |")
    L.append("| --- | --- |")
    for r in rows:
        L.append(f"| `{r['name']}` | {', '.join(r['hk'][:5])} |")

    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
