"""Variant C: hybrid. L2-normalise A and B separately, then concatenate as
[alpha * A, (1 - alpha) * B].

### The thing that is easy to get wrong

Both halves are unit vectors, so after scaling, the cosine between two hybrid
vectors is not a linear blend of the two cosines. Writing a = cos_A(x,y) and
b = cos_B(x,y):

    numerator   = alpha^2 * a + (1-alpha)^2 * b
    denominator = alpha^2 + (1-alpha)^2          (both vectors have this norm)

    cos_C = (alpha^2 * a + (1-alpha)^2 * b) / (alpha^2 + (1-alpha)^2)

The weights are **quadratic** in alpha, not linear. So alpha=0.7 is not "70%
text": it is 0.49 vs 0.09, i.e. 84% text and 16% tags. Only alpha=0.5 gives the
equal blend its number suggests. `effective_weights()` prints this, because
reporting "alpha=0.7" as though it meant 70/30 would misdescribe the result.

Dimensionality: A is 384-d and B is 128-d, so C is 512-d. The *count* of
dimensions carries no weight of its own here -- each half is unit-norm before
scaling, so a 384-d half and a 128-d half contribute equally at alpha=0.5.

Usage:
    python -m embed.variant_c --a variant_a_int8_mean --b variant_b
    python -m embed.variant_c --alphas 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .common import load_games, load_vectors, save_vectors


def effective_weights(alpha: float) -> tuple[float, float]:
    """Actual contribution of each half to the hybrid cosine.

    Returns (text_share, tag_share), summing to 1.
    """
    wa, wb = alpha ** 2, (1.0 - alpha) ** 2
    total = wa + wb
    return wa / total, wb / total


def build(XA: np.ndarray, XB: np.ndarray, alpha: float) -> np.ndarray:
    """Concatenate scaled halves. Inputs must already be L2-normalised."""
    if XA.shape[0] != XB.shape[0]:
        raise ValueError(f"row mismatch: A has {XA.shape[0]}, B has {XB.shape[0]}")
    return np.hstack([alpha * XA, (1.0 - alpha) * XB]).astype(np.float32)


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="variant_a_int8_mean", help="text variant name")
    ap.add_argument("--b", default="variant_b", help="tag variant name")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--prefix", default="variant_c")
    args = ap.parse_args()

    g = load_games()
    XA, ap_ids, meta_a = load_vectors(args.a)
    XB, bp_ids, meta_b = load_vectors(args.b)

    # Row alignment is the failure this check exists to prevent: a silent
    # mismatch pairs one game's text vector with another game's tag vector and
    # every downstream number still looks plausible.
    if not np.array_equal(ap_ids, bp_ids):
        print("appid order differs between A and B; refusing to concatenate")
        return 1
    if not np.array_equal(ap_ids, g.appids):
        print("appid order differs from the catalog; refusing to concatenate")
        return 1

    XA, XB = _unit(XA), _unit(XB)
    print(f"A  {args.a:<24} {XA.shape}  {meta_a.get('method', '')}")
    print(f"B  {args.b:<24} {XB.shape}  {meta_b.get('method', '')}")
    print()

    for alpha in args.alphas:
        X = build(XA, XB, alpha)
        ta, tb = effective_weights(alpha)
        name = f"{args.prefix}_a{str(alpha).replace('.', '')}"
        # A hybrid is only clean for tag-prediction if BOTH halves are: a
        # leaky text half would reintroduce the masked tag through the
        # concatenation, however clean the tag half is.
        clean = (
            bool(meta_a.get("holdout")) or "holdout" in args.a
        ) and (
            bool(meta_b.get("holdout")) or "holdout" in args.b
        )
        meta = {
            "variant": "C",
            "method": f"hybrid [{alpha}*A, {1 - alpha}*B]",
            "dims": int(X.shape[1]),
            "alpha": alpha,
            "effective_text_share": ta,
            "effective_tag_share": tb,
            "source_a": args.a,
            "source_b": args.b,
            "holdout": clean,
            "n_games": len(g),
        }
        save_vectors(name, X, g.appids, meta)
        print(f"  alpha={alpha:<4}  dims={X.shape[1]}   "
              f"effective weights: text {ta * 100:4.1f}% / tags {tb * 100:4.1f}%"
              f"   -> {name}")

    print("\nNote: weights are quadratic in alpha, so alpha=0.7 means 84/16,")
    print("not 70/30. Only alpha=0.5 is the equal blend it looks like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
