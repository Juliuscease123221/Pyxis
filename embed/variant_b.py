"""Variant B: tag co-occurrence embeddings. PPMI + truncated SVD, no neural model.

Pipeline: sparse games x tags matrix -> PPMI weighting -> truncated SVD to 128
dims -> L2 normalise.

Why PPMI rather than raw counts or plain TF-IDF. The matrix is dominated by a
handful of near-universal tags: "Indie" is on 33,318 of 56,129 games and
"Singleplayer" on 33,365. Those carry almost no information about what a game
*is*. Pointwise mutual information divides the observed co-occurrence by what
independence would predict, so a tag that appears everywhere scores near zero
while a tag that appears on a specific, coherent set of games scores high. The
positive clip (PPMI) discards the negative half, which is the noisy half:
"these two tags co-occur less than chance" is a weak, high-variance signal on
sparse data.

Tag rank as a weight. The dump gives tag *names in vote order* but not the vote
counts themselves, so exact vote weights are unavailable. Rank is the only
surviving trace of the vote distribution. Hollow Knight's first tag is
Metroidvania and its seventh is Indie; treating those as equally true throws
away real signal. `--weight rank` applies a 1/(1+rank) decay before PPMI.
Binary weighting is kept as a baseline and both are reported, because the
decay's shape is a guess and the comparison is the honest way to present it.

Usage:
    python -m embed.variant_b                  # rank-weighted, 128 dims
    python -m embed.variant_b --weight binary
    python -m embed.variant_b --dims 64
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .common import Games, choose_masked_tags, l2_normalize, load_games, save_vectors


def build_matrix(g: Games, weight: str = "rank", decay: float = 1.0,
                 masked: dict[int, str] | None = None
                 ) -> tuple[sparse.csr_matrix, list[str]]:
    """Games x tags sparse matrix.

    binary  every tag on a game counts 1.0
    rank    tag at position r counts 1/(1 + decay*r), so the most-voted tag
            dominates and the long tail still contributes
    """
    masked = masked or {}
    vocab: dict[str, int] = {}
    for tags in g.tags:
        for t in tags:
            if t not in vocab:
                vocab[t] = len(vocab)

    rows, cols, vals = [], [], []
    for i, tags in enumerate(g.tags):
        hide = masked.get(i)
        for r, t in enumerate(tags):
            if t == hide:
                continue  # held out at BUILD time, so scoring it is not circular
            rows.append(i)
            cols.append(vocab[t])
            vals.append(1.0 if weight == "binary" else 1.0 / (1.0 + decay * r))

    M = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(len(g), len(vocab)),
    )
    names = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return M, names


def ppmi(M: sparse.csr_matrix, smoothing: float = 0.75) -> sparse.csr_matrix:
    """Positive pointwise mutual information over a games x tags matrix.

        PMI(g,t) = log( P(g,t) / (P(g) * P(t)^alpha) )
        PPMI     = max(PMI, 0)

    `smoothing` is Levy & Goldberg's context distribution smoothing: raising the
    tag marginal to alpha<1 flattens it, which damps the boost that PMI
    otherwise gives to very rare tags. Without it the rarest tags in the
    vocabulary -- some appearing on a handful of games -- dominate the SVD.
    """
    M = M.tocsr().astype(np.float64)
    total = M.sum()
    if total == 0:
        raise ValueError("empty tag matrix")

    row_sum = np.asarray(M.sum(axis=1)).ravel()   # P(game)
    col_sum = np.asarray(M.sum(axis=0)).ravel()   # P(tag)

    p_row = row_sum / total
    p_col = col_sum / total
    p_col_s = p_col ** smoothing
    p_col_s = p_col_s / p_col_s.sum()

    out = M.tocoo()
    p_joint = out.data / total
    denom = p_row[out.row] * p_col_s[out.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(np.where(denom > 0, p_joint / denom, 1.0))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0] = 0.0  # positive clip

    keep = pmi > 0
    return sparse.csr_matrix(
        (pmi[keep], (out.row[keep], out.col[keep])), shape=M.shape
    )


def build(g: Games, dims: int = 128, weight: str = "rank",
          smoothing: float = 0.75, seed: int = 0,
          masked: dict[int, str] | None = None) -> tuple[np.ndarray, dict]:
    """Full variant B pipeline. Returns (vectors, metadata)."""
    t0 = time.perf_counter()
    M, vocab = build_matrix(g, weight=weight, masked=masked)
    t_matrix = time.perf_counter() - t0

    t0 = time.perf_counter()
    P = ppmi(M, smoothing=smoothing)
    t_ppmi = time.perf_counter() - t0

    # dims must stay under the vocabulary size
    k = min(dims, min(P.shape) - 1)
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=k, random_state=seed, algorithm="randomized",
                       n_iter=7)
    X = svd.fit_transform(P)
    t_svd = time.perf_counter() - t0

    X = l2_normalize(X)
    ev = float(svd.explained_variance_ratio_.sum())

    meta = {
        "variant": "B",
        "method": "tag co-occurrence PPMI + truncated SVD",
        "dims": int(k),
        "weight": weight,
        "smoothing": smoothing,
        "vocab_size": len(vocab),
        "n_games": len(g),
        "holdout": bool(masked),
        "matrix_nnz": int(M.nnz),
        "ppmi_nnz": int(P.nnz),
        "density": float(M.nnz / (M.shape[0] * M.shape[1])),
        "explained_variance": ev,
        "seconds": {
            "matrix": round(t_matrix, 2),
            "ppmi": round(t_ppmi, 2),
            "svd": round(t_svd, 2),
        },
    }
    return X, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--weight", choices=["rank", "binary"], default="rank")
    ap.add_argument("--smoothing", type=float, default=0.75)
    ap.add_argument("--name", default=None, help="output name (default variant_b)")
    ap.add_argument("--holdout", action="store_true",
                    help="remove each game's masked eval tag before building,"
                         " making tag-prediction non-circular")
    args = ap.parse_args()

    print("loading games...")
    g = load_games()
    print(f"  {len(g):,} games")

    masked = choose_masked_tags(g) if args.holdout else None
    if masked:
        print(f'  holding out 1 tag for {len(masked):,} games')

    X, meta = build(g, dims=args.dims, weight=args.weight,
                    smoothing=args.smoothing, masked=masked)

    name = args.name or ("variant_b_holdout" if args.holdout else "variant_b")
    path = save_vectors(name, X, g.appids, meta)

    print(f"\nvariant B  [{args.weight}-weighted]")
    print(f"  vocabulary       {meta['vocab_size']:,} tags")
    print(f"  matrix           {meta['n_games']:,} x {meta['vocab_size']:,}"
          f"  nnz={meta['matrix_nnz']:,}  density={meta['density'] * 100:.2f}%")
    print(f"  ppmi nnz         {meta['ppmi_nnz']:,}"
          f"  ({meta['ppmi_nnz'] / meta['matrix_nnz'] * 100:.1f}% survived the positive clip)")
    print(f"  dims             {meta['dims']}")
    print(f"  explained var    {meta['explained_variance'] * 100:.1f}%")
    print(f"  time             matrix {meta['seconds']['matrix']}s"
          f"  ppmi {meta['seconds']['ppmi']}s  svd {meta['seconds']['svd']}s")
    print(f"  saved            {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
