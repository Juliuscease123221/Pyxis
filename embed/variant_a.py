"""Variant A: text embeddings from bge-small-en-v1.5 via ONNX Runtime on CPU.

No GPU on this machine, so the SPEC.md plan of "batch 256 on GPU" is not
available. Instead: ONNX Runtime, batch 32, max_length capped at 192 tokens,
with int8 dynamic quantisation available and measured against fp32.

Why each of those:

* **ONNX over PyTorch.** ONNX Runtime's CPU execution provider is faster than
  eager PyTorch for inference-only transformer workloads, and it drops the
  torch dependency entirely -- this project never trains anything.
* **int8 dynamic quantisation, measured not assumed.** Weights become int8 and
  dequantise per-op; the model goes 133MB -> 34MB and matmuls run on integer
  kernels. `--compare-fp32` reports both the per-game cosine against fp32 *and*
  the top-10 neighbourhood overlap, because a high mean cosine can still hide
  heavy neighbour reordering -- and neighbour order is the entire product here.
* **max_length 192.** Steam long descriptions run to thousands of tokens, and
  past the opening paragraph they are feature bullets, system requirements and
  award quotes: marketing filler that adds tokens without adding signal.
  Attention cost is quadratic in sequence length, so the cap is both a quality
  and a throughput decision. Measured token lengths are mean 89 / p95 118, so
  192 truncates only a long tail.
* **Length-sorted batching.** Padding is pure waste: a batch mixing a 20-token
  and a 192-token input pads everything to 192. Sorting by token length before
  batching groups similar lengths, then the original order is restored on
  write-back. Measured at 1.25x, with padding waste falling 35.4% -> 2.2%.

Pooling is deliberately left as a measured choice rather than a fixed one; see
`_pool`.

Usage:
    python -m embed.variant_a --benchmark 1000      # measure, project, stop
    python -m embed.variant_a --compare-fp32 600    # what did int8 cost?
    python -m embed.variant_a --all                 # int8+fp32 x mean+cls
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from .common import (
    ROOT,
    Games,
    build_text,
    choose_masked_tags,
    l2_normalize,
    load_games,
    save_vectors,
)

MODEL_DIR = ROOT / "data" / "models" / "bge-small"
INT8 = MODEL_DIR / "model_int8.onnx"
FP32 = MODEL_DIR / "model_fp32.onnx"
DIMS = 384


def make_session(model_path: Path, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads or max(1, os.cpu_count() or 4)
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def make_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def _pool(last_hidden: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """CLS or mean pooling.

    SPEC.md specifies mean pooling. BGE's own model card specifies CLS for the
    v1.5 models -- the [CLS] token is what their contrastive objective actually
    trained, and mean pooling is off-spec for this family. Rather than pick by
    authority, both are computed from the same forward pass and decided on the
    downstream metrics.
    """
    if how == "cls":
        return last_hidden[:, 0]
    m = mask[..., None].astype(np.float32)
    return (last_hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)


def _prepare(tokenizer, texts: list[str], max_length: int):
    """Token lengths and the length-sorted visit order."""
    ids = tokenizer(texts, truncation=True, max_length=max_length,
                    padding=False)["input_ids"]
    lengths = np.asarray([len(x) for x in ids], dtype=np.int32)
    return lengths, np.argsort(lengths, kind="stable")


def _feed(tokenizer, batch: list[str], max_length: int, input_names: set[str]):
    feats = tokenizer(batch, truncation=True, max_length=max_length,
                      padding=True, return_tensors="np")
    inputs = {k: v.astype(np.int64) for k, v in feats.items() if k in input_names}
    if "token_type_ids" in input_names and "token_type_ids" not in inputs:
        inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    return feats, inputs


def encode(texts: list[str], session, tokenizer, batch_size: int = 32,
           max_length: int = 192, pooling: str = "mean",
           sort_by_length: bool = True, progress: bool = False
           ) -> tuple[np.ndarray, dict]:
    """Encode `texts` to L2-normalised vectors, preserving input order."""
    n = len(texts)
    input_names = {i.name for i in session.get_inputs()}
    lengths, order = _prepare(tokenizer, texts, max_length)
    if not sort_by_length:
        order = np.arange(n)

    out = np.zeros((n, DIMS), dtype=np.float32)
    padded = real = 0
    t0 = time.perf_counter()

    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        feats, inputs = _feed(tokenizer, [texts[i] for i in idx],
                              max_length, input_names)
        hidden = session.run(None, inputs)[0]
        out[idx] = _pool(hidden, feats["attention_mask"], pooling)
        padded += inputs["input_ids"].size
        real += int(feats["attention_mask"].sum())

        if progress and (s // batch_size) % 40 == 0:
            done = min(s + batch_size, n)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"    {done:>7,}/{n:,}  {rate:6.1f} games/s"
                  f"  eta {(n - done) / rate / 60:5.1f} min", end="\r")

    elapsed = time.perf_counter() - t0
    stats = {
        "seconds": elapsed,
        "games_per_sec": n / elapsed if elapsed else 0.0,
        "mean_tokens": float(lengths.mean()),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "max_tokens": int(lengths.max()),
        "pad_waste": 1.0 - (real / padded) if padded else 0.0,
    }
    return l2_normalize(out), stats


def encode_multi(texts: list[str], session, tokenizer, batch_size: int = 32,
                 max_length: int = 192,
                 poolings: tuple[str, ...] = ("mean", "cls"),
                 progress: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Encode once, pool several ways.

    The forward pass dominates cost and pooling is a reduction over an array we
    already hold, so computing mean and CLS together makes the pooling
    comparison free rather than a second full encode.
    """
    n = len(texts)
    input_names = {i.name for i in session.get_inputs()}
    lengths, order = _prepare(tokenizer, texts, max_length)
    outs = {p: np.zeros((n, DIMS), dtype=np.float32) for p in poolings}
    t0 = time.perf_counter()

    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        feats, inputs = _feed(tokenizer, [texts[i] for i in idx],
                              max_length, input_names)
        hidden = session.run(None, inputs)[0]
        for pl in poolings:
            outs[pl][idx] = _pool(hidden, feats["attention_mask"], pl)

        if progress and (s // batch_size) % 40 == 0:
            done = min(s + batch_size, n)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"    {done:>7,}/{n:,}  {rate:6.1f} games/s"
                  f"  eta {(n - done) / rate / 60:5.1f} min", end="\r")

    elapsed = time.perf_counter() - t0
    stats = {
        "seconds": elapsed,
        "games_per_sec": n / elapsed if elapsed else 0.0,
        "mean_tokens": float(lengths.mean()),
        "p95_tokens": float(np.percentile(lengths, 95)),
    }
    return {p: l2_normalize(v) for p, v in outs.items()}, stats


def benchmark(g: Games, n: int, batch_size: int, max_length: int,
              pooling: str) -> None:
    """Measure on a sample, project the full run, encode nothing else."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
    texts = [build_text(g, int(i)) for i in idx]
    tok = make_tokenizer()

    print(f"model      {INT8.name}  ({INT8.stat().st_size / 1e6:.1f} MB)")
    print(f"threads    {os.cpu_count()} logical cores")
    print(f"batch      {batch_size}   max_length {max_length}   pooling {pooling}")
    print(f"sample     {len(texts):,} games\n")

    sess = make_session(INT8)
    encode(texts[:batch_size * 2], sess, tok, batch_size, max_length, pooling)

    results = {}
    for label, sort in (("length-sorted", True), ("input-order", False)):
        _, st = encode(texts, sess, tok, batch_size, max_length, pooling,
                       sort_by_length=sort)
        results[label] = st
        print(f"{label:<16} {st['games_per_sec']:7.1f} games/s"
              f"   {st['seconds']:6.2f}s for {len(texts):,}"
              f"   padding waste {st['pad_waste'] * 100:4.1f}%")

    best = max(results.values(), key=lambda s: s["games_per_sec"])
    ts = results["length-sorted"]
    print(f"\ntoken lengths   mean {ts['mean_tokens']:.0f}"
          f"   p95 {ts['p95_tokens']:.0f}   max {ts['max_tokens']} (cap {max_length})")

    secs = len(g) / best["games_per_sec"]
    speedup = (results["length-sorted"]["games_per_sec"]
               / results["input-order"]["games_per_sec"])
    print("\n" + "=" * 62)
    print(f"PROJECTION for the full catalog ({len(g):,} games)")
    print("=" * 62)
    print(f"  throughput     {best['games_per_sec']:.1f} games/s")
    print(f"  projected      {secs / 60:.1f} min  ({secs / 3600:.2f} h)")
    print(f"  length-sorting {speedup:.2f}x vs input order")
    print("=" * 62)


def compare_fp32(g: Games, n: int, batch_size: int, max_length: int,
                 pooling: str) -> None:
    """What did int8 quantisation actually cost? Measured, not assumed.

    Reports per-game cosine against fp32 and -- the number that matters for a
    nearest-neighbour map -- how much of each game's top-10 neighbourhood is
    preserved. A high mean cosine routinely coexists with heavy reordering.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
    texts = [build_text(g, int(i)) for i in idx]
    tok = make_tokenizer()

    print(f"comparing int8 vs fp32 on {len(texts):,} games\n")
    X = {}
    for label, path in (("int8", INT8), ("fp32", FP32)):
        sess = make_session(path)
        X[label], st = encode(texts, sess, tok, batch_size, max_length, pooling)
        print(f"  {label:<5} {st['games_per_sec']:6.1f} games/s"
              f"   {st['seconds']:6.2f}s   {path.stat().st_size / 1e6:6.1f} MB")

    cos = (X["int8"] * X["fp32"]).sum(axis=1)
    print("\nper-game cosine(int8, fp32)")
    print(f"  mean   {cos.mean():.4f}")
    print(f"  min    {cos.min():.4f}")
    print(f"  p1     {np.percentile(cos, 1):.4f}")
    print(f"  <0.99  {(cos < 0.99).mean() * 100:.1f}% of games")

    k = 10
    s_i, s_f = X["int8"] @ X["int8"].T, X["fp32"] @ X["fp32"].T
    np.fill_diagonal(s_i, -np.inf)
    np.fill_diagonal(s_f, -np.inf)
    keep = sum(
        len(set(np.argpartition(-s_i[r], k)[:k])
            & set(np.argpartition(-s_f[r], k)[:k])) / k
        for r in range(len(texts))
    )
    print(f"\ntop-{k} neighbourhood overlap: {keep / len(texts) * 100:.1f}%")
    print("\nA high mean cosine with a much lower neighbourhood overlap means")
    print("int8 preserves direction but reorders near-ties, which is exactly")
    print("what a k-NN map is sensitive to.")


def encode_all(g: Games, batch_size: int, max_length: int) -> None:
    """Encode the catalog under int8 and fp32, mean and CLS, from two passes.

    Four candidate vector sets, so precision and pooling are both settled on
    downstream metrics rather than on a proxy like cosine-against-fp32.
    """
    texts = [build_text(g, i) for i in range(len(g))]
    tok = make_tokenizer()

    for prec, path in (("int8", INT8), ("fp32", FP32)):
        print(f"\n{prec}: encoding {len(texts):,} games "
              f"(batch {batch_size}, max_length {max_length})")
        sess = make_session(path)
        outs, st = encode_multi(texts, sess, tok, batch_size, max_length,
                                progress=True)
        print(f"\n  {st['games_per_sec']:.1f} games/s, {st['seconds'] / 60:.1f} min")
        for pl, X in outs.items():
            meta = {
                "variant": "A",
                "precision": prec,
                "pooling": pl,
                "method": f"bge-small-en-v1.5 ONNX {prec}, {pl} pooling, CPU",
                "dims": int(X.shape[1]),
                "batch_size": batch_size,
                "max_length": max_length,
                "n_games": len(g),
                **st,
            }
            name = f"variant_a_{prec}_{pl}"
            save_vectors(name, X, g.appids, meta)
            print(f"    saved {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=int, default=0,
                    help="measure on N games, project the full run, and stop")
    ap.add_argument("--compare-fp32", type=int, default=0,
                    help="cosine + neighbourhood agreement of int8 vs fp32 on N games")
    ap.add_argument("--all", action="store_true",
                    help="encode int8+fp32 x mean+cls (2 passes, 4 outputs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--name", default="variant_a")
    ap.add_argument("--holdout", action="store_true",
                    help="drop each game's masked eval tag from its text, so"
                         " tag-prediction is not circular for A either")
    args = ap.parse_args()

    if not INT8.exists():
        print(f"missing {INT8}; run the quantisation step first")
        return 1

    g = load_games()
    print(f"catalog: {len(g):,} games\n")

    if args.benchmark:
        benchmark(g, args.benchmark, args.batch_size, args.max_length, args.pooling)
        return 0
    if args.compare_fp32:
        compare_fp32(g, args.compare_fp32, args.batch_size, args.max_length,
                     args.pooling)
        return 0
    if args.all:
        encode_all(g, args.batch_size, args.max_length)
        return 0

    masked = choose_masked_tags(g) if args.holdout else {}
    texts = [build_text(g, i, skip_tag=masked.get(i)) for i in range(len(g))]
    if masked:
        print(f"holding out 1 tag from the text of {len(masked):,} games")
    sess, tok = make_session(INT8), make_tokenizer()
    print(f"encoding {len(texts):,} games (batch {args.batch_size}, "
          f"max_length {args.max_length}, {args.pooling} pooling)")
    X, st = encode(texts, sess, tok, args.batch_size, args.max_length,
                   args.pooling, progress=True)
    print(f"\n  {st['games_per_sec']:.1f} games/s, {st['seconds'] / 60:.1f} min")

    meta = {
        "variant": "A",
        "precision": "int8",
        "pooling": args.pooling,
        "method": "bge-small-en-v1.5 ONNX int8, CPU",
        "dims": int(X.shape[1]),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_games": len(g),
        **st,
    }
    print(f"  saved {save_vectors(args.name, X, g.appids, meta)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
