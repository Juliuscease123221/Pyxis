"""Benchmarks: scaling to 1M points, format size, and parse cost.

Scaling is the claim this library makes, so it is measured on synthetic point
sets rather than only on whatever the caller happens to have. The generator
produces clustered data, not uniform noise -- uniform points make a quadtree
look better than it is, because every tile fills evenly and no subdivision is
ever wasted.

Reported:
  * build time vs point count (10k / 100k / 1M)
  * bytes per tile, binary vs JSON, raw and gzipped
  * **parse time** for each, which is the cost the size comparison omits
  * tiles per viewport at each zoom
  * viewport -> tile-list query time
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from .format import PointSet, decode, decode_json, encode, encode_json
from .quadtree import build, viewport_tiles


def synth(n: int, n_clusters: int = 60, seed: int = 0) -> PointSet:
    """Clustered synthetic points with a heavy-tailed importance distribution.

    Both properties matter. Uniform points would flatter the quadtree; a
    uniform importance distribution would flatter naive selection, because the
    top-N would already be spatially spread.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-50, 50, size=(n_clusters, 2))
    weights = rng.dirichlet(np.full(n_clusters, 0.7))
    counts = rng.multinomial(n, weights)
    xs, ys, cats = [], [], []
    for c, (cx, cy) in zip(counts, centres):
        if c == 0:
            continue
        spread = rng.uniform(0.6, 4.0)
        xs.append(rng.normal(cx, spread, c))
        ys.append(rng.normal(cy, spread, c))
        cats.append(np.full(c, len(cats), dtype=np.uint32))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    cat = np.concatenate(cats).astype(np.uint32)
    imp_raw = rng.pareto(1.2, len(x)) + 1.0
    order = np.argsort(imp_raw)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    imp = np.round(ranks / max(len(x) - 1, 1) * 65535).astype(np.uint16)
    return PointSet(x, y, np.arange(len(x), dtype=np.uint32), cat, imp)


def bench_build(sizes, max_zoom, max_points, strategy="stratified") -> list[dict]:
    rows = []
    for n in sizes:
        ps = synth(n)
        t0 = time.perf_counter()
        ts = build(ps, max_zoom=max_zoom, max_points=max_points,
                   strategy=strategy)
        secs = time.perf_counter() - t0
        st = ts.stats()
        blobs = [encode(t.points, bbox=t.bounds) for t in ts.tiles]
        total = sum(len(b) for b in blobs)
        rows.append({
            "n": n, "seconds": secs, "tiles": st["n_tiles"],
            "records": st["total_point_records"],
            "bytes": total,
            "us_per_point": secs / n * 1e6,
        })
    return rows


def bench_format(ps: PointSet, sample_tiles: list) -> dict:
    """Size and parse cost, binary vs JSON, raw and gzipped."""
    res = {"binary": {}, "json": {}}
    for name, enc, dec in (("binary", encode, decode),
                           ("json", encode_json, decode_json)):
        raw_sizes, gz_sizes, parse_times = [], [], []
        for t in sample_tiles:
            blob = enc(t.points) if name == "json" else enc(t.points, bbox=t.bounds)
            raw_sizes.append(len(blob))
            gz_sizes.append(len(gzip.compress(blob, 6)))
            t0 = time.perf_counter()
            for _ in range(5):
                dec(blob)
            parse_times.append((time.perf_counter() - t0) / 5)
        res[name] = {
            "mean_raw": float(np.mean(raw_sizes)),
            "mean_gzip": float(np.mean(gz_sizes)),
            "mean_parse_ms": float(np.mean(parse_times)) * 1e3,
        }
    b, j = res["binary"], res["json"]
    res["ratio"] = {
        "raw": j["mean_raw"] / max(b["mean_raw"], 1),
        "gzip": j["mean_gzip"] / max(b["mean_gzip"], 1),
        "parse": j["mean_parse_ms"] / max(b["mean_parse_ms"], 1e-9),
    }
    return res


def bench_query(ts, reps: int = 2000) -> list[dict]:
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = ts.bounds
    out = []
    for z in range(ts.max_zoom + 1):
        # a viewport covering ~1/4 of the extent at this zoom
        span = (maxx - minx) / max(1 << z, 1) * 2
        counts, t0 = [], time.perf_counter()
        for _ in range(reps):
            cx = rng.uniform(minx, maxx)
            cy = rng.uniform(miny, maxy)
            view = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
            counts.append(len(viewport_tiles(ts.bounds, z, view)))
        secs = (time.perf_counter() - t0) / reps
        out.append({"z": z, "mean_tiles": float(np.mean(counts)),
                    "us_per_query": secs * 1e6})
    return out


def run(sizes=(10_000, 100_000, 1_000_000), max_zoom=5, max_points=1000,
        out: Path | None = None) -> dict:
    print("=" * 72)
    print("BUILD SCALING  (clustered synthetic points, Pareto importance)")
    print("=" * 72)
    print(f"{'points':>10}{'seconds':>10}{'us/point':>11}{'tiles':>9}"
          f"{'records':>12}{'MB':>9}")
    print("-" * 72)
    build_rows = bench_build(sizes, max_zoom, max_points)
    for r in build_rows:
        print(f"{r['n']:>10,}{r['seconds']:>10.2f}{r['us_per_point']:>11.2f}"
              f"{r['tiles']:>9,}{r['records']:>12,}{r['bytes'] / 1e6:>9.1f}")

    if len(build_rows) > 1:
        a, b = build_rows[0], build_rows[-1]
        factor = b["n"] / a["n"]
        got = b["seconds"] / max(a["seconds"], 1e-9)
        print(f"\n  {factor:.0f}x the points cost {got:.1f}x the time "
              f"(linear would be {factor:.0f}x)")

    print("\n" + "=" * 72)
    print("FORMAT  (binary vs JSON, measured on 50 tiles of the 100k set)")
    print("=" * 72)
    ps = synth(100_000)
    ts = build(ps, max_zoom=max_zoom, max_points=max_points)
    sample = [t for t in ts.tiles if len(t.points) > 100][:50]
    fmt = bench_format(ps, sample)
    print(f"{'':<10}{'raw B':>12}{'gzip B':>12}{'parse ms':>12}")
    for k in ("binary", "json"):
        f = fmt[k]
        print(f"{k:<10}{f['mean_raw']:>12,.0f}{f['mean_gzip']:>12,.0f}"
              f"{f['mean_parse_ms']:>12.3f}")
    r = fmt["ratio"]
    print(f"\n  JSON is {r['raw']:.1f}x larger raw, {r['gzip']:.1f}x gzipped,"
          f" and {r['parse']:.1f}x slower to parse")
    print("  gzip closes much of the size gap; it does nothing for parse cost,")
    print("  which is why size alone understates the case for binary.")

    print("\n" + "=" * 72)
    print("VIEWPORT QUERY")
    print("=" * 72)
    print(f"{'zoom':>6}{'mean tiles':>13}{'us/query':>11}")
    q = bench_query(ts)
    for r_ in q:
        print(f"{r_['z']:>6}{r_['mean_tiles']:>13.1f}{r_['us_per_query']:>11.2f}")

    result = {"build": build_rows, "format": fmt, "query": q,
              "params": {"max_zoom": max_zoom, "max_points": max_points}}
    if out:
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return result
