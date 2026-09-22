"""tiles CLI.

    python -m tiles build points.parquet out/
    python -m tiles build points.parquet out/ --strategy naive --max-zoom 5
    python -m tiles bench --sizes 10000 100000 1000000
    python -m tiles inspect out/

The input needs columns x, y, id, category, importance. Nothing in this
library knows what those mean.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from .format import encode, encode_json
from .io import load_points
from .quadtree import build


def cmd_build(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    cols = json.loads(args.columns) if args.columns else None
    ps = load_points(args.input, cols)
    t_load = time.perf_counter() - t0
    print(f"loaded {len(ps):,} points from {args.input}  ({t_load:.2f}s)")
    print(f"  bbox {tuple(round(v, 3) for v in ps.bbox())}")
    print(f"  {len(np.unique(ps.category)):,} distinct categories")

    t0 = time.perf_counter()
    ts = build(ps, max_zoom=args.max_zoom, max_points=args.max_points,
               strategy=args.strategy, grid=args.grid)
    t_build = time.perf_counter() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.perf_counter()
    for tile in ts.tiles:
        blob = encode(tile.points, soa=not args.aos, bbox=tile.bounds)
        if args.gzip:
            blob = gzip.compress(blob, 6)
        p = out / f"{tile.tid.z}" / f"{tile.tid.x}"
        p.mkdir(parents=True, exist_ok=True)
        fn = p / (f"{tile.tid.y}.bin.gz" if args.gzip else f"{tile.tid.y}.bin")
        fn.write_bytes(blob)
        total += len(blob)
    t_write = time.perf_counter() - t0

    st = ts.stats()
    meta = {
        "version": 1,
        "bounds": list(ts.bounds),
        "max_zoom": ts.max_zoom,
        "max_points": ts.max_points,
        "strategy": ts.strategy,
        "layout": "aos" if args.aos else "soa",
        "gzip": bool(args.gzip),
        "n_points": len(ps),
        "n_tiles": st["n_tiles"],
        "tiles_per_zoom": st["tiles_per_zoom"],
        "points_per_zoom": st["points_per_zoom"],
        "bytes_total": total,
        "seconds": {"load": round(t_load, 2), "build": round(t_build, 2),
                    "write": round(t_write, 2)},
    }
    (out / "tiles.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nbuilt {st['n_tiles']:,} tiles in {t_build:.2f}s, "
          f"wrote in {t_write:.2f}s")
    print(f"  tiles per zoom   {st['tiles_per_zoom']}")
    print(f"  points per zoom  {st['points_per_zoom']}")
    print(f"  total bytes      {total / 1e6:.2f} MB"
          f"   mean {total / max(st['n_tiles'], 1):,.0f} B/tile")
    print(f"  redundancy       {st['total_point_records'] / len(ps):.2f}x"
          f"  (point records / input points)")
    print(f"\nwrote {out}/tiles.json")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    out = Path(args.out)
    meta = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    print(json.dumps(meta, indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run

    run(sizes=args.sizes, max_zoom=args.max_zoom,
        max_points=args.max_points, out=Path(args.out) if args.out else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tiles", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a tile directory from a point file")
    b.add_argument("input")
    b.add_argument("out")
    b.add_argument("--max-zoom", type=int, default=5)
    b.add_argument("--max-points", type=int, default=1000)
    b.add_argument("--strategy", choices=["stratified", "naive"],
                   default="stratified")
    b.add_argument("--grid", type=int, default=8)
    b.add_argument("--gzip", action="store_true")
    b.add_argument("--aos", action="store_true",
                   help="interleaved layout instead of struct-of-arrays")
    b.add_argument("--columns", default=None,
                   help='JSON map, e.g. \'{"importance":"weight"}\'')
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect", help="print a tile directory's metadata")
    i.add_argument("out")
    i.set_defaults(func=cmd_inspect)

    n = sub.add_parser("bench", help="scaling and format benchmarks")
    n.add_argument("--sizes", type=int, nargs="+",
                   default=[10_000, 100_000, 1_000_000])
    n.add_argument("--max-zoom", type=int, default=5)
    n.add_argument("--max-points", type=int, default=1000)
    n.add_argument("--out", default=None)
    n.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
