"""Render tile contents to PNG, to compare selection strategies by eye.

A documentation and debugging tool, not part of the library's runtime path.
`matplotlib` is imported lazily so it stays an optional extra -- the library
itself needs only numpy.

The comparison this exists for: at low zoom a tile shows a few hundred of
several thousand points, and *which* few hundred decides what the map looks
like. Naive selection takes the highest-importance points and leaves
low-importance regions visibly empty; stratified selection keeps the spatial
shape. The difference is obvious in a picture and invisible in a size table.

Usage:
    python -m tiles render points.parquet out.png --zoom 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .io import load_points
from .quadtree import build


def render_comparison(ps, zooms, max_points, grid, path: Path,
                      point_size: float = 1.6) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = ["naive", "stratified"]
    fig, axes = plt.subplots(len(zooms), len(strategies) + 1,
                             figsize=(4.2 * (len(strategies) + 1),
                                      4.0 * len(zooms)),
                             squeeze=False)

    # Build deeper than the zooms being shown. Leaf tiles are exhaustive by
    # design, so if the displayed zoom IS the leaf level every strategy shows
    # every point and the comparison shows nothing.
    depth = max(max(zooms) + 1, 5)
    built = {s: build(ps, max_zoom=depth, max_points=max_points,
                      strategy=s, grid=grid) for s in strategies}

    for r, z in enumerate(zooms):
        # reference: everything, faintly
        ax = axes[r][0]
        ax.scatter(ps.x, ps.y, s=point_size * 0.35, c="#b8bec9",
                   linewidths=0, rasterized=True)
        ax.set_title(f"all {len(ps):,} points" if r == 0 else "",
                     fontsize=11)
        ax.set_ylabel(f"zoom {z}", fontsize=12)

        for c, s in enumerate(strategies, start=1):
            ax = axes[r][c]
            tiles = [t for t in built[s].tiles if t.tid.z == z]
            xs = np.concatenate([t.points.x for t in tiles]) if tiles else []
            ys = np.concatenate([t.points.y for t in tiles]) if tiles else []
            ax.scatter(ps.x, ps.y, s=point_size * 0.25, c="#e8ebef",
                       linewidths=0, rasterized=True)
            ax.scatter(xs, ys, s=point_size, c="#1f4e79", linewidths=0,
                       rasterized=True)
            n_shown = len(xs)
            ax.set_title(f"{s} — {n_shown:,} of {len(ps):,} shown"
                         if r == 0 else f"{s} — {n_shown:,} shown",
                         fontsize=11)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#d0d5dd")

    fig.suptitle("Tile selection: which points survive the budget",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def coverage_stats(ps, zooms, max_points, grid) -> list[dict]:
    """A number to go with the picture: how much of the plane stays occupied."""
    out = []
    b = ps.bbox()
    G = 32

    def occupied(x, y) -> int:
        gx = np.clip(((x - b[0]) / max(b[2] - b[0], 1e-9) * G).astype(int), 0, G - 1)
        gy = np.clip(((y - b[1]) / max(b[3] - b[1], 1e-9) * G).astype(int), 0, G - 1)
        return len(set(zip(gx.tolist(), gy.tolist())))

    full = occupied(ps.x, ps.y)
    for s in ("naive", "stratified"):
        ts = build(ps, max_zoom=max(max(zooms) + 1, 5), max_points=max_points,
                   strategy=s, grid=grid)
        for z in zooms:
            tiles = [t for t in ts.tiles if t.tid.z == z]
            if not tiles:
                continue
            xs = np.concatenate([t.points.x for t in tiles])
            ys = np.concatenate([t.points.y for t in tiles])
            out.append({"strategy": s, "zoom": z, "shown": int(len(xs)),
                        "cells_occupied": occupied(xs, ys),
                        "cells_total": full})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--zoom", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-points", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=8)
    args = ap.parse_args(argv)

    ps = load_points(args.input)
    print(f"loaded {len(ps):,} points")

    stats = coverage_stats(ps, args.zoom, args.max_points, args.grid)
    print(f"\n{'strategy':<12}{'zoom':>6}{'shown':>10}"
          f"{'cells occupied':>17}{'of full':>10}")
    print("-" * 56)
    for s in stats:
        print(f"{s['strategy']:<12}{s['zoom']:>6}{s['shown']:>10,}"
              f"{s['cells_occupied']:>17,}"
              f"{s['cells_occupied'] / s['cells_total'] * 100:>9.0f}%")

    p = render_comparison(ps, args.zoom, args.max_points, args.grid,
                          Path(args.out))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
