"""Quadtree tiling: which points to send for a given viewport, at a given zoom.

The quadtree here is *only* a delivery structure. It answers "what should the
browser download", never "what does this group of points mean" -- that is the
caller's cluster hierarchy, which this library never sees. It receives an
opaque `category` integer per point and passes it through untouched.

### Construction

The dataset bbox is subdivided into quadrants, recursively, to `max_zoom`. At
every zoom level the whole extent is covered:

    z=0   1 tile       top N of everything
    z=1   4 tiles      top N per quadrant
    z=2   16 tiles     ...

A point may appear in a coarse tile *and* again in finer ones. That redundancy
is the point: each tile is independently renderable, so the viewer can draw a
z=1 tile immediately without having fetched z=0, and panning never produces
holes. It costs storage roughly linear in max_zoom, which is why max_zoom is a
parameter rather than a constant.

**Leaf tiles are exhaustive.** At the deepest level every point appears in
exactly one tile, with no importance selection applied, so nothing is ever
unreachable by zooming. The library's central test asserts exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .format import PointSet
from .selection import select


@dataclass
class TileId:
    z: int
    x: int
    y: int

    def key(self) -> str:
        return f"{self.z}/{self.x}/{self.y}"


@dataclass
class Tile:
    tid: TileId
    bounds: tuple[float, float, float, float]
    points: PointSet
    is_leaf: bool
    total_in_bounds: int = 0


@dataclass
class TileSet:
    bounds: tuple[float, float, float, float]
    max_zoom: int
    max_points: int
    strategy: str
    tiles: list[Tile] = field(default_factory=list)

    def stats(self) -> dict:
        per_z: dict[int, int] = {}
        pts_z: dict[int, int] = {}
        for t in self.tiles:
            per_z[t.tid.z] = per_z.get(t.tid.z, 0) + 1
            pts_z[t.tid.z] = pts_z.get(t.tid.z, 0) + len(t.points)
        return {"tiles_per_zoom": dict(sorted(per_z.items())),
                "points_per_zoom": dict(sorted(pts_z.items())),
                "n_tiles": len(self.tiles),
                "total_point_records": sum(len(t.points) for t in self.tiles)}


def tile_bounds(root: tuple[float, float, float, float], z: int,
                x: int, y: int) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = root
    n = 1 << z
    w = (maxx - minx) / n
    h = (maxy - miny) / n
    return (minx + x * w, miny + y * h, minx + (x + 1) * w, miny + (y + 1) * h)


def _cell_of(ps: PointSet, root, z: int) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = root
    n = 1 << z
    w = max(maxx - minx, 1e-12)
    h = max(maxy - miny, 1e-12)
    cx = np.clip(((ps.x - minx) / w * n).astype(np.int64), 0, n - 1)
    cy = np.clip(((ps.y - miny) / h * n).astype(np.int64), 0, n - 1)
    return cx, cy


def build(ps: PointSet, max_zoom: int = 5, max_points: int = 1000,
          strategy: str = "stratified", grid: int = 8,
          bounds: tuple[float, float, float, float] | None = None) -> TileSet:
    """Build every tile from z=0 to z=max_zoom."""
    root = bounds if bounds is not None else ps.bbox()
    # nudge the max edge so points exactly on it still land inside
    minx, miny, maxx, maxy = root
    root = (minx, miny, maxx + (maxx - minx) * 1e-9 + 1e-12,
            maxy + (maxy - miny) * 1e-9 + 1e-12)

    ts = TileSet(bounds=root, max_zoom=max_zoom, max_points=max_points,
                 strategy=strategy)

    for z in range(max_zoom + 1):
        cx, cy = _cell_of(ps, root, z)
        key = cy * (1 << z) + cx
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
        starts = np.searchsorted(key_sorted, np.unique(key_sorted), side="left")
        ends = np.searchsorted(key_sorted, np.unique(key_sorted), side="right")

        for k, s, e in zip(np.unique(key_sorted), starts, ends):
            members = order[s:e]
            gx = int(k % (1 << z))
            gy = int(k // (1 << z))
            b = tile_bounds(root, z, gx, gy)
            sub = ps.take(members)
            if z == max_zoom:
                chosen = np.arange(len(sub))          # leaves are exhaustive
            else:
                chosen = select(sub, max_points, b, strategy, grid)
            ts.tiles.append(Tile(
                tid=TileId(z, gx, gy), bounds=b,
                points=sub.take(chosen),
                is_leaf=(z == max_zoom),
                total_in_bounds=len(sub),
            ))
    return ts


def viewport_tiles(root: tuple[float, float, float, float], z: int,
                   view: tuple[float, float, float, float]) -> list[TileId]:
    """Tile ids at zoom `z` intersecting `view`. The renderer's hot query."""
    minx, miny, maxx, maxy = root
    n = 1 << z
    w = max(maxx - minx, 1e-12)
    h = max(maxy - miny, 1e-12)
    vx0, vy0, vx1, vy1 = view

    x0 = max(0, min(n - 1, int((vx0 - minx) / w * n)))
    x1 = max(0, min(n - 1, int((vx1 - minx) / w * n)))
    y0 = max(0, min(n - 1, int((vy0 - miny) / h * n)))
    y1 = max(0, min(n - 1, int((vy1 - miny) / h * n)))
    return [TileId(z, x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)]
