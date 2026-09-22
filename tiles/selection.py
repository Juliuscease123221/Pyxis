"""Which points go in a tile: two strategies, so the difference can be seen.

A tile holds at most N points. Choosing *which* N is the decision that
determines what a zoomed-out map looks like.

**naive** -- the top N by importance. Simple, and wrong in a specific way: at
low zoom the whole map becomes whatever the importance scalar favours. On a
game catalog that is the AAA titles, and whole regions of the map render empty
because everything in them scores low. The viewer concludes those regions are
sparse when they are merely unpopular.

**stratified** -- subdivide the tile into a g x g grid and take the top
N/(g*g) from each cell. Spatial shape is preserved because every occupied cell
contributes something, while importance still decides *which* point represents
each cell. A dense region of low-importance points stays visible as a dense
region.

Neither knows what importance means. It is an opaque scalar; the library only
requires that larger is more important.
"""

from __future__ import annotations

import numpy as np

from .format import PointSet


def _desc(importance: np.ndarray) -> np.ndarray:
    """Importance widened to a signed type, for descending sorts.

    `importance` is uint16. Negating it does not produce a descending order --
    it wraps: `-uint16(1)` is 65535, so `argsort(-imp)` ranks the *least*
    important first. The resulting map still looks plausible (the points are
    spatially spread either way), which is why this is worth a named helper
    and a regression test rather than an inline minus sign.
    """
    return importance.astype(np.int32)


def naive(ps: PointSet, n: int, **_: object) -> np.ndarray:
    """Indices of the top-n points by importance."""
    if len(ps) <= n:
        return np.arange(len(ps))
    imp = _desc(ps.importance)
    idx = np.argpartition(-imp, n)[:n]
    return idx[np.argsort(-imp[idx])]


def stratified(ps: PointSet, n: int, bounds: tuple[float, float, float, float],
               grid: int = 8) -> np.ndarray:
    """Top points per cell of a grid x grid subdivision of `bounds`.

    Cells are allocated an equal share of the budget. Empty cells give their
    share back: the leftover is redistributed by importance across everything
    not already chosen, so a tile always returns min(n, len(ps)) points rather
    than silently under-filling in sparse regions.
    """
    m = len(ps)
    if m <= n:
        return np.arange(m)

    minx, miny, maxx, maxy = bounds
    w = max(maxx - minx, 1e-12)
    h = max(maxy - miny, 1e-12)
    gx = np.clip(((ps.x - minx) / w * grid).astype(np.int64), 0, grid - 1)
    gy = np.clip(((ps.y - miny) / h * grid).astype(np.int64), 0, grid - 1)
    cell = gy * grid + gx

    per_cell = max(1, n // (grid * grid))
    imp = _desc(ps.importance)
    order = np.argsort(-imp, kind="stable")
    cell_sorted = cell[order]

    taken: list[np.ndarray] = []
    counts = np.zeros(grid * grid, dtype=np.int64)
    # one pass over importance-sorted points, keeping the first `per_cell` of each cell
    keep_mask = np.zeros(m, dtype=bool)
    for pos, c in enumerate(cell_sorted):
        if counts[c] < per_cell:
            counts[c] += 1
            keep_mask[order[pos]] = True
    taken.append(np.flatnonzero(keep_mask))

    chosen = taken[0]
    if len(chosen) < n:
        # redistribute the unused budget by importance
        rest = order[~keep_mask[order]]
        chosen = np.concatenate([chosen, rest[:n - len(chosen)]])
    elif len(chosen) > n:
        chosen = chosen[np.argsort(-imp[chosen])][:n]

    return chosen[np.argsort(-imp[chosen])]


STRATEGIES = {"naive": naive, "stratified": stratified}


def select(ps: PointSet, n: int, bounds, strategy: str = "stratified",
           grid: int = 8) -> np.ndarray:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; "
                         f"expected one of {sorted(STRATEGIES)}")
    if strategy == "naive":
        return naive(ps, n)
    return stratified(ps, n, bounds, grid)
