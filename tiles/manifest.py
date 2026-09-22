"""The category manifest: everything the renderer needs that is not a point.

Point records carry a **leaf category id only**. The renderer walks upward
through `parent` pointers in this manifest to find ancestors, rather than each
point carrying its full ancestor chain. For 1M points and a depth-5 hierarchy
that is the difference between 4 bytes and 20 bytes per point -- 16MB of
duplicated chain that is identical for every point in a category.

The manifest is loaded once, up front, so level-of-detail decisions need no
network round trip.

This library does not know what a category *is*. It validates the shape of the
records and passes the rest through:

    id          int    the value that appears in point records
    parent      int|None
    label       str    display text; opaque
    confidence  str    a tier name; opaque, but see below
    bbox        [minx, miny, maxx, maxy]
    centroid    [x, y]
    size        int    points in this category's whole subtree
    purity      float|None   0..1, how cleanly the region belongs to it

`confidence` and `purity` are carried because a renderer needs them to decide
*how* to draw a label, not just whether: a low-confidence label can be dimmed
rather than dropped, and a low-purity category can be declined as a region
while its children are drawn instead. The library does not interpret either --
it only guarantees they survive the round trip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED = ("id", "parent", "label", "bbox", "centroid", "size")
OPTIONAL = ("confidence", "purity", "containment", "depth")


@dataclass
class CategoryNode:
    id: int
    parent: int | None
    label: str
    bbox: list[float]
    centroid: list[float]
    size: int
    depth: int = 0
    confidence: str | None = None
    purity: float | None = None
    containment: float | None = None


def validate(nodes: list[dict]) -> list[str]:
    """Structural problems a renderer would trip over. Returns messages."""
    problems: list[str] = []
    ids = {n["id"] for n in nodes if "id" in n}

    for i, n in enumerate(nodes):
        missing = [k for k in REQUIRED if k not in n]
        if missing:
            problems.append(f"node {i}: missing {missing}")
            continue
        if n["parent"] is not None and n["parent"] not in ids:
            problems.append(f"node {n['id']}: parent {n['parent']} not in manifest")
        if len(n["bbox"]) != 4:
            problems.append(f"node {n['id']}: bbox must have 4 values")
        if len(n["centroid"]) != 2:
            problems.append(f"node {n['id']}: centroid must have 2 values")

    # cycles / unreachable roots
    by_id = {n["id"]: n for n in nodes if "id" in n}
    for nid in list(by_id):
        seen, cur, hops = {nid}, by_id[nid].get("parent"), 0
        while cur is not None and hops < len(by_id) + 1:
            if cur in seen:
                problems.append(f"node {nid}: parent chain contains a cycle")
                break
            seen.add(cur)
            cur = by_id.get(cur, {}).get("parent")
            hops += 1
    return problems


def write(nodes: list[dict], path: Path, strict: bool = True) -> dict:
    problems = validate(nodes)
    if problems and strict:
        raise ValueError("manifest invalid:\n  " + "\n  ".join(problems[:10]))
    payload = {"version": 1, "n_categories": len(nodes), "categories": nodes}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return {"n": len(nodes), "bytes": Path(path).stat().st_size,
            "problems": problems}


def read(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
