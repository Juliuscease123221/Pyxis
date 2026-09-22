"""Binary tile format: encode/decode, plus the JSON baseline it is measured against.

This module knows nothing about the domain it is tiling. A point is
`(x, y, id, category, importance)` and `category` is an opaque integer.

### Layout: struct-of-arrays, not array-of-structs

SPEC.md specifies an interleaved record layout::

    points:  x:f32 y:f32 id:u32 cluster:u32 importance:u16

and justifies binary over JSON because it maps "straight into a typed array".
Those two things are incompatible. That record is 18 bytes, so consecutive `x`
values sit 18 bytes apart; `Float32Array` requires a 4-byte-aligned, contiguous
run of floats. Interleaved data can only be read through `DataView`, field by
field, in a JavaScript loop -- which is most of the parse cost the format was
meant to avoid.

So the payload is **struct-of-arrays**: all x, then all y, then all ids, then
all categories, then all importances. Each block is contiguous and aligned, so
the browser does::

    const xs = new Float32Array(buf, offset, count);   // zero copy

which is what the spec wanted. The AoS variant is implemented too
(`encode(..., soa=False)`) and `bench.py` measures the difference rather than
asserting it.

### Header

    magic   4   b"ATLS"
    version 2   u16
    flags   2   u16   bit 0: 1 = struct-of-arrays
    count   4   u32
    bbox   16   4 x f32  (minx, miny, maxx, maxy)
                = 28 bytes, 4-byte aligned

SPEC.md gives count 2 bytes. That caps a tile at 65,535 points, which is
fine at the default 1,000 but makes the format unable to express a coarse
whole-dataset tile -- and this library benchmarks to 1,000,000 points. 4 bytes
costs two bytes per tile and removes the ceiling.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"ATLS"
VERSION = 1
FLAG_SOA = 1
HEADER = struct.Struct("<4sHHI4f")   # 28 bytes
assert HEADER.size == 28


@dataclass
class PointSet:
    """A tile's payload. Domain-agnostic by construction."""

    x: np.ndarray            # float32 (n,)
    y: np.ndarray            # float32 (n,)
    ids: np.ndarray          # uint32  (n,)
    category: np.ndarray     # uint32  (n,)  opaque cluster/class id
    importance: np.ndarray   # uint16  (n,)

    def __len__(self) -> int:
        return int(len(self.ids))

    def bbox(self) -> tuple[float, float, float, float]:
        if len(self) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (float(self.x.min()), float(self.y.min()),
                float(self.x.max()), float(self.y.max()))

    def take(self, idx) -> "PointSet":
        return PointSet(self.x[idx], self.y[idx], self.ids[idx],
                        self.category[idx], self.importance[idx])


def encode(ps: PointSet, soa: bool = True,
           bbox: tuple[float, float, float, float] | None = None) -> bytes:
    """Serialise a PointSet. `soa=False` gives the interleaved layout."""
    n = len(ps)
    bb = bbox if bbox is not None else ps.bbox()
    flags = FLAG_SOA if soa else 0
    head = HEADER.pack(MAGIC, VERSION, flags, n, *bb)

    if soa:
        body = b"".join((
            ps.x.astype("<f4", copy=False).tobytes(),
            ps.y.astype("<f4", copy=False).tobytes(),
            ps.ids.astype("<u4", copy=False).tobytes(),
            ps.category.astype("<u4", copy=False).tobytes(),
            ps.importance.astype("<u2", copy=False).tobytes(),
        ))
        # pad so the whole tile is a multiple of 4 bytes
        body += b"\x00" * ((-len(body)) % 4)
    else:
        rec = np.zeros(n, dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("id", "<u4"),
            ("cat", "<u4"), ("imp", "<u2")]))
        rec["x"] = ps.x
        rec["y"] = ps.y
        rec["id"] = ps.ids
        rec["cat"] = ps.category
        rec["imp"] = ps.importance
        body = rec.tobytes()
    return head + body


def decode(buf: bytes) -> PointSet:
    """Inverse of `encode`. Used by the tests and the benchmark's parse timing."""
    magic, version, flags, n, *bb = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    off = HEADER.size

    if flags & FLAG_SOA:
        def block(dtype, count):
            nonlocal off
            a = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
            off += a.nbytes
            return a
        x = block("<f4", n)
        y = block("<f4", n)
        ids = block("<u4", n)
        cat = block("<u4", n)
        imp = block("<u2", n)
        return PointSet(x, y, ids, cat, imp)

    rec = np.frombuffer(buf, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("id", "<u4"),
        ("cat", "<u4"), ("imp", "<u2")]), count=n, offset=off)
    return PointSet(rec["x"], rec["y"], rec["id"], rec["cat"], rec["imp"])


def encode_json(ps: PointSet) -> bytes:
    """The baseline the binary format is measured against.

    Object-per-point, which is what a naive implementation emits and what the
    browser then has to allocate.
    """
    return json.dumps([
        {"x": float(a), "y": float(b), "id": int(c),
         "cluster": int(d), "importance": int(e)}
        for a, b, c, d, e in zip(ps.x, ps.y, ps.ids, ps.category, ps.importance)
    ]).encode("utf-8")


def decode_json(buf: bytes) -> PointSet:
    rows = json.loads(buf)
    return PointSet(
        np.asarray([r["x"] for r in rows], dtype=np.float32),
        np.asarray([r["y"] for r in rows], dtype=np.float32),
        np.asarray([r["id"] for r in rows], dtype=np.uint32),
        np.asarray([r["cluster"] for r in rows], dtype=np.uint32),
        np.asarray([r["importance"] for r in rows], dtype=np.uint16),
    )
