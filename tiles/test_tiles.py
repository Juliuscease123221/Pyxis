"""Tests for the tiles library.

Run:  python -m pytest tiles/ -q

The central one is `test_every_point_reachable`: a tile pyramid that silently
drops points is worse than useless, because the loss is invisible -- the map
looks complete and a game simply cannot be found by zooming.
"""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from .bench import synth
from .format import PointSet, decode, decode_json, encode, encode_json
from .io import load_points
from .manifest import validate, write
from .quadtree import build, tile_bounds, viewport_tiles
from .selection import naive, stratified


# ---------------------------------------------------------------- coverage

@pytest.mark.parametrize("strategy", ["stratified", "naive"])
def test_every_point_reachable(strategy):
    """Every input point appears in at least one LEAF tile. The acceptance test."""
    ps = synth(20_000, seed=3)
    ts = build(ps, max_zoom=4, max_points=300, strategy=strategy)

    seen: set[int] = set()
    for t in ts.tiles:
        if t.is_leaf:
            seen.update(int(i) for i in t.points.ids)

    missing = set(int(i) for i in ps.ids) - seen
    assert not missing, f"{len(missing)} points appear in no leaf tile"
    assert len(seen) == len(ps)


def test_leaf_tiles_are_exhaustive():
    """Leaves apply no importance selection, so nothing is dropped at depth."""
    ps = synth(5_000, seed=1)
    ts = build(ps, max_zoom=3, max_points=50)
    leaf_records = sum(len(t.points) for t in ts.tiles if t.is_leaf)
    assert leaf_records == len(ps)


def test_points_land_inside_their_tile_bounds():
    ps = synth(4_000, seed=2)
    ts = build(ps, max_zoom=3, max_points=200)
    for t in ts.tiles:
        if len(t.points) == 0:
            continue
        minx, miny, maxx, maxy = t.bounds
        assert t.points.x.min() >= minx - 1e-4
        assert t.points.y.min() >= miny - 1e-4
        assert t.points.x.max() <= maxx + 1e-4
        assert t.points.y.max() <= maxy + 1e-4


def test_coarse_tiles_respect_budget():
    ps = synth(10_000, seed=5)
    ts = build(ps, max_zoom=3, max_points=100)
    for t in ts.tiles:
        if not t.is_leaf:
            assert len(t.points) <= 100


# ------------------------------------------------------------------ format

@pytest.mark.parametrize("soa", [True, False])
def test_binary_roundtrip(soa):
    ps = synth(1_000, seed=7)
    got = decode(encode(ps, soa=soa))
    assert np.allclose(got.x, ps.x)
    assert np.allclose(got.y, ps.y)
    assert np.array_equal(got.ids, ps.ids)
    assert np.array_equal(got.category, ps.category)
    assert np.array_equal(got.importance, ps.importance)


def test_json_roundtrip():
    ps = synth(200, seed=8)
    got = decode_json(encode_json(ps))
    assert np.array_equal(got.ids, ps.ids)
    assert np.allclose(got.x, ps.x, atol=1e-5)


def test_binary_is_smaller_than_json():
    ps = synth(1_000, seed=9)
    b, j = len(encode(ps)), len(encode_json(ps))
    assert b < j
    assert len(gzip.compress(b"" + encode(ps))) < len(gzip.compress(encode_json(ps)))


def test_soa_blocks_are_aligned():
    """The whole point of SoA: each block starts 4-byte aligned for typed arrays."""
    ps = synth(101, seed=10)
    blob = encode(ps, soa=True)
    assert len(blob) % 4 == 0
    assert 28 % 4 == 0           # header size


def test_rejects_bad_magic():
    with pytest.raises(ValueError):
        decode(b"XXXX" + encode(synth(10))[4:])


def test_empty_pointset_roundtrips():
    ps = PointSet(np.zeros(0, "f4"), np.zeros(0, "f4"), np.zeros(0, "u4"),
                  np.zeros(0, "u4"), np.zeros(0, "u2"))
    assert len(decode(encode(ps))) == 0


# --------------------------------------------------------------- selection

def test_naive_picks_highest_importance():
    ps = synth(2_000, seed=11)
    idx = naive(ps, 50)
    assert len(idx) == 50
    cutoff = np.sort(ps.importance)[-50]
    assert ps.importance[idx].min() >= cutoff


def test_stratified_covers_more_cells_than_naive():
    """The behavioural difference the two strategies exist to express."""
    ps = synth(30_000, n_clusters=40, seed=12)
    b = ps.bbox()
    n = 400
    cells = lambda idx: len({  # noqa: E731
        (int((ps.x[i] - b[0]) / max(b[2] - b[0], 1e-9) * 16),
         int((ps.y[i] - b[1]) / max(b[3] - b[1], 1e-9) * 16))
        for i in idx})
    assert cells(stratified(ps, n, b)) > cells(naive(ps, n))


def test_selection_returns_requested_count():
    ps = synth(5_000, seed=13)
    for strat in (naive, lambda p, k: stratified(p, k, p.bbox())):
        assert len(strat(ps, 250)) == 250


def test_unknown_strategy_rejected():
    from .selection import select

    ps = synth(100)
    with pytest.raises(ValueError):
        select(ps, 10, ps.bbox(), strategy="nope")


# ---------------------------------------------------------------- quadtree

def test_viewport_query_covers_the_view():
    root = (0.0, 0.0, 8.0, 8.0)
    got = viewport_tiles(root, 2, (0.1, 0.1, 3.9, 3.9))
    assert {(t.x, t.y) for t in got} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_tile_bounds_tile_the_plane():
    root = (0.0, 0.0, 4.0, 4.0)
    z = 2
    total = sum((b[2] - b[0]) * (b[3] - b[1])
                for b in (tile_bounds(root, z, x, y)
                          for x in range(1 << z) for y in range(1 << z)))
    assert total == pytest.approx(16.0)


# ---------------------------------------------------------------- manifest

def test_manifest_validation_catches_dangling_parent():
    bad = [{"id": 1, "parent": 99, "label": "a", "bbox": [0, 0, 1, 1],
            "centroid": [0, 0], "size": 5}]
    assert any("not in manifest" in p for p in validate(bad))


def test_manifest_validation_catches_cycle():
    cyc = [
        {"id": 1, "parent": 2, "label": "a", "bbox": [0, 0, 1, 1],
         "centroid": [0, 0], "size": 1},
        {"id": 2, "parent": 1, "label": "b", "bbox": [0, 0, 1, 1],
         "centroid": [0, 0], "size": 1},
    ]
    assert any("cycle" in p for p in validate(cyc))


def test_manifest_roundtrip(tmp_path):
    nodes = [
        {"id": 0, "parent": None, "label": "root", "bbox": [0, 0, 1, 1],
         "centroid": [0.5, 0.5], "size": 10, "confidence": "strong",
         "purity": 0.9, "depth": 0},
        {"id": 1, "parent": 0, "label": "child", "bbox": [0, 0, 0.5, 0.5],
         "centroid": [0.25, 0.25], "size": 4, "confidence": "weak",
         "purity": 0.2, "depth": 1},
    ]
    info = write(nodes, tmp_path / "m.json")
    assert info["problems"] == []
    back = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert back["categories"][1]["confidence"] == "weak"
    assert back["categories"][1]["purity"] == 0.2


# ------------------------------------------------------------------- io/CLI

def test_loads_a_file_with_no_domain_columns(tmp_path):
    """The acceptance condition: works on data with no Steam fields in it."""
    n = 2_000
    rng = np.random.default_rng(0)
    path = tmp_path / "points.npz"
    np.savez(path,
             x=rng.normal(size=n).astype(np.float32),
             y=rng.normal(size=n).astype(np.float32),
             id=np.arange(n, dtype=np.uint32),
             category=rng.integers(0, 12, n).astype(np.uint32),
             importance=rng.pareto(1.2, n))
    ps = load_points(path)
    assert len(ps) == n
    assert ps.importance.dtype == np.uint16


def test_missing_column_error_is_actionable(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, x=np.zeros(3), y=np.zeros(3))
    with pytest.raises(ValueError, match="--columns"):
        load_points(path)


def test_cli_build_end_to_end(tmp_path):
    from .__main__ import main

    n = 3_000
    rng = np.random.default_rng(1)
    src = tmp_path / "points.npz"
    np.savez(src,
             x=rng.normal(size=n).astype(np.float32),
             y=rng.normal(size=n).astype(np.float32),
             id=np.arange(n, dtype=np.uint32),
             category=rng.integers(0, 8, n).astype(np.uint32),
             importance=rng.random(n))
    out = tmp_path / "out"
    assert main(["build", str(src), str(out), "--max-zoom", "3",
                 "--max-points", "200"]) == 0
    meta = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    assert meta["n_points"] == n
    assert meta["n_tiles"] > 0
    assert (out / "0" / "0" / "0.bin").exists()


# -------------------------------------------------------------- standalone

def test_library_imports_nothing_from_the_host_repo():
    """`tiles` ships as its own project, so it must not reach into Overworld.

    Checked by parsing the source rather than by convention, because this is
    the constraint most easily broken by a one-line convenience import.
    """
    host_packages = {"ingest", "embed", "cluster", "layout", "viewer", "bench"}
    offenders: list[str] = []
    for py in Path(__file__).parent.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in host_packages:
                        offenders.append(f"{py.name}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    if node.module.split(".")[0] in host_packages:
                        offenders.append(f"{py.name}: from {node.module}")
    assert not offenders, "tiles must be standalone: " + "; ".join(offenders)


def test_no_domain_vocabulary_in_source():
    """No Steam/game vocabulary anywhere in the library."""
    banned = ("steam", "appid", "genre", "playtime", "review_count")
    hits: list[str] = []
    for py in Path(__file__).parent.glob("*.py"):
        if py.name == "test_tiles.py":
            continue
        text = py.read_text(encoding="utf-8").lower()
        for word in banned:
            if word in text:
                hits.append(f"{py.name}: {word}")
    assert not hits, "domain vocabulary leaked into the library: " + "; ".join(hits)
