"""Shared loading and text construction for the three embedding variants.

All variants must embed the *same* games in the *same* row order, or the
comparison in variants.md is meaningless and the hybrid concatenation in
variant C silently pairs one game's text vector with another game's tag vector.
`load_games()` is therefore the single source of row order for the whole phase:
appid ascending, no exceptions.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest.db import DEFAULT_DB, connect

ROOT = Path(__file__).resolve().parent.parent
VEC_DIR = ROOT / "data" / "vectors"

# Steam descriptions are HTML-ish and carry boilerplate. Strip the markup
# before it reaches a tokenizer; "<br>" tokens are pure noise.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Games:
    """Row-aligned arrays for the filtered catalog.

    Every array is indexed by the same row number. `appids[i]`, `names[i]`,
    `tags[i]` and `genres[i]` all describe the same game.
    """

    appids: np.ndarray            # int64 (n,)
    names: list[str]
    tags: list[list[str]]
    genres: list[list[str]]
    short_desc: list[str]
    long_desc: list[str]
    review_count: np.ndarray      # int64 (n,)

    def __len__(self) -> int:
        return len(self.appids)

    def index_of(self, appid: int) -> int | None:
        hit = np.searchsorted(self.appids, appid)
        if hit < len(self.appids) and self.appids[hit] == appid:
            return int(hit)
        return None


def clean_html(s: str | None) -> str:
    if not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def load_games(db: str | Path = DEFAULT_DB, min_tags: int = 0) -> Games:
    """Load the filtered catalog in a fixed row order (appid ascending)."""
    conn: sqlite3.Connection = connect(db)
    rows = conn.execute(
        "SELECT appid, name, short_desc, long_desc, tags, genres, review_count"
        " FROM games ORDER BY appid ASC"
    ).fetchall()
    conn.close()

    appids, names, tags, genres, shorts, longs, revs = [], [], [], [], [], [], []
    for r in rows:
        t = json.loads(r["tags"] or "[]")
        if len(t) < min_tags:
            continue
        appids.append(int(r["appid"]))
        names.append(r["name"] or "")
        tags.append(t)
        genres.append(json.loads(r["genres"] or "[]"))
        shorts.append(clean_html(r["short_desc"]))
        longs.append(clean_html(r["long_desc"]))
        revs.append(int(r["review_count"] or 0))

    return Games(
        appids=np.asarray(appids, dtype=np.int64),
        names=names,
        tags=tags,
        genres=genres,
        short_desc=shorts,
        long_desc=longs,
        review_count=np.asarray(revs, dtype=np.int64),
    )


def build_text(g: Games, i: int, desc_chars: int = 600,
               skip_tag: str | None = None) -> str:
    """The variant A input string for game `i`.

    Order is deliberate: name, then tags, then genres, then description. With
    max_length capped at 192 tokens the tail is truncated, so the highest-signal
    fields must come first. Tags lead the semantic content because they are
    thousands of players voting on what the game actually is, where the
    description is the publisher's own marketing copy.

    `skip_tag` removes one tag from the string. The tag-prediction metric masks
    a tag and asks whether neighbours recover it; since this template *contains*
    the tags, leaving the masked one in makes that score circular for variant A
    in exactly the way it is for variant B. Passing it here removes the leak at
    build time so A and B are scored on the same footing.
    """
    parts = [g.names[i].strip() or "Untitled"]
    tags = [t for t in g.tags[i] if t != skip_tag] if skip_tag else g.tags[i]
    if tags:
        parts.append("Tags: " + ", ".join(tags[:12]).lower() + ".")
    if g.genres[i]:
        parts.append("Genres: " + ", ".join(g.genres[i]).lower() + ".")
    desc = g.short_desc[i] or g.long_desc[i]
    if desc:
        parts.append(desc[:desc_chars])
    return " ".join(p for p in parts if p)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows are left as zeros, not NaN."""
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def save_vectors(name: str, X: np.ndarray, appids: np.ndarray, meta: dict) -> Path:
    """Persist a variant's vectors plus the appid order that produced them."""
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    path = VEC_DIR / f"{name}.npz"
    np.savez_compressed(
        path,
        X=X.astype(np.float16),
        appids=appids,
        meta=json.dumps(meta),
    )
    return path


def load_vectors(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X float32, appids, meta) for a saved variant."""
    d = np.load(VEC_DIR / f"{name}.npz", allow_pickle=False)
    return (
        d["X"].astype(np.float32),
        d["appids"],
        json.loads(str(d["meta"])),
    )


def choose_masked_tags(g: "Games", seed: int = 0, min_tags: int = 4) -> dict[int, str]:
    """Pick one tag to hold out per eligible game, for the tag-prediction metric.

    Held out from the *middle* of the vote ranking, not the top: masking the
    single defining tag makes the task near-impossible, and masking the last
    makes it trivial. Deterministic in `seed`, so every variant is scored
    against exactly the same held-out set.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, str] = {}
    for i, tags in enumerate(g.tags):
        if len(tags) < min_tags:
            continue
        out[i] = tags[int(rng.integers(1, min(len(tags), 10)))]
    return out
