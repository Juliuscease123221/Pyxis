"""SQLite schema and connection helpers for the Pyxis ingest stage.

The schema is fixed by SPEC.md Phase 1. `games` holds one row per Steam app;
`failures` tracks appids the crawler could not fetch so retries hit only those.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "pyxis.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
  appid INTEGER PRIMARY KEY,
  name TEXT,
  short_desc TEXT,
  long_desc TEXT,
  tags TEXT,          -- JSON array, ordered by vote count
  genres TEXT,        -- JSON array
  release_date TEXT,
  review_count INTEGER,
  review_score REAL,
  price REAL,
  fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS failures (
  appid INTEGER PRIMARY KEY,
  status TEXT,
  tries INTEGER
);

CREATE INDEX IF NOT EXISTS idx_games_review_count ON games(review_count);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the Pyxis database, creating the schema if needed.

    WAL mode lets the background crawler write while evaluation scripts read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for the two tables, for progress reporting."""
    g = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    f = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    return {"games": g, "failures": f}
