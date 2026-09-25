"""Load the FronkonGames dump into the Pyxis SQLite database, with filtering.

Two source files, because neither is complete on its own:

  games.csv                 has community Tags (the deciding field) but no
                            short_description.
  data/train-*.parquet      has short_description and detailed_description but
                            its Tags column was flattened to empty lists during
                            the parquet conversion.

We take tags from the CSV and short_desc from the parquet, joined on appid.

Filtering, per SPEC.md: drop DLC, soundtracks, demos, videos and software;
drop anything under 10 reviews (no tags, clusters as noise, pure render cost).

Usage:
    python -m ingest.load_dump
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .csv_schema import read_kwargs
from .db import DEFAULT_DB, connect
from .evaluate_dumps import parse_tags

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "games.csv"
PARQUET_PATH = ROOT / "data" / "fronkon.parquet"

MIN_REVIEWS = 10

# Name-based rejects, in two tiers.
#
# Tier 1 words are unambiguous wherever they appear: no shipped game is called
# "... Playtest" or "... Soundtrack" and also a game.
#
# Tier 2 words appear inside legitimate game titles and must be anchored.
# "Movie" alone would drop The LEGO Movie - Videogame; "DLC" alone would drop
# DLC Quest; "Artwork" alone would drop Please, Touch The Artwork; "Trailer"
# alone would drop Trailer Shop Simulator. Each of those is a real game with
# thousands of reviews, so these are anchored to the trailing-product or
# parenthetical forms that actually mark a non-game SKU.
#
# Non-game videos are caught by SOFTWARE_GENRES (Documentary/Movie/Short/
# Episodic) rather than by name, which is the more reliable signal.
NAME_REJECT = re.compile(
    r"""(
      # --- tier 1: unambiguous ---
        \bdemo\b | \bplaytest\b | \bbeta\ test\b
      | \bsoundtrack\b | \boriginal\ score\b | \bmusic\ pack\b
      | \bart\s?book\b | \bwallpaper\b
      | \bseason\ pass\b | \bexpansion\ pass\b | \bcontent\ pack\b
      | \bteaser\b | \bdedicated\ server\b | \btest\ server\b
      | \bbenchmark\b | \bupgrade\ pack\b | \bskin\ pack\b
      | \bcosmetic\ pack\b | \bepisode\ \d+\ pack\b

      # --- tier 2: anchored, because the bare word hits real games ---
      | \(\s*dlc\s*\)        # "Kingdom Hearts III + Re Mind (DLC)"
      | [-–—:]\s*dlc\b       # "Some Game - DLC"
      | \bdlc\ pack\b
      | \bost\b\s*$          # "Sinless + OST"
      | [-–—:]\s*ost\b
      | \bartwork\ pack\b
      | \btrailer\ pack\b
      | \bsdk\b\s*$
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Steam genres that mark an app as non-game software or a video.
SOFTWARE_GENRES = {
    "Utilities",
    "Design & Illustration",
    "Animation & Modeling",
    "Video Production",
    "Audio Production",
    "Photo Editing",
    "Web Publishing",
    "Software Training",
    "Game Development",
    "Accounting",
    "Documentary",
    "Episodic",
    "Short",
    "Movie",
    "Tutorial",
}


def classify_drop(name: str, genres: list[str], reviews: int) -> str | None:
    """Return a drop reason, or None to keep the row.

    Order matters only for reporting: we attribute each dropped row to the
    first rule that caught it, so the counts in the summary sum to the total.
    """
    if not name or not str(name).strip():
        return "no_name"
    if NAME_REJECT.search(str(name)):
        return "name_pattern"
    if genres and SOFTWARE_GENRES.issuperset(genres):
        # every genre is a software/video genre -> not a game
        return "software_genre"
    if reviews < MIN_REVIEWS:
        return "few_reviews"
    return None


def load_short_descriptions() -> dict[int, str]:
    """appid -> short_description, from the parquet (the CSV has no such column)."""
    if not PARQUET_PATH.exists():
        print(f"  ! {PARQUET_PATH.name} missing; short_desc will be empty")
        return {}
    tbl = pq.read_table(PARQUET_PATH, columns=["appID", "short_description"])
    df = tbl.to_pandas()
    df = df[df["short_description"].notna()]
    out: dict[int, str] = {}
    for appid, desc in zip(df["appID"], df["short_description"]):
        try:
            out[int(appid)] = str(desc)
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--csv", default=str(CSV_PATH))
    args = ap.parse_args()

    print("Loading short descriptions from parquet...")
    shorts = load_short_descriptions()
    print(f"  {len(shorts):,} short descriptions")

    conn = connect(args.db)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    drops: dict[str, int] = {}
    kept = 0
    total = 0

    # Map the CSV's human-readable headers onto safe identifiers so that
    # itertuples gives us named access instead of positional `_8` fields.
    RENAME = {
        "AppID": "appid",
        "Name": "name",
        "Release date": "release_date",
        "About the game": "long_desc",
        "Positive": "positive",
        "Negative": "negative",
        "Price": "price",
        "Genres": "genres",
        "Tags": "tags",
    }

    print(f"\nStreaming {args.csv} ...")
    reader = pd.read_csv(args.csv, **read_kwargs(chunksize=20_000))
    for chunk in reader:
        missing = [c for c in RENAME if c not in chunk.columns]
        if missing:
            raise SystemExit(f"CSV is missing expected columns: {missing}")
        chunk = chunk.rename(columns=RENAME)
        rows = []
        for r in chunk.itertuples(index=False):
            total += 1
            try:
                appid = int(r.appid)
            except (TypeError, ValueError):
                drops["bad_appid"] = drops.get("bad_appid", 0) + 1
                continue

            name = r.name or ""
            genres = parse_tags(r.genres)
            tags = parse_tags(r.tags)

            try:
                pos = int(float(r.positive))
            except (TypeError, ValueError):
                pos = 0
            try:
                neg = int(float(r.negative))
            except (TypeError, ValueError):
                neg = 0
            reviews = pos + neg

            reason = classify_drop(str(name), genres, reviews)
            if reason:
                drops[reason] = drops.get(reason, 0) + 1
                continue

            score = (pos / reviews) if reviews else None
            try:
                price = float(r.price)
            except (TypeError, ValueError):
                price = None

            long_desc = r.long_desc
            if not isinstance(long_desc, str):
                long_desc = ""

            rows.append((
                appid,
                str(name),
                shorts.get(appid, ""),
                long_desc,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(genres, ensure_ascii=False),
                str(r.release_date or ""),
                reviews,
                score,
                price,
                now,
            ))
            kept += 1

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO games (appid, name, short_desc, long_desc,"
                " tags, genres, release_date, review_count, review_score, price,"
                " fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        print(f"  {total:>7,} read  {kept:>7,} kept", end="\r")

    print(f"\n\nread   {total:,}")
    print(f"kept   {kept:,}")
    print("dropped:")
    for reason, n in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:18s} {n:>8,}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
