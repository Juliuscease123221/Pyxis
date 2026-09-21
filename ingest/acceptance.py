"""Phase 1 acceptance check.

SPEC.md: "SQLite file with 50k+ filtered games, >=80% having 3+ community
tags."

Exits non-zero if either condition fails, so it can gate the phase.

Usage:
    python -m ingest.acceptance
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .db import DEFAULT_DB, connect

MIN_GAMES = 50_000
MIN_TAGS = 3
MIN_COVERAGE = 0.80


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    conn = connect(args.db)
    rows = conn.execute("SELECT appid, name, tags, genres, review_count FROM games").fetchall()
    n = len(rows)

    tag_counts = []
    tag_freq: Counter[str] = Counter()
    n_short = n_long = 0
    for r in rows:
        tags = json.loads(r["tags"] or "[]")
        tag_counts.append(len(tags))
        tag_freq.update(tags)
    for r in conn.execute(
        "SELECT SUM(LENGTH(short_desc)>0), SUM(LENGTH(long_desc)>0) FROM games"
    ):
        n_short, n_long = r[0] or 0, r[1] or 0

    with_tags = sum(1 for c in tag_counts if c >= MIN_TAGS)
    coverage = with_tags / n if n else 0.0
    median_tags = sorted(tag_counts)[n // 2] if n else 0

    print("=" * 62)
    print("PHASE 1 ACCEPTANCE")
    print("=" * 62)
    print(f"database                       {args.db}")
    print(f"games                          {n:>9,}")
    print(f"games with >={MIN_TAGS} tags            {with_tags:>9,}  ({coverage * 100:.1f}%)")
    print(f"median tags per game           {median_tags:>9}")
    print(f"distinct tags                  {len(tag_freq):>9,}")
    print(f"games with a short_desc        {n_short:>9,}  ({n_short / n * 100:.1f}%)")
    print(f"games with a long_desc         {n_long:>9,}  ({n_long / n * 100:.1f}%)")

    print("\n20 most common tags:")
    for tag, c in tag_freq.most_common(20):
        print(f"  {tag:<28s} {c:>7,}")

    ok_count = n >= MIN_GAMES
    ok_cov = coverage >= MIN_COVERAGE
    print("\n" + "-" * 62)
    print(f"  [{'PASS' if ok_count else 'FAIL'}]  50k+ filtered games        "
          f"{n:,} / {MIN_GAMES:,}")
    print(f"  [{'PASS' if ok_cov else 'FAIL'}]  >=80% with 3+ tags         "
          f"{coverage * 100:.1f}% / {MIN_COVERAGE * 100:.0f}%")
    print("-" * 62)

    passed = ok_count and ok_cov
    print(f"\nPHASE 1: {'PASS' if passed else 'FAIL'}")
    conn.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
