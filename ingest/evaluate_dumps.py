"""Phase 1 dump evaluation: pick a dump by community-tag coverage.

SPEC.md names community tags as the deciding field. Official genres are a
coarse 20-value taxonomy; user-voted tags ("Souls-like", "Bullet Hell") are the
clustering signal we actually want. This script reports coverage so the choice
is made on a number rather than a README claim.

Usage:
    python -m ingest.evaluate_dumps data/games.csv
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

from .csv_schema import read_kwargs

# Columns we need for the coverage question. Reading a subset keeps the
# 400MB CSV manageable.
USECOLS = ["AppID", "Name", "Positive", "Negative", "Genres", "Tags", "Categories"]

MIN_REVIEWS = 10  # SPEC.md filter threshold
MIN_TAGS = 3  # acceptance requires >=80% of survivors having this many


def parse_tags(value: object) -> list[str]:
    """Tags arrive as a comma-joined string or a repr'd list/dict.

    The FronkonGames CSV writes SteamSpy's ``{tag: votes}`` dict as a plain
    comma-separated string in vote order. Older rows may carry a literal dict.
    Return a list of tag names in the order given.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s in {"[]", "{}"}:
        return []
    if s[0] in "[{":
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, dict):
            # dict is {tag: votes}; sort by votes descending
            return [k for k, _ in sorted(parsed.items(), key=lambda kv: -kv[1])]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def main(path: str) -> int:
    csv_path = Path(path)
    print(f"Evaluating dump: {csv_path}  ({csv_path.stat().st_size / 1e6:.0f} MB)\n")

    chunks = pd.read_csv(
        csv_path, **read_kwargs(usecols=USECOLS, chunksize=50_000)
    )
    frames = []
    for ch in chunks:
        ch["n_tags"] = ch["Tags"].map(lambda v: len(parse_tags(v)))
        ch["n_genres"] = ch["Genres"].map(lambda v: len(parse_tags(v)))
        # pandas 3 infers these as string dtype; coerce before adding or the
        # '+' concatenates instead of summing.
        pos = pd.to_numeric(ch["Positive"], errors="coerce").fillna(0)
        neg = pd.to_numeric(ch["Negative"], errors="coerce").fillna(0)
        ch["reviews"] = pos + neg
        frames.append(ch[["AppID", "Name", "n_tags", "n_genres", "reviews"]])
    df = pd.concat(frames, ignore_index=True)

    print(f"total rows                     {len(df):>8,}")
    print(f"rows with >=1 tag              {(df.n_tags >= 1).sum():>8,}"
          f"  ({(df.n_tags >= 1).mean() * 100:5.1f}%)")
    print(f"rows with >={MIN_TAGS} tags              "
          f"{(df.n_tags >= MIN_TAGS).sum():>8,}"
          f"  ({(df.n_tags >= MIN_TAGS).mean() * 100:5.1f}%)")

    survivors = df[df.reviews >= MIN_REVIEWS]
    print(f"\n--- after the >={MIN_REVIEWS}-review filter "
          f"(DLC/soundtrack filtering happens at load) ---")
    print(f"survivors                      {len(survivors):>8,}")
    if len(survivors):
        cov = (survivors.n_tags >= MIN_TAGS).mean()
        print(f"survivors with >={MIN_TAGS} tags         "
              f"{(survivors.n_tags >= MIN_TAGS).sum():>8,}  ({cov * 100:5.1f}%)")
        print(f"median tags per survivor       {survivors.n_tags.median():>8.0f}")
        print(f"median genres per survivor     {survivors.n_genres.median():>8.0f}")

        print("\nACCEPTANCE (Phase 1):")
        ok_count = len(survivors) >= 50_000
        ok_cov = cov >= 0.80
        print(f"  50k+ filtered games       {'PASS' if ok_count else 'FAIL'}"
              f"  ({len(survivors):,})")
        print(f"  >=80% with 3+ tags        {'PASS' if ok_cov else 'FAIL'}"
              f"  ({cov * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/games.csv"))
