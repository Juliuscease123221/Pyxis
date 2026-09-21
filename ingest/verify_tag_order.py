"""Verify that stored tag order is by vote count, not alphabetical.

The Phase 2 text template takes the leading tags as the most-voted ones. If the
join or the dump had alphabetised them, "2D" and "Action" would lead every
string and the top-N truncation would systematically select the alphabetically
earliest tags rather than the defining ones -- quietly, with no error anywhere.

Method: for each probe game, fetch SteamSpy live (which returns {tag: votes}),
and compare the stored order against the vote-descending order by Spearman
rank correlation. Also test against the alphabetical ordering as a control, so
a pass means "matches votes AND does not merely match the alphabet".

Usage:
    python -m ingest.verify_tag_order
"""

from __future__ import annotations

import json
import sys
import time

import requests

from .crawl import STEAMSPY_URL, USER_AGENT
from .db import connect

# Well-known games with unambiguous defining tags.
PROBES = {
    367520: "Hollow Knight",
    620: "Portal 2",
    292030: "The Witcher 3: Wild Hunt",
    570: "Dota 2",
    413150: "Stardew Valley",
    1145360: "Hades",
    105600: "Terraria",
    268910: "Cuphead",
    588650: "Dead Cells",
    322330: "Don't Starve Together",
    546560: "Half-Life: Alyx",
    252490: "Rust",
    236850: "Europa Universalis IV",
    1174180: "Red Dead Redemption 2",
    391540: "Undertale",
}


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy. Both inputs are already rank vectors."""
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> int:
    conn = connect()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"{'game':<28}{'n':>4}{'vs votes':>10}{'vs alpha':>10}  stored order (first 5)")
    print("-" * 104)

    vote_scores: list[float] = []
    alpha_scores: list[float] = []
    checked = 0

    for appid, label in PROBES.items():
        row = conn.execute("SELECT name, tags FROM games WHERE appid = ?", (appid,)).fetchone()
        if row is None:
            print(f"{label:<28}  not in db (filtered out)")
            continue
        stored = json.loads(row["tags"] or "[]")
        if len(stored) < 3:
            print(f"{label:<28}  only {len(stored)} stored tags, skipping")
            continue

        resp = session.get(
            STEAMSPY_URL, params={"request": "appdetails", "appid": appid}, timeout=20
        )
        live = (resp.json() or {}).get("tags") or {}
        time.sleep(1.05)
        if not isinstance(live, dict) or not live:
            print(f"{label:<28}  no live tag data, skipping")
            continue

        common = [t for t in stored if t in live]
        if len(common) < 3:
            print(f"{label:<28}  only {len(common)} tags in common, skipping")
            continue

        stored_rank = list(range(len(common)))
        by_votes = sorted(common, key=lambda t: -int(live[t]))
        vote_rank = [by_votes.index(t) for t in common]
        by_alpha = sorted(common, key=str.lower)
        alpha_rank = [by_alpha.index(t) for t in common]

        rv = spearman(stored_rank, vote_rank)
        ra = spearman(stored_rank, alpha_rank)
        vote_scores.append(rv)
        alpha_scores.append(ra)
        checked += 1

        print(f"{row['name'][:27]:<28}{len(common):>4}{rv:>10.3f}{ra:>10.3f}"
              f"  {', '.join(stored[:5])}")

    if not checked:
        print("\nno games could be checked")
        return 1

    mv = sum(vote_scores) / checked
    ma = sum(alpha_scores) / checked
    print("-" * 104)
    print(f"mean rank correlation vs VOTE order  {mv:>7.3f}   (want ~1.0)")
    print(f"mean rank correlation vs ALPHA order {ma:>7.3f}   (want ~0.0)")

    ok = mv > 0.90 and ma < 0.5
    print(f"\nTAG ORDER: {'PASS - ordered by vote count' if ok else 'FAIL'}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
