"""Resumable SteamSpy crawler: fills tag gaps and refreshes review counts.

Why SteamSpy rather than the official Steam API: community tags are the
deciding field for clustering (SPEC.md Phase 1) and the official
``appdetails`` endpoint does not return them -- they live on the store page.
SteamSpy's ``appdetails`` returns the vote-ranked tag dict plus current
positive/negative counts in a single call, which is exactly the gap the dump
leaves.

Resumability, per SPEC.md:
  * every game is written to SQLite immediately, one row at a time
  * on startup the set of already-crawled appids is loaded and skipped
  * failures go to their own table with a status and a try count, so a retry
    pass hits only those

Rate limit: SteamSpy documents 1 request/second for appdetails. The default
here is slightly above that. Nothing downstream blocks on this process -- the
dump already clears the Phase 1 acceptance bar on its own, so the crawler is
enrichment and can be interrupted at any point.

Usage:
    python -m ingest.crawl                  # gaps first, then refresh by rank
    python -m ingest.crawl --mode gaps      # only games with <3 tags
    python -m ingest.crawl --mode retry     # only previously failed appids
    python -m ingest.crawl --limit 500
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

from .db import DEFAULT_DB, connect

STEAMSPY_URL = "https://steamspy.com/api.php"
USER_AGENT = "overworld-steam-map/0.1 (dataset research; contact via repo)"
MIN_TAGS = 3

CRAWL_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS crawl_log (
  appid INTEGER PRIMARY KEY,
  fetched_at TEXT,
  n_tags INTEGER
);
"""

_stop = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    """Finish the in-flight request, then exit cleanly at the top of the loop."""
    global _stop
    _stop = True
    print("\n  interrupt received - finishing current request, then stopping")


def build_worklist(conn: sqlite3.Connection, mode: str, limit: int | None) -> list[int]:
    """Appids to crawl, most valuable first, excluding ones already done.

    gaps    games whose dump tags are missing or too few -- the actual holes
    retry   appids in `failures`, for a second pass
    all     gaps first, then everything else by review count descending, so an
            interrupted run has still refreshed the games most people see
    """
    done = {r[0] for r in conn.execute("SELECT appid FROM crawl_log")}

    def _filter(rows):
        out = [r[0] for r in rows if r[0] not in done]
        return out[:limit] if limit else out

    if mode == "retry":
        rows = conn.execute(
            "SELECT appid FROM failures WHERE tries < 3 ORDER BY tries, appid"
        ).fetchall()
        return [r[0] for r in rows][:limit] if limit else [r[0] for r in rows]

    gaps = conn.execute(
        "SELECT appid FROM games"
        " WHERE tags IS NULL OR tags = '[]' OR json_array_length(tags) < ?"
        " ORDER BY review_count DESC",
        (MIN_TAGS,),
    ).fetchall()
    work = _filter(gaps)
    if mode == "gaps":
        return work

    if limit is None or len(work) < limit:
        rest = conn.execute(
            "SELECT appid FROM games"
            " WHERE json_array_length(tags) >= ?"
            " ORDER BY review_count DESC",
            (MIN_TAGS,),
        ).fetchall()
        remaining = None if limit is None else limit - len(work)
        extra = [r[0] for r in rest if r[0] not in done]
        work += extra if remaining is None else extra[:remaining]
    return work


def fetch(session: requests.Session, appid: int, timeout: float = 20.0) -> dict | None:
    """One SteamSpy appdetails call. Returns the payload, or None if unusable.

    Raises requests exceptions to the caller, which records them as failures.
    """
    resp = session.get(
        STEAMSPY_URL,
        params={"request": "appdetails", "appid": appid},
        timeout=timeout,
    )
    if resp.status_code == 429:
        raise requests.HTTPError("429 rate limited", response=resp)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        return None
    # SteamSpy answers unknown appids with a body whose name is null.
    if not isinstance(data, dict) or not data.get("name"):
        return None
    return data


def apply_update(conn: sqlite3.Connection, appid: int, data: dict, now: str) -> int:
    """Write one crawled record. Returns the number of tags found.

    Tags are stored ordered by vote count descending, matching the dump's
    convention so downstream code needs no special case.
    """
    raw_tags = data.get("tags") or {}
    if isinstance(raw_tags, dict):
        tags = [t for t, _ in sorted(raw_tags.items(), key=lambda kv: -int(kv[1] or 0))]
    elif isinstance(raw_tags, list):
        tags = [str(t) for t in raw_tags]
    else:
        tags = []

    pos = int(data.get("positive") or 0)
    neg = int(data.get("negative") or 0)
    reviews = pos + neg
    score = (pos / reviews) if reviews else None

    # Only overwrite tags when the crawl actually found some: a transient
    # empty response must not erase good dump data.
    if tags:
        conn.execute(
            "UPDATE games SET tags = ?, review_count = ?, review_score = ?,"
            " fetched_at = ? WHERE appid = ?",
            (json.dumps(tags, ensure_ascii=False), reviews, score, now, appid),
        )
    elif reviews:
        conn.execute(
            "UPDATE games SET review_count = ?, review_score = ?, fetched_at = ?"
            " WHERE appid = ?",
            (reviews, score, now, appid),
        )

    conn.execute(
        "INSERT OR REPLACE INTO crawl_log (appid, fetched_at, n_tags) VALUES (?,?,?)",
        (appid, now, len(tags)),
    )
    conn.execute("DELETE FROM failures WHERE appid = ?", (appid,))
    conn.commit()
    return len(tags)


def record_failure(conn: sqlite3.Connection, appid: int, status: str) -> None:
    conn.execute(
        "INSERT INTO failures (appid, status, tries) VALUES (?, ?, 1)"
        " ON CONFLICT(appid) DO UPDATE SET status = excluded.status,"
        " tries = failures.tries + 1",
        (appid, status[:200]),
    )
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--mode", choices=["all", "gaps", "retry"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.05,
                    help="seconds between requests (SteamSpy allows ~1/sec)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    conn = connect(args.db)
    conn.executescript(CRAWL_LOG_SCHEMA)

    work = build_worklist(conn, args.mode, args.limit)
    already = conn.execute("SELECT COUNT(*) FROM crawl_log").fetchone()[0]
    print(f"mode={args.mode}  already crawled={already:,}  queued={len(work):,}")
    if not work:
        print("nothing to do")
        return 0
    eta_h = len(work) * args.delay / 3600
    print(f"delay={args.delay}s  estimated {eta_h:.1f}h  (resumable; Ctrl-C is safe)\n")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    ok = miss = fail = tags_gained = 0
    now_start = time.time()

    for i, appid in enumerate(work, 1):
        if _stop:
            break
        t0 = time.time()
        try:
            data = fetch(session, appid)
            if data is None:
                miss += 1
                record_failure(conn, appid, "no_data")
                conn.execute(
                    "INSERT OR REPLACE INTO crawl_log (appid, fetched_at, n_tags)"
                    " VALUES (?,?,0)",
                    (appid, datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )
                conn.commit()
            else:
                n = apply_update(
                    conn, appid, data,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                ok += 1
                tags_gained += n
        except requests.HTTPError as e:
            fail += 1
            record_failure(conn, appid, f"http:{e}")
            if "429" in str(e):
                print("\n  rate limited - backing off 60s")
                time.sleep(60)
        except requests.RequestException as e:
            fail += 1
            record_failure(conn, appid, f"net:{type(e).__name__}")

        if i % 25 == 0 or i == len(work):
            rate = i / max(time.time() - now_start, 1e-6)
            print(f"  {i:>7,}/{len(work):,}  ok={ok:,} miss={miss:,} fail={fail:,}"
                  f"  {rate:.2f}/s", end="\r")

        elapsed = time.time() - t0
        if elapsed < args.delay:
            time.sleep(args.delay - elapsed)

    print(f"\n\nstopped after {i:,} requests")
    print(f"  ok={ok:,}  no_data={miss:,}  failed={fail:,}")
    if ok:
        print(f"  mean tags on success: {tags_gained / ok:.1f}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
