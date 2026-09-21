"""Corrected column layout for the FronkonGames games.csv dump.

The published CSV has a malformed header: it carries 39 names while every data
row carries 40 fields. The cause is a missing comma between ``Discount`` and
``DLC count``, which the upstream writer emitted as the single token
``DiscountDLC count``.

Read naively, pandas sees one fewer header name than data fields and silently
promotes the first data column (AppID) to the index, shifting every named
column one position to the left. ``Tags`` then reads the ``Screenshots``
column, which is why a naive coverage check reports plausible-looking
nonsense. Always read this file with ``names=COLUMNS, header=0,
index_col=False``.

Verified against the file: 3000/3000 sampled data rows have exactly 40 fields,
and the alignment below reproduces sane values for Windows/Mac/Linux (bools),
Positive/Negative (ints) and Screenshots/Movies (URLs).
"""

from __future__ import annotations

COLUMNS: list[str] = [
    "AppID",                       # 0
    "Name",                        # 1
    "Release date",                # 2
    "Estimated owners",            # 3
    "Peak CCU",                    # 4
    "Required age",                # 5
    "Price",                       # 6
    "Discount",                    # 7   <- split from 'DiscountDLC count'
    "DLC count",                   # 8   <- split from 'DiscountDLC count'
    "About the game",              # 9
    "Supported languages",         # 10
    "Full audio languages",        # 11
    "Reviews",                     # 12
    "Header image",                # 13
    "Website",                     # 14
    "Support url",                 # 15
    "Support email",               # 16
    "Windows",                     # 17
    "Mac",                         # 18
    "Linux",                       # 19
    "Metacritic score",            # 20
    "Metacritic url",              # 21
    "User score",                  # 22
    "Positive",                    # 23
    "Negative",                    # 24
    "Score rank",                  # 25
    "Achievements",                # 26
    "Recommendations",             # 27
    "Notes",                       # 28
    "Average playtime forever",    # 29
    "Average playtime two weeks",  # 30
    "Median playtime forever",     # 31
    "Median playtime two weeks",   # 32
    "Developers",                  # 33
    "Publishers",                  # 34
    "Categories",                  # 35
    "Genres",                      # 36
    "Tags",                        # 37
    "Screenshots",                 # 38
    "Movies",                      # 39
]

assert len(COLUMNS) == 40, "corrected header must have 40 fields"


def read_kwargs(**extra: object) -> dict:
    """Keyword arguments for ``pandas.read_csv`` that read the dump correctly."""
    kw: dict = {
        "names": COLUMNS,
        "header": 0,        # skip the malformed header row, use COLUMNS instead
        "index_col": False,  # never promote a data column to the index
        "low_memory": False,
        "on_bad_lines": "warn",
    }
    kw.update(extra)
    return kw
