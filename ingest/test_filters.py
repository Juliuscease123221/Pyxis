"""Tests for the Phase 1 ingest filters.

Run: python -m pytest ingest/test_filters.py -q

The keep-list is the important half. Every name in it is a real game that an
earlier, looser version of NAME_REJECT threw away, several with thousands of
reviews. They are regression cases: if someone re-broadens a pattern to a bare
\\bmovie\\b or \\bdlc\\b, these fail.
"""

from __future__ import annotations

import pytest

from .csv_schema import COLUMNS
from .load_dump import NAME_REJECT, classify_drop
from .evaluate_dumps import parse_tags

# Real games an over-broad filter wrongly dropped.
KEEP = [
    "The LEGO Movie - Videogame",
    "The LEGO NINJAGO Movie Video Game",
    "The LEGO Movie 2 Videogame",
    "DLC Quest",
    "Please, Touch The Artwork",
    "Trailer Shop Simulator",
    "She Sees Red - Interactive Movie",
    "Joe Danger 2: The Movie",
    "The Executive - Movie Industry Tycoon",
    "Democracy 3",
    "Ghostrunner",
    "Hollow Knight",
    "Half-Life 2: Episode One",
    "Serious Sam HD",
]

# Non-game SKUs that must not reach the map.
DROP = [
    "Gothic Playable Teaser",
    "Abbot's Book Demo",
    "MetaWare High School (Demo)",
    "Black Myth: Wukong Benchmark Tool",
    "Z1 Battle Royale: Test Server",
    "Wallpaper Engine",
    "Celeste Soundtrack",
    "Sinless + OST",
    "KINGDOM HEARTS III + Re Mind (DLC)",
    "Game Playtest",
    "Some Game - Season Pass",
    "Nier Artbook",
]


@pytest.mark.parametrize("name", KEEP)
def test_real_games_survive_name_filter(name):
    assert not NAME_REJECT.search(name), f"{name!r} is a real game but was dropped"


@pytest.mark.parametrize("name", DROP)
def test_non_games_are_dropped(name):
    assert NAME_REJECT.search(name), f"{name!r} is not a game but survived"


def test_review_threshold():
    assert classify_drop("Some Game", ["Action"], 9) == "few_reviews"
    assert classify_drop("Some Game", ["Action"], 10) is None


def test_software_genres_dropped_only_when_every_genre_is_software():
    # pure software -> drop
    assert classify_drop("Blender-like", ["Animation & Modeling"], 500) == "software_genre"
    # a game that also carries a software genre -> keep
    assert classify_drop("Game Dev Tycoon", ["Simulation", "Game Development"], 500) is None


def test_empty_name_dropped():
    assert classify_drop("", ["Action"], 500) == "no_name"


def test_parse_tags_forms():
    assert parse_tags("Metroidvania,Souls-like,2D") == ["Metroidvania", "Souls-like", "2D"]
    assert parse_tags("") == []
    assert parse_tags("[]") == []
    assert parse_tags(None) == []
    # dict form sorts by vote count descending
    assert parse_tags("{'Indie': 10, 'Metroidvania': 99}") == ["Metroidvania", "Indie"]


def test_csv_schema_is_forty_columns():
    """The dump's own header says 39; the data rows carry 40."""
    assert len(COLUMNS) == 40
    assert COLUMNS[7] == "Discount"
    assert COLUMNS[8] == "DLC count"
    assert COLUMNS[37] == "Tags"
