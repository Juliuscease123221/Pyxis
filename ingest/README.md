# ingest — Phase 1

Turns the Steam catalog into `data/pyxis.db`, filtered and tagged.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 1. fetch the dump (two files; neither is complete on its own)
curl -L -o data/games.csv \
  https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/games.csv
curl -L -o data/fronkon.parquet \
  https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/data/train-00000-of-00001.parquet

# 2. check tag coverage before committing to the dump
python -m ingest.evaluate_dumps data/games.csv

# 3. load + filter into SQLite
python -m ingest.load_dump

# 4. gate the phase
python -m ingest.acceptance

# 5. enrichment, runs alongside everything else (resumable, Ctrl-C safe)
python -m ingest.crawl --mode all
```

## Files

| File | Job |
| --- | --- |
| `csv_schema.py` | The corrected 40-column layout. The published CSV's header has 39 names for 40 fields; read it any other way and every column silently shifts. |
| `evaluate_dumps.py` | Tag-coverage report. The deciding field for choosing a dump. |
| `load_dump.py` | CSV+parquet → SQLite, with the DLC/demo/soundtrack/software filters. |
| `crawl.py` | Resumable SteamSpy crawler. Fills tag gaps, refreshes review counts. |
| `acceptance.py` | Phase 1 gate. Exits non-zero on failure. |
| `test_filters.py` | Regression tests, mostly real games an over-broad filter wrongly dropped. |

## Result

```
games                      56,129
games with >=3 tags        55,014  (98.0%)
median tags per game           18
distinct tags                 451
short_desc / long_desc    100.0% / 99.8%
```

## Two things worth knowing

**The parquet's `Tags` column is empty.** All 124,146 rows, typed
`list<element: null>`. Tags come from `games.csv`; only `short_description`
comes from the parquet. Using the parquet alone — the obvious choice, it being
smaller and typed — silently loses the single most important field.

**Tag coverage and the review threshold select for the same games.** Coverage
is 66% over the raw dump and 98% after dropping games with under 10 reviews,
because tags are player votes and unreviewed games have no voters.
