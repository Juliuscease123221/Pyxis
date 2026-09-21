# EXPLAIN.md

Running rationale log. Each phase appends while the reasoning is fresh; Phase 9
reorganises this into the full design-decisions / walkthrough / question-bank
document.

---

## Phase 1 — Data

**Acceptance: PASS.** 56,129 filtered games, 98.0% with 3+ community tags
(bar: 50k+ and ≥80%).

### Which dump, and why

Three candidates were named. Only one was evaluable without credentials:

| Dump | Verdict |
| --- | --- |
| Kaggle `hubertsidorowicz/steam-games-dataset-daily-updates` | **Not evaluated.** Kaggle requires an API token; no `~/.kaggle/kaggle.json` on this machine and no CLI. Cannot be checked for tag coverage without credentials. |
| HuggingFace `FronkonGames/steam-games-dataset` | **Chosen.** 125,855 rows, open access, and it carries SteamSpy community tags. |
| GitHub `vintagedon/steam-dataset-2025` | Read as prior art. The repo is 27MB of code and notebooks, not a bundled catalog dump; it collects from the Steam API itself, so using it means running its crawler rather than downloading a table. Rejected as a *starting point* for exactly the reason SPEC.md says to start from a dump: nothing should be blocked on a crawl. |

The deciding field was tag coverage, measured rather than assumed
(`ingest/evaluate_dumps.py`):

- 125,855 total rows; 66.2% have any tag at all
- after the ≥10-review filter: 56,662 survivors, **97.7%** with 3+ tags

The coverage gap is almost entirely in games we drop anyway. Tags come from
player votes, and a game with under 10 reviews has nobody voting. Tag
availability and the review threshold select for the same games, which is why
coverage jumps from 66% to 98% the moment the filter is applied. That is a
convenient accident of the domain, not a property of the dump.

**Two files, not one.** Neither published artifact is complete:

- `games.csv` (401MB) has the `Tags` column.
- `data/train-00000-of-00001.parquet` (184MB) has `short_description`, which
  the CSV lacks — but its own `Tags` column is typed `list<element: null>` and
  is empty for all 124,146 rows; the parquet conversion dropped it.

So tags come from the CSV, `short_desc` from the parquet, joined on appid. Had
we taken the obvious path of using only the parquet (smaller, typed, faster),
the deciding field would have been silently absent and every downstream cluster
would have been built on marketing copy alone.

### The malformed header

`games.csv` ships a 39-name header over 40-field data rows. Upstream emitted
`Discount` and `DLC count` as one token, `DiscountDLC count`, having dropped the
comma between them.

Read naively, pandas sees one fewer name than fields and silently promotes the
first data column (`AppID`) to the DataFrame index, shifting every named column
one place left. The failure is quiet and it is not uniform: `df['AppID']`
returns game *names*, which is loud enough to catch, but columns after the merged
token land back in near-alignment, so a coverage check can return a plausible
number while the loader is reading names into the appid field. Ours did exactly
that — the first load kept 23 of 125,855 rows.

The fix is `ingest/csv_schema.py`: an explicit 40-name `COLUMNS` list, always
read with `names=COLUMNS, header=0, index_col=False`. Alignment was verified
positionally (3000/3000 sampled rows carry 40 fields) and then semantically
against a known row — appid 367520 returns *Hollow Knight*, 403,641 positive /
12,305 negative, matching a live SteamSpy call exactly, with tags led by
Metroidvania and Souls-like.

**What would break if it were wrong:** everything, invisibly. Embeddings built
on a shifted frame would encode screenshot URLs as tag text and still produce a
map that looks fine from a distance.

### Filtering

Drops, from 125,855 to 56,129:

| Rule | Dropped |
| --- | --- |
| `few_reviews` (<10) | 60,534 |
| `name_pattern` | 8,288 |
| `software_genre` | 904 |

The name filter needed two passes. The first version matched bare `\bmovie\b`,
`\bdlc\b`, `\bartwork\b` and `\btrailer\b`, which is wrong: *The LEGO Movie -
Videogame* (5,264 reviews), *DLC Quest* (6,244), *Please, Touch The Artwork*,
*Trailer Shop Simulator* and *The LEGO NINJAGO Movie Video Game* (7,179) are all
real games. Auditing the drops by review count surfaced them — only 96 of the
8,358 name-drops had ≥10 reviews at all, and roughly ten of those 96 were false
positives.

The filter is now two-tier: unambiguous words (`playtest`, `soundtrack`,
`artbook`, `season pass`, `benchmark`) match anywhere; ambiguous ones are
anchored to the shapes that actually mark a non-game SKU — `(DLC)`, `- DLC`,
`OST` at end of title, `artwork pack`. Non-game *videos* are caught by genre
(`Documentary`, `Movie`, `Short`, `Episodic`) rather than by name, which is the
more reliable signal. `ingest/test_filters.py` pins all 26 names as regression
cases.

`software_genre` drops only when *every* genre on the app is a software genre,
so *Game Dev Tycoon* (Simulation + Game Development) survives while a pure
`Animation & Modeling` tool does not.

**Considered and rejected:** filtering on the `Categories` column, and trusting
the dataset's "Only published games, no DLCs" claim. The claim is false — the
very first row of the CSV is *Black Dragon Mage Playtest*.

### The crawler

SteamSpy, not the official Steam API. Community tags do not exist in Steam's
`appdetails` response; they live on the store page. SteamSpy's `appdetails`
returns the vote-ranked tag dict *and* current review counts in one call, which
is precisely the gap the dump leaves.

Resumability is per-row, not per-batch: each game commits immediately, a
`crawl_log` table records completions, and startup subtracts it from the
worklist. `failures` carries status and try count so a retry pass touches only
those. Verified by running twice — the second run skipped the first 12 appids
and advanced to the next 12.

Worklist order is deliberate: the ~1,100 games with under 3 tags first, then
everything else by review count descending. An interrupted crawl has therefore
already fixed the actual holes and refreshed the games most people will look at.

Rate limit 1.05 s/request (SteamSpy documents ~1/sec), so a full pass over 56k
games is ~16 hours. **Nothing downstream blocks on it** — the dump alone clears
acceptance, which is the fallback SPEC.md's risk table anticipated.

### Known weaknesses at this phase

- **Kaggle dump never compared.** It is the freshest of the three and claims
  136k+ games; it may have better coverage. Credentials would settle it.
- **Tags are a snapshot**, dated to the dump's build, not to today. The crawler
  refreshes them but will take ~16h to cover the catalog.
- **`review_count` mixes vintages** until the crawl completes: dump-loaded rows
  hold the dump's counts, crawled rows hold today's. Importance ranking in
  Phase 5 will be slightly inconsistent across the catalog until then.
- **451 distinct tags** is the whole vocabulary. Steam caps displayed tags at 20
  per game, so tag 21+ is invisible to us — a long-tail signal we simply do not
  have.
- **The 10-review threshold is a guess** inherited from SPEC.md, not a tuned
  parameter. It drops 60,534 games, by far the largest cut. Nothing has been
  measured about what is in there.
