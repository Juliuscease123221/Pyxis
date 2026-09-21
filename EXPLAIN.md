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

### Filtering

Drops, from 125,855 to 56,129:

| Rule | Dropped |
| --- | --- |
| `few_reviews` (<10) | 60,534 |
| `name_pattern` | 8,288 |
| `software_genre` | 904 |

The filter is two-tier: unambiguous words (`playtest`, `soundtrack`, `artbook`,
`season pass`, `benchmark`) match anywhere; ambiguous ones are anchored to the
shapes that actually mark a non-game SKU — `(DLC)`, `- DLC`, `OST` at end of
title, `artwork pack`. That split was forced by bug 3 below. Non-game *videos*
are caught by genre (`Documentary`, `Movie`, `Short`, `Episodic`) rather than by
name, which is the more reliable signal.

`software_genre` drops only when *every* genre on the app is a software genre,
so *Game Dev Tycoon* (Simulation + Game Development) survives while a pure
`Animation & Modeling` tool does not.

**Considered and rejected:** filtering on the `Categories` column, and trusting
the dataset's "Only published games, no DLCs" claim. The claim is false — the
very first row of the CSV is *Black Dragon Mage Playtest*.

---

## Phase 1 — bugs found, how they surfaced, what they would have cost

Three real defects, all of the same species: **silent, plausible-looking
corruption of the input**. None of them raises an exception. Each produces a
dataset that loads, passes a casual glance, and yields a map that looks fine
from a distance. That is the failure mode this phase exists to catch, and it is
why SPEC.md says to validate upstream before moving downstream.

### Bug 1 — the parquet's `Tags` column is empty

**Symptom.** `pyarrow` reported the column's type as `list<element: null>`.
Every one of the 124,146 rows held an empty list; tag coverage was exactly 0.0%.
The `genres` column in the same file was populated normally, so the file looked
healthy.

**How it was found.** By reading the schema before reading the data. The type
`list<element: null>` is only producible when *every* value is an empty list —
arrow has no non-null element to infer a type from. The type annotation itself
was the tell, ahead of any row inspection. A coverage count then confirmed it:
0 of 124,146.

**Root cause.** Upstream, SteamSpy's tags are a `{tag: votes}` dict. The
parquet conversion appears to have coerced that dict to a list and dropped the
contents.

**What it would have cost.** The parquet is the file you would naturally
choose — 184MB against 401MB, typed, columnar, faster, and it is the one the
HuggingFace viewer shows by default. Taking it alone loses community tags
entirely, which are the deciding field for the whole project. Clustering would
then have run on store descriptions only: marketing copy in which, per
SPEC.md's own warning, every game claims to be an atmospheric epic adventure.
The map would have rendered beautifully and grouped games by *ad-copy style*
rather than gameplay — and nothing downstream would have flagged it, because
there is no error to raise. The differentiating claim of the project would have
been quietly false.

**Fix.** Tags from `games.csv`, `short_desc` from the parquet, joined on appid
(`ingest/load_dump.py`). Both files are required; neither is sufficient.

### Bug 2 — 39-name header over 40-field rows

**Symptom.** The first full load kept **23 rows out of 125,855**, attributing
125,809 of the drops to `bad_appid`. `int(r.appid)` was raising `ValueError:
invalid literal for int() with base 10: 'Black Dragon Mage Playtest'` — the
appid field contained a game *name*.

**How it was found.** The absurd keep count (23) made it impossible to miss;
the interesting part was diagnosing it. Printing the header alongside a data row
showed `About the game = '0'` and `Metacritic score = 'False'` — text and
booleans in numeric fields, all shifted by one from index 8 onward. Counting
fields confirmed it: the header carries 39 names, and 3000 of 3000 sampled data
rows carry 40. Upstream dropped the comma between `Discount` and `DLC count`,
emitting the single token `DiscountDLC count`.

Given one fewer header name than data fields, pandas silently promotes the first
data column (`AppID`) to the DataFrame index and shifts every named column one
place left.

**What it would have cost.** This one is nastier than it first appears, because
the corruption is *non-uniform*. The merged header token swallows two data
columns, so columns positioned after it drift back into near-alignment while
columns before it are badly wrong. The practical consequence: my
`evaluate_dumps.py` tag-coverage check returned **the same 97.7% both before and
after the fix** — the `Tags` column happened to land correctly under `usecols`.
So the coverage number, the thing gating the dump choice, looked completely
healthy while the loader was writing game names into the primary key.

Had the keep count not been so obviously broken, a slightly luckier shift would
have produced a database that loaded cleanly with `review_count` reading the
`Score rank` column — plausible integers, wrong field. Importance ranking in
Phase 5 and the frontier scoring in Phase 7 both key off `review_count`, so the
zoomed-out view would have shown a confidently wrong selection of games with no
symptom anywhere.

**Fix.** `ingest/csv_schema.py` pins an explicit 40-name `COLUMNS` list, always
read via `names=COLUMNS, header=0, index_col=False`. Verified positionally
(3000/3000 rows carry 40 fields) and then *semantically* against ground truth:
appid 367520 returns *Hollow Knight*, 403,641 positive / 12,305 negative,
matching a live SteamSpy call exactly. Positional verification alone would not
have been enough — it proves the count, not the alignment.

### Bug 3 — the name filter discarded real games

**Symptom.** None visible. The filter dropped 8,358 rows and the surviving
count still cleared the 50k bar, so every acceptance check passed.

**How it was found.** By deliberately auditing the drops rather than the keeps —
listing everything the name filter rejected, sorted by review count descending.
The top of that list was self-evidently wrong: *The LEGO NINJAGO Movie Video
Game* (7,179 reviews), *DLC Quest* (6,244), *The LEGO Movie - Videogame*
(5,264), *She Sees Red - Interactive Movie* (3,076), *Joe Danger 2: The Movie*,
*Please, Touch The Artwork*, *Trailer Shop Simulator*. My patterns `\bmovie\b`,
`\bdlc\b`, `\bartwork\b` and `\btrailer\b` were matching words that occur inside
legitimate game titles.

Only 96 of the 8,358 name-drops had ≥10 reviews at all, and about ten of those
were genuine false positives — roughly 0.02% of the catalog.

**What it would have cost.** Less than the other two, and worth stating
honestly: ~10 games out of 56,129 is a rounding error in any aggregate metric.
The cost is not statistical, it is demo-facing. These are recognisable titles,
and the cluster they belong to is one a viewer is likely to zoom into. A map
that silently lacks the LEGO games in its licensed-platformer region is wrong in
exactly the way a person notices and a benchmark does not.

The broader lesson is the transferable one: **acceptance checks measure what
survived, never what was discarded.** A filter can only fail in the direction
the checks do not look. Auditing drops is not optional.

**Fix.** Two-tier patterns (above), plus `ingest/test_filters.py` pinning all 26
probe names — 14 real games that must survive, 12 non-games that must not — as
regression cases, so re-broadening a pattern fails a test rather than silently
shrinking the catalog.

### Tag ordering — checked, not a bug

Not a defect, but the same class of risk, so it was verified rather than
assumed. The Phase 2 text template takes the *leading* tags as the most-voted
ones; had the dump or the join alphabetised them, "2D" and "Action" would head
every string and top-N truncation would select the alphabetically earliest tags
instead of the defining ones — silently, with no error.

`ingest/verify_tag_order.py` fetches live SteamSpy vote counts for 15 well-known
games and rank-correlates stored order against vote-descending order, with
alphabetical order as a control:

```
mean rank correlation vs VOTE order    1.000   (want ~1.0)
mean rank correlation vs ALPHA order  -0.071   (want ~0.0)
```

All 15 probes scored exactly 1.000 against votes. Stored order is vote order,
and the leading tags are the defining ones — *Hollow Knight* → Metroidvania,
Souls-like, Platformer; *The Witcher 3* → Open World, RPG, Story Rich; *Europa
Universalis IV* → Grand Strategy, Strategy, Historical. The control matters: a
high vote correlation would be meaningless if vote order and alphabetical order
happened to coincide.

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

---

## Phase 2 — Embeddings

**Acceptance: PASS.** `embed/variants.md` scores all three variants on both
metrics, chooses one, and records the reasoning. Chosen: **variant C, hybrid at
α = 0.5** — an equal blend of bge-small text embeddings and tag-co-occurrence
embeddings, 512-d.

Full reasoning lives in [embed/variants.md](embed/variants.md). The points that
belong in a design log rather than a results table:

### The comparison nearly ranked itself backwards

Both A and B take tags as input — A's template carries a `Tags:` line, B is
built from nothing else — so scoring either on a masked tag measures
memorisation. The leaky numbers said B beat A 80.5% to 57.4%. With the masked
tags removed at build time the honest gap is 52.9% to 50.1%: **2.8 points, not
23**.

The subtle part is that the leak size differs by variant (B −27.6 pts, A −7.3
pts), because tags are B's only input and one line of A's. Only B *looks* like
it needs a holdout, so the natural move — fix B, leave A — would have been
wrong in both directions simultaneously.

### A proxy metric was badly pessimistic

int8 quantisation preserved only **71.8%** of each game's top-10 neighbourhood
against fp32, at mean cosine 0.9725. For a product that is entirely nearest
neighbours, that reads as disqualifying.

It was not. Encoding both in full and scoring downstream gave 57.4% vs 57.0%
tag prediction and identical 1.28x genre lift. The reordering sits among
near-ties that are semantically interchangeable — swapping rank 4 and rank 7
among a dozen equivalent metroidvanias moves the statistic and moves nothing a
viewer sees. int8 ships: 1.6x faster, 4x smaller, no measurable cost.

The general lesson: a proxy that is *cheap* is not thereby *conservative*. It
was wrong in the safe-looking direction, which is the direction that gets
believed.

### A hypothesis tested and rejected

Variant A retrieved on apparent title collisions — Hollow Knight → "Hollow
Floor", "Hollowed"; Cuphead → "Gunheart", "Head Shot", "SQUAREHEAD". The obvious
diagnosis was that the template leads with the name.

Encoding without the title refuted it: Hollow Knight still returns "Hollow
Floor", Cuphead still returns "Gunheart". The collisions are driven by
description prose, not names — every one of those games advertises itself as a
fast, punishing action game with striking art. That is exactly the failure
SPEC.md predicted from marketing copy; the title was a red herring, and the
name stays in the template.

### Metrics that did not earn their place

Genre agreement scored 1.28–1.30x for **every variant**, including pure text.
Steam has ~20 genres for 56,129 games and `Indie` alone covers 59%, so random
pairs already agree 75.4% of the time. It is in SPEC.md, it is reported, and
it contributed nothing. Saying so is more useful than presenting it as
corroboration.

Top-tag agreement discriminated well (57.5% → 84.5%) but is circular: it asks
whether an embedding clusters by its own input, which structurally favours
tag-derived variants. Reported beside the held-out number, never instead of it.

That left the α tie-break to 15 hand-picked spot checks — a sample of 15,
selected by the person with a view on the answer. Worth stating plainly.

### Why α=0.5 over α=0.7

The clean metrics disagreed: α=0.7 is 1.8 points better at recovering a masked
tag, α=0.5 is 5.6 points better at clustering games sharing their defining tag.
They pull apart because masked-tag prediction rewards inferring tags from prose
— the confounded capability.

Spot checks favoured α=0.5 on 3 of 4 probes. The deciding case was Europa
Universalis IV: α=0.7 drops *Victoria II*, *Victoria 3* and *Imperator: Rome*
for the *Making History* series. Those Paradox titles are near-identical in
play, and a map that separates them is wrong in a way a viewer notices
immediately. α=0.5 is also the only α that means what its number says.

### The quadratic-weight trap

Both halves are unit vectors, so for `[α·A, (1−α)·B]` the hybrid cosine is
`(α²·cos_A + (1−α)²·cos_B) / (α² + (1−α)²)`. Weights go as **α²**. α=0.7 is
84/16, not 70/30. Reporting it as 70/30 would misdescribe the result by a
factor of two.

Relatedly, dimension counts carry no weight of their own: A is 384-d and B is
128-d, but each is unit-norm before scaling, so they contribute equally at
α=0.5 despite the 3:1 size difference.

### Variant B is the quiet result

453 tags, no neural model, **3.4 seconds**, and it beats the transformer on
every clean metric (52.9% vs 50.1% tag prediction; 83.6% vs 57.5% top-tag). The
chosen hybrid beats B by 3.1 points — real, but modest against a 165x build-time
gap. If the transformer half were cut, B alone would ship.

SPEC.md predicted this ("B often wins; that is a finding worth reporting").
It was right, and for the reason given: thousands of players voting on what a
game *is* beats one publisher describing what they wish it were.

### Known weaknesses at this phase

- α was searched at three points, not tuned; the optimum is between 0.5 and 0.7
  and was not located.
- Spot checks decided the tie-break on a self-selected sample of 15.
- Every metric here is k-NN in embedding space. Whether these clusters survive
  HDBSCAN and UMAP into something navigable is Phases 3–4, and nothing yet
  proves it.
- Review counts still mix dump and crawl vintages — irrelevant to embeddings,
  relevant to Phase 5 importance ranking.
