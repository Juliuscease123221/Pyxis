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

---

## Phase 3 — Cluster hierarchy and labels

**Acceptance: PASS.** `data/cluster/tree_k12_labelled.json` — walking root to
leaf gives labels a person would say out loud. Full reasoning in
[cluster/RATIONALE.md](cluster/RATIONALE.md).

```
Hollow Knight  →  Platformer   →  Metroidvania / Exploration  →  Souls-like / Difficult / Dark Fantasy
Terraria       →  Simulation   →  Open World Survival Craft   →  Online Co-Op / Co-op
Dota 2         →  Free to Play →  Multiplayer / Shooter       →  PvP / RTS / MOBA
```

**Shipped: recursive spherical k-means, not HDBSCAN.** The HDBSCAN path was
built, collapsed, labelled and read before being rejected on evidence. It stays
in the repo because the comparison is the result.

### PCA quietly undid the Phase 2 decision

SPEC.md asks for 50 dims at ~90% variance. On this embedding that needs 208
dims; 50 gives 54.6%. The two constraints are incompatible and one has to go.

The worse problem was invisible. Variant C is `[0.5·A₃₈₄, 0.5·B₁₂₈]`; both
halves carry equal total variance, but the tag half packs it into 128 dims
while the text half spreads it over 384, so PCA takes tag directions first. At
50 dims the clustering input is **85.8% tag loading** — the α=0.5 blend chosen
in Phase 2, nullified by a preprocessing step, with every number still looking
reasonable.

The shipped path sidesteps it: k-means does not need the density contrast PCA
was there to rescue, so it runs on the full 512-d vector and the blend survives
exactly as chosen. Had the pipeline stayed on HDBSCAN, `--mode balanced` (PCA
each half separately, rejoin at parity) was the fix.

### The tree was a caterpillar, and the reported noise was the wrong number

HDBSCAN returned **84.2% noise** — but that is EOM *flat-label selection*,
which this pipeline never uses. The condensed tree assigns 100% of games to
some node (verified: 56,129 rows, 56,129 unique). The honest figure is games
falling out at the root: **20.6%** at min_samples=1, 39.8% at 5.

Quoting 84.2% would have made a structural problem look like a data problem.

The real defect was shape: 562 nodes, **depth 143**, branching factor 2 at
every level — one spine shedding a small cluster at each step while the bulk
continued down. SPEC.md predicts "a cluster divides into itself plus four
noise points"; the condensed tree has already absorbed that, so what survives
is the same pathology applied to *clusters*. The warning was right, one level
above where it was expected.

Collapse measured: 562 → 304 nodes, depth 143 → 6, mean fan-out 4.61. **210
spine merges, 0 small-branch dissolves, 0 pass-through dissolves.** The two
rules SPEC.md describes fired zero times; the entire restructuring came from
spine collapse, which is not in the spec.

### Every threshold bought the same trade-off

| spine_ratio | unclustered | biggest node |
| ---: | ---: | ---: |
| 0.3 | 63.7% | 28.8% |
| 0.9 | 20.6% | 79.4% |

Two descriptions of one fact: half the catalog is a connected region with no
internal density structure, and tuning only chooses whether to call it "noise"
or "one enormous cluster". Reading the labels ended it — the 49% node came out
"Visual Novel / Action Roguelike / Turn-Based Strategy", and Hollow Knight's
path terminated at **"Golf / Sokoban / Pinball"**.

Steam is a continuum. There is no density valley between action-roguelikes and
action-platformers. HDBSCAN was answering honestly; the question does not suit
it. Confirmed across 40+ configurations, with cosine geometry, and on the pure
tag space — every one gave ≥73% noise or 2–3 mega-clusters.

### Labelling: the metric improved while the product got worse

Sibling scoping is the requirement; the denominator took three attempts.

1. `tf·log(1 + n/df)` — a tag on every sibling still scores log 2, and generic
   tags have the highest tf, so they win. **90.0%** of children reused a parent
   term: "Action → Action → Action Indie", verbatim.
2. `tf·log(n/df)` — now zero for a tag on every sibling, which kills generic
   terms *and* the defining term of any coherent cluster. Whatever tag sits in
   exactly one sibling wins, however rare. Stardew Valley's neighbourhood came
   out **"Memes / Naval Combat / Pirates"**. Repetition fell to 17.7% while the
   labels got considerably worse — the metric moved the wrong way round.
3. **Prevalence × lift**, which ships:
   `score = p_c(t) · log((p_c(t)+ε)/(p_rest(t)+ε))`, coverage on both sides.

Both failures share one cause: `df` is a *binary presence count* and cannot
tell "on 90% of every sibling" from "on 2% of every sibling". The fix is to
compare degrees rather than presence against absence.

SPEC.md warns that special-casing common tags signals wrong scoping. True —
and the corollary is that the scope must be a *graded* sibling comparison. With
that, commonness handles itself: 20.2% repetition, 0 unnamed nodes, no
stopword list anywhere.

### Noise handling, as decided

The shipped tree has **0% unclustered** — k-means places every game in a leaf,
so nothing is soft-assigned and nothing renders specially.

For the HDBSCAN path both options SPEC.md asks to choose between are
implemented in `assign_noise`: `soft` (nearest leaf centroid by cosine, every
game gets a position, the cost being that an isolated game inherits a label
that is a small lie) and `keep` (left at the root as explicit unclustered
territory for the map to render dimmer and unlabelled). `soft` was the default
there. Both record the assignment per game so a viewer can distinguish core
members from soft-assigned ones. Neither lets noise disappear.

A double-counting bug in the first implementation reported 184.2% coverage:
soft-assigned games were appended to their nearest leaf while still counted at
the root. Fixed by reassigning rather than appending; the report now prints
distinct-vs-total and flags duplicate placements.

### Phase 8 substitution: NMI against tags, not Steam genres

Phase 8 specifies NMI against Steam genres. Phase 2 already showed those genres
carry almost no information here — genre agreement was **1.28–1.30× over
baseline for every embedding variant**, including pure text, because ~20 genres
cover 56,129 games and `Indie` alone covers 59%. A metric that could not
separate three very different embeddings will not separate cluster depths.

**Plan:** measure NMI against **community tags** (453 values, demonstrated
discriminative power: 57.5% → 84.5% on top-tag agreement), and report genre NMI
beside it explicitly labelled as the uninformative baseline — so the
substitution is visible rather than a quiet swap.

### Known weaknesses at this phase

- Every k-means split is forced; the method cannot say "no structure here", so
  some depth-3 leaves are arbitrary slices of a continuum.
- Depth 4 holds 0.9% of games — `min_split=120` halts before the depth cap, so
  the map gets three levels of real semantics, not six.
- "Indie / Casual / VR" (5,522 games) is a residue bucket, not a genre.
- Fallout 4 lands under "FPS / Shooter"; Dota 2 reaches MOBA only at depth 3.
- k was chosen from two candidates (8, 12), not searched.
- Labels are raw tag terms joined by slashes; LLM tidying is cut line 1.

---

## Phase 3 (revised) — what actually shipped, and the correction that got there

The section above records Phase 3 as it happened. Its conclusion — that
HDBSCAN cannot work on this data — was **wrong in its reasoning**, and the
correction changed the shipped tree. Both are kept: the sequence is the
result.

**Shipped:** `tree_leiden_labelled.json` — multi-resolution Leiden over a
cosine k-NN graph on the full 512-d vectors, nested by membership containment.

### Headline result: PCA silently reversed a decision from the previous phase

This is the most transferable finding in the project so far, and it is worth
stating on its own.

Variant C is `[0.5·A₃₈₄, 0.5·B₁₂₈]` — a deliberate 50/50 blend of text and tag
embeddings, chosen in Phase 2 on measured evidence. Both halves carry equal
*total* variance by construction. But the tag half packs that variance into
128 dimensions while the text half spreads it over 384, so the densest
directions in the concatenated space are overwhelmingly tag directions. PCA
takes directions in variance order:

| retained dims | text loading | tag loading |
| ---: | ---: | ---: |
| 50 | 14.2% | **85.8%** |
| 128 | 12.3% | 87.7% |
| 256 | 51.0% | 49.0% |

**At the spec's 50 dims, the clustering input is 86% tag** — the α=0.5 blend,
undone by a preprocessing step that looks like a neutral efficiency measure.
Nothing errors. Every downstream number stays plausible. The phase that chose
the blend and the phase that discards it are in different files.

Two further things fall out of the same measurement:

- SPEC.md's "PCA to 50 dims, keep ~90% variance" is **not satisfiable here**:
  90% needs 208 of 512 dims, and 50 dims gives 54.6%. Those are two
  constraints, not one, and the data only permits one of them.
- The general lesson: *a dimensionality reduction is a modelling decision, not
  a neutral preprocessing step.* Anywhere a pipeline concatenates
  heterogeneous feature blocks and then reduces, check what survives.
  `prepare.py::half_mass()` is the two-line diagnostic that caught it.

### The correction: "never cluster on 2D" is not "never reduce"

Phase 3's first pass ran HDBSCAN only on 512-d cosine space and on PCA of it,
because SPEC.md says to cluster in high-dimensional space and never on 2D
coordinates. I read that as barring any reduction before clustering. It is
not: the rule is about *display* coordinates, which are optimised for
legibility and pack unrelated regions together to fill the plane.

A **separate** UMAP reduction — 5 dims, `metric='cosine'`, `min_dist=0.0`,
never rendered, never reused as the layout — is a different object, and is the
standard BERTopic/DataMapPlot pipeline. The two trees SPEC.md insists on
keeping apart stay apart: this feeds only the cluster tree, and Phase 4 fits
its own layout.

The diagnosis in the original section ("Steam is a continuum") was correct as
far as it went. What it missed is that **the absence of density contrast was a
property of the 512-d space, not of the catalog**:

| space | pairwise CV | 10-NN contrast | games at root |
| --- | ---: | ---: | ---: |
| 512-d cosine | 0.106 | 1.60 | 39.8% |
| PCA-50 | 0.122 | 1.78 | 39.8% |
| **UMAP 5-d** | **0.386** | **12.83** | **0.0%** |

Condensed-tree forking nodes went 110 → 407. The collapse trade-off that §2
called inescapable — 63.7% unclustered to hold the biggest node under 30% —
became 2.3% at the same balance. Forty-plus HDBSCAN configurations had been
swept in the wrong space; the sweep was thorough and the space was wrong.

### Four methods, judged by reading paths

| | k-means k=12 | UMAP→HDBSCAN | gated k-means | **Leiden** |
| --- | ---: | ---: | ---: | ---: |
| nodes / leaves | 1,064 / 908 | 371 / ~310 | 1,028 / 737 | 1,432 / 897 |
| unclustered | 0% | 3.0% | 0% | **0%** |
| biggest node | 11.8% | 15.2% | 15.0% | 12.9% |
| stuck at depth 1 | 0% | **33.6%** | 0% | 0% |
| parent-term repetition | 20.2% | 26.4% | 29.4% | 38.5% |

- **UMAP→HDBSCAN** gave the cleanest territories but stranded a third of the
  catalog at depth 1, and put Dota 2 in an 8,550-game "First-Person / Horror /
  Psychological Horror" node that never subdivided for it. Its children were
  individually coherent while their parent had no honest name — HDBSCAN's
  top-level merge order is arbitrary.
- **Gated bisecting k-means** made splits *earned* (54 refused for lack of
  structure), but silhouette rises monotonically as k falls, so it chose k=2
  for 201 of 317 splits and rebuilt the binary cascade — "Platformer →
  Platformer → Platformer", 37.5% repetition. Picking the widest k within 85%
  of the best score cut that to 29.4%; top-level placement stayed poor.
- **Leiden** read best. Community detection asks which games are more connected
  to each other than to the rest of the graph, rather than where the gaps are —
  the right question for a continuum.

```
Europa Universalis IV → Strategy / RTS → Turn-Based Strategy / Historical
                      → Turn-Based Combat / Hex Grid → Historical / Military
                      → 4X / Economy / Grand Strategy
Hollow Knight         → Platformer → 2D Platformer → Metroidvania / Exploration
                      → Metroidvania / Souls-like / Dark Fantasy
```

**The repetition metric is misleading here and was overruled.** It counts any
shared top-3 term between parent and child, so it penalises `Platformer → 2D
Platformer → Metroidvania` — a legitimate refinement, and exactly what a
zoomable map wants. It was built to catch "Action → Action → Action Indie",
where the child adds nothing, and cannot tell the two apart. Leiden scores
worst on it and reads best.

### Residue buckets: lift, not coverage

A node whose label does not distinguish it is now flagged rather than
presented as a genre. The motivating case — `tree_k12`'s 5,522-game "Indie /
Casual / VR", 10% of the catalog — has **94% top-term coverage** but **lift
1.68** against its siblings, lowest of twelve territories where real genres
score 5.4 to 60.9.

Coverage cannot catch it; lift can. Threshold 2.0, about the 10th percentile
of the lift distribution. This is not a stopword list in disguise: nothing
names a tag, the test is structural, and a ubiquitous tag fails it precisely
because being everywhere is what makes it uninformative.

In the shipped tree 183 of 1,432 nodes are flagged, covering 22,074 games
(39%), including two of the sixteen territories. That number is high and is
not a detector bug — it is the honest report that a large minority of the
catalog has no confident label. Phase 6 must render those dimmer and
unlabelled rather than assert a genre over them.

### Phase 6 reframing: why screen-footprint LOD is right for *this* tree

SPEC.md justifies footprint-based LOD by unbalanced depth — "dense regions
like 2D indie platformers nest 6–7 levels; sparse ones like flight sims stop
at 2", so depth-keyed thresholds leave half the map unlabelled.

**That argument does not hold for the shipped tree.** Leiden's levels come
from resolution sweeps, so depth is near-uniform: 83.1% of games land at depth
5 and 16.7% at depth 4. Nothing nests to 7 and nothing stops at 2.

The justification is therefore different, and stronger:

> **Cluster size varies by three orders of magnitude at the same depth.** At
> depth 1 the territories run from 276 games (Hidden Object) to 7,239 (Visual
> Novel) — a 26× spread. Across the whole tree, sizes run 16 to 7,239 with a
> median of 62.

A depth-keyed rule shows all sixteen depth-1 labels at the same zoom, so the
276-game territory is either an unreadable speck while Visual Novel is legible,
or legible while Visual Novel has long since needed subdividing. Screen
footprint keys the decision to what the viewer can actually see: a label
appears when its cluster occupies a usable fraction of the viewport, whatever
depth it sits at. Small territories surface later, large ones subdivide
sooner, and no level is tuned by hand.

The conclusion in SPEC.md is right; the premise needs replacing with the one
above. Recorded here so the Phase 6 rationale matches the tree that shipped.

### Known weaknesses

- **Containment purity: median 0.90, and 50% of nodes below 0.9.** Leiden is
  flat per resolution and the hierarchy is imposed afterwards by majority
  containment, so 709 fine communities straddle two coarse parents. The tree is
  an approximate nesting, not a strict one; `purity` is stored per node.
- 39% of games sit under a residue-flagged node.
- Dota 2 enters under "FPS / Shooter" and reaches MOBA only at depth 4.
- Resolutions {0.5, 2, 8, 32, 128} span a wide range but were not tuned; the
  k-NN graph is k=20, unswept.
- The HDBSCAN negative result and all four trees stay reproducible in the repo.

---

## Phase 3 (final) — recursive Leiden, exact nesting

**Shipped:** `tree_rleiden_labelled.json` — recursive Leiden, purity 1.000 by
construction.

Multi-resolution Leiden inferred parenthood by majority containment, so
nothing forced a fine community to sit inside one coarse community: 709 of
them straddled two, median containment purity 0.90 with half the nodes below
it. That is fatal rather than untidy — Phase 4's whole job is keeping children
inside their parent's region, and a child whose members belong to two parents
gets torn across two territories, which reads as a rendering bug.

**Recursive Leiden removes the failure mode instead of measuring it.** Each
community is re-clustered on a k-NN graph rebuilt from *only its own members*,
so a child is computed from its parent's membership and cannot straddle. The
subgraph rebuild also changes the question at depth: "nearest" becomes nearest
*among strategy games* rather than nearest in the whole catalog.

| | multi-resolution | **recursive** |
| --- | ---: | ---: |
| containment purity | 0.90 median, 50% below 0.9 | **1.000, exact** |
| nodes / leaves | 1,432 / 897 | 1,117 / 874 |
| biggest node | 12.9% | **10.0%** |
| top-level territories | 16 | 27 |
| top-term repeat | 9.9% | **2.8%** |
| build time | ~35s | 39s |

Recursive also reads better: Terraria, Valheim, Rust and Fallout 4 all land
under *Survival / Crafting / Open World Survival Craft*, where the
multi-resolution tree scattered them.

### The repetition metric was wrong, and is now fixed rather than overruled

The previous section overruled a 38.5% "repetition" score on the grounds that
it penalised legitimate refinement. That was the right call for the wrong
reason — the honest move is to fix the metric, not to argue past it.

The old metric counted **any** shared top-3 term between parent and child, so
it could not distinguish `Platformer → 2D Platformer → Metroidvania` (the
child narrowing, which is what a zoomable map wants) from `Action → Action →
Action Indie` (the child restating the parent). Keyed on the **top term only**,
with three outcomes:

| | multi-resolution | recursive |
| --- | ---: | ---: |
| **repeat** — child restates parent (degenerate) | 9.9% | **2.8%** |
| refinement — child narrows onto a secondary parent term | 36.0% | 36.4% |
| novel — child introduces a new term | 54.1% | 60.8% |

Leiden's alarming 38.5% was almost entirely refinement. The fixed metric
defends the tree on its own, and no override is needed.

### Confidence tiers, not a binary residue flag

Dropping a label wherever lift < 2.0 left grey continents — two of sixteen
territories in one tree — which reads as unfinished rather than as honest. A
viewer is better served by "this region is loosely Casual games, shown
faintly" than by nothing.

| tier | rule | nodes | games | renderer |
| --- | --- | ---: | ---: | --- |
| strong | lift ≥ 4 | 645 | 54.8% | normal |
| normal | lift ≥ 2 | 271 | 26.0% | normal |
| weak | lift < 2 | 114 | 11.3% | dimmed / italic |
| none | coverage < 25% | 86 | 7.9% | no label |

**80.8% of games sit under a normally-labelled node**, 11.3% under a dimmed
one, and only 7.9% under a node with no label at all — against "39% residue"
under the binary rule. Largest dimmed node: "Indie / Action / Simulation",
2,433 games, lift 1.39.

---

## Phase 4 — layout, and hierarchy consistency measured before any fix

The display UMAP is a **separate object** from the clustering reduction: 2
dims, `min_dist=0.05`, `n_neighbors=25`, rendered; the clustering one is 5
dims, `min_dist=0.0`, never rendered. Parallel branches off the same vectors,
as the architecture requires.

**One deviation:** SPEC.md says to lay out from the 50-dim PCA. Phase 3
showed PCA-50 of variant C is 85.8% tag loading, so laying out from it would
put the map in a different space from the tree. This runs on the full 512-d
vectors with cosine, matching both.

### Measured, before attempting any fix

Regions are **trimmed** convex hulls (central 95% by distance to centroid);
an untrimmed hull is hostage to its worst outlier. Note that "fraction of
members inside the cluster's own hull" is 1.0 by construction and measures
nothing, so the informative pair is:

- **containment** — fraction of a node's members inside its *parent's* region
- **purity** — of catalog points inside a node's region, the fraction that
  actually belong to it

| depth | nodes | containment mean | median | <0.9 | purity mean | median | <0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27 | 0.851 | 0.999 | 22% | 0.518 | 0.528 | 44% |
| 2 | 224 | 0.929 | 0.976 | 15% | 0.305 | 0.268 | 80% |
| 3 | 676 | 0.948 | 0.957 | 12% | 0.182 | 0.137 | 95% |
| 4 | 189 | 0.949 | 0.951 | 13% | 0.116 | 0.086 | 98% |

**Containment is good and does not degrade with depth** — mean 0.85–0.95,
median 0.95–0.999, with only 12–22% of nodes below 0.9. Children largely stay
inside their parents. That is the number SPEC.md expected to be worst at
depth, and it is not.

**Purity is poor and collapses with depth** — 0.52 at depth 1 down to 0.12 at
depth 4, with 95–98% of deep nodes below 0.5.

### Part of the purity number is a metric artifact, and part is real

Purity is strongly size-correlated, which it must be: a 40-game cluster's hull
sitting in a dense region of a 56,129-point map contains many non-members no
matter how good the layout is.

| cluster size | mean purity |
| --- | ---: |
| < 50 | 0.169 |
| 50–100 | 0.187 |
| 100–300 | 0.222 |
| 300–1,000 | 0.349 |
| 1,000+ | 0.443 |

But size does not explain the variance *within* a depth level. The depth-1
territories are all of comparable size and differ wildly:

```
Visual Novel / Interactive Fiction   4,505   purity 0.93
Action Roguelike / Action RPG        3,330   purity 0.91
Platformer / 2D Platformer           5,630   purity 0.88
Horror / Walking Simulator           4,398   purity 0.79
Management / Simulation              3,353   purity 0.69
Shooter / FPS / Violent              4,411   purity 0.38
Puzzle / Relaxing / Logic            3,844   purity 0.13
RPG / Turn-Based Combat / JRPG       3,272   purity 0.08
```

So this is **not** uniformly bad layout. Most territories are cohesive; a few
are genuinely shattered across the map. *RPG / Turn-Based Combat / JRPG* at
0.08 and *Puzzle* at 0.13 are coherent in 512-d and fragmented in 2D — UMAP
has scattered them, which is exactly the distortion Phase 4's fixes exist to
address.

That also says which fix to reach for: the problem is concentrated in a
handful of top-level territories rather than spread evenly, so the recursive
layout (lay out territories as points, allocate regions, run UMAP within each,
affine-transform into place) targets it directly, where a global constrained
UMAP would pay a cost everywhere to fix a few places.

No fix has been applied yet; the numbers above are the baseline against which
any fix must be judged.

### Phase 7 constrains which layout fix is allowed

SPEC.md offers three fixes for hierarchy inconsistency and presents them as
increasing effort: accept-and-prune, recursive layout, constrained UMAP. The
ordering implies recursive layout is the middle option — more work than
pruning, less than constraining. On effort, that is right. On consequences, it
is the most destructive of the three, and the reason is in a different phase.

**Recursive layout breaks Phase 7.** Frontier recommendations are defined as
territory *adjacent to* the user's hot zones: a cluster near their
playtime-weighted centroid, surfaced because it is near. The explainability
SPEC.md claims for the feature — "the user can see why, and see what else is
nearby" — is entirely a property of the geometry.

Recursive layout allocates each top-level territory an arbitrary region and
runs a separate UMAP inside it. The distance between two territories then
reflects the packing algorithm, not the data. SPEC.md states the consequence
in one line ("cross-region distances become meaningless") without connecting
it to the phase that depends on them. That is the connection: recursive layout
would fix purity and silently remove the basis of the personal layer, and
nothing in Phase 4's acceptance check would notice, because containment and
purity would both *improve*.

So the constraint is recorded before the choice is made, not after:

> **Any layout fix must keep cross-territory distance data-driven.**
> Recursive layout is available and implemented-shaped, but it is a last
> resort, and if it ever ships, EXPLAIN.md must state plainly that Phase 7's
> adjacency scoring is no longer meaningful and the frontier feature is
> reduced to within-territory recommendations.

Both fixes tried below satisfy the constraint. Supervised UMAP adds an
attraction term toward territory labels while the layout still optimises the
real neighbour graph; chained UMAP is an ordinary unsupervised reduction of a
space that itself came from the data. Neither imposes a region on anything.

### Purity lift: normalising away the size bias

Raw purity is size-correlated (0.169 under 50 games, 0.443 over 1,000), so
cross-level comparison is meaningless without normalisation. `purity_lift =
observed / expected`, where expected comes from a **size-matched random
baseline** — clusters of the same size drawn at random from the catalog, hulls
computed identically. Measured empirically rather than derived, because hull
geometry matters. Same lift construction as the label scoring.

### Three layouts, measured side by side

| layout | depth-1 purity | lift d1 / d2 / d3 | containment d1 |
| --- | ---: | ---: | ---: |
| baseline (512→2) | 0.518 | 8.0 / 48.0 / 72.5 | 0.851 |
| chained (512→5→2) | **0.130** | 1.1 / 4.5 / 23.4 | 0.870 |
| supervised tw=0.01 | **1.000** | 21.7 / 125.7 / 150.1 | 0.867 |
| supervised tw=0.1 | 1.000 | 21.7 / 125.8 / 146.1 | 0.887 |
| supervised tw=0.5 | 1.000 | 21.6 / 125.6 / 143.5 | 0.844 |

**Chained failed outright** — purity lift 1.1 at depth 1 is indistinguishable
from random, and every territory collapsed to ~0.10. Reducing an already-
reduced space compounds distortion rather than inheriting its structure. This
was the variant I expected to win by construction and it lost worst, which is
the argument for measuring rather than reasoning about it.

**Supervised worked spectacularly** on the stated metric: every depth-1
territory reaches purity 1.000, including the shattered ones (RPG/JRPG 0.08 →
1.00, Puzzle 0.13 → 1.00, Shooter/FPS 0.38 → 1.00), with lift roughly doubled
at every depth.

### Which is exactly when to be suspicious

Purity and containment both reward separation. A layout that shattered the
catalog into 27 disjoint blobs would score perfectly on both and be useless —
and that is what recursive layout, the fix ruled out for breaking Phase 7,
would produce. Purity 1.000 is the number that should trigger a check, not end
the search.

`layout/fidelity.py` measures the two things that would actually break:

| layout | territory arrangement (Spearman, Phase 7) | 10-NN overlap (Phase 6) |
| --- | ---: | ---: |
| **baseline** | **0.542** | **12.2%** |
| chained | 0.475 | 15.3% |
| supervised tw=0.01 | 0.452 | 10.4% |
| supervised tw=0.1 | 0.421 | 10.3% |
| supervised tw=0.3 | 0.318 | 10.6% |
| supervised tw=0.5 | 0.320 | 10.3% |

Supervision has a **floor effect**: purity saturates at 1.000 even at
`target_weight=0.01`, the lowest value tested, while arrangement fidelity
still falls 0.542 → 0.452. The trade cannot be tuned away, only chosen.

And the cost is visible, not just numeric. Each territory's nearest
neighbouring territory:

```
                        baseline                supervised tw=0.01
RTS               ->    Tower Defense           Action Roguelike
Platformer        ->    RPG                     Education
Horror            ->    Online Co-Op            Clicker
Action Roguelike  ->    Shoot 'Em Up            Survival
Visual Novel      ->    Sexual Content          Sexual Content
```

The baseline's adjacencies are the ones a person would draw. Supervised
produces *Platformer next to Education* and *Horror next to Clicker* — which
is precisely the Phase 7 failure the constraint above was written to prevent,
arriving through a gentler mechanism than recursive layout.

### Decision: keep the baseline, prune rather than force

**Supervised UMAP is rejected despite winning the metric it was tried for.**
It buys purity with the currency Phase 7 spends.

The deeper reason is that the low-purity territories are *not places*. Puzzle,
RPG and Shooter are cross-cutting attributes — a puzzle game can live in a
dozen neighbourhoods — so forcing them into one blob does not reveal
structure, it asserts structure that is not there. The baseline layout is
telling the truth about them.

Evidence that this is the right reading: the fragmented territories'
**children** are markedly more contiguous than the territories themselves,
while cohesive territories' children are not.

```
territory            size   purity   mean child purity
RPG                 3,272     0.08            0.30      <- children 4x better
Puzzle              3,844     0.13            0.26
Indie               2,433     0.12            0.24
Platformer          5,630     0.88            0.33      <- parent already fine
Visual Novel        4,505     0.93            0.40
```

RPG is not a region; its sub-genres are. So SPEC.md's fix #1 —
accept-and-prune — applies **per territory** rather than per level: a
territory whose purity falls below threshold is not drawn as one labelled
region, and its children are surfaced in its place. Cheapest of the three
fixes, and here also the most honest.

This composes with the confidence tiers from Phase 3: a territory can be
low-confidence in *label* (weak lift) or low-confidence in *place* (low
purity), and the renderer should treat those separately.

### What was rejected, and why

| fix | status | reason |
| --- | --- | --- |
| chained UMAP | rejected | purity lift 1.1 — no better than random |
| supervised UMAP | rejected | buys purity by degrading territory adjacency (0.542 → 0.452) and neighbour preservation (12.2% → 10.4%); breaks Phase 7's basis |
| recursive layout | not attempted | would destroy cross-territory adjacency outright |
| **accept-and-prune, per territory** | **chosen** | keeps adjacency data-driven; declines to assert that cross-cutting attributes are places |

### Honest caveats

- **Neighbour overlap is low for every layout** (10–15%). That is inherent to
  512-d → 2-d, not a defect of any variant, but it does mean Phase 6's hover
  behaviour should read from the tree and the high-dimensional neighbours,
  not from screen proximity.
- **Baseline territory Spearman is 0.542**, which is moderate, not good. Phase
  7 adjacency is meaningful but not precise, and the frontier scoring should
  be presented with that in mind rather than as a strong geometric claim.
- The prune threshold has not yet been chosen; it needs the Phase 6 renderer
  to calibrate against.

### Provenance of the two fixes

Both layout fixes were proposed by the project owner, not derived here, and
the outcomes split:

- **Supervised UMAP** was their first suggestion and is the one that worked on
  the target metric — it was also the one this analysis then rejected, on a
  check their own Phase 7 constraint implied.
- **Chained UMAP (512→5→2)** was their second suggestion, offered with the
  reasoning that contiguity would follow "by construction" because the display
  space would be a further reduction of the space the communities were found
  in. That reasoning is sound and the data disagreed: purity lift 1.1, no
  better than random. Reducing an already-reduced space compounds distortion
  rather than inheriting structure.

Recorded because a plausible mechanism predicted the opposite of the
measurement, which is the case worth keeping.

### Note for Phase 6: make the 10–15% overlap a feature, not a caveat

Every layout tested preserves only 10–15% of each game's high-dimensional
10-NN on screen. That is inherent to 512-d → 2-d, not a defect of any variant,
and it means screen proximity is a navigational aid rather than a metric
statement.

Rather than hide that, **the renderer should show it**. On hover or select,
highlight the game's true high-dimensional nearest neighbours wherever they
land, with connecting lines or halos:

- neighbours clustered nearby → the map is locally faithful, and the viewer
  can see that it is
- neighbours scattered across three territories → that game genuinely resists
  placement, which is information about the game, not a rendering failure

This makes the limitation legible instead of buried, and it is the same
information the accept-and-prune rule uses at territory level, surfaced at the
level of a single game. It needs the high-dimensional neighbour lists
precomputed and shipped alongside the tiles — a k-NN list per game, which is
cheap at k=10.

**The overlap number belongs in the README**, stated plainly. Every embedding
atlas has this property and almost none report it. The map is a navigational
aid, not a metric space, and saying so directly is a strength rather than an
admission.

---

## Phase 5 — Tile engine (standalone library)

**Acceptance: PASS.** `python -m tiles build points.parquet out/` builds from
a parquet containing only `x, y, id, category, importance` — no domain fields
— and `test_every_point_reachable` asserts every input point appears in at
least one leaf tile, for both selection strategies. Verified independently on
the real 56,129-game dataset: 56,129 in, 56,129 in leaf tiles, 0 missing.

57 tests pass (26 in `tiles/`, 31 in `ingest/`).

### Standalone, enforced rather than intended

`tiles/` must ship as its own project, so the constraint is tested rather than
trusted:

- `test_library_imports_nothing_from_the_host_repo` parses every source file's
  AST for imports of `ingest`, `embed`, `cluster`, `layout`, `viewer`, `bench`.
  Convention would not survive one convenience import.
- `test_no_domain_vocabulary_in_source` greps for domain words. **It failed on
  first run** — my own docstrings said "games", "genres", "Steam" and used
  `review_count` as an example column name. The library was already structurally
  clean; its prose was not.

Everything Atlas-specific lives in `export_tiles.py` at the repo root, which
flattens the catalog, cluster tree and layout into five anonymous columns plus
a manifest. Nothing in `tiles/` would need to change for a different dataset.

### A bug the tests caught that review would not have

`naive` selection ranked points by `np.argsort(-importance)`. `importance` is
`uint16`, and negating an unsigned array **wraps** rather than negating:

```
importance:       [    0     1   100 65535]
-importance:      [    0 65535 65436     1]
argsort(-imp):    [    0 65535   100     1]   <- least important first
```

It selected the *least* important points in every coarse tile. The failure is
invisible by inspection: the points are still spatially distributed, the tile
counts are right, the file sizes are right, and a rendered map looks entirely
plausible — it is simply showing the wrong thousand games. Only an assertion
comparing selected importance against the true top-N cutoff caught it.
`_desc()` now widens to int32 with the reasoning in a docstring, because the
inline minus sign reads as obviously correct.

### Struct-of-arrays, because the spec's own justification requires it

SPEC.md specifies interleaved records and justifies binary over JSON because
it maps "straight into a typed array". Those are incompatible: the record is 18
bytes, so consecutive `x` values sit 18 bytes apart, and `Float32Array` needs a
contiguous 4-byte-aligned run. Interleaved data can only be read through
`DataView`, field by field, in a JS loop — which is most of the parse cost the
format exists to avoid.

The payload is therefore struct-of-arrays, giving the zero-copy view the spec
wanted. The interleaved layout is still implemented (`--aos`) so the choice
stays measurable. `count` is also widened from 2 bytes to 4: two bytes caps a
tile at 65,535 points, which the 1M benchmark would exceed.

### Binary vs JSON: the size argument is weaker than claimed, the parse argument is decisive

| | raw | gzipped | parse |
| --- | ---: | ---: | ---: |
| binary | 15,450 B | 10,555 B | **0.003 ms** |
| JSON | 86,052 B | 22,669 B | 0.748 ms |
| ratio | 5.6× | **2.1×** | **233×** |

SPEC.md estimates "JSON ≈ 80KB, binary ≈ 16KB" — close to the measured raw
figures. But **gzip narrows 5.6× to 2.1×**, and any real deployment gzips. On
size alone the custom format would be hard to justify.

Parse cost is where it is won, and that is the number a size table omits: 233×
slower, and gzip does nothing for it. Across a ~200-tile LRU cache that is the
difference between a frame and a stall. The honest form of the argument is
"JSON costs 2× the bytes and 233× the parse", not "JSON is 5× bigger".

### Selection strategy, measured and pictured

![naive vs stratified](../tiles/docs/selection.png)

| strategy | zoom 0 | zoom 1 |
| --- | ---: | ---: |
| naive — top N by importance | 45% of the plane occupied | 71% |
| stratified — top N per grid cell | **59%** | **92%** |

Occupancy is the fraction of a 32×32 grid over the data extent holding at least
one shown point. Naive concentrates on high-importance regions and leaves
others visibly empty — a viewer reads them as sparse when they are merely
unpopular. Stratified keeps the shape while importance still decides which
point represents each cell. Stratified is the default.

One thing the first rendering attempt got wrong and the picture exposed: it
built with `max_zoom` equal to the zoom being displayed, so the displayed level
*was* the leaf level, leaves are exhaustive by design, and both strategies
showed all 40,000 points. The comparison rendered identically for both and
looked like a bug in selection rather than in the harness.

### Scaling to 1M

| points | seconds | µs/point | tiles | records | MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.02 | 1.67 | 781 | 44,735 | 0.8 |
| 100,000 | 0.11 | 1.08 | 913 | 221,580 | 4.0 |
| 1,000,000 | 1.07 | 1.07 | 931 | 1,185,226 | 21.4 |

100× the points costs **64.5×** the time — sub-linear, because tile count
saturates once every quadrant is occupied (781 → 931 tiles across two orders of
magnitude) and per-point work is a bucket assignment. Viewport→tile-list is
~4 µs and flat across zoom levels, being arithmetic on bounds rather than a
tree walk.

Synthetic data is clustered with Pareto importance, deliberately: uniform
points would flatter the quadtree, and uniform importance would flatter naive
selection by spreading the top-N automatically.

### Manifest

Point records carry a **leaf category id only**; the renderer walks up via
`parent`. At 1M points and depth 5, per-point ancestor chains would cost ~16 MB
of data identical for every point in a category.

The manifest carries `id`, `parent`, `label`, `bbox`, `centroid`, `size`,
`depth`, `confidence`, `purity`, `containment` — confirmed on the real export:
1,117 categories, 0 validation problems, confidence on all 1,117 and
purity/containment on 1,116 (the root has no parent region). `confidence` and
`purity` are opaque to the library but carried because the renderer needs to
*dim* an uncertain label rather than drop it, and to decline drawing a region
for a category whose points are scattered — the two decisions Phases 3 and 4
left it.

### Known weaknesses

- **~2.9× redundancy** at `max_zoom=5` on the real data. Storage is linear in
  zoom depth; the trade buys independently renderable tiles.
- **No merging of sparse tiles.** A quadrant with three points still produces a
  file. At 485 leaf tiles this is not yet worth fixing.
- **`importance` is rank-normalised into uint16 on load.** Order is preserved
  exactly, magnitudes are not. Selection only uses order, but a caller wanting
  magnitude-weighted rendering would need the raw value alongside.
- **Single-threaded**, which 1M points in ~1 s has not made worth changing.
- **Parse timings are Python**, not JavaScript. The 233× ratio reflects
  `json.loads` plus object allocation against `np.frombuffer`; the browser
  ratio will differ in magnitude while the direction holds, since `JSON.parse`
  also allocates one object per point where the typed-array path allocates
  nothing. Worth re-measuring in Phase 6 with real browser numbers.
