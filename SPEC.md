# Pyxis

A zoomable map of the Steam catalog where zoom levels come from a semantic cluster
hierarchy, not geometric tiles.

Zoomed out: broad territories. Zoom in: a territory splits into its real sub-genres.
Zoom again: splits further. Every label describes an actual group of similar games.

## The differentiator

WizMap (ACL 2023, arXiv 2306.09328) builds multi-resolution labels from a quadtree over
the 2D layout. Its limitations section says this is "sensitive to tile size selection"
and that clustering-based alternatives "warrant exploration." This project is that
alternative: labels come from an HDBSCAN condensed tree, and the quadtree is used only
for deciding what to send to the browser.

Three things that must hold:

1. Labels derive from a semantic cluster hierarchy, not spatial tiles
2. Level-of-detail is keyed to a cluster's screen footprint, not its depth in the tree
3. A personal layer: the user's own library and playtime as a density overlay, with
   recommendations as unexplored territory adjacent to their hot zones

Out of scope: accounts, multi-user, mobile-first, live refresh.

## Ground rules

- Do not use Plotly, deck.gl, or any charting library for the main renderer. Raw WebGL
  via regl. The point of the project is that the renderer was written here.
- Do not build the viewer before embeddings are validated. Bad embeddings produce a
  beautiful meaningless map. Validate upstream before moving downstream, every phase.
- `tiles/` is a standalone library with its own tests, CLI, and benchmarks. It ships as a
  separate project. No imports from the rest of the repo.
- Cluster in high-dimensional space, never on 2D coordinates. Clustering after UMAP finds
  structure in UMAP's distortions.
- Every phase ends with its acceptance check passing. Do not proceed past a failing
  check; report it instead.
- Python 3.11+, uv or venv. Frontend is vanilla JS + regl, no framework, no build step
  beyond esbuild if needed.
- Commit after each phase with the acceptance check result in the message.
- After each phase, append a short rationale note to EXPLAIN.md (see Phase 9): what was
  decided, what else was considered, why. Write it while the reasoning is fresh, not at
  the end from memory.

## Architecture

```
Steam data (~50k games)
    └── embeddings (384-dim vectors)
            ├── HDBSCAN  → cluster tree → c-TF-IDF labels
            └── UMAP     → x, y coordinates
                    └── quadtree → binary tiles → WebGL viewer
```

Clustering and UMAP are parallel branches off the same vectors. Neither feeds the other.

Two trees, different jobs:

|            | Cluster tree            | Quadtree               |
| ---------- | ----------------------- | ---------------------- |
| Built from | 384-dim vectors         | 2D coordinates         |
| Answers    | what does this group mean | what to send the browser |
| Drives     | labels, point colors    | tile loading, LOD      |
| Balanced   | no                      | yes                    |

Keeping them separate is the whole design. Do not merge them.

Per-game record at the end: `x`, `y` (fixed forever), `ancestors` (cluster id chain
leaf→root), `importance` (review count), metadata for tooltips.

Repo layout:

```
pyxis/
  ingest/    data acquisition + resumable crawler
  embed/     vectorization, three variants
  cluster/   HDBSCAN tree extraction + labeling
  layout/    UMAP + hierarchy-consistency pass
  tiles/     quadtree builder, binary serialization  [STANDALONE]
  viewer/    WebGL front end
  bench/     benchmarks + evaluation
```

## Phase 1 — Data

Start from a dump so nothing is blocked; run the crawler alongside to enrich.

Dumps to evaluate (pick by tag coverage):

- Kaggle `hubertsidorowicz/steam-games-dataset-daily-updates` — 136k+, freshest
- HuggingFace `FronkonGames/steam-games-dataset` — ~85k, well documented
- GitHub `vintagedon/steam-dataset-2025` — full catalog, also read as prior art

The deciding field: **community tags**. Steam's user-voted tags ("Souls-like", "Bullet
Hell", "Cozy") are the strongest clustering signal — far better than the official genre
list. They live on the store page, not the official API, so coverage varies. Check tag
coverage before committing to a dump.

Crawler (fills gaps, mainly tags and current review counts):

- Rate limit ~200 requests / 5 min on the store API
- Resumable: write each game to SQLite immediately; on startup load completed appids and
  skip
- Separate `failures` table with status and try count so retries hit only those
- Start it in the background in hour zero

Schema:

```sql
CREATE TABLE games (
  appid INTEGER PRIMARY KEY,
  name TEXT,
  short_desc TEXT,
  long_desc TEXT,
  tags TEXT,          -- JSON array, ordered by vote count
  genres TEXT,        -- JSON array
  release_date TEXT,
  review_count INTEGER,
  review_score REAL,
  price REAL,
  fetched_at TEXT
);
CREATE TABLE failures (appid INTEGER PRIMARY KEY, status TEXT, tries INTEGER);
```

Filtering: drop DLC, soundtracks, demos, videos, software. Drop games with <10 reviews
(no tags, cluster as noise, pure rendering cost). Expect 50–80k survivors.

**Acceptance:** SQLite file with 50k+ filtered games, ≥80% having 3+ community tags.

## Phase 2 — Embeddings

Build three variants and choose empirically. The comparison matters more than the winner.

**A. Text.** One string per game, encoded with `bge-small-en-v1.5` (384-dim) on GPU:

```
Hollow Knight. Tags: metroidvania, souls-like, difficult,
atmospheric, 2d platformer. Genres: action, adventure, indie.
Forge your own path in Hollow Knight...
```

Batch 256, mean-pool, L2-normalize, store float16.

**B. Tag co-occurrence.** No text model. Sparse games × tags matrix → PPMI weighting →
truncated SVD to 128 dims.

**C. Hybrid.** L2-normalize A and B separately, concatenate as `[α·A, (1−α)·B]`. Try
α ∈ {0.3, 0.5, 0.7}.

Why this is a real decision: store descriptions are marketing copy — every game claims to
be an atmospheric epic adventure. Weight them heavily and clusters reflect ad-copy style
rather than gameplay. Community tags are thousands of players voting on what a game
actually is. B often wins; that is a finding worth reporting.

Evaluation — run before building anything downstream:

1. **Spot checks.** Print 10 nearest neighbors for 15 well-known games. Hollow Knight
   should return Ori, Blasphemous, Dead Cells. If it returns unrelated indies with similar
   ad copy, the input text is wrong — fix it now.
2. **Genre agreement.** Hold out Steam genres. Per game, fraction of 10-NN sharing ≥1
   genre. Compare A/B/C against a random-pairs baseline.
3. **Tag prediction.** Mask one tag per game; check whether neighbors' tags recover it.
   Precision@5.

**Acceptance:** `variants.md` with all three scored on both metrics, one chosen, reasoning
written down.

## Phase 3 — Cluster hierarchy and labels

This is the differentiating piece of the whole project. Implement it fully, and write a
`cluster/RATIONALE.md` explaining every design choice made here — it feeds EXPLAIN.md at
the end.

**Step 1: PCA to 50 dims.** HDBSCAN degrades in high dimensions. Keep ~90% variance.

**Step 2: HDBSCAN, keep the condensed tree.** The usual call returns flat labels and
throws away the nesting. The condensed tree records, as density threshold falls, where
each cluster splits and where points drop out as noise.

```python
clusterer = HDBSCAN(min_cluster_size=25, min_samples=5)
clusterer.fit(X_pca)
tree = clusterer.condensed_tree_.to_pandas()
```

**Step 3: condensed tree → clean n-ary tree.** This is the hard part and it is where the
interview questions live.

- Collapse runs where a cluster "splits" into one real child plus noise points — not real
  branches
- Drop branches below a minimum size, reassign members upward
- Cap depth at ~6
- Record per node: id, parent, member ids, size, centroid, 2D bbox (filled after Phase 4)

**Step 4: label each node with c-TF-IDF.** Concatenate each cluster's members' tags and
description terms into one meta-document; score terms by in-cluster frequency against
cross-cluster frequency.

Critical: score against **sibling clusters**, not the global corpus. Otherwise every
cluster under "action" gets labeled "action" — common inside, rare outside. Scoring within
the parent surfaces what distinguishes a child from its siblings.

**Step 5 (optional): LLM name tidying.** Top 10 c-TF-IDF terms + 5 example game names →
2–3 word label. `roguelite, deckbuild, card, run-based` → "Deckbuilding Roguelites". Keep
raw terms as fallback, cache results.

**Acceptance:** JSON tree where walking root→leaf gives labels a person would actually say.

## Phase 4 — 2D layout

Baseline. UMAP from 50-dim PCA, `n_neighbors=25`, `min_dist=0.05`. Low `min_dist` keeps
clusters tight, which matters at 2-pixel dots.

Measure the violation first. Per cluster node:

- **Containment** — fraction of members inside the cluster's own 2D hull
- **Purity** — fraction of points inside that hull that actually belong

Report per depth level. Expect top two levels fine, deeper levels messy. This measurement
is itself a result.

Three fixes, increasing effort:

1. **Accept and prune** — only expose levels clearing a containment threshold. Cheapest,
   often enough.
2. **Recursive layout** — lay out top-level clusters as points, allocate regions, run UMAP
   within each cluster, affine-transform into its region. Guarantees containment;
   cross-region distances become meaningless.
3. **Constrained UMAP** — attraction term pulling points toward parent centroid, weighted
   by depth. Better global structure; needs tuning.

Start with 1. Implement 2 if time allows. Report containment for whichever ships.

Then: compute each cluster's 2D bbox and centroid, write back into the tree. Phase 6 needs
them.

**Acceptance:** every cluster node has a bbox; containment per level recorded in a table.

## Phase 5 — Tile engine [STANDALONE LIBRARY]

Problem: 50k points, ~2000 useful pixels. Sending everything is slow and pointless.

**Construction.** Recursively subdivide the bbox into quadrants. Each tile holds at most N
points (start N=1000), chosen by importance. A point may appear in a coarse tile and again
in finer ones — that redundancy makes each tile independently renderable.

```
z=0  1 tile      top 1000 games, whole map
z=1  4 tiles     top 1000 per quadrant
z=2  16 tiles    ...
z=5  1024 tiles  every remaining game
```

**Importance selection matters.** Naive top-N by review count makes zoomed-out views
all-AAA and leaves indie regions looking empty. Better: stratify within the tile —
subdivide into a small grid, take top points per cell. Preserves spatial shape while
favoring notable games. Implement both and compare screenshots.

**Binary format, not JSON.** JSON for 1000 points ≈ 80KB plus parse cost; binary ≈ 16KB
mapping straight into a typed array.

```
header:  magic(4) version(2) count(2) bbox(16)
points:  x:f32 y:f32 id:u32 cluster:u32 importance:u16
```

`struct.pack` or `numpy.tobytes`; read with `DataView`/`Float32Array`. Gzip on top.

Also emit a **label manifest** — one JSON with every cluster node: id, parent, label,
centroid, bbox, size. Loaded once up front so label decisions need no network round trips.

Benchmarks the library reports:

- Build time vs point count (10k / 100k / 1M)
- Bytes per tile, binary vs JSON, gzipped and raw
- Tiles per viewport at each zoom
- Query time for viewport → tile list

**Acceptance:** `python -m tiles build points.parquet out/` produces a tile directory; a
test asserts every point appears in ≥1 leaf tile.

## Phase 6 — Renderer and semantic zoom

Raw WebGL via regl.

**Point rendering.** One draw call, `gl.POINTS`, attribute buffers for position / cluster
id / importance. Vertex shader applies view transform and sizes by importance; fragment
shader draws a soft circle, colors by active cluster level. 50k points = one draw call,
comfortably 60fps.

**Tile loading.** On viewport change, compute intersecting tiles at current zoom, request
missing, LRU cache ~200 tiles. Debounce during active panning — loading mid-drag causes
stutter. Render cached immediately, upgrade as tiles arrive.

**LOD keyed to screen footprint, not depth.**

The obvious approach ties depth to zoom (depth 1 at zoom 0–2, etc.). It breaks because the
tree is unbalanced: dense regions like 2D indie platformers nest 6–7 levels; sparse ones
like flight sims stop at 2. Fixed thresholds leave half the map unlabeled at high zoom.

Instead:

```
footprint = cluster.bbox.area × zoom² / viewport.area
visible   = 0.05 < footprint < 0.60
opacity   = smoothstep over both edges
```

Too large = not zoomed in yet. Too small = not worth naming. Dense regions reveal deeper
clusters sooner because children grow on screen sooner. No blank patches, no per-level
tuning.

**Cross-fades.** Opacity from the same smoothstep — parent fades out as children fade in.
Points recolor by active ancestor, so a blob visibly splits into colored sub-regions
without any point moving. Point positions are fixed forever.

**Label collision.** Greedy placement in descending cluster-size order; skip any label
whose box hits one already placed.

**Interaction.** Hover tooltip (nearest point via spatial lookup, not linear scan), click
for detail panel, search that flies to a game, breadcrumb showing current cluster path.

**Acceptance:** zoom from whole catalog to a single game, labels resolve smoothly
throughout, frame counter stays near 60.

## Phase 7 — Personal layer

**Pull library.** `IPlayerService/GetOwnedGames` with a Steam API key → appids + minutes
played for a public profile.

**Heat overlay.** KDE over owned games weighted by log playtime, rendered as a second
WebGL layer beneath the points. Bandwidth tuned to read as territory, not dots.

**Frontier recommendations.** Per cluster node:

```
score = w1·adjacency + w2·quality + w3·(1 − ownership)
```

where `adjacency` = inverse distance from cluster centroid to the user's playtime-weighted
centroid, `quality` = mean review score of members, `ownership` = fraction already owned.
Rank clusters, surface top games in each.

Output reads as: "You have 400 hours in metroidvanias and 200 in soulslikes, but own
nothing in the adjacent cluster containing Nine Sols and Blasphemous 2."

Explainability falls out of the geometry — the user can see why and see what else is
nearby.

**Acceptance:** enter a Steam ID → heat overlay + five ranked frontier clusters with top
games.

## Phase 8 — Evaluation and benchmarks

Do not skip this. Without numbers it is a pretty picture.

| Metric                | How                                                              | Why                        |
| --------------------- | ---------------------------------------------------------------- | -------------------------- |
| FPS vs point count    | Instrument render loop at 10k/50k/200k, vs Plotly and deepscatter | Headline number            |
| Tile payload          | Bytes/tile, binary vs JSON, gzipped and raw                       | Justifies custom format    |
| Time to first paint   | Cold load → first rendered frame                                  | What users feel            |
| Label quality (NMI)   | Normalized mutual information vs Steam genre tags, per depth      | Proves labels mean something |
| Hierarchy containment | Members inside own 2D hull, per depth                             | Justifies layout work      |
| Embedding variants    | Genre agreement + tag prediction for A/B/C                        | Shows the choice was measured |
| Build time            | End-to-end on 50k games, per stage                                | Shows profiling            |

On NMI: Steam genres are a coarse human taxonomy. High MI at shallow depths means
top-level clusters recover known genres. The interesting result is at deeper levels where
NMI necessarily drops — you are finding structure the genre list does not encode. Report
the curve and interpret it. Falling NMI at depth is not a bug.

**Fair benchmarking.** Same machine, browser, dataset. Median of five runs. State the
methodology.

README structure: demo GIF of the zoom → numbers table → architecture → the WizMap
limitation addressed, cited → what is next and what does not work yet.

**Acceptance:** every number above in a table reproducible from the scripts.

## Phase 9 — EXPLAIN.md

Implement everything. Nothing is left as a TODO.

Because of that, this phase is mandatory and is not a cut line. Produce `EXPLAIN.md` at
the repo root containing:

**1. Design decisions and their alternatives.** For every non-obvious choice, state what
was chosen, what else was considered, and why the alternative lost. At minimum: HDBSCAN
condensed tree vs. k-means vs. flat clustering; clustering in 384-d vs. 2-d; the chosen
embedding variant vs. the other two; screen-footprint LOD vs. depth-based; binary tile
format vs. JSON; sibling-scoped c-TF-IDF vs. global; the layout-consistency fix chosen vs.
the other two.

**2. A walkthrough of every non-trivial function**, in the order data flows through it.
Name the file, say what goes in, what comes out, and what would break if it were wrong.

**3. An interview question bank** — at least 25 questions an interviewer could ask about
this codebase, each with the answer, drawn from what was actually built rather than from
generic theory. Include the eight below and go well past them:

1. Why HDBSCAN's condensed tree instead of k-means or flat clustering?
2. Why cluster in 384 dimensions rather than on the 2D coordinates?
3. How do you keep child clusters inside their parent's region, and what breaks if you
   don't?
4. Why screen footprint rather than tree depth for level of detail?
5. Why a binary tile format — what is the actual cost of JSON here?
6. Why score c-TF-IDF against siblings rather than the whole corpus?
7. What does falling NMI at deeper levels mean — is that a bug?
8. What would break at 10 million points?

**4. Known weaknesses.** What does not work well, what was cut, what the numbers do not
show. Be specific and unflattering.

Write it as though the reader has to defend this code in a technical interview tomorrow,
because they do.

## Cut lines, in order

1. LLM label prettifying — raw c-TF-IDF terms are fine
2. Recursive layout — ship plain UMAP, report containment honestly
3. Personal overlay — painful, it is the best demo, but additive
4. Search and breadcrumbs — nice, not load-bearing
5. Deepest hierarchy levels — three good levels beat six bad ones

**Never cut:** benchmark numbers, README, demo GIF, EXPLAIN.md.

## Known risks

| Risk                                  | Fallback                                                             |
| ------------------------------------- | -------------------------------------------------------------------- |
| Dump lacks community tags             | Genres + description text; note the degradation                       |
| Condensed tree is a mess after collapsing | Recursive k-means, fixed branching — less principled, always balanced |
| Poor hierarchy containment            | Expose top 3 levels only; report the number                           |
| Labels read as gibberish              | Score against siblings not globally; add LLM naming                   |
| WebGL fights you                      | Canvas 2D fallback at reduced point count; fix WebGL after            |
| Crawler rate-limited into uselessness | Dump alone suffices; crawler is enrichment                            |

The one that actually bites: **Phase 3 step 3.** Turning the condensed tree into a clean
taxonomy looks like plumbing and is not. The tree is full of degenerate splits where a
cluster "divides" into itself plus four noise points. Collapsing those correctly while
preserving real branches takes iteration. Budget more time than feels reasonable.
