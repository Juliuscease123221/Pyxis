# cluster/RATIONALE.md — Phase 3 design decisions

The differentiating piece of the project. Feeds EXPLAIN.md.

**Shipped:** `tree_leiden_labelled.json` — multi-resolution Leiden over a
cosine k-NN graph on the full 512-d vectors, nested by membership containment,
labelled with sibling-scoped prevalence×lift over community tags.

**Four methods were built, labelled and read.** All four stay in the repo,
reproducible, because the comparison is the result. §10 has the head-to-head;
§1–9 below document the first two attempts in the order they happened, and the
PCA finding in §1 is the most transferable thing in this file.

> **Correction, recorded because it changed the outcome.** §2–3 conclude that
> HDBSCAN cannot work here. That conclusion was *wrong in its reasoning*: I had
> read SPEC.md's "cluster in high-dimensional space, never on 2D" as barring
> any reduction before clustering, so I only ever ran HDBSCAN on 512-d cosine
> space and on PCA of it. The rule is about *display* coordinates. A separate
> UMAP reduction to 5–15 dims, tuned for clustering and never rendered, is the
> standard BERTopic/DataMapPlot pipeline and is a different object entirely.
> Applying it moved density contrast from 1.60 to 12.83 and unclustered games
> from 39.8% to 0.0%. See §10.

---

## 1. PCA: the spec's two constraints cannot both hold

SPEC.md asks for "PCA to 50 dims, keep ~90% variance". On this embedding
those are incompatible:

| dims | variance retained |
| ---: | ---: |
| 50 | 54.6% |
| 100 | 72.5% |
| **208** | **90.0%** |

90% needs 208 of 512 dims. The embedding is high-rank — unsurprising for a
concatenation of a 384-d transformer output with a 128-d SVD basis.

**Worse, PCA silently undid the Phase 2 decision.** Variant C is
`[0.5·A₃₈₄, 0.5·B₁₂₈]`. Both halves carry equal *total* variance by
construction, but the tag half packs it into 128 dims while the text half
spreads it over 384. PCA takes the densest directions first:

| dims | text loading | tag loading |
| ---: | ---: | ---: |
| 50 | 14.2% | **85.8%** |
| 128 | 12.3% | 87.7% |
| 256 | 51.0% | 49.0% |

At 50 dims the clustering input is 86% tag. The α=0.5 blend chosen in Phase 2
would have been nullified by a preprocessing step, invisibly — the numbers all
look fine, and nothing downstream would have said a word.

`prepare.py --mode balanced` was written to fix this (PCA each half to 25 dims,
L2-normalise, rejoin at parity). In the end the shipped path avoids the problem
entirely by **not using PCA**: k-means does not need the density contrast PCA
was there to rescue, so it runs on the full 512-d vector and the blend survives
exactly as chosen. `half_mass()` remains as the diagnostic that caught this.

## 2. HDBSCAN produced a caterpillar, not a hierarchy

Fitted at SPEC.md's settings (min_cluster_size 25, min_samples 5, 50 dims):

```
flat clusters returned      41
games assigned to NOISE     47,281  (84.2%)
condensed tree              226 nodes, max depth 53, branching factor 2 everywhere
```

**The 84.2% is the wrong number and should not be quoted.** It is HDBSCAN's
EOM *flat-label selection*, which this pipeline never uses. The condensed tree
assigns **100%** of games to some node — verified: 56,129 rows, 56,129 unique.
The honest "unclustered" figure is games falling out at the root, which is
20.6% at `min_samples=1` and 39.8% at 5 — the 20–40% range anticipated.

The real defect is the *shape*. Depth 53 with branching 2 at every level is one
spine that sheds a small cluster at each step while the bulk of the catalog
continues down. Read literally, 52 of 53 levels are the same undifferentiated
mass.

### This is SPEC.md's warning, one level up

SPEC.md predicts "a cluster 'divides' into itself plus four noise points".
The *condensed* tree has already absorbed that — condensing is exactly what
turns dropped points into point-children rather than sibling branches. What
survives is the same pathology applied to *clusters*: the spine is the cluster
dividing into itself plus one real group. The warning was right; the mechanism
sits one level above where it was expected.

### Collapse: three rules, before/after

`tree.collapse` applies, in order:

1. **Spine collapse** — a child holding ≥ `spine_ratio` of its parent is the
   parent continuing, not a subdivision. Merge it up, promote its children.
   Repeated, this walks the entire spine and gathers every shed cluster into
   one wide fan-out. This is the rule that does the work.
2. **Dissolve small** — below `min_branch`, members go to the parent.
3. **Pass-through** — a node with one surviving child re-describes its parent
   and costs every descendant a level. Make it transparent.

Measured (min_cluster_size 15, min_samples 1, 20 dims, spine_ratio 0.5):

```
before          562 cluster nodes, max depth 143
spine merges    210
dissolved small   0
pass-through      0
depth-cap cuts   18
after           304 cluster nodes, max depth 6, mean fan-out 4.61
```

Rules 2 and 3 fired **zero** times. The entire restructuring is rule 1. That is
worth stating plainly: the collapse SPEC.md describes (drop small branches,
collapse unary runs) was not what this tree needed, because the condensed tree
does not contain those defects. It needed spine collapse, which is not in the
spec.

### The trade-off no threshold escapes

| spine_ratio | unclustered | biggest node |
| ---: | ---: | ---: |
| 0.3 | 63.7% | 28.8% |
| 0.5 | 50.7% | 48.3% |
| 0.7 | 30.0% | 69.7% |
| 0.9 | 20.6% | 79.4% |

These are the same fact twice. Half the catalog is one connected region with no
internal density structure; the only choice is whether to call it "noise" or
"one enormous cluster". Tuning moves the label, not the problem.

### Reading the labels settled it

At the most balanced setting the paths came out:

```
Hollow Knight  →  Visual Novel / Action Roguelike / Turn-Based Strategy
               →  Precision Platformer / 2D Platformer
               →  Golf / Sokoban / Pinball

Portal 2       →  …  →  Philosophical / Music-Based Procedural Generation / Hero Shooter
```

The depth-1 node holding 49% of the catalog was labelled "Visual Novel / Action
Roguelike / Turn-Based Strategy". That is not a labelling bug — it is an honest
description of a node containing everything.

## 3. Why HDBSCAN was the wrong tool, not a badly-tuned one

Steam is a **continuum**. There is no density valley between action-roguelikes
and action-platformers; the catalog fills the space between them. A
density-based method asked to find separated regions correctly reports one
connected region and some outlying islands. HDBSCAN answered honestly.

Checked, so this is not an excuse:

- Swept dims {10,20,50} × min_cluster_size {15,25,40,50,100,400} × min_samples
  {1,3,5,15} × {eom, leaf} — 40+ configurations. Every one either produced
  ≥73% flat noise or collapsed to 2–3 mega-clusters.
- L2-normalising after PCA (cosine geometry, matching how the embeddings were
  built and evaluated) changed nothing material: 77–88% noise.
- Clustered the **pure tag space** in case the text half was blurring density.
  Same picture: 73–78%.
- Distance concentration is severe — CV of pairwise distance 0.12, 10-NN
  contrast 1.78. That is a space without density contrast.

## 4. Recursive spherical k-means — the documented fallback

SPEC.md's risk table: *"Condensed tree is a mess after collapsing → Recursive
k-means, fixed branching — less principled, always balanced."* Taken.

**What it gives up.** Cluster count is imposed, not discovered. Every split is
forced whether or not a boundary exists, so a k-means cut through a continuum
is arbitrary at the margin: two near-identical games either side get different
labels. HDBSCAN would have declined to split there. There is no notion of an
outlier — every game is in a cluster whether it belongs or not.

**What it keeps.** The differentiator is untouched. SPEC.md's claim is that
labels come from a *semantic cluster hierarchy* rather than spatial tiles, and
this is still a hierarchy built in 512-d embedding space, never on 2D
coordinates. The WizMap contrast holds exactly as stated.

**What it gains.** 0% unclustered, no soft assignment, uniform branching so no
level is empty, and every node large enough for c-TF-IDF to say something.

Spherical: vectors are L2-normalised, so Euclidean k-means maximises cosine
similarity — the metric Phase 2 was built and evaluated on.

k=12 over k=8: biggest territory 11.8% vs 16.5%, parent-term repetition 20.2%
vs 23.1%, and at k=8 racing games and party games were buried inside a
territory labelled "Shooter / Action / FPS".

## 5. Noise handling

**Shipped tree: 0% unclustered.** k-means places every game in a leaf, so the
question does not arise, and nothing is soft-assigned or rendered specially.

For the HDBSCAN path, `assign_noise` implements both options SPEC.md asks to
be chosen between:

- `soft` — assign each unclustered game to the nearest leaf centroid by cosine.
  Every game gets a position; the cost is that an isolated game inherits a
  label that is a small lie.
- `keep` — leave them at the root as explicit unclustered territory, for the
  map to render dimmer and unlabelled.

`soft` was the default there, because the deliverable is a map and a game with
no position is worse than one with an approximate position. Both record the
assignment per game so the viewer can still distinguish core members from
soft-assigned ones. **Neither option lets noise disappear**, which was the
requirement.

A double-counting bug in the first implementation reported coverage of 184.2%:
soft-assigned games were *added* to their nearest leaf while still being
counted at the root. Fixed by reassigning rather than appending; the report now
prints distinct-vs-total and flags any duplicate placement.

## 6. Labelling: two failures before the working scoping

Sibling scoping is the requirement. Getting the *denominator* right took three
attempts, and the first two failed in opposite directions.

**Attempt 1 — `tf · log(1 + n/df)`** (BERTopic's smoothed form). A tag on every
sibling still scores `log 2 = 0.69`, and generic tags also carry the highest
tf, so they win outright. **90.0%** of children reused a top-3 term from their
parent — the "Action → Action → Action Indie" failure, verbatim.

**Attempt 2 — `tf · log(n/df)`.** A tag on every sibling now scores exactly 0,
which does kill generic terms — and also kills the *defining* term of a
coherent cluster, since every child of a farming-sim cluster carries "Farming
Sim". With the real terms zeroed, whatever tag sits in exactly one sibling wins
however rare. Stardew Valley's neighbourhood came out **"Memes / Naval Combat /
Pirates"**. Repetition fell to 17.7% while the labels got worse — a case where
the metric improved and the product did not.

Both failures share a cause: `df` is a **binary presence count**. It cannot
distinguish "on 90% of every sibling" from "on 2% of every sibling".

**Attempt 3 — prevalence × lift, which ships:**

```
p_c(t)    = fraction of this cluster's games carrying t
p_rest(t) = mean coverage of t across the other siblings
score     = p_c(t) · log( (p_c(t) + ε) / (p_rest(t) + ε) )
```

The log ratio is distinctiveness; the leading `p_c` is prevalence. A term must
be common *here* and less common *there*:

- 90% here, 85% siblings → near-zero log, suppressed
- 2% here, 0% elsewhere → large log, but `p_c` kills it
- 70% here, 10% elsewhere → wins, correctly

A genuinely defining tag survives even when every sibling carries some of it,
because the ratio compares *degrees* rather than presence against absence.

**There is no stopword list, and there must not be one.** SPEC.md is right
that special-casing common tags signals wrong scoping — but the corollary is
that the scope must be a *graded* comparison against siblings, not a binary
one. Commonness is then handled by the arithmetic: 20.2% repetition, 0 unnamed
nodes, no vocabulary hand-tuning anywhere.

## 9. Note for Phase 8: NMI against Steam genres will not work

Phase 8 specifies NMI against Steam genres as the label-quality metric. Phase 2
already established those genres carry almost no information here: genre
agreement scored **1.28–1.30× over baseline for every embedding variant**,
including pure text, because there are ~20 genres for 56,129 games and `Indie`
alone covers 59%. A metric that could not separate three very different
embeddings will not separate cluster depths either.

**Plan:** measure NMI against **community tags** instead — 453 values with real
discriminative power, already shown to separate variants cleanly (57.5% → 84.5%
on top-tag agreement). Report genre NMI alongside it, explicitly as the
uninformative baseline it is, so the substitution is visible rather than a
quiet swap. Recorded in EXPLAIN.md.

---

## 10. The rework: a reduction built for clustering

§2–3 rejected HDBSCAN on the grounds that Steam is a continuum. The continuum
claim is true. The conclusion drawn from it was not, because the diagnosis
stopped one step short: **512-d cosine space has no density contrast, and that
is a property of the space, not of the catalog.**

A separate UMAP reduction, at 5–15 dims with `min_dist=0.0` and
`metric='cosine'`, never rendered and never reused as the display layout, is
not what SPEC.md's rule forbids. The rule exists because a 2D layout packs
unrelated regions together to fill the plane, so clustering on it finds
UMAP's distortions. A 5-d reduction tuned for clustering has no such job.

The two trees SPEC.md insists on keeping separate remain separate: this
reduction feeds only the cluster tree, and Phase 4 fits its own layout.

### Density contrast, measured

| space | pairwise CV | 10-NN contrast |
| --- | ---: | ---: |
| 512-d cosine (rejected in §3) | 0.106 | 1.60 |
| PCA-50 of it | 0.122 | 1.78 |
| **UMAP 5-d, cosine, min_dist 0** | **0.386** | **12.83** |

An 8× gain in contrast. HDBSCAN's behaviour changes completely:

| | 512-d / PCA | UMAP 5-d |
| --- | ---: | ---: |
| games falling out at root | 39.8% | **0.0%** |
| condensed-tree nodes | 220 | 814 |
| forking nodes | 110 | **407** |
| flat `labels_` noise | 82.4% | 33.5% |

Swept `n_components` {5,10,15} × `n_neighbors` {15,30} × `min_cluster_size`
{25,50} × `min_samples` {5,10}; every combination landed at 0.0–0.2% root-out
against 39.8%. `umap_c5_n15` had the most forking structure.

The collapse trade-off from §2 simply disappears. At `spine_ratio` 0.3 the
512-d tree gave 63.7% unclustered; the UMAP tree gives **2.3%**, with the same
biggest-node share.

### Four candidates, judged by reading

| | k-means k=12 | UMAP→HDBSCAN | gated k-means on UMAP | **Leiden** |
| --- | ---: | ---: | ---: | ---: |
| nodes | 1,064 | 371 | 1,028 | **1,432** |
| leaves | 908 | ~310 | 737 | 897 |
| unclustered | 0% | 3.0% | 0% | **0%** |
| biggest node | 11.8% | 15.2% | 15.0% | 12.9% |
| max depth | 4 | 5 | 6 | 5 |
| parent-term repetition | **20.2%** | 26.4% | 29.4% | 38.5% |
| games stuck at depth 1 | 0% | **33.6%** | 0% | 0% |

Numbers did not decide it; paths did.

- **UMAP→HDBSCAN** produced the cleanest *territories* but left a third of the
  catalog at depth 1 with no finer label, and put Dota 2 in an 8,550-game node
  labelled "First-Person / Horror / Psychological Horror" that never
  subdivided for it. The depth-1 partition is HDBSCAN's merge order, which is
  arbitrary at the top: its children were individually coherent (Visual Novel,
  Shooter, Crafting/Survival, Hentai) while their parent had no honest name.
- **Gated bisecting k-means** fixed coverage and made splits earned (54
  refused), but silhouette rises monotonically as k falls, so it chose k=2 for
  201 of 317 splits and rebuilt the binary cascade: paths read "Platformer →
  Platformer → Platformer", 37.5% repetition. Taking the *widest* k within 85%
  of the best score cut that to 29.4%, but top-level placement stayed poor —
  Hollow Knight under "Shoot 'Em Up / Arcade".
- **Leiden** read best by a clear margin.

```
Europa Universalis IV  →  Strategy / RTS / Tower Defense
                       →  Turn-Based Strategy / Turn-Based / Historical
                       →  Turn-Based Combat / Turn-Based / Hex Grid
                       →  Historical / Military / War
                       →  4X / Economy / Grand Strategy

Hollow Knight          →  Platformer / 2D Platformer / Puzzle-Platformer
                       →  2D Platformer / 2D / Side Scroller
                       →  Metroidvania / Action-Adventure / Exploration
                       →  Metroidvania / Souls-like / Dark Fantasy

Dota 2                 →  FPS / Shooter  →  …  →  MOBA / Third Person / PvP
```

Sixteen top-level territories, all coherent, biggest 12.9%, nothing
unclustered. Community detection asks "which games are more connected to each
other than to the rest of the graph" rather than "where are the gaps", which
is the right question for a continuum — exactly as predicted.

### Why the repetition metric is misleading here

Leiden scores worst on parent-term repetition (38.5%) and reads best. The
metric counts any shared top-3 term between parent and child, which penalises
`Platformer → 2D Platformer → Metroidvania` — a legitimate refinement, and the
thing a zoomable map *wants*. It was built to catch "Action → Action → Action
Indie", where the child adds nothing, and it cannot tell that apart from
genuine narrowing. Reported, but not trusted over reading.

### Residue buckets, fixed

A node whose label does not actually distinguish it is now flagged rather than
presented as a genre. **Lift, not coverage, is the signal.** The motivating
case, `tree_k12`'s 5,522-game "Indie / Casual / VR", has 94% top-term coverage
— a coverage test passes it easily — but lift 1.68 against its siblings, the
lowest of twelve territories:

```
Action Roguelike / Bullet Hell   lift 60.9      Simulation / Management  lift  5.4
Visual Novel / Anime             lift 31.6      Indie / Casual / VR      lift  1.68  <- residue
```

Threshold 2.0 (roughly the 10th percentile of the lift distribution: p10 2.07,
p25 3.17, p50 5.61), with a weak 25% coverage floor. This is not a stopword
list: nothing names a tag, the test is structural, and a globally ubiquitous
tag fails it precisely because being everywhere is what makes it
uninformative.

In the shipped tree, 183 of 1,432 nodes are flagged, covering 22,074 games
(39%). Two of the sixteen territories — "Indie / Action / Strategy" (2,397)
and "Free to Play / VR / Indie" (1,275) — are among them and must render as
unclustered ground, not as genres.

That 39% is high and is not a bug in the detector: it is the honest report that
a large minority of the catalog sits in regions whose best available label is
weak. Phase 6 should render those dimmer and unlabelled rather than assert a
genre over them.

### k was searched, not picked

`gated_tree.py` searches k at every node (2..12) and reports the distribution;
the shipped Leiden tree has no k at all — resolution {0.5, 2, 8, 32, 128}
produces 16 → 45 → 119 → 360 → 892 communities, with community count emerging
from the graph rather than being imposed.

## 11. Shipped tree

```
games                 56,129        unclustered            0 (0.0%)
cluster nodes          1,432        leaves               897
max depth                  5        mean fan-out        2.67
largest cluster        12.9%        unnamed nodes          0
residue nodes    183 (22,074 games) build time         ~35s
```

Clusters per depth: 1 / 16 / 45 / 119 / 360 / 892.
Games land at depth 4 (16.7%) and depth 5 (83.1%).

## 12. Known weaknesses

- **Containment purity is only median 0.90, and 50% of nodes fall below 0.9.**
  Leiden is flat at each resolution; the hierarchy is imposed afterwards by
  majority containment, so a fine community that straddles two coarse ones is
  assigned to whichever holds more of it. 709 nodes straddle. The tree is
  therefore an *approximate* nesting, not a strict one, and `purity` is stored
  per node so the viewer can see where it is weak. A strictly nested
  alternative (agglomerating communities level by level) was not built.
- **39% of games sit under a residue-flagged node.** Honest, but it means a
  large part of the map has no confident label.
- **Dota 2 still enters under "FPS / Shooter"** at depth 1 and only reaches
  MOBA at depth 4. Its tag profile (Free to Play, Multiplayer, Strategy) is
  genuinely shooter-adjacent in this embedding.
- **Resolutions {0.5, 2, 8, 32, 128} were chosen to span a wide range**, not
  tuned. Levels are consequently uneven — 16 → 45 is a 2.8× fan-out, 360 → 892
  is 2.5×, but depth 1→2 carries far more semantic weight than 4→5.
- **The k-NN graph is k=20, unswept.** Graph density materially affects
  modularity clustering and was not explored.
- **Labels remain raw tag terms joined by slashes.** LLM tidying is cut line 1.
