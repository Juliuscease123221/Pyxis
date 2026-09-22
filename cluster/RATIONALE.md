# cluster/RATIONALE.md — Phase 3 design decisions

The differentiating piece of the project. Feeds EXPLAIN.md.

**Shipped:** `tree_k12_labelled.json` — recursive spherical k-means, k=12,
max_depth 5, labelled with sibling-scoped prevalence×lift over community tags.

**Headline:** HDBSCAN's condensed tree was built, collapsed, labelled and read —
and **rejected on the evidence**. It is kept in the repo, reproducible, because
the comparison is the result.

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

## 7. Shipped tree

```
games                 56,129        unclustered            0 (0.0%)
cluster nodes          1,064        leaves               908
max depth                  4        mean fan-out        6.78
largest cluster        11.8%        unnamed nodes          0
parent-term repetition 20.2%        build time          ~17s
```

Clusters per depth: 1 / 12 / 144 / 898 / 10.
Games land at depth 2 (19.5%), depth 3 (79.5%), depth 4 (0.9%).

Effectively **three good levels**, which is SPEC.md's own cut-line preference
("three good levels beat six bad ones") reached by measurement rather than by
cutting.

## 8. Known weaknesses

- **Every split is forced.** k-means has no way to say "this region has no
  internal structure". Some depth-3 leaves are arbitrary slices of a continuum,
  and their labels are correspondingly thin.
- **Depth 4 is nearly unused** (0.9% of games). `min_split=120` stops recursion
  before the depth cap, so the depth-6 ceiling in SPEC.md is never reached.
  The map gets three zoom levels of real semantics, not six.
- **Top-level territories are uneven in quality.** "Indie / Casual / VR" (5,522
  games) is a catch-all, not a genre. It is the residue bucket that any fixed-k
  partition of a continuum produces.
- **Misplacements are visible.** Fallout 4 sits under "FPS / Shooter /
  First-Person" rather than with open-world RPGs, because its tag profile
  genuinely leans shooter. Dota 2 reaches MOBA only at depth 3.
- **k=12 was chosen from two candidates** (8 and 12), on top-level label quality
  and balance. Not a search.
- **Labels are raw tag terms joined by slashes.** LLM tidying is cut line 1 and
  was not done; "Souls-like / Difficult / Dark Fantasy" is readable but is not
  a name a person would write.
- **No quantitative label-quality metric yet.** That is Phase 8 — with the
  substitution noted below.

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
