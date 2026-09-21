# Phase 2 — embedding variants

**Chosen: variant C, hybrid at α = 0.5** (`variant_c_a05`), an equal blend of
bge-small text embeddings and tag-co-occurrence embeddings, 512-d.

Catalog: 56,129 games. Every variant embeds the same games in the same row
order (appid ascending) and is scored against the same seeded held-out tag set.
Generated numbers live in [variants_data.md](variants_data.md); this file is
the authored decision.

---

## Results

Only variants built *without* the masked tags appear here. See
[Circularity](#circularity-the-metric-that-lied) for why that matters.

| variant | dims | genre lift | top-tag | tag pred @5 |
| --- | ---: | ---: | ---: | ---: |
| A — text only (`variant_a_int8_cls_holdout`) | 384 | 1.28x | 57.5% | 50.1% |
| B — tags only (`variant_b_holdout`) | 128 | 1.29x | 83.6% | 52.9% |
| C — α=0.3, 16% text (`variant_c_holdout_a03`) | 512 | 1.29x | 84.0% | 53.5% |
| **C — α=0.5, 50/50 (`variant_c_holdout_a05`)** | **512** | **1.30x** | **84.5%** | **56.0%** |
| C — α=0.7, 84% text (`variant_c_holdout_a07`) | 512 | 1.30x | 78.9% | **57.8%** |

Build cost, 6 CPU cores, no GPU:

| variant | full catalog | notes |
| --- | ---: | --- |
| B | **3.4 s** | no neural model at all |
| A (int8) | 9.4 min | 99.1 games/s, 34 MB model |
| A (fp32) | 14.7 min | 63.6 games/s, 133 MB model |
| C | ~1 s | concatenation, once A and B exist |

## The three metrics, and how much to trust each

**Genre agreement** — fraction of a game's 10-NN sharing ≥1 Steam genre.
Nearly useless in absolute terms: `Indie` alone is on 59% of the catalog, so
random pairs already agree **75.4%** of the time. Even the lift barely separates
anything (1.28x–1.30x across every variant, including the pure text one). Steam
has ~20 genres for 56,129 games; the metric saturates. Reported because
SPEC.md asks for it, but it did not inform the decision.

**Top-tag agreement** — fraction of a game's 10-NN also carrying its single
most-voted tag, restricted to games whose top tag is not one of the 12 most
common. This is the most *discriminating* automated metric (57.5% → 84.5% across
variants) but it is structurally circular: it asks whether an embedding clusters
by its own input, which necessarily favours tag-derived variants. It cannot be
used to crown B or C over A on its own.

**Tag prediction @5** — mask one tag per game (drawn from the middle of the vote
ranking), ask whether the 10-NN's tags recover it. The only metric with a real
holdout, and therefore the one with the strongest claim to fairness — but see
below for what it rewards.

**Spot checks** — 10-NN for 15 well-known games, read by a human. Crude, and the
only thing that caught the actual failure modes. Both automated metrics rank
α=0.7 at or near the top; the spot checks are what revealed that α=0.7 quietly
drops *Victoria II*, *Victoria 3* and *Imperator: Rome* from Europa Universalis
IV's neighbourhood.

## Circularity: the metric that lied

Both A and B take tags as input — A's text template contains a `Tags: ...` line,
B is built from nothing else. Scoring either on a tag it was built from measures
memorisation, not generalisation.

The inflation is not marginal:

| variant | tag pred @5, leaky | tag pred @5, held out | gap |
| --- | ---: | ---: | ---: |
| B | 80.5% | 52.9% | **−27.6 pts** |
| A | 57.4% | 50.1% | −7.3 pts |
| C (α=0.5) | 80.7% | 56.0% | −24.7 pts |

Two things follow. First, the leaky numbers would have ranked B far above A
(80.5% vs 57.4%) when the honest gap is 2.8 points. Second, the *size* of the
leak is itself a signal: B loses 27.6 points because tags are its only input,
A loses 7.3 because tags are one line of a longer string. A comparison that
mixed leaky and clean numbers — the natural thing to do, since only B obviously
needed a holdout — would have been wrong in both directions at once.

`--holdout` on `variant_a` and `variant_b` removes the masked tags at build
time. `variant_c` refuses to mark a hybrid clean unless *both* halves are, since
a leaky text half reintroduces the tag through the concatenation.

## Why α = 0.5 rather than α = 0.7

The two clean metrics disagree, and the disagreement is the interesting part:

| | α=0.5 | α=0.7 |
| --- | ---: | ---: |
| tag prediction @5 | 56.0% | **57.8%** |
| top-tag agreement | **84.5%** | 78.9% |
| genre lift | 1.30x | 1.30x |

α=0.7 is 1.8 points better at recovering a *masked* tag; α=0.5 is 5.6 points
better at clustering games that share their *defining* tag. Those pull apart
because masked-tag prediction rewards inferring tags from prose — a real
capability, and precisely the one SPEC.md warns is confounded, since store
descriptions are marketing copy. Recovering "Atmospheric" from an atmospheric-
sounding blurb is not evidence that two games play alike.

The spot checks break the tie, and they favour α=0.5 on 3 of 4 probes:

| probe | α=0.5 | α=0.7 |
| --- | --- | --- |
| Europa Universalis IV | Terra Invicta, **Victoria II**, **Victoria 3** | Making History 1 & 2 |
| Cuphead | Mago, **Enchanted Portals** *(a Cuphead clone)* | Battletoads, Levelhead, Foxyland 2 |
| Hollow Knight | Lost Wish, WarriOrb, **Aeterna Noctis** | Void Memory, OUTBUDDIES DX |
| Undertale | Unlucky Seven, Heartbound, OneShot | **DELTARUNE**, **OMORI** |

α=0.7 wins Undertale outright — DELTARUNE and OMORI are the canonical answers
and the tag-only side never finds them. But it loses the Paradox grand-strategy
cluster for EU IV, which is a worse failure: those games are near-identical in
play and a map that separates them is wrong in a way a viewer will notice.

α=0.5 is also the only α that means what its number says (see below), which
makes it the easier default to defend.

## The hybrid weights are quadratic, not linear

Both halves are unit vectors before scaling, so for the concatenation
`[α·A, (1−α)·B]`:

```
cos_C(x,y) = (α²·cos_A + (1−α)²·cos_B) / (α² + (1−α)²)
```

The weights go as **α²**, not α. So:

| α | nominal | actual text / tags |
| ---: | --- | --- |
| 0.3 | 30 / 70 | 15.5% / 84.5% |
| 0.5 | 50 / 50 | 50.0% / 50.0% |
| 0.7 | 70 / 30 | **84.5% / 15.5%** |

Reporting "α=0.7" as 70/30 would misdescribe the result by a factor of two.
Only α=0.5 is the equal blend it appears to be. `effective_weights()` prints
this alongside every build.

Note also that the *dimension counts* carry no weight: A is 384-d and B is 128-d,
but each half is unit-norm before scaling, so they contribute equally at α=0.5
despite the 3:1 size difference.

## What the text model actually gets wrong

Variant A retrieves games whose **descriptions** read alike, not games that play
alike. The clearest case:

```
Cuphead  ->  Gunheart, Battlegun, SnOut 2, Head Shot, DEATHPIT 3000
```

My first diagnosis was that the template leads with the game title, so the
encoder was matching title tokens — "Hollow Knight" → "Hollow Floor",
"Hollowed"; "Cuphead" → "Gunheart", "Head Shot", "SQUAREHEAD". That hypothesis
was **tested and rejected**: `variant_a_int8_cls_noname` drops the title
entirely, and Hollow Knight still returns "Hollow Floor" while Cuphead still
returns "Gunheart". The collisions survive without the name.

So the real cause is the one SPEC.md predicted from the start — generic
indie-game marketing prose. Every one of those games describes itself as a
fast, punishing action game with striking art. The title was a red herring.

Removing the name is a marginal aggregate gain (top-tag 57.8% → 60.2%, genre
lift unchanged) but costs a clear qualitative win: Undertale loses OMORI. The
name stays, matching SPEC.md's template.

## Decisions that went the other way from the spec

| decision | SPEC.md | shipped | why |
| --- | --- | --- | --- |
| Pooling | mean | **CLS** | BGE's own model card specifies CLS for v1.5 — it is what their contrastive objective trained. CLS also measured better on every A variant (57.4% vs 56.1% int8; 57.0% vs 55.9% fp32). Small margin, but two independent reasons point the same way. |
| Precision | fp32 implied (GPU) | **int8** | No GPU. int8 is 1.6x faster at full scale and **indistinguishable downstream**: 57.4% vs 57.0% tag prediction, identical 1.28x genre lift. |
| Batch size | 256 | **32** | CPU, per instruction. |
| Max length | unspecified | **192** | Measured token lengths are mean 89 / p95 118, so 192 truncates only a tail. |

### The int8 result is not what the intermediate number suggested

A direct int8-vs-fp32 comparison on 600 games looked alarming:

```
per-game cosine(int8, fp32)     mean 0.9725
top-10 neighbourhood overlap    71.8%
```

28% of each game's nearest neighbours change under quantisation. That looks
disqualifying for a map built entirely on nearest neighbours — and mean cosine
of 0.97 is exactly the kind of reassuring aggregate that hides it.

But encoding both in full and scoring them downstream showed **no measurable
difference**. The reordering is confined to near-ties that are semantically
interchangeable: swapping rank 4 and rank 7 among a dozen equally-similar
metroidvanias changes the overlap statistic and changes nothing a viewer would
see. The proxy metric was more pessimistic than reality, and the only way to
know that was to measure the thing we actually cared about.

## Variant B deserves specific credit

SPEC.md predicted "B often wins; that is a finding worth reporting." It
essentially did: B alone beats A alone on every clean metric (52.9% vs 50.1%
tag prediction, 83.6% vs 57.5% top-tag), from **453 tags and no neural model**,
in **3.4 seconds**. The chosen hybrid beats B by 3.1 points of tag prediction
and 0.9 of top-tag — real, but modest against a 165x build-time difference.

If the transformer half had to be cut, B alone would be a defensible ship.

Implementation notes on B:

- **PPMI over raw counts.** `Indie` is on 33,318 of 56,129 games and
  `Singleplayer` on 33,365; raw co-occurrence is dominated by tags that say
  nothing. PMI divides observed co-occurrence by what independence predicts, so
  ubiquitous tags score near zero. The positive clip discards the negative half,
  which is the noisy half on sparse data — 95.1% of nonzeros survive it.
- **Context-distribution smoothing at α=0.75** (Levy & Goldberg). Without it,
  PMI's boost to rare tags lets a handful of tags on a few dozen games dominate
  the SVD.
- **Rank weighting, 1/(1+r).** The dump preserves tag *order* but not vote
  *counts*, so rank is the only surviving trace of the vote distribution.
  Hollow Knight's first tag is Metroidvania and its seventh is Indie; treating
  those as equally true discards real signal.
- 128 dims from a 453-tag vocabulary retains 64.6% of variance.

## Known weaknesses

- **Genre agreement never discriminated anything.** Every variant scored
  1.28–1.30x. The metric is in SPEC.md and is reported, but it contributed
  nothing to the decision and should not be presented as though it did.
- **Top-tag agreement is circular** and structurally favours the variants that
  won. It is reported alongside the held-out number, never instead of it.
- **α was searched at three points**, not tuned. The true optimum is somewhere
  between 0.5 and 0.7 and was not located.
- **Spot checks are 15 games chosen by me**, which is a sample size of 15 and a
  selection I am not neutral about. They decided the α tie-break.
- **No evaluation of the map itself.** Every metric here is k-NN in the
  embedding space. Whether these clusters survive HDBSCAN and UMAP into
  something a person can navigate is Phases 3 and 4, and nothing yet proves it.
- **The catalog is mid-crawl.** Review counts mix dump vintages with freshly
  crawled ones, so `importance` is slightly inconsistent — irrelevant to
  embeddings, relevant to Phase 5.
