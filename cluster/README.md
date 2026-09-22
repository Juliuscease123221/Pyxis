# cluster — Phase 3

Builds the semantic cluster hierarchy that drives the map's labels and zoom
levels. This is the differentiating piece: labels come from this tree, never
from spatial tiles.

Design reasoning and the rejected alternatives are in
[RATIONALE.md](RATIONALE.md).

## Reproduce

```bash
# shipped path: recursive spherical k-means on the full 512-d variant C
python -m cluster.kmeans_tree --k 12 --max-depth 5 --out tree_k12
python -m cluster.label  --tree tree_k12
python -m cluster.stats  --tree tree_k12_labelled
python -m cluster.paths  --tree tree_k12_labelled --random 20

# rejected path, kept because the comparison is the result
python -m cluster.prepare --mode joint --dims 50
python -m cluster.tree --input pca --dims 20 --normalize \
    --min-cluster-size 15 --min-samples 1 --min-branch 15 \
    --spine-ratio 0.5 --out data/cluster/tree_main.json
python -m cluster.label --tree tree_main
python -m cluster.paths --tree tree_main_labelled
```

Everything under `data/cluster/` is regenerable; the k-means tree builds in
~17s.

## Files

| File | Job |
| --- | --- |
| `kmeans_tree.py` | **Shipped.** Recursive spherical k-means taxonomy. |
| `label.py` | Sibling-scoped prevalence×lift labelling. |
| `paths.py` | Prints root→leaf label paths. The actual acceptance test. |
| `stats.py` | Tree shape, balance, noise accounting, label health. |
| `tree.py` | HDBSCAN condensed tree → n-ary taxonomy, incl. spine collapse. |
| `prepare.py` | PCA, with the modality-imbalance diagnostic. |
| `sweep.py` | HDBSCAN parameter sweep. |
| `tune_collapse.py` | Collapse tuning, reusing one fit per (mcs, ms). |

## Shipped tree

```
games                 56,129        unclustered            0 (0.0%)
cluster nodes          1,064        leaves               908
max depth                  4        mean fan-out        6.78
largest cluster        11.8%        unnamed nodes          0
parent-term repetition 20.2%        build time          ~17s
```

Clusters per depth: 1 / 12 / 144 / 898 / 10. Games land at depth 2 (19.5%),
depth 3 (79.5%), depth 4 (0.9%) — three levels of real semantics.

Top-level territories:

```
6,620  Horror / Psychological Horror / Mystery
5,878  Simulation / Management / Sandbox
5,808  Puzzle / Relaxing / Casual
5,522  Indie / Casual / VR                       <- residue bucket, see RATIONALE
5,376  RPG / Turn-Based Combat / Fantasy
4,651  Platformer / 2D Platformer / Precision Platformer
4,630  FPS / Shooter / First-Person
4,387  Visual Novel / Anime / Romance
4,228  Local Multiplayer / Arcade / 4 Player Local
3,659  Strategy / RTS / Tactical
3,177  Action Roguelike / Bullet Hell / Rogue-lite
2,193  Free to Play / Massively Multiplayer / VR
```

## Three things worth knowing

**HDBSCAN was rejected on evidence, not skipped.** Its condensed tree here is a
caterpillar — depth 143, branching 2 everywhere, one spine shedding a cluster
per level. Steam is a continuum, so a density method correctly reports one
connected region. Reading the labels settled it: Hollow Knight's path ended at
"Golf / Sokoban / Pinball".

**The 84% noise figure is wrong and should not be quoted.** That is HDBSCAN's
EOM flat-label selection, which this pipeline never uses. The condensed tree
covers 100% of games; the honest unclustered figure is 20.6–39.8%.

**The label scoping must compare degrees, not presence.** `df` as a binary
count fails in both directions — smoothed idf lets generic tags win (90% parent
repetition), unsmoothed idf kills the defining tags too ("Memes / Naval Combat
/ Pirates"). Prevalence×lift with coverage on both sides fixes it, with no
stopword list.
