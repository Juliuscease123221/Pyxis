# cluster — Phase 3

Builds the semantic cluster hierarchy that drives the map's labels and zoom
levels. This is the differentiating piece: labels come from this tree, never
from spatial tiles.

Design reasoning and the rejected alternatives are in
[RATIONALE.md](RATIONALE.md).

## Reproduce

```bash
# shipped path: recursive Leiden on the full 512-d variant C
python -m cluster.leiden_tree --recursive --out tree_leiden
python -m cluster.label  --tree tree_leiden
python -m cluster.stats  --tree tree_leiden_labelled
python -m cluster.paths  --tree tree_leiden_labelled --random 20

# rejected paths, kept because the comparison is the result
python -m cluster.kmeans_tree --k 12 --max-depth 5 --out tree_k12
python -m cluster.prepare --mode joint --dims 50
python -m cluster.tree --input pca --dims 20 --normalize     --min-cluster-size 15 --min-samples 1 --min-branch 15     --spine-ratio 0.5 --out data/cluster/tree_main.json
```

Everything under `data/cluster/` is regenerable; the shipped tree builds in
~39s.

## Files

| File | Job |
| --- | --- |
| `leiden_tree.py` | **Shipped.** Recursive Leiden over per-community k-NN subgraphs. |
| `kmeans_tree.py` | Rejected alternative: recursive spherical k-means. |
| `label.py` | Sibling-scoped prevalence×lift labelling. |
| `paths.py` | Prints root→leaf label paths. The actual acceptance test. |
| `stats.py` | Tree shape, balance, noise accounting, label health. |
| `tree.py` | HDBSCAN condensed tree → n-ary taxonomy, incl. spine collapse. |
| `prepare.py` | PCA, with the modality-imbalance diagnostic. |
| `sweep.py` | HDBSCAN parameter sweep. |
| `tune_collapse.py` | Collapse tuning, reusing one fit per (mcs, ms). |

## Shipped tree

Recursive Leiden. Each community is re-clustered on a k-NN graph rebuilt from
*only its own members*, so a child cannot straddle two parents and the nesting
is exact by construction.

```
games                 56,129        unclustered            0 (0.0%)
cluster nodes          1,117        leaves               874
max depth                  4        territories           27
largest cluster        10.0%        unnamed nodes          0
containment purity     1.000        top-term repeat     2.8%
  (exact)                           build time          ~39s
```

Clusters per depth: 1 / 27 / 224 / 676 / 189.

Top-level territories, with the Phase 4 purity that decides whether the viewer
draws them at all (threshold 0.35):

```
5,630  0.88  Platformer / 2D Platformer / Puzzle-Platformer
4,505  0.93  Visual Novel / Interactive Fiction / Romance
4,411  0.38  Shooter / FPS / Violent
4,398  0.79  Horror / Walking Simulator / Psychological Horror
3,844  0.13  Puzzle / Relaxing / Logic                     <- pruned
3,353  0.69  Management / Simulation / Building
3,330  0.91  Action Roguelike / Action RPG / Hack and Slash
3,272  0.08  RPG / Turn-Based Combat / JRPG                <- pruned
2,900  0.53  Local Multiplayer / Sports / 4 Player Local
2,433  0.12  Indie / Action / Simulation                   <- pruned
2,187  0.12  RTS / Strategy / War                          <- pruned
2,004  0.52  Shoot 'Em Up / Arcade / Bullet Hell
1,890  0.61  Point & Click / Hidden Object / Adventure
1,791  0.93  Racing / Driving / Automobile Sim
```

11 of 27 territories are pruned from the map. That is load-bearing rather than
tidiness: footprint correlates *inversely* with purity, so without pruning the
LOD would preferentially label the least coherent categories. See EXPLAIN.md.

## Three things worth knowing

**HDBSCAN was rejected on evidence, not skipped.** Its condensed tree here is a
caterpillar — depth 143, branching 2 everywhere, one spine shedding a cluster
per level. Steam is a continuum, so a density method correctly reports one
connected region. Reading the labels settled it: Hollow Knight's path ended at
"Golf / Sokoban / Pinball".

**Recursive Leiden replaced multi-resolution Leiden**, which inferred
parenthood by majority containment and left 709 communities straddling two
parents (median containment purity 0.90). Rebuilding each subgraph from only
its own members makes straddling impossible: containment purity 1.000 exact,
and top-term repetition falls 9.9% -> 2.8%.

**The 84% noise figure is wrong and should not be quoted.** That is HDBSCAN's
EOM flat-label selection, which this pipeline never uses. The condensed tree
covers 100% of games; the honest unclustered figure is 20.6–39.8%.

**The label scoping must compare degrees, not presence.** `df` as a binary
count fails in both directions — smoothed idf lets generic tags win (90% parent
repetition), unsmoothed idf kills the defining tags too ("Memes / Naval Combat
/ Pirates"). Prevalence×lift with coverage on both sides fixes it, with no
stopword list.
