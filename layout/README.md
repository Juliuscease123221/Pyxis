# layout — Phase 4

The 2D coordinates people look at, plus the hierarchy-consistency measurement
that decides whether they need fixing.

## Two UMAPs, and they are not the same object

| | `cluster/umap_reduce.py` | `layout/layout.py` |
| --- | --- | --- |
| dims | 5 | 2 |
| `min_dist` | 0.0 | 0.05 |
| rendered | never | yes |
| purpose | density contrast for HDBSCAN/Leiden | the map |

Neither is derived from the other — parallel branches off the same vectors,
as SPEC.md's architecture requires. The `chained` variant deliberately
breaks that independence as an experiment; see below.

## Reproduce

```bash
python -m layout.layout --fit --measure          # baseline + consistency table
python -m layout.variants --fit supervised chained --target-weight 0.1 0.3 0.5
python -m layout.variants --compare              # side-by-side
```

## What is measured

Regions are **trimmed** convex hulls — the hull of the central 95% of members
by distance to centroid. An untrimmed hull is hostage to its worst outlier:
one stray member stretches the region across half the map and drives every
purity number to noise.

- **containment** — fraction of a node's members inside its *parent's* region.
  Note that "members inside their own hull" is 1.0 by construction and
  measures nothing; the parent's hull is the informative denominator.
- **purity** — of catalog points inside a node's region, the fraction that
  belong to it.
- **purity lift** — purity divided by a **size-matched random baseline**:
  clusters of the same size drawn at random, hulls computed identically. Raw
  purity is size-correlated (0.169 under 50 games, 0.443 over 1,000), so
  cross-level comparison needs the normalisation. Same lift construction as
  the cluster labels.

## The constraint on fixes

**Any layout fix must keep cross-territory distance data-driven.** Phase 7's
frontier recommendations are defined as territory *adjacent to* the user's hot
zones, so if adjacency becomes an artifact of a packing algorithm the feature
loses its basis — and Phase 4's own acceptance check would not notice, because
containment and purity would both improve.

That rules out SPEC.md's recursive layout (fix #2) as anything but a last
resort, despite it being the middle option by effort. The two variants here
both satisfy the constraint.

## Files

| File | Job |
| --- | --- |
| `layout.py` | Baseline 2D UMAP; trimmed hulls; containment/purity per depth; writes bbox + centroid per node for Phase 6. |
| `variants.py` | Supervised and chained layouts, purity lift, side-by-side comparison. |
