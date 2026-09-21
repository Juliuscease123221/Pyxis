# Phase 2 — embedding variants

Catalog: **56,129 games**. All variants share the same row order (appid ascending) and are scored against the same seeded held-out tag set.


## How to read these numbers

**Genre agreement** is the fraction of each game's 10 nearest neighbours sharing at least one Steam genre. The absolute value is nearly meaningless: `Indie` alone is on 59% of the catalog, so random pairs already agree **75.4%** of the time. Read the lift column.


**Tag prediction** masks one tag per game (drawn from the middle of the vote ranking) and asks whether the 10 nearest neighbours' tags recover it, precision@5. It is only meaningful for variants built *without* those tags — see the circularity note below.


## Results

| variant | dims | genre lift | tag pred @5 | method |
| --- | ---: | ---: | ---: | --- |
| `variant_b_holdout` | 128 | 1.29x | 52.9% | tag co-occurrence PPMI + truncated SVD |

### Scored, but circular on tag prediction

These variants were built from inputs containing the masked tag, so their tag-prediction score is inflated and is **not** comparable with the table above. Listed for completeness.

| variant | dims | genre lift | tag pred @5 (inflated) | method |
| --- | ---: | ---: | ---: | --- |
| `variant_b` | 128 | 1.30x | 80.5% | tag co-occurrence PPMI + truncated SVD |
| `variant_a_int8_cls` | 384 | 1.28x | 57.4% | bge-small-en-v1.5 ONNX int8, cls pooling, CPU |
| `variant_a_fp32_cls` | 384 | 1.28x | 57.0% | bge-small-en-v1.5 ONNX fp32, cls pooling, CPU |
| `variant_a_int8_mean` | 384 | 1.28x | 56.1% | bge-small-en-v1.5 ONNX int8, mean pooling, CPU |
| `variant_a_fp32_mean` | 384 | 1.28x | 55.9% | bge-small-en-v1.5 ONNX fp32, mean pooling, CPU |

## Spot check: Hollow Knight's 5 nearest neighbours

The metric that catches an embedding which has learned marketing register rather than gameplay.

| variant | neighbours |
| --- | --- |
| `variant_a_fp32_cls` | Candle Knight, Elypse, Hollow Floor, Hollowed, Slavania |
| `variant_a_fp32_mean` | Hollowed, Hollow Floor, Elypse, Candle Knight, Slavania |
| `variant_a_int8_cls` | Elypse, Hollow Floor, Void Memory, Corrupt, ANIMAL WELL |
| `variant_a_int8_mean` | Hollow Floor, Elypse, Void Memory, Hollow Witch, Corrupt |
| `variant_b` | WarriOrb, Lost Wish: In the desperate world, WarriOrb: Prologue, Unbound: Worlds Apart, The Guise |
| `variant_b_holdout` | Lost Wish: In the desperate world, The Guise, WarriOrb: Prologue, Unbound: Worlds Apart, Aeterna Noctis |
