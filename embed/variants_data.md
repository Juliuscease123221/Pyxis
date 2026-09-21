# Phase 2 — embedding variants

Catalog: **56,129 games**. All variants share the same row order (appid ascending) and are scored against the same seeded held-out tag set.


## How to read these numbers

**Genre agreement** is the fraction of each game's 10 nearest neighbours sharing at least one Steam genre. The absolute value is nearly meaningless: `Indie` alone is on 59% of the catalog, so random pairs already agree **75.4%** of the time. Read the lift column.


**Tag prediction** masks one tag per game (drawn from the middle of the vote ranking) and asks whether the 10 nearest neighbours' tags recover it, precision@5. It is only meaningful for variants built *without* those tags — see the circularity note below.


## Results

| variant | dims | genre lift | top-tag | tag pred @5 | method |
| --- | ---: | ---: | ---: | ---: | --- |
| `variant_c_holdout_a07` | 512 | 1.30x | 78.9% | 57.8% | hybrid [0.7*A, 0.30000000000000004*B] |
| `variant_c_holdout_a05` | 512 | 1.30x | 84.5% | 56.0% | hybrid [0.5*A, 0.5*B] |
| `variant_c_holdout_a03` | 512 | 1.29x | 84.0% | 53.5% | hybrid [0.3*A, 0.7*B] |
| `variant_b_holdout` | 128 | 1.29x | 83.6% | 52.9% | tag co-occurrence PPMI + truncated SVD |
| `variant_a_int8_cls_holdout` | 384 | 1.28x | 57.5% | 50.1% | bge-small-en-v1.5 ONNX int8, CPU |

### Scored, but circular on tag prediction

These variants were built from inputs containing the masked tag, so their tag-prediction score is inflated and is **not** comparable with the table above. Listed for completeness.

| variant | dims | genre lift | top-tag | tag pred @5 (inflated) | method |
| --- | ---: | ---: | ---: | ---: | --- |
| `variant_c_a03` | 512 | 1.30x | 84.2% | 80.9% | hybrid [0.3*A, 0.7*B] |
| `variant_c_a05` | 512 | 1.30x | 84.4% | 80.7% | hybrid [0.5*A, 0.5*B] |
| `variant_b` | 128 | 1.30x | 83.9% | 80.5% | tag co-occurrence PPMI + truncated SVD |
| `variant_c_a07` | 512 | 1.30x | 78.9% | 73.8% | hybrid [0.7*A, 0.30000000000000004*B] |
| `variant_a_int8_cls_noname` | 384 | 1.29x | 60.2% | 59.0% | bge-small-en-v1.5 ONNX int8, CPU |
| `variant_a_int8_cls` | 384 | 1.28x | 57.8% | 57.4% | bge-small-en-v1.5 ONNX int8, cls pooling, CPU |
| `variant_a_fp32_cls` | 384 | 1.28x | 57.8% | 57.0% | bge-small-en-v1.5 ONNX fp32, cls pooling, CPU |
| `variant_a_int8_mean` | 384 | 1.28x | 57.3% | 56.1% | bge-small-en-v1.5 ONNX int8, mean pooling, CPU |
| `variant_a_fp32_mean` | 384 | 1.28x | 57.1% | 55.9% | bge-small-en-v1.5 ONNX fp32, mean pooling, CPU |

## Spot check: Hollow Knight's 5 nearest neighbours

The metric that catches an embedding which has learned marketing register rather than gameplay.

| variant | neighbours |
| --- | --- |
| `variant_a_fp32_cls` | Candle Knight, Elypse, Hollow Floor, Hollowed, Slavania |
| `variant_a_fp32_mean` | Hollowed, Hollow Floor, Elypse, Candle Knight, Slavania |
| `variant_a_int8_cls` | Elypse, Hollow Floor, Void Memory, Corrupt, ANIMAL WELL |
| `variant_a_int8_cls_holdout` | Hollow Floor, Souldiers, Elypse, Candle Knight, ANIMAL WELL |
| `variant_a_int8_cls_noname` | Void Memory, Sheba: A New Dawn, Hollow Floor, The Little Brave, Elypse |
| `variant_a_int8_mean` | Hollow Floor, Elypse, Void Memory, Hollow Witch, Corrupt |
| `variant_b` | WarriOrb, Lost Wish: In the desperate world, WarriOrb: Prologue, Unbound: Worlds Apart, The Guise |
| `variant_b_holdout` | Lost Wish: In the desperate world, The Guise, WarriOrb: Prologue, Unbound: Worlds Apart, Aeterna Noctis |
| `variant_c_a03` | WarriOrb, Lost Wish: In the desperate world, WarriOrb: Prologue, Unbound: Worlds Apart, The Guise |
| `variant_c_a05` | Lost Wish: In the desperate world, Unbound: Worlds Apart, WarriOrb, Aeterna Noctis, WarriOrb: Prologue |
| `variant_c_a07` | Void Memory, Unbound: Worlds Apart, Aeterna Noctis, OUTBUDDIES DX, Valdis Story: Abyssal City |
| `variant_c_holdout_a03` | Lost Wish: In the desperate world, The Guise, WarriOrb: Prologue, Unbound: Worlds Apart, Aeterna Noctis |
| `variant_c_holdout_a05` | Lost Wish: In the desperate world, The Guise, Unbound: Worlds Apart, Aeterna Noctis, Nine Sols |
| `variant_c_holdout_a07` | Void Memory, Aeterna Noctis, Unbound: Worlds Apart, Lillusion, Aestik |
