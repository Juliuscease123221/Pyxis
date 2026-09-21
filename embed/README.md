# embed — Phase 2

Three embedding variants over the same 56,129 games, compared on the same
metrics. The comparison is the deliverable; see [variants.md](variants.md) for
the numbers and the choice.

## Reproduce

```bash
# variant B first: no neural model, ~3s on CPU, usable immediately
python -m embed.variant_b
python -m embed.variant_b --holdout          # for fair tag-prediction scoring

# variant A: fetch + quantise the model, then measure before committing
python -m embed.variant_a --benchmark 1000   # throughput + projection, encodes nothing
python -m embed.variant_a --compare-fp32 600 # what int8 actually costs
python -m embed.variant_a --all              # int8+fp32 x mean+cls, 2 passes

# variant C: hybrid, instant once A and B exist
python -m embed.variant_c --a variant_a_int8_cls --b variant_b

# evaluation
python -m embed.evaluate --variant variant_b --spot-only
python -m embed.compare                      # writes variants.md
```

Model setup (one time):

```bash
mkdir -p data/models/bge-small
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -sL -o data/models/bge-small/$f \
    https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/$f
done
curl -sL -o data/models/bge-small/model_fp32.onnx \
  https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \
  quantize_dynamic('data/models/bge-small/model_fp32.onnx', \
                   'data/models/bge-small/model_int8.onnx', \
                   weight_type=QuantType.QInt8)"
```

## Files

| File | Job |
| --- | --- |
| `common.py` | Row order, text template, save/load. The single source of row order for the phase. |
| `variant_a.py` | bge-small via ONNX Runtime on CPU. Benchmarking, int8-vs-fp32, mean-vs-CLS. |
| `variant_b.py` | Tag co-occurrence: PPMI + truncated SVD. |
| `variant_c.py` | Hybrid concatenation with the quadratic-weight correction. |
| `evaluate.py` | Spot checks, genre agreement, tag prediction. |
| `compare.py` | Scores every built variant, writes `variants.md`. |

## Measured on this machine (6 CPU cores, no GPU)

| | throughput | full catalog | model size |
| --- | ---: | ---: | ---: |
| variant B (PPMI+SVD) | — | **3.4 s** | — |
| variant A int8 | 99.1 games/s | 9.4 min | 34 MB |
| variant A fp32 | 63.6 games/s | 14.7 min | 133 MB |

Length-sorted batching is worth 1.25x, cutting padding waste from 35.4% to
2.2%. Token lengths are mean 89 / p95 118 against the 192 cap.

## Three things worth knowing

**Row order is load-bearing.** Every variant embeds the same games in the same
order (appid ascending). `variant_c.py` refuses to concatenate if the appid
arrays disagree, because a silent mismatch pairs one game's text vector with
another game's tag vector and every downstream number still looks plausible.

**Tag prediction is circular unless the tag is held out at build time.** Both A
and B take tags as input. Scoring them on a tag they were built from inflated
variant B from 52.9% to 80.5%. `--holdout` on either variant removes the masked
tags before building.

**Hybrid weights are quadratic, not linear.** Both halves are unit vectors, so
`cos_C = (a^2*cos_A + (1-a)^2*cos_B) / (a^2 + (1-a)^2)`. alpha=0.7 means 84/16,
not 70/30.
