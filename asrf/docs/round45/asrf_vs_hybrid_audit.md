# Round45 ASRF-only vs Hybrid evaluation

## Scope and definitions

This frozen comparison covers the same 36 held-out trajectories across PP,
plug, wipe, pour, and unscrew. ASRF-only is the final SF ASB argmax timeline
collapsed into consecutive-equal runs. Hybrid uses frozen SF labels and SF/r5
BRB boundary evidence, then assigns each predicted segment its majority SF
label. This is the raw, uncompressed Round45 comparison; no new inference or
metric computation was performed for this staging copy.

## Metrics

Temporal F1 uses Hungarian interval-IoU matching and ignores skill identity.
Semantic F1 and Edit are label-sensitive. Boundary precision/recall/F1 use a
separate one-to-one matching rule with a ±20-frame tolerance. These metric
families are not interchangeable: a high temporal interval score does not
mean the skill label is correct.

| Metric | ASRF-only | Hybrid | Difference | Paired 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Temporal F1@10 | 0.857036 | 0.875487 | +0.018451 | [-0.018846, +0.058169] |
| Temporal F1@25 | 0.845197 | 0.875487 | +0.030290 | [-0.009524, +0.074464] |
| Temporal F1@50 | 0.719416 | 0.841912 | +0.122496 | [+0.049610, +0.194821] |
| Edit score | 0.678403 | 0.757063 | +0.078659 | [+0.038824, +0.120484] |
| Semantic F1@50 | 0.646282 | 0.720204 | +0.073922 | [+0.042159, +0.106729] |
| Boundary F1@20 | 0.170969 | 0.316932 | +0.145963 | Not reported in aggregate bootstrap table |

The temporal F1@50 absolute gain is 0.122496 (17.027% relative). Paired
trajectory bootstrap used 2,000 resamples with seed 450045. Temporal F1@50
improved on 18 trajectories, was unchanged on 9, and degraded on 9.

## Boundary and family results

Boundary precision/recall/F1@20 changed from 0.158359/0.196115/0.170969 for
ASRF-only to 0.289487/0.360873/0.316932 for Hybrid. Mean absolute error among
matched boundaries changed from 9.693333 to 8.811458 frames.

Family temporal F1@50 (ASRF-only → Hybrid) was PP 0.918350 → 0.917636
(`N=11`), plug 0.718787 → 0.801901 (`N=6`), wipe 0.537453 → 0.752276
(`N=7`), pour 0.644010 → 0.886739 (`N=9`), and unscrew 0.642045 → 0.718944
(`N=3`). The full aggregate, family, and bootstrap tables are in
[`../../results/asrf_vs_hybrid/`](../../results/asrf_vs_hybrid/).

## Limitations

The test population is 36 trajectories. Hybrid did not globally reduce
fragmentation: mean predicted segment count rose from 9.22 to 9.81, and the
number of trajectories with more predicted than GT segments rose from 21 to
28. Hybrid retains SF frame labels and aggregates them per segment; it adds no
new learned semantic evidence, so temporal gains do not establish independent
skill-identity improvement. Boundary metrics are a distinct evaluation and
their bootstrap interval is not provided in the aggregate table.

Final-test GT was used only for evaluation after the model and Hybrid
configuration were frozen; it did not select checkpoints or parameters.
