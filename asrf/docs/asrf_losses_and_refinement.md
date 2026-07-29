# ASRF losses and refinement (round 3)

Reference implementation: official ASRF commit
`9623f1e8d9a1171333a4eeb65d190997b6c44a95`.

## Boundary targets

For a valid label sequence `y[0:T]`, the target is `b[0] = 1` and
`b[t] = 1` exactly when `y[t] != y[t-1]`. The final frame is not marked solely
because it is final. Right padding is zero. Timestamp annotation ends are
exclusive and are converted to heatmap columns before targets are generated.
The round-3 audit covered train p1–p10 using train and validation splits and
test p1–p5; all 15 trajectories matched.

## Classification and smoothing losses

Class weights use the official median-frequency formula:

```text
frequency[c] = count[c] / sum(count)
weight[c] = median(frequency) / frequency[c]
```

CE consumes logits directly, excludes invalid frames, and uses the weighted
PyTorch mean over valid targets. Each of four ASB stage losses is computed
independently and stage-averaged.

TMSE computes adjacent log-softmax differences, squares them, and clamps each
element at `tau^2`, with `tau=4.0`. GS-TMSE multiplies that penalty by

```text
exp(-||feature[t] - feature[t-1]||_2 / (2*sigma^2))
```

with `sigma=1.0`. Both frames in a pair must be valid. The official code does
not detach the similarity input. Official ASRF receives precomputed I3D
features; this CITR adaptation passes HeatmapEncoder output as the analogous
pre-TCN feature representation.

## Boundary and combined losses

BRB uses masked `BCEWithLogitsLoss` with the official reciprocal positive
ratio weight, `total_valid_boundary_frames / positive_boundary_frames`. The
Each trajectory's valid-frame BCE is averaged before the batch average, and
the four stage losses are then averaged. The combined objective is:

```text
L_ASRF = (mean CE + 1.0 * mean GS-TMSE) + 0.1 * mean BRB BCE
```

The smoothing weight `1.0` and boundary coefficient `0.1` are from the
checked-in official 50Salads/Breakfast configuration used as this baseline.

## Boundary selection and intervals

Values strictly below the threshold are zeroed, so threshold equality
survives. Frame zero is selected when valid. Interior peaks require strict
greater-than comparisons on both neighbors; plateaus are rejected and the
final frame is not an interior peak. Round 3 defaults to `theta_p=0.5`.

Selected boundaries become sorted, deduplicated start indices. Intervals use
`[start, end)`, begin at zero, end at the valid length, and cover every valid
frame exactly once.

For each interval, official majority voting counts raw ASB argmax labels. If
counts tie and class scores are available, the class with the largest summed
original score wins; exact score-sum ties keep the lowest class ID. The
optional mean-probability method is diagnostic only. No duration, grammar, or
transition constraint is applied.

Known limitation: if a BRB interval contains a majority of the wrong ASB
class, majority voting propagates that wrong class over the interval.
