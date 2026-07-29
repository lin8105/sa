# ASRF interface specification

This is the independent ASRF interface. Round 3 implements the encoder,
shared temporal extractor, ASB, BRB, in-memory boundary targets, losses, and
inference refinement. Training and dataset evaluation remain planned.

## Dataset sample

```text
heatmap:          FloatTensor[3, 88, T]
labels:           LongTensor[T]
boundary_targets: FloatTensor[T]
valid_mask:       BoolTensor[T]
trajectory_id:    str
timestamps:       optional LongTensor[T]
segments:         structured annotation metadata
```

`heatmap` is loaded from `citr_fingerprint_pure.png` after RGB conversion.
Its temporal width is the source image width and is never resized. Boundary
targets follow the official convention: frame zero is a boundary and every
frame whose label differs from the preceding frame is a boundary. The final
frame is not marked solely because it is final; right-padded positions are
zero.

## Batch

```text
heatmap:          FloatTensor[B, 3, 88, T_max]
labels:           LongTensor[B, T_max]
boundary_targets: FloatTensor[B, T_max]
valid_mask:       BoolTensor[B, T_max]
```

Temporal padding is applied only on the right. Padded labels use `-100` and
must be excluded by `valid_mask`.

## Planned model interfaces

```text
HeatmapEncoder:
  [B, 3, 88, T] -> [B, 128, T]

Shared long-term feature extractor:
  [B, 128, T] -> [B, 64, T]

ASB stage output:
  [B, 7, T]

BRB stage output:
  [B, 1, T]
```

The intended configuration has four ASB predictions and four BRB predictions:
one initial output plus three refinement outputs. The official implementation
returns all four during training and only the final output in evaluation mode;
the ASRF project will preserve that distinction explicitly.

## Inference refinement interface

Inputs:

- final ASB probabilities or logits;
- final BRB probabilities or logits;
- valid mask.

Outputs:

- raw ASB labels;
- selected boundary indices;
- predicted intervals;
- majority class per interval;
- refined frame labels.

Refinement uses BRB local maxima at configurable `theta_p` (default `0.5`),
half-open intervals `[start, end)`, and official majority voting. An optional
mean-probability result is diagnostic and is not used inside model forward.

## Round-3 loss interface

ASB losses consume stage logits, labels, and `[B,T]` valid masks. TMSE and
GS-TMSE use valid adjacent pairs only. GS-TMSE receives HeatmapEncoder
features as the CITR analogue of the official precomputed feature input.
BRB consumes stage logits and binary targets with a training-split positive
weight. All four stage losses are returned separately and averaged.
