# ASRF/Hybrid Temporal Skill Segmentation

This project segments robot demonstrations into temporally localized skills. It
adapts the Action Segment Refinement Framework (ASRF) to RGB CITR
task-fingerprint timelines: an **Action Segmentation Branch (ASB)** predicts
**WHAT** known skill is active, while a **Boundary Regression Branch (BRB)**
predicts **WHERE** skill transitions occur. A project-specific Hybrid combines
both signals to produce the final temporal segmentation.

## Motivation

Early experiments used **MS-TCN**, which performed well on the predefined known
skills but provided less reliable boundaries when trajectories contained unseen
skill patterns. This motivated the move to **ASRF**, where semantic recognition
and boundary localization are modeled separately through ASB and BRB.

## Method

The model processes the complete trajectory while preserving one output
position per input frame. It does not compress the entire demonstration into a
single trajectory embedding.

```text
ASB: WHAT known skill is active at each frame?
BRB: WHERE is there evidence of a skill transition?
```

Both branches operate on shared temporal features but solve different
subproblems.

## From Video ASRF to Robot CITR ASRF

The original ASRF architecture was proposed for temporal action segmentation
and combines long-term temporal features with a framewise Action Segmentation
Branch and a class-agnostic Boundary Regression Branch.

This project retains that WHAT/WHERE decomposition but adapts the input
representation to robot interaction trajectories. Instead of conventional video
features, the network operates on RGB heatmaps generated from **CITR
(Coordinate-Invariant Task Representation)** features.

The CITR HeatmapEncoder used here is project-specific and is not part of the
original ASRF architecture.

## CITR Input Representation

Each robot trajectory is represented by an 11-row time-aligned CITR
fingerprint.

The first ten rows describe interaction quantities:

```text
ff
ftau
tautau
fv
tauv
vv
fw
tauw
vw
ww
```

The eleventh row contains the reversed normalized gripper signal.

The resulting numeric fingerprint is rendered as an RGB heatmap and supplied
to ASRF as:

```text
[B, 3, 88, T]
```

where:

* `B` is batch size,
* `88` is heatmap height,
* `T` is the original trajectory length.

The temporal axis is preserved throughout the network.

## Heatmap Encoder

The project-specific HeatmapEncoder converts the RGB CITR heatmap into a
frame-aligned learned feature sequence:

```text
RGB CITR timeline [B,3,88,T]
        │
        ├─ Conv2D 3→16, kernel 5×5
        │  BatchNorm + ReLU
        │
        ├─ MaxPool 2×1
        │
        ├─ Conv2D 16→32, kernel 3×5
        │  BatchNorm + ReLU
        │
        ├─ MaxPool 2×1
        │
        ├─ Conv2D 32→128, kernel 3×5
        │  BatchNorm + ReLU
        │
        └─ mean over heatmap height
                    ↓
             [B,128,T]
```

Pooling is applied only along the heatmap-height dimension, so temporal
resolution remains unchanged.

The encoder therefore produces one learned 128-D feature vector for every
trajectory frame rather than one embedding for the entire demonstration.

## Long-Term Temporal Feature Extractor

The encoded sequence is projected from 128 to 64 channels and processed by a
stack of non-causal dilated temporal convolutions:

```text
[B,128,T]
     ↓
1×1 temporal projection
     ↓
[B,64,T]
     ↓
dilation 1
     ↓
dilation 2
     ↓
dilation 4
     ↓
dilation 8
     ↓
...
     ↓
dilation 512
     ↓
[B,64,T]
```

Each residual temporal layer contains:

* a kernel-3 dilated convolution,
* ReLU,
* a kernel-1 convolution,
* dropout (`p=0.5`),
* a residual connection.

There is no temporal pooling.

With the ten-layer dilation schedule used by the frozen model, a single stack
has a theoretical receptive field of 2,047 frames. The model therefore retains
frame-level output resolution while allowing each prediction to use long-range
temporal context.

## WHAT — Action Segmentation Branch

The **Action Segmentation Branch (ASB)** performs dense semantic
classification.

The shared 64-D temporal feature at every frame is projected to the predefined
skill classes, followed by three temporal refinement stages:

```text
shared temporal features
        ↓
initial ASB prediction
        ↓
refinement stage 1
        ↓
refinement stage 2
        ↓
refinement stage 3
        ↓
framewise skill prediction
```

Each refinement stage receives the previous stage's softmax probability
sequence.

For the frozen seven-skill model, the final output has shape:

```text
[B, 7, T]
```

ASB is therefore a framewise temporal semantic model rather than a segment-level
classifier.

## WHERE — Boundary Regression Branch

The **Boundary Regression Branch (BRB)** predicts whether each frame is close to
a transition between two skills.

Its output is:

```text
[B, 1, T]
```

and contains a class-agnostic boundary probability for every frame.

BRB does not predict the identity of either skill. Its task is only to provide
independent evidence about **where a temporal transition occurs**.

```text
ASB
→ semantic identity

BRB
→ temporal transition location
```

## Why Separate WHAT and WHERE?

ASB is trained on a fixed set of known skills. When an unseen skill appears, its
frames must still be assigned to one of those known semantic classes, so changes
in ASB labels are not a reliable way to determine the boundaries of an unseen
skill.

BRB instead predicts transitions without requiring a skill identity. This
class-agnostic boundary signal allows temporal boundaries to be proposed even
when the semantic class itself is not represented in the ASB vocabulary.

## Training Objective

All ASB and BRB stages are supervised during training.

The ASB objective combines class-weighted framewise cross-entropy with
Gaussian-similarity-weighted temporal mean-square error (GS-TMSE):

```text
L_ASB = L_CE + 1.0 × L_GS-TMSE
```

Cross-entropy trains semantic skill recognition, while GS-TMSE encourages
temporal consistency without forcing smoothing across strong feature
transitions.

The BRB uses positive-weighted binary cross-entropy because true transition
frames are sparse.

The complete objective is:

```text
L_total = L_ASB + 0.1 × L_BRB
```

For the frozen configuration:

```text
GS-TMSE tau   = 4
GS-TMSE sigma = 1
```

The BRB positive-class weight is determined from the fitting split.

## Hybrid Segmentation

A direct **ASRF-only** segmentation is obtained by collapsing consecutive frames
with the same final ASB label:

```text
CITR
 ↓
ASRF
 ↓
final ASB labels
 ↓
ASRF-only segmentation
```

The project **Hybrid** additionally combines the semantic timeline with
independently predicted boundary evidence:

```text
ASB semantic timeline ────────┐
                              ├─ Hybrid
BRB boundary evidence ────────┘
                                  ↓
                         temporal skill segments
```

ASB remains the source of semantic identity, while BRB contributes transition
locations.

## ASRF-only vs Hybrid

Both methods were evaluated on the same frozen 36-trajectory Round45 test set.

| Metric         | ASRF-only |       Hybrid |        Change |
| -------------- | --------: | -----------: | ------------: |
| Edit           |  0.678403 | **0.757063** | **+0.078659** |
| Semantic F1@10 |  0.736021 | **0.795095** | **+0.059075** |
| Semantic F1@25 |  0.706831 | **0.775150** | **+0.068318** |
| Semantic F1@50 |  0.646282 | **0.720204** | **+0.073922** |
| Boundary F1@20 |  0.170969 | **0.316932** | **+0.145963** |

Semantic F1@50 improves from `0.646282` to `0.720204`, corresponding to an
absolute gain of `0.073922` and a relative improvement of approximately
`11.44%`.

The original Round45 interval-only evaluation, which matches temporal intervals
without requiring the skill labels to agree, increased from `0.719416` to
`0.841912` at F1@50 (`+0.122496`, or `+17.03%`).

Hybrid improved F1@50 on 18 of the 36 trajectories, left 9 unchanged, and
degraded 9.

The strongest improvement is in boundary detection. Boundary F1@20 increases
from `0.170969` to `0.316932`, while the mean number of predicted segments
changes from `9.22` to `9.81`. Hybrid therefore does not simply reduce
over-segmentation; its main contribution is recovering or relocating transition
boundaries so that the resulting segments align better with the temporal
structure.

Because Hybrid retains ASB as its source of semantic evidence, it does not add
new semantic classification capability.

## Limitations

### Closed-Set Semantic Vocabulary

ASB is trained on a predefined set of semantic skills. An unseen skill therefore
cannot be represented as a new semantic class by ASB itself.

### Boundary Detection Does Not Identify the Skill

BRB can provide class-agnostic transition evidence, including around behaviors
that are not represented in the ASB vocabulary, but it does not determine what
the unseen skill is.

### Hybrid Does Not Eliminate Over-Segmentation

Although Hybrid improves boundary and segment-level metrics, the frozen
comparison does not show a reduction in the overall number of predicted
segments. Its benefit is better understood as improved transition recovery and
boundary placement.

## References

* Y. Ishikawa, S. Kasai, Y. Aoki, and H. Kataoka.
  **“Alleviating Over-Segmentation Errors by Detecting Action Boundaries.”**
  IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.
  [Paper](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html) ·
  [Original implementation](https://github.com/yiskw713/asrf)

* P. So, R. I. C. Muchacho, R. J. Kirschner, A. Swikir, L. F. C. Figueredo,
  F. J. Abu-Dakka, and S. Haddadin.
  **“CITR: A Coordinate-Invariant Task Representation for Robotic Manipulation.”**
  IEEE International Conference on Robotics and Automation (ICRA), 2024,
  pp. 17501–17507.
  [DOI: 10.1109/ICRA57147.2024.10611312](https://doi.org/10.1109/ICRA57147.2024.10611312)
