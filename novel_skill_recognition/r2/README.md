# Novel-Skill Recognition

This module determines whether a robot-skill segment predicted by ASRF/Hybrid belongs to the known skill distribution **OLD** or should be rejected as **UNKNOWN**.

The system has two main components:

* **ASRF/Hybrid** provides the temporal segment boundaries and predicts the known-skill label.
* **R2** evaluates each predicted segment independently and checks whether its interaction pattern is sufficiently supported by previously observed OLD data.

R2 is therefore **not a seven-class skill classifier**. If R2 accepts a segment as OLD, the ASRF/Hybrid prediction is kept. If R2 rejects it, the final output becomes `UNKNOWN`.

---

## Method

### Segment Representation

The system operates at the segment level because ASRF already provides the temporal skill proposals. R2 only needs to decide whether each predicted segment belongs to the known interaction distribution.

Each frozen ASRF-predicted interval is extracted directly from the canonical numeric CITR representation.

The first ten rows contain interaction features, while the eleventh row contains the normalized gripper signal. Each segment is resampled to:

```text
[11, 128]
```

and then processed independently by R2.

As a result, every predicted segment receives its own embedding and novelty score. R2 does not generate a single embedding for the entire trajectory.

### Trajectory-Shared Normalization

Earlier segment-local normalization removed the influence of novel frames, but it also removed useful relative magnitude information between segments. The final version therefore uses one shared normalization reference for all segments within the same trajectory.

The ten interaction channels use one max-absolute scale per channel, shared across all segments from that trajectory.

In the current evaluation, this scale is computed using only the **GT-labeled OLD frames** of the trajectory. Novel frames do not contribute to the scale. The gripper channel keeps its fixed canonical normalization.

For example, if two segments have maximum interaction magnitudes of 2 and 20, segment-local normalization would scale both to approximately 1. With a shared trajectory scale of 20, their relative magnitudes are preserved as 0.1 and 1.0.

Controlled experiments showed that replacing segment-local normalization with OLD-only trajectory-shared normalization substantially recovered OLD/novel separation, while the R2 architecture, training objective, scorer, and data split remained unchanged.

---

## From PM-5 to R2

PM-5 was originally designed to learn a skill-discriminative representation. Later experiments showed that strong separation between the seven known skills does not necessarily produce the most useful embedding space for detecting unseen skills, which motivated the move to R2.

The name `PM-5` comes from the earlier **parameter-matched temporal-granularity comparison**. PM-5 was the five-phase version of the `PM-5 / PM-8 / PM-10` family and was selected as the preferred parameter-matched representation.

PM-5 used:

```text
Cross-Entropy
+
Supervised Contrastive Loss
```

with the seven known skill labels. This encourages segments from the same skill to form similar embeddings and allows one prototype to be constructed for each known skill.

R2 removes this seven-class objective.

Instead of describing the known world using a small number of skill prototypes, R2 keeps a **distributed, multimodal OLD embedding space** made up of individual segment instances.

The aim is therefore not to force all OLD samples into one compact cluster, or even to form seven perfectly separated clusters. A legitimate OLD sample only needs to be supported by nearby OLD examples.

---

## R2 Encoder

Experiments with distributed OLD representations showed that local support was more useful for novelty detection than representing each skill with a single prototype. R2 therefore uses a known-only segment representation together with local kNN scoring.

Its encoder is:

```text
[11,128] CITR segment
        ↓
Conv1d 11 → 64, kernel 5
        ↓
Residual block, dilation 1
        ↓
Residual block, dilation 2
        ↓
Residual block, dilation 4
        ↓
Residual block, dilation 8
        ↓
Temporal mean pooling
        ↓
Linear 64 → 128
        ↓
L2 normalization
        ↓
128-D segment embedding
```

Each residual block contains two kernel-3 temporal convolutions with GroupNorm, GELU, and a residual connection.

The final R2 model does not contain a semantic seven-skill classification head.

### Contrastive Training

R2 is trained only on OLD TRAIN segments.

For each segment, two mildly perturbed versions are generated using small crop/resampling perturbations, temporal shifts, mild amplitude perturbation, and sensor noise.

The frozen augmentation parameters are a 122-of-128-frame crop resampled to 128 steps, a temporal roll from −2 to +2 frames, per-channel amplitude scaling within ±5%, and Gaussian noise at 2% of OLD TRAIN channel standard deviation. The three frozen seeds (84, 184, 284) are embedded independently; their embeddings are combined by coordinate-wise median followed by L2 normalization.

The two views of the same segment form a positive pair, while other segment instances in the batch are treated as negatives.

The training objective follows the NT-Xent formulation introduced by Chen et al. in [SimCLR, ICML 2020](https://proceedings.mlr.press/v119/chen20j.html).

Semantic skill labels are not used as contrastive class targets. In other words, R2 is not explicitly trained to force all `grasp`, `lift`, or other same-skill segments into a single cluster.

---

## OLD Memory and Novelty Detection

Because the OLD population is multimodal, the final detector keeps individual OLD examples rather than representing each skill with only one prototype.

After R2 training, every legal OLD TRAIN segment is encoded and stored as an individual memory embedding.

The memory contains examples from all known skills, but their semantic labels are not used by the novelty scorer.

For a test segment with embedding `z`, R2 finds the 20 nearest embeddings in the OLD memory using cosine distance:

```text
distance = 1 - cosine_similarity
```

The novelty score is the mean distance to these 20 nearest OLD examples.

```text
small mean distance
→ strong local OLD support
→ OLD

large mean distance
→ weak OLD support
→ UNKNOWN
```

The frozen operating threshold is calibrated using **OLD DEV only**, with a target 10% false-unknown rate:

```text
threshold = 0.3693014085292816
```

A score at or above this threshold is classified as `UNKNOWN`.

No novel test segment was used to train R2, build the OLD memory, or calibrate the threshold.

---

## Held-Out Test Results

The final frozen population contains:

```text
30 trajectories
313 ASRF-predicted segments
238 matched OLD
75 matched novel
0 unmatched
```

All R2 queries use the exact frozen ASRF-predicted intervals rather than GT segment boundaries.

| Test Set      | Trajectories | ASRF Segments |    AUROC ↑ | AUPR-OOD ↑ |    FPR95 ↓ |     OSCR ↑ |
| ------------- | -----------: | ------------: | ---------: | ---------: | ---------: | ---------: |
| Held-out Test |           30 |           313 | **0.9513** | **0.8887** | **0.3403** | **0.8811** |

95% trajectory-level bootstrap confidence intervals:

| Metric   | Result |           95% CI |
| -------- | -----: | ---------------: |
| AUROC    | 0.9513 | [0.9114, 0.9831] |
| AUPR-OOD | 0.8887 | [0.8154, 0.9532] |
| FPR95    | 0.3403 | [0.0353, 0.5150] |
| OSCR     | 0.8811 | [0.8325, 0.9305] |

The confidence intervals are based on 1,000 trajectory-level bootstrap replicates, and all 1,000 replicates were valid for every metric.

The trajectory figures visualize the 19 original Clean Test trajectories; the headline quantitative results use the full 30-trajectory held-out population. See [`figures/clean_test/`](figures/clean_test/) and the [frozen score table](results/held_out_test_asrf_segment_r2_scores.csv).

Recompute the published metrics with [`evaluation/compute_metrics.py`](evaluation/compute_metrics.py). The complete-population inference entry point is [`evaluation/infer_held_out_test.py`](evaluation/infer_held_out_test.py); it consumes preprocessed `[N,11,128]` ASRF-segment tensors plus the three frozen checkpoints and OLD-memory release assets, and verifies every segment ID and frame interval against the frozen score manifest. Public model/source functions are in [`src/r2.py`](src/r2.py), and sanitized freeze/split/threshold metadata is in [`metadata/`](metadata/).

The scripts use Python with NumPy, pandas, scikit-learn, and PyTorch. Canonical raw CITR extraction and the original robot recordings are not bundled; inference starts from canonical numeric CITR converted into the documented segment tensor format.

AUROC, AUPR-OOD, and FPR95 evaluate R2's novelty ranking. OSCR evaluates the combined:

```text
ASRF/Hybrid known-skill prediction
+
R2 novelty rejection
```

pipeline.

---

## Interpretation

The final AUROC of approximately `0.95` and AUPR-OOD of approximately `0.89` indicate strong overall separation between OLD and novel ASRF-predicted segments.

The results support the main design hypothesis:

> A known-only segment-level contrastive representation combined with local OLD-memory support can generalize to unseen skill families without using target-novel examples during training or threshold calibration.

R2 should not be understood as making the entire OLD population globally compact. Instead, it preserves a distributed OLD manifold: legitimate OLD segments tend to have nearby OLD support, while novel interaction patterns are more likely to fall outside these locally supported regions.

The normalization experiments also indicate that relative interaction magnitude between segments is useful for novelty detection. Normalizing each segment independently can remove part of this information.

---

## Current Limitations and Failure Analysis

### 1. Test-Time Normalization Still Requires GT OLD Information

The current trajectory-shared normalization addresses the loss of relative magnitude caused by segment-local normalization, but it still requires the GT OLD-frame mask for each test trajectory.

GT is **not** used to:

* define ASRF segment boundaries;
* train R2;
* construct the OLD memory;
* calibrate the novelty threshold.

However, GT is still needed to determine the trajectory normalization scale.

The current method should therefore be viewed as a **mechanism-validating evaluation**, rather than a fully online deployment procedure.

A deployment-ready version would need to replace this GT-dependent scale with a reference that is available at inference time, such as a fixed TRAIN-derived scale or another robust online normalization method.

### 2. High-Recall Rejection Remains Weaker

Although AUROC and AUPR-OOD are strong, FPR95 is:

```text
0.3403
```

with a wide confidence interval:

```text
[0.0353, 0.5150]
```

This means that achieving at least 95% novel recall still requires rejecting a non-trivial fraction of legitimate OLD segments.

The combination of strong overall ranking and weaker FPR95 suggests that most OLD and novel samples are well separated, but a smaller difficult subset still overlaps.

The uncertainty is also influenced by the limited evaluation population of 30 trajectories and 75 novel predicted segments.

### 3. R2 Detects Novelty but Does Not Identify the Novel Skill

R2 only outputs:

```text
OLD
or
UNKNOWN
```

An UNKNOWN segment is not automatically identified as `wipe`, `pour`, `insert`, `unscrew`, or another new skill.

A complete open-world system therefore still needs a later stage for human confirmation, candidate association, clustering, or few-shot skill registration.

### 4. Continual Few-Shot Expansion Is Not Yet Reliable

Later experiments showed that turning UNKNOWN segments into stable new skills is more difficult than simply detecting them.

Some skills such as `pour` and `unscrew` could be promoted successfully from a small number of demonstrations, while `wipe` generalized less consistently. Continual adaptation could also reduce OLD retention or affect the detection of other still-unseen skills.

This reflects a **stability–plasticity trade-off**:

```text
strong adaptation
→ learn the new skill
→ risk damaging the existing OLD space

conservative adaptation
→ preserve OLD
→ risk weak new-skill generalization
```

For this reason, the current R2 should mainly be regarded as a **frozen novel-skill detector**, rather than a complete continual-learning solution.
