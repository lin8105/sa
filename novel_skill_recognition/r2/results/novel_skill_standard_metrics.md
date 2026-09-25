# Novel-skill standard metrics

## Primary held-out test

The primary population is the combined set of all 30 fully trajectory-held-out test trajectories. It contains 19 trajectories from the original `data/test/` folder and 11 manually held out from the original `data/train/` pool. The 313 frozen ASRF/Hybrid-predicted segments match the frozen Round91 manifest exactly: 238 matched OLD segments, 75 matched novel segments, and no unmatched predictions.

| Test set | Trajectories | ASRF segments | Matched OLD | Matched novel | Unmatched | AUROC ↑ (95% CI) | AUPR-OOD ↑ (95% CI) | FPR95 ↓ (95% CI) | OSCR ↑ (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-out Test | 30 | 313 | 238 | 75 | 0 | 0.951 (0.911–0.983) | 0.889 (0.815–0.953) | 0.340 (0.035–0.515) | 0.881 (0.832–0.930) |

| Metric | Point estimate | Bootstrap median | 95% percentile CI | Valid replicates |
|---|---:|---:|---:|---:|
| AUROC | 0.951317 | 0.952148 | [0.911448, 0.983111] | 1,000 |
| AUPR-OOD | 0.888739 | 0.889471 | [0.815410, 0.953216] | 1,000 |
| FPR95 | 0.340336 | 0.320551 | [0.035274, 0.515035] | 1,000 |
| OSCR | 0.881064 | 0.881473 | [0.832479, 0.930498] | 1,000 |

## Protocol

- The 7 OLD labels are `reach`, `grasp`, `lift`, `transport`, `place`, `release`, and `retreat`. A matched GT skill outside this set is novel/UNKNOWN (positive class), including `pour_recover` and `rotation`. The GT match supplies only the evaluation label: each R2 query is the exact frozen ASRF `[start_frame, end_frame)` interval.
- R2 novelty is the mean cosine distance to the 20 nearest frozen OLD TRAIN memory embeddings; higher scores indicate more novelty. The OLD/UNKNOWN decision uses the unchanged OLD-DEV 10%-FUR threshold `0.3693014085292816` (UNKNOWN iff score ≥ threshold). AUROC, AUPR-OOD, and FPR95 use raw novelty scores, not thresholded predictions. FPR95 is the OLD false-positive rate at the first ROC operating point whose novel TPR reaches at least 95%; this sweep is descriptive and does not alter the deployed threshold.
- OSCR combines the frozen ASRF/Hybrid predicted skill and R2 rejection score. At each novelty cutoff, CCR is correctly labeled OLD segments accepted as OLD divided by all OLD segments; FPR is novel segments incorrectly accepted as OLD divided by all novel segments. OSCR is trapezoidal area under CCR versus FPR; it is not a seven-class R2 classifier metric.
- Confidence intervals use 1,000 trajectory-level bootstrap replicates (seed 94094), resampling complete trajectories with replacement and retaining all segments, including repeated copies. Each metric had 1,000 valid replicates.

## Freeze and limitation

Frozen Round94 checkpoints (seeds 84/184/284), aggregate-embedding policy, OLD memory, cosine kNN-20 scorer, canonical CITR preprocessing, normalization rule, and OLD-DEV threshold were reused. The 11 held-out training-pool trajectories had zero overlap with R2 TRAIN, DEV, OLD memory, or threshold calibration and were not used for model selection. ASRF was not rerun; its exact Round91 frozen segment manifest was reused. No model, threshold, split, or postprocessing setting was changed. CUDA was unavailable for the additional inference, so those 113 segments were embedded on CPU. A CPU parity rerun of all 200 previously saved original-test segments differed from the existing CUDA scores by at most `9.95e-5` (median `9.76e-6`); no frozen OLD/UNKNOWN decision changed.

As in the frozen Round94 protocol, test-time trajectory-shared scales for the first ten channels use each trajectory's GT OLD-frame mask. GT does not set query boundaries or tune the model/threshold, but this preprocessing is GT-dependent and is not fully online.

The per-segment input to these metrics is [`held_out_test_asrf_segment_r2_scores.csv`](held_out_test_asrf_segment_r2_scores.csv). Source-split metrics are diagnostic only and are recorded in [`novel_skill_provenance_diagnostic.md`](novel_skill_provenance_diagnostic.md); neither provenance subgroup replaces the combined primary population.
