# ASRF Round 7A — BRB positive-weight ablation

## Verdict

The reciprocal weight is too sensitive for peak precision, but reducing it does
not solve BRB without a semantic trade-off. On all 12 current test trajectories
at the official threshold 0.50:

| weight | raw F1@50 | official F1@50 | boundary F1@±33 | false peaks | missed | harmed trajectories |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reciprocal 528.48 | 0.949 | 0.958 | 0.259 | 427 | 8 | 8 |
| 200 | 0.868 | 0.936 | 0.313 | 277 | 17 | 10 |
| 100 | 0.909 | 0.733 | 0.573 | 48 | 31 | 19 |
| 50 | 0.934 | 0.371 | 0.357 | 8 | 64 | 24 |
| 25 | 0.914 | 0.000 | 0.000 | 0 | 84 | 24 |

The best boundary-oriented candidate is pw=100. The best semantic-preserving
fixed-weight compromise is pw=200, but its raw F1@50 drop is material and it
does not dominate the reciprocal baseline. The recommended next experiment is
target/calibration work with pw=200 or pw=100 held fixed; this round does not
justify deploying either as an unconditional replacement.

## Implementation and target audit

The exact implementation audit is in
`docs/current_asb_brb_pipeline.md`. HeatmapEncoder -> shared temporal
extractor -> separate ASB/BRB branches is unchanged. Raw ASB does not use BRB;
official refinement uses predicted BRB peaks and majority voting; oracle
refinement uses ground-truth boundaries only.

The target remains single-frame: frame 0 is positive and each subsequent
positive is a canonical label transition. The final frame is not marked merely
because it is final. On `multitask_train`: 264 positive frames, 139255 negative
frames, ratio 0.0018922153971860462, and weight
`(264+139255)/264 = 528.4810606060606`. No target widening or smoothing was
introduced.

## Data and training integrity

The integrity artifact confirms 40 train, 13 validation, 5 pour, 3 pp, and 4
wipe test trajectories; w1, w2, w3, and w4 occur exactly once. There is no
train/validation overlap, test leakage, duplicated physical trajectory,
canonical-label violation, or two-scan instability. The external dataset was
read-only.

Training used seed 42 and independent initialization for all new models:

| weight | best epoch | stop epoch | validation total loss | checkpoint SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 200 | 19 | 34 | 0.359985928 | `cd67b2b395a7fb4fa726b40155480e9dc88c01b004facd5b3e8fccc5f6015037` |
| 100 | 25 | 40 | 0.349673876 | `79387541266d9aa4c9373cc5de186ff1ee71a18b0305c2860892cc69bc4d95af` |
| 50 | 17 | 32 | 0.348212335 | `d58bc7a094590880070d428c50cf9eec1fdcb9c56bba621bff08c175b50dcadb` |
| 25 | 19 | 34 | 0.353952967 | `9ee249bff6dde124e5757eaa5d6a63d2a416ee803ccbcdb3359edba49c1ddbd5` |

The frozen reciprocal checkpoint remains
`ad557bc5b10bc00d1582c3a1d82897e81173f6abc83dfc2220a2fb96ee2c0241`; the
frozen pour-only checkpoint remains
`586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358`.

## Task and transition observations

At official threshold 0.50, pw=200 retains test pour/pp/wipe official F1@50
of 0.901/1.000/0.932, while pw=100 gives 0.701/0.896/0.651. The composite
w4 official F1@50 is 0.895 for pw=200, 0.600 for pw=100, 0.190 for pw=50,
and 0 for pw=25; reciprocal is 1.000.

The transition table is in
`outputs/brb_ablation_round7a/per_transition_boundary_metrics.csv`. Reciprocal
is strongest on lift -> transport, transport -> pour, and pour -> pour_recover,
but weak on wipe -> lift. pw=200 reduces probability amplitude broadly but
preserves most transition recall; pw=100 gains boundary precision while losing
reach -> grasp and wipe -> lift recall. pw=50/25 become broadly under-sensitive.

## Artifacts

- `outputs/brb_ablation_round7a/baseline_manifest.json`
- `outputs/brb_ablation_round7a/split_integrity.json`
- `outputs/brb_ablation_round7a/target_audit.json`
- `outputs/brb_ablation_round7a/model_comparison.csv`
- `outputs/brb_ablation_round7a/fixed_threshold_metrics.csv`
- `outputs/brb_ablation_round7a/peak_diagnostics.csv`
- `outputs/brb_ablation_round7a/refinement_effect_by_trajectory.csv`
- `outputs/brb_ablation_round7a/per_transition_boundary_metrics.csv`
- `outputs/brb_ablation_round7a/figures/`

All four primary figures and all 20 representative trajectory figures
(pour/p1, pp/pp_c1, wipe/w1, wipe/w4 for every experiment) were opened and
visually inspected. Full per-class precision/recall/F1, confusion matrices,
all boundary tolerances, collapsed sequences, over/under-segmentation, and
threshold diagnostics are in the per-experiment JSON/CSV artifacts.
