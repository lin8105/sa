# Source-provenance diagnostic

This supplementary check compares the two source-provenance groups within the frozen, combined held-out test. Both groups are trajectory-unseen by R2 development; `source_split` is descriptive only and was not used for selection, calibration, or tuning. The primary result remains the combined 30-trajectory evaluation.

| Provenance | Trajectories | ASRF segments | OLD / novel | AUROC (95% CI) | AUPR-OOD (95% CI) | FPR95 (95% CI) | OSCR (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original `data/test/` | 19 | 200 | 162 / 38 | 0.955 (0.910–0.988) | 0.893 (0.781–0.966) | 0.321 (0.024–0.685) | 0.895 (0.839–0.948) |
| Manually held-out `data/train/` | 11 | 113 | 76 / 37 | 0.941 (0.889–1.000) | 0.881 (0.804–1.000) | 0.513 (0.000–0.580) | 0.850 (0.760–0.957) |

Each interval is a 95% percentile interval from 1,000 trajectory-level bootstrap replicates (seed 94094); all 1,000 replicates were valid for every metric in both groups. Ranking results are directionally similar, while point FPR95 is higher in the 11-trajectory held-out-training-pool group (0.513 vs 0.321). The intervals overlap broadly, so this small descriptive comparison does not establish a robust provenance effect; the FPR95 point difference should nevertheless remain visible rather than being hidden by aggregation.

## Isolation checks

- The 11 IDs are exactly the frozen Round94 strict-held-out selection and appear as `train` trajectories in the frozen Round91 ASRF manifest.
- All 73 matching candidate records in the Round94 R2 TRAIN manifest are marked `selected_holdout_trajectory=1` and `included=0`.
- Held-out trajectory overlap with included R2 TRAIN, included R2 DEV, and OLD memory is zero. The threshold manifest reports zero held-out trajectory segments and zero existing-test segments.
- The 11 frozen GT-OLD trajectory scales were reused from `round94_test_scales.json`; each scale's `citr_features.csv` source hash matches. GT OLD masks are used at test time for normalization, as disclosed in the primary report.
- The evaluated R2 score is on each exact ASRF-predicted interval, not a GT interval. No model, threshold, or split selection used either subgroup's test labels.
