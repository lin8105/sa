# Public source mapping

| Public file | Historical authoritative source / basis |
|---|---|
| `src/r2.py` | `asrf/scripts/run_round84f_nuisance_strict_old_new_feasibility.py` (R2 encoder, residual blocks, augmentations, OLD-only NT-Xent training); `asrf/scripts/run_round93_r2_trajectory_pure.py` (endpoint-aligned segment resampling); `asrf/scripts/run_round94_exposure_free_full_old_norm.py` (OLD-only trajectory scale, segment slicing, cosine kNN, OLD-DEV threshold, three-seed median aggregation and memory). |
| `evaluation/infer_held_out_test.py` | Clean extraction of Round94 embedding and scoring behavior in `asrf/scripts/run_round94_exposure_free_full_old_norm.py`; full-population score inventory in `held_out_test_asrf_segment_r2_scores.csv`. |
| `evaluation/compute_metrics.py` | Implements the published metric contract and OSCR/bootstrap definitions documented in `results/novel_skill_standard_metrics.md`; validated against the frozen 313-segment score table. |
| `src/__init__.py`, `evaluation/__init__.py` | New Python package markers; no scientific behavior. |

The public code does not replace canonical raw CITR extraction. It accepts aligned canonical numeric CITR and preserves the frozen Round94 segment, normalization, encoder, and scoring behavior.
