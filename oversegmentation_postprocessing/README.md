# Joint physical compression and merge

This package applies a frozen nine-signal physical low-activity test, then the unchanged canonical one-sided segmentation merge. Compression removes sustained inactive interior frames while preserving transition context and the historical gripper-event / skill-survival handling. The merge consolidates short redundant predictions using local motion similarity; segments containing physical gripper events are protected.

The additional signal limits were frozen from the 79 TRAIN trajectories only, after inspecting P99.5, P99.9, P99.95, P99.99, and the maximum of the historical Round35-removed frames. The selected finite envelope exactly reproduces the Round35 retained mapping on TRAIN (79/79 trajectories; zero differing frames). Some added-signal limits reach the TRAIN velocity-low candidate maximum because lower limits changed the historical mapping. Thresholds were not selected or adjusted using TEST.

On the full 36-trajectory TEST mapping audit, only 1 trajectory was exactly identical to Round35; there were 15,626 retained-index differences across the other 35. On the 34 trajectories with frozen Round91 bundles (two plug trajectories lack bundles), macro F1@50 was 0.839763 raw, 0.899273 after compression, and 0.927906 after merge. Boundary recall ±20 frames was 0.372297, 0.706033, and 0.676306 respectively. Thus the frozen TEST result is not an exact reproduction of historical Round45; the merge improved F1 over compression while reducing boundary recall. No TEST-driven tuning followed.

| Signal | Limit | Unit | TRAIN basis |
|---|---:|---|---|
| Linear velocity `||v||` | 0.02600749068 | m/s | Frozen Round35 |
| Angular velocity `||ω||` | 0.0477127692 | rad/s | Frozen Round35 |
| Linear acceleration `||a||` | 133.32933144573843 | m/s² | Velocity-low candidate maximum |
| Angular acceleration `||α||` | 281.1981816308892 | rad/s² | Velocity-low candidate maximum |
| Force magnitude `||F||` | 65.78368493108684 | N | Velocity-low candidate maximum |
| Torque magnitude `||τ||` | 2.811585602507774 | N·m | Velocity-low candidate maximum |
| Force change `||dF/dt||` | 692801.9009054365 | N/s | Velocity-low candidate maximum |
| Torque change `||dτ/dt||` | 40039.03073424154 | N·m/s | Velocity-low candidate maximum |
| Gripper velocity `|dg/dt|` | 1.0104263346161215e-12 | m/s | Just above Round35-removed maximum |

Temporal parameters remain Round35: minimum activity run 0.5 s, 0.6 s transition context per side, 10-frame skill survival, no gap bridging, and 0.5 s gripper-event context. Merge parameters remain the canonical Round40 defaults: fraction 0.80, local window 5, one pass, and whole-segment physical gripper-event protection.

The merge code is unchanged. Its Round40 fitting-derived scaler and motion thresholds are bundled byte-for-byte as `merge_scaler.yaml` and `merge_motion_thresholds.yaml` so the wrapper does not silently fall back to untracked local output files.

Example from the repository root (using the project's required interpreter):

```bash
PROJECT_PYTHON=/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python
"$PROJECT_PYTHON" oversegmentation_postprocessing/postprocess_oversegmentation.py \
  --bundle-dir asrf/outputs/0/round91_end_to_end_open_world_transition_audit_v002/bundles/standard \
  --data-root /media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data \
  --split test \
  --output-dir asrf/outputs/joint_physical_postprocess_v001
```

`physical_thresholds.json` records the frozen values and TRAIN-only provenance. The runner reuses the validated Round35 signal builder, canonical bundle compressor, and canonical merge implementation; it does not retrain or rerun ASRF inference.
