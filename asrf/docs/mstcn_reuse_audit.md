# MSTCN reuse audit

This audit was performed by read-only inspection of
`/home/yue/Documents/zsc_Franka/mstcn` at commit
`fa603a99592c5df0eb349e3ac7ce00d1bda42d0e`. ASRF does not import from that
checkout at runtime. In round 1, only the small pour split files and label
configuration were copied; no MSTCN source, output, checkpoint, or dataset was
copied.

| Concern | MSTCN source and symbol | Copy/adapt decision | Dependencies and hidden assumptions | Tests | ASRF destination |
|---|---|---|---|---|---|
| Heatmap loading | `src/seg_learning/data/dataset.py`: `_load_heatmap_tensor`, `load_trajectory_sample` | Rewrite cleanly; preserve behavior | PIL RGB conversion, float range `[0,1]`, source PNG is `[H,T]`; width must equal timestamp rows; no horizontal resize | `tests/test_dataset.py`, `tests/test_dataset_validation.py` | `src/asrf/data/dataset.py` (round 1) |
| Temporal mapping | `src/seg_learning/data/validation.py`: `load_demo_heatmap_column_mapping`, `load_heatmap_column_mapping` | Rewrite cleanly | `citr_features.csv` has one increasing `timestamp_us` per heatmap column; optional explicit mapping is supported by MSTCN | `tests/test_dataset.py`, validation tests | `src/asrf/data/dataset.py` (round 1) |
| Segment parsing | `src/seg_learning/data/annotations.py`: `load_segments_csv`, `detect_annotation_format` | Adapted behavior | Timestamp ends are exclusive; frame ends are inclusive; rows must be non-empty and unambiguous | `tests/test_annotation_conversion.py` | `src/asrf/data/annotations.py` (round 1) |
| Frame labels | `convert_segments_to_frame_labels` in the same file | Adapted behavior | Full coverage is required when background is disabled; overlapping segments fail | Annotation-conversion tests | `src/asrf/data/annotations.py` (round 1) |
| Aliases | `normalize_label_name`, `LabelMapping` | Adapted behavior | Alias chains are resolved; `pick -> reach`, `translation -> transport` | Annotation-conversion tests | `src/asrf/data/labels.py` (round 1) |
| Canonical classes | `configs/labels_pour.yaml` | Copied verbatim | Seven contiguous IDs; aliases are metadata, not extra classes | Label/config tests | `configs/labels_pour.yaml` |
| Split parsing | `src/seg_learning/data/validation.py`: `read_split_file` | Adapted behavior | One trajectory ID per non-empty line; split files are repository-local | Dataset/validation tests | `src/asrf/data/dataset.py`, `splits/` |
| Valid masks and batching | `src/seg_learning/data/collate.py`: `trajectory_collate_fn` | Rewrite cleanly | Right-only temporal padding; labels use `-100`; boolean mask excludes padding | `tests/test_collate.py` | `src/asrf/data/collate.py` |
| Frame accuracy | `src/seg_learning/evaluation/metrics.py`: `frame_accuracy` | Planned clean adaptation | Ignore index and valid masks must be applied before aggregation | `tests/test_metrics.py` | `src/asrf/evaluation/metrics.py` (later round) |
| Edit score | `edit_score`, segment helpers in `evaluation/metrics.py` | Planned clean adaptation | Consecutive labels collapse; normalized Levenshtein score is in `[0,1]` | `tests/test_metrics.py` | `src/asrf/evaluation/metrics.py` |
| F1@10/25/50 | `segmental_f1`, `f1_at_10`, `f1_at_25`, `f1_at_50` | Planned clean adaptation | Inclusive frame segments and one-to-one matching | `tests/test_metrics.py`, `test_segments.py` | `src/asrf/evaluation/metrics.py` |
| Per-class metrics | `per_class_precision`, `per_class_recall`, `per_class_f1` | Planned clean adaptation | Confusion matrix rows are targets and columns are predictions | `tests/test_metrics.py` | `src/asrf/evaluation/metrics.py` |
| Confusion matrix | `confusion_matrix` | Planned clean adaptation | Invalid/padded labels are filtered before counting | `tests/test_metrics.py` | `src/asrf/evaluation/metrics.py` |
| Visualization | `src/seg_learning/evaluation/visualization.py`: `save_aligned_trajectory`, `save_annotation_vs_prediction`, `save_prediction_only` | Planned clean adaptation | Heatmap and frame axes share the same temporal columns; frame endpoints are inclusive | `tests/test_visualization.py`, prediction-export tests | `src/asrf/evaluation/visualization.py` |
| HeatmapEncoder | `src/seg_learning/models/heatmap_encoder.py`: `HeatmapEncoder` | Planned adaptation in round 2 | Input `[B,3,H,T]`; two height-only pools; output `[B,128,T]`; temporal width is never changed | `tests/test_heatmap_encoder.py` | `src/asrf/models/heatmap_encoder.py` |

The MSTCN configuration helpers resolve repository-relative paths from the
package location, while the ASRF implementation uses its own resolver. The
only absolute data path retained by ASRF is the intentional shared external
dataset path under `/media/.../seg_learning/data`.

