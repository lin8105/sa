# Full dataset audit — ASRF round 3.5

Audit date: 2026-07-22. The external dataset was read-only. The audit script
is [audit_full_dataset.py](../scripts/audit_full_dataset.py); generated tables
are under `outputs/round3_5_data_audit/`.

## Inventory

The task directories are `pour`, `pick and place`, and `wipe` under train, and
`pour` and `pp` under test. There are 80 train trajectory directories and 8
test trajectory directories. All current recordings are direct children of a
task directory; no nested run directories were observed. Empty directories
were retained as incomplete candidates rather than ignored.

A valid recording requires `citr_fingerprint_pure.png`, `segments.csv`, and
readable timestamp information. Every valid heatmap is RGB with height 88 and
its width equals the `citr_features.csv` timestamp count. Alternative files
are detected in the inventory but never silently substituted.

| root/task | directories | valid | incomplete |
|---|---:|---:|---:|
| train/pour | 25 | 16 | 9 |
| train/pick and place | 30 | 25 | 5 |
| train/wipe | 25 | 9 | 16 |
| test/pour | 5 | 5 | 0 |
| test/pp | 3 | 3 | 0 |

There are 58 valid recordings: 50 train and 8 test. The 30 incomplete train
directories are empty: `p17`–`p25`, `pp26`–`pp30`, and `w10`–`w25`.

## Current split coverage

The current split files reference 15 physical paths: train p1–p8 (8),
validation p9–p10 (2), test p1–p2 (2), and optional test p3–p5 (3).
Therefore p1–p8 alone expands to eight train recordings. The 15-recording
count used in the earlier round-3 audit is the union of p1–p10 in train/val
and p1–p5 in test. It is not an expansion of p1–p8 alone. There are 42 valid
recordings not referenced by current split files.

## Labels and annotation quality

Unique raw labels are `grasp`, `lift`, `pick`, `place`, `pour`,
`pour_recover`, `reach`, `retreat`, `translation`, `transport`, and `wipe`.
Known project aliases are retained only as audit metadata: `pick -> reach` and
`translation -> transport`. No spelling or capitalization variants were
observed. No zero/negative durations, overlaps, temporal gaps, or heatmap /
timestamp width mismatches were found in the 58 valid recordings.

`segments.csv.bak` is present for some recordings, but primary
`segments.csv` was used. Every valid recording also has alternative
`citr_fingerprint.png` and `video_timestamps.csv`; these were not substituted.

The important conflicts are ontology questions rather than malformed files:
`pick` occurs only in pour while `reach` occurs in the other tasks;
`translation` occurs only in pour while `transport` occurs in the other tasks;
and `place`, `retreat`, `pour_recover`, and `wipe` have task-specific usage.
These require manual review before merging.

## Sequence and transition diversity

Pick-and-place is fixed across all 25 train recordings:

```text
reach -> grasp -> lift -> transport -> place
```

The three pp test recordings use the same sequence. Pour has four observed
patterns, including direct `lift -> pour`, `lift -> translation -> pour`, and
optional `pour_recover -> translation -> place`. Wipe has repeated wipe cycles
and optional retreat, including `place -> wipe`, `wipe -> lift`, and
`place -> retreat`.

Observed requested transitions include `lift -> transport -> place` in
pick-and-place, pp, and wipe; `lift -> pour` in 7 pour recordings;
`lift -> translation` in 12 pour recordings; `pour_recover -> translation`
in 12 pour recordings; `pour_recover -> place` in 6 pour recordings; and
repeated `lift -> transport -> place` after wipe in 8 wipe recordings. No
unobserved transition is proposed as valid.

## Pick-and-place compatibility

Exact directories are `train/pick and place/pp1`–`pp25`, with empty `pp26`–
`pp30`, and `test/pp/pp_c1`–`pp_c3`. The 28 valid recordings use only
`reach`, `grasp`, `lift`, `transport`, and `place`. Heatmaps are `[3,88,T]`,
with train T 2416–3120 and test T 2470–2699.

The current seven-class pour configuration can represent pick-and-place only
after confirming that `pick/reach` and `translation/transport` are
interchangeable. No new class IDs are technically required if those aliases
are approved, but automatic inclusion is not yet safe. Pick-and-place adds
repeated `transport -> place` examples and useful task variation, while its
fixed sequence cannot by itself validate alias boundary semantics. The three
pp test recordings are suitable for cross-task generalization after family
leakage review.

## Boundary statistics

Across all 58 valid recordings and 196,256 frames: 315 internal boundaries,
58 frame-0 boundaries, and 373 positives including frame 0. The positive
ratio is `0.001900579` and reciprocal weight `526.155496`; excluding frame 0,
the ratio is `0.001605046` and reciprocal weight `623.034921`.

For training-only valid recordings (50, 170,126 frames), the reciprocal weight
including frame 0 is `528.341615`. The round-3 pour training split used 32,035
frames and weight `508.492063`; all compatible training data changes the
estimate modestly, so the loss implementation remains unchanged.

Task and transition values are in `boundary_statistics.csv`.

## Duplicate and figure audit

No exact segment or heatmap hash duplicate was found. `pp7` and `pp10` share
the same temporal width and collapsed sequence, but their segment and heatmap
hashes differ; this is a structural match, not evidence of duplication.

Five representative figures were visually inspected: pour train/test,
pick-and-place train, wipe train, and pp test. They show the expected RGB
heatmap structure and annotation boundaries, with no obvious offset, missing
tail, shape anomaly, or duplicated recording. The x-axis is the original
heatmap-column coordinate; only display aspect scaling is used.

## Recommendation

Keep the pour-only baseline for fair MSTCN comparison, manually review the two
alias pairs, then create a separately versioned multi-task experiment using
draft group splits. Initially hold out task-specific `pour`, `pour_recover`,
`wipe`, and `retreat` for open-set research.

## Snapshot stability

The external tree was not stable during this audit: `train/wipe/w9` was
observed empty in an earlier read and complete in the final read. No ASRF
command writes external data; another process was apparently populating the
shared tree. The CSV/JSON outputs reflect the final 58-recording snapshot.
Re-run the audit immediately before training and require two identical
read-only inventories.
