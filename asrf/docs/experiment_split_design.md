# Experiment split design

No final multi-task split files are created. The dataset exposes task and
trajectory IDs but no reliable operator, object, recording-session, or batch
metadata. These proposals use trajectory IDs provisionally and require review
before training.

## Experiment A — pour closed-set baseline

Use Proposal 1 only:

- train: `train/pour/p1`–`p8`;
- validation: `train/pour/p9`–`p10`;
- test: `test/pour/p1`–`p5`.

This is an 8/2/5 recording split with seven known classes and the existing
MSTCN comparison policy. No frame-level or segment-level random splitting is
allowed.

## Experiment B — multi-task closed-set ASRF (draft)

Use Proposal 2 after alias review. A provisional stratified draft is:

- train: pour p1–p12, pick-and-place pp1–pp20, wipe w1–w4;
- validation: pour p13–p16, pick-and-place pp21–pp25, wipe w5–w9;
- test: test pour p1–p5 and test pp_c1–pp_c3.

This gives a provisional 36/14/8 recording split. The test set is external to
train and is not used for model selection or statistics. The grouping key is
task plus trajectory ID; the assumption that numeric IDs are independent
recording families is unverified. Replace this draft if session/operator/
object metadata reveals family leakage.

All proposal-2 classes are represented in the provisional training groups,
including pour/pour_recover, wipe, and retreat. Validation coverage must still
be checked after manual annotation review.

## Experiment C — open-set skill discovery (draft)

- known training: pour plus pick-and-place;
- held-out task candidate: wipe, using wipe only for test;
- held-out skill candidates: `wipe`, `retreat`, `pour`, or `pour_recover`;
- alternate task holdout: train pick-and-place plus wipe and test pour-specific
  skills.

There is no independent test/wipe directory, so wipe holdout is currently an
internal draft protocol, not an unbiased final test. Do not call a skill
unknown if an alias or equivalent raw label occurred in training.

## Leakage rules

Never split frames or segments from one recording across partitions. Keep all
recordings from one family together once family metadata is found. Do not use
test heatmaps, annotations, class counts, boundary ratios, or transitions for
training decisions. Recheck the possible-duplicate table before finalizing
groups.
