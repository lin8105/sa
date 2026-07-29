# Previous MSTCN local-probe audit

The previous experiment is under the read-only MSTCN checkout at
`/home/yue/Documents/zsc_Franka/mstcn/outputs/pour_local_probe`.

## What it classified

`src/seg_learning/models/local_probe.py` defines `FrozenEncoderLocalProbe`:

- the frozen MSTCN HeatmapEncoder produces `[B, 128, T]` local features;
- a trainable pointwise `Conv1d(kernel_size=1)` classifies every frame;
- the encoder is frozen and kept in evaluation mode;
- it uses the seven-class pour ontology only;
- the training and evaluation split is pour-only.

Thus the experiment classified frames, not independent complete ground-truth
skill segments. Its saved `segment_diagnostic_standard.csv` and
`segment_diagnostic_optional.csv` summarize frame predictions inside each
annotated segment, but they do not train or evaluate one classifier sample per
segment. They can show majority/mean-probability behavior, not a clean
context-free segment recognition rate.

## Meaning of 0.485 and 0.473

The saved standard and optional metrics report overall frame accuracy:

- standard: `0.4850132441098564`;
- optional: `0.47306791569086654`.

They are aggregate frame-weighted values across the evaluated pour
trajectories. They are not per-skill segment recognition rates. The saved
`per_class_metrics.csv` contains frame-level precision, recall, and F1, while
the segment diagnostic contains one row per ground-truth segment with
majority/mean-probability labels, but no independently trained segment sample
prediction table.

The standard run used the seven classes `reach`, `grasp`, `lift`, `transport`,
`pour`, `pour_recover`, and `place`. The optional run has the same class set
and differs in its evaluation split/configuration. Neither result is used as
the answer to the current nine-class segment question.

The current ASRF probes instead crop each ground-truth segment before feature
extraction and report support-normalized per-skill recall, precision, F1, and
confusion matrices.
