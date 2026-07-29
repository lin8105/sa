# Open-set evaluation protocol

This is a future evaluation design only; no unknown-detection code is added.

## Protocols

1. **Leave-one-skill-out:** remove one canonical skill from training and keep
   it only in test. If aliases are merged, hold out the complete alias group.
2. **Held-out task:** train on two tasks and test task-specific skills from a
   third. The current dataset lacks an independent wipe test root, so this is
   presently a draft internal protocol.
3. **Known/unknown transition:** include known skills around unknown segments
   to test boundary behavior, not only isolated unknown clips.

## Metrics

Report AUROC, AUPR, FPR@95TPR, known-class accuracy, unknown-frame F1,
unknown-segment F1, and open-set macro F1. When unknown ground truth exists,
also report ARI, NMI, and cluster purity. Retain both micro/macro and
per-task results.

## Recommended first holdouts

The best initial candidates are `pour`, `pour_recover`, `wipe`, and `retreat`:
they are task-specific and raw-label unambiguous. `wipe` appears in nine
train recordings, `retreat` in six, and `pour`/`pour_recover` in all 21 valid
pour recordings. `pick`/`reach` and `translation`/`transport` are poor unknown
candidates until alias status is resolved because an equivalent label may
remain in training.

Unknown labels must be defined from annotations before evaluation. A skill is
not unknown merely because a model predicts it poorly.
