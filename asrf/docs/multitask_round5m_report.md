# ASRF multi-task round 5M

This experiment is a separate nine-class closed-set model. It was initialized
from random weights and did not load `outputs/pour_baseline/best.pt`.

The stable dataset inventory contains 53 valid training recordings and 11
valid test recordings. The deterministic train/validation split contains 40
and 13 recordings respectively:

- train: `train/pour/p1..p12`, `train/pick and place/pp1..pp20`,
  `pp26..pp27`, and `train/wipe/w1..w6`;
- validation: `train/pour/p13..p16`, `train/pick and place/pp21..pp25`,
  `pp28`, and `train/wipe/w7..w9`.

The nine classes are `reach`, `grasp`, `lift`, `transport`, `pour`,
`pour_recover`, `place`, `wipe`, and `retreat`. Only the audited aliases
`pick -> reach` and `translation -> transport` are used.

The model has 1,239,672 parameters. Training used the full configured
architecture, CPU, batch size 1, seed 42, Adam at 0.0005, and the official
reciprocal BRB positive weight computed from the multi-task training split.
The run stopped at epoch 53; validation-total-loss selection chose epoch 38.

Validation-only BRB calibration selected threshold 0.90 for macro internal
boundary F1 at ±33 frames. Threshold 0.50 remains the official baseline and is
reported separately. Test evaluation was performed only after this threshold
was frozen.

Machine-readable inventories, split statistics, calibration, checkpoint
manifests, per-task summaries, per-class metrics, prediction CSVs, and figures
are under `outputs/multitask_baseline/`. The frozen seven-class pour baseline
and the MSTCN checkpoint were hash-verified before and after this round.

This is not an unknown-skill experiment: wipe and retreat are known classes in
the training ontology, and no unknown detector was added.
