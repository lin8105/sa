# ASRF

ASRF means **Action Segment Refinement Framework**. This is an independent
implementation and adaptation for CITR-based robotic skill segmentation. It is
not the MSTCN repository and does not import runtime code from it.

The reference implementation is the official
[yiskw713/asrf repository](https://github.com/yiskw713/asrf), which implements
the WACV 2021 paper *Alleviating Over-segmentation Errors by Detecting Action
Boundaries* by Yuchi Ishikawa et al. The existing comparison baseline is
`/home/yue/Documents/zsc_Franka/mstcn`.

## Environment and data

Use the approved interpreter:

```bash
cd /home/yue/Documents/zsc_Franka/asrf
/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python scripts/check_environment.py
```

The shared dataset is read in place:

```text
/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data
```

No raw trajectory data is stored in this repository. The copied pour split
files and label configuration are small independent metadata files. The
package namespace is `asrf`:

```bash
PYTHONPATH=src /media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python -c "import asrf; print(asrf.__version__)"
```

## Status

Round 4 adds the deterministic pour-only trainer, validation-only model
selection, checkpoint/resume support, stage-aware diagnostics, and p9/p10
validation exports. The trained artifacts are under
`outputs/pour_baseline/`. No p1–p5 test trajectory was used.

Round 3 added in-memory boundary targets, training-only statistics, masked
CE/TMSE/GS-TMSE/BRB losses, the combined multi-stage objective, official
boundary peaks, half-open segment construction, majority-vote refinement, and
diagnostic mean-probability refinement.

See [the interface specification](docs/asrf_interface_spec.md),
[the official mapping](docs/asrf_official_mapping.md), and
[the round plan](docs/implementation_rounds.md).
