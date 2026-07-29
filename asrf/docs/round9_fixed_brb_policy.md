# ASRF Round 9 fixed BRB policy

All Round 9 experiments use the Round 8 hard-window target with radius 5 frames:

```yaml
boundary_target_mode: hard_window
boundary_window_radius: 5
boundary_include_frame_zero: true
boundary_include_final_frame: false
```

At the dataset's 100 Hz sampling rate this is a clipped ±0.05 second target window. The BRB and
ASB architectures, optimizer family, peak extraction, local-maximum rule, official threshold
`0.50`, majority-vote refinement, boundary loss coefficient, and post-processing are unchanged.

Round 9 does not use single-frame, radius-10, radius-20, or Gaussian targets. It does not use NMS,
minimum peak distance, or prominence filtering. The repository-local default is
`configs/round9/default_hard_window_r5.yaml`; the Round 8 configuration is preserved unchanged.

The plug primary initialization is the verified Round 8 `hard_window_r5` checkpoint. Its ten
existing ASB rows were copied by canonical index into the historical twelve-class head; its former Plug phase and
`insert` rows are newly initialized.
