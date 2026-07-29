# Multi-task split protocol

The primary split is deterministic and trajectory-level with seed 42. No
reliable operator, object, session, or recording-family metadata was present,
so those fields are explicitly unavailable in the manifest. Exact heatmap and
annotation hashes were checked across train and validation.

Training uses 40 recordings:

- pour: p1–p12;
- pick and place: pp1–pp20, pp26–pp27;
- wipe: w1–w6.

Validation uses 13 recordings:

- pour: p13–p16;
- pick and place: pp21–pp25, pp28;
- wipe: w7–w9.

This gives every one of the nine classes positive training coverage and
validation coverage. Test recordings are never included in either split.
