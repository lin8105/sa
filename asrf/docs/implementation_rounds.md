# Implementation rounds

1. **Round 1:** project creation, audits, data interfaces.
2. **Round 2:** HeatmapEncoder, long-term extractor, ASB, BRB.
3. **Round 3:** boundary targets, losses, peak selection, majority-vote refinement — complete; no training or evaluation performed.
4. **Round 4:** strict pour-only training and validation on p1–p10 — complete;
   primary selection used validation total loss only, with no test access.
5. **Round 5:** standard and optional-transition evaluation, comparison with MSTCN.
