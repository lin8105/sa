# Current ASB/BRB/refinement pipeline

This audit describes the implementation used by the frozen multi-task ASRF
baseline and retained for Round 7A. No post-processing behavior is changed in
this round.

1. `src/asrf/models/heatmap_encoder.py:HeatmapEncoder.forward` consumes RGB
   heatmaps `[B,3,88,T]`, applies height-only pooling, and returns
   `[B,128,T]`. Temporal width is preserved.
2. `src/asrf/models/feature_extractor.py:LongTermFeatureExtractor.forward`
   applies the 1x1 projection and the shared ten-layer non-causal dilated
   residual stack, returning `[B,64,T]`.
3. `src/asrf/models/model.py:ASRFModel.forward` sends the shared sequence to
   separate `ActionSegmentationBranch` and `BoundaryRegressionBranch`
   instances.
4. `src/asrf/models/asb.py:ActionSegmentationBranch.forward` produces class
   logits and softmax probabilities at every stage; the final stage is the
   per-frame ASB class-probability sequence.
5. `src/asrf/models/brb.py:BoundaryRegressionBranch.forward` produces one
   sigmoid boundary probability per frame at every stage; the final stage is
   the BRB boundary-probability sequence.
6. `src/asrf/refinement/peaks.py:select_boundary_peaks` first applies the
   threshold, preserves frame 0 as the sequence-start boundary, and selects
   strict interior local maxima. The final valid frame is not an interior
   candidate.
7. `src/asrf/refinement/segments.py:construct_segments` sorts unique selected
   boundary indices, forces start 0, and partitions the valid sequence into
   half-open `[start,end)` intervals.
8. `src/asrf/refinement/refine.py:refine_asrf_predictions` calls peak
   selection, then applies `_vote_one` to each predicted interval. The exact
   official aggregation is `voting="majority"` in
   `src/asrf/refinement/majority_vote.py:_vote_one`; ties are resolved by the
   summed ASB probabilities and then the lower class id.
9. Raw ASB does not use BRB: it is `final_asb.argmax(dim=1)` in
   `src/asrf/refinement/refine.py:refine_asrf_predictions` and does not use
   BRB.
10. Oracle refinement is diagnostic-only in
    `scripts/evaluate_multitask.py:_oracle_refinement`: it constructs
    boundaries from ground-truth labels and applies the same majority vote.

The training path is `scripts/train_multitask_asrf.py:main` ->
`asrf.training.trainer:ASRFTrainer`. The primary checkpoint is selected only
by validation `total_loss` in `ASRFTrainer.train`. Round 7A changes only the
configured BRB positive weight consumed by `ASRFLoss`; it does not change
target width, branch architecture, peak extraction, voting, splits, or the
ASB loss terms.
