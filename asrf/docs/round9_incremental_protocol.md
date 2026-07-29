# ASRF Round 9 incremental target-family protocol

Round 9 uses a fixed `base_pp10` dataset (`train/pick and place/pp1` through `pp10`) and adds
nested target-family trajectories. Every model starts independently from the verified Round 8
This is a historical Round 9 protocol. Its twelve-class checkpoint is legacy and is not compatible with `round12_multiskill_v2`.
rows. The BRB target is fixed to `hard_window`, radius `5`, and the official inference threshold is
`0.50`.

Primary training sets are `base_pp10 + pour-{3,5,all}`, `base_pp10 + wipe-{3,5,all}`, and
`base_pp10 + plug-{3,5,all}`. A fixed common validation set is drawn from `pp11`–`pp20`; primary
test sets are pour `p1,p2`, wipe `w1,w2`, and every valid independent test/plug trajectory.

The nine primary runs are staged and trained in the specified order. Larger models are never
initialized from smaller-subset checkpoints, and test data is not used for validation, selection,
or subset construction.
