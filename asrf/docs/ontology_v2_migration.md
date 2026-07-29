# Round 12 ontology migration

The active model-facing ontology is `round12_multiskill_v2`:

`reach=0, grasp=1, lift=2, transport=3, pour=4, pour_recover=5, place=6, release=7, wipe=8, retreat=9, insert=10`.

Only `pull_out -> lift` and `extract -> lift` are runtime aliases. The legacy
`align` label is not mapped to `place` at runtime. Dataset annotations must be
manually edited before a Round 12 audit or training run can pass.

Migration note: old `align` ID 10 was removed; old `insert` ID 11 becomes new
`insert` ID 10. Old checkpoints and prototype banks fail with an ontology
version mismatch, and classifier rows are never automatically reinterpreted.
New checkpoints and prototype banks must carry `ontology_metadata` containing
the version, label IDs, aliases, and class count.
