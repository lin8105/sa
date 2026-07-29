# Multi-task label ontology proposals

These are proposals only. `configs/labels_pour.yaml` was not changed. Counts
use all 58 currently valid recordings unless stated otherwise.

## Proposal 1 — strict pour-only baseline

| ID | class | aliases |
|---:|---|---|
| 0 | reach | pick |
| 1 | grasp | |
| 2 | lift | |
| 3 | transport | translation |
| 4 | pour | |
| 5 | pour_recover | |
| 6 | place | |

This includes 21 valid pour recordings: 16 train and 5 test. The existing
comparison protocol uses 8 train, 2 validation, and 5 test recordings. Pour
frame/segment totals are reach 13,721/21; grasp 8,294/21; lift 6,471/21;
transport 9,537/29; pour 15,752/21; pour_recover 9,241/21; place 16,205/21.

Advantage: exact baseline comparability. Risk: task-order shortcut and low
transition diversity. The aliases require confirmation outside the existing
pour baseline.

## Proposal 2 — conservative multi-task ontology

After manual confirmation, merge only the two existing project aliases and
keep task-specific skills separate:

```text
reach, grasp, lift, transport, place, pour, pour_recover, wipe, retreat
```

Aliases are `pick -> reach` and `translation -> transport`. It includes all
58 valid recordings. Frame/segment totals are reach 37,197/58; grasp
22,411/58; lift 23,679/64; transport 24,323/72; place 52,269/64; pour
15,752/21; pour_recover 9,241/21; wipe 8,134/9; retreat 3,250/6.

Advantage: shared physical skills plus task-specific actions. Risk: alias
merges may hide boundary-definition differences. Manual review must compare
the physical start/end criteria for pick vs reach and translation vs transport.

## Proposal 3 — expanded raw multi-task ontology

Keep every observed raw label distinct:

```text
grasp, lift, pick, place, pour, pour_recover, reach, retreat,
translation, transport, wipe
```

Counts are grasp 22,411/58; lift 23,679/64; pick 13,721/21; place
52,269/64; pour 15,752/21; pour_recover 9,241/21; reach 23,476/37;
retreat 3,250/6; translation 9,537/29; transport 14,786/43; wipe 8,134/9.

Advantage: no automatic semantic collapse and best support for unknown-skill
research. Risk: more classes and severe task-specific imbalance. `release`,
`plug`, `unplug`, `push`, `recover`, and `return` were not observed and must
not be added as names alone.

## Decision

Proposal 1 is immediately reproducible. Proposal 2 is the recommended
multi-task candidate after alias review. Proposal 3 is safest for open-set
research if its class imbalance is accepted.
