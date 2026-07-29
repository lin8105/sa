# Multi-task closed-set ontology

The multi-task experiment uses nine known classes. `pick` and `translation`
are retained as raw annotation aliases only after the round-5M audit found
consistent boundary and gripper-position evidence.

| ID | Canonical class | Definition |
|---:|---|---|
| 0 | reach | Empty gripper purposefully approaches the next target object. |
| 1 | grasp | Gripper closes and establishes a stable object constraint. |
| 2 | lift | A held object is raised from its support. |
| 3 | transport | The manipulator moves while holding an object. |
| 4 | pour | A container is tilted to pour. |
| 5 | pour_recover | The container returns from pouring orientation. |
| 6 | place | A held object is brought to and settled at its target. |
| 7 | wipe | A held tool or object performs the wiping interaction. |
| 8 | retreat | The empty gripper or manipulator withdraws from the completed interaction region. |

Aliases:

- `pick -> reach`: all 21 pick segments occur at the beginning of pour
  recordings, are followed by `grasp`, and have the open gripper-position
  reference.
- `translation -> transport`: all 29 translation segments occur after `lift`
  or `pour_recover`, before `pour` or `place`, and have a closed
  gripper-position signal.

The `robot_states.csv:is_grasped` field was present but zero throughout the
audited occurrences, so it was not used as the decisive held-object signal.
The position channel was used explicitly and is recorded in
`outputs/multitask_baseline/alias_review.csv`. No `release`, `unknown`, or
additional recover class is included.
