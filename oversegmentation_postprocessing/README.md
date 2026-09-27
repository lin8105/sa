# Over-Segmentation Post-Processing

This module reduces over-segmentation in robot skill segmentation through two post-processing steps: **physics-guided timeline compression** and **segment merging**.

The method works directly on frozen ASRF predictions and does not require retraining or modifying the segmentation model.

## Motivation

Several boundary-refinement strategies were evaluated, including ASB/BRB-assisted merging, boundary protection, hard-negative training, and short-fragment consolidation. While these methods could reduce false boundaries, they could also remove valid transitions.

The current pipeline therefore complements model-based boundary predictions with the robot's physical behavior, using motion and interaction signals to guide post-processing more conservatively.

## What It Does

ASRF predictions can contain long redundant periods and short fragmented segments, which introduce unnecessary boundaries into the final skill sequence.

This module addresses the problem in two stages:

1. **Physics-Guided Timeline Compression** shortens redundant low-motion parts of the trajectory while keeping important transitions and physical interaction.
2. **Segment Merging** removes unnecessary boundaries between neighboring predicted segments when their local motion patterns are sufficiently similar.

## Method

### 1. Physics-Guided Timeline Compression

The first stage looks for sustained low-motion regions that contain little useful temporal information.

Instead of relying on motion alone, the compression rule considers several synchronized physical signals, including:

* linear and angular motion;
* linear and angular acceleration;
* force and torque;
* changes in force and torque;
* gripper motion.

These signals help distinguish truly redundant low-motion periods from situations where the robot is moving very little but is still physically interacting with the environment.

For example, a robot may remain almost stationary while pressing against an object or maintaining contact during insertion. Such periods should not be treated in the same way as an idle pause.

The compression therefore follows a simple idea:

**redundant low-motion period → compress**
**motion or transition → keep**
**meaningful physical interaction → keep**
**gripper manipulation event → keep**

Only sustained regions are considered for compression. Context around the beginning and end of each region is preserved so that important transitions are not removed.

The method also keeps context around gripper opening and closing events and ensures that each skill retains a minimum temporal extent.

The resulting retained-frame mapping is applied consistently to all synchronized trajectory data, including:

* CITR features;
* ASRF predictions;
* boundary predictions;
* gripper signals;
* ground-truth labels when used for evaluation.

No temporal interpolation, stretching, or uniform downsampling is performed. The retained frames remain in their original order.

All physical thresholds are determined from the TRAIN trajectories and frozen before TEST evaluation. No TEST data are used to adjust the compression rule.

### 2. Segment Merging

After timeline compression, some short redundant predicted segments may still remain.

The second stage examines these candidate segments together with their neighboring segments and compares their local motion characteristics.

The comparison uses physical and trajectory information such as:

* CITR features;
* force and torque information;
* gripper state;
* linear and angular motion;
* velocity and acceleration.

A local window of **5 frames** is used around the candidate region. If the candidate is sufficiently similar to one neighboring segment, it is merged into that side and the unnecessary boundary is removed.

The final configuration uses:

* similarity fraction: **0.80**
* local window: **5 frames**
* **one-pass** merging

The merge is deliberately conservative. It does not simply combine every short segment.

Segments containing a physical gripper opening or closing event are protected from deletion. This helps preserve manipulation-critical regions such as grasping or releasing even when nearby segments appear similar.

## Results

The physical compression thresholds were selected using the **79 TRAIN trajectories** and frozen before TEST evaluation.

Segmentation metrics are reported on **34 TEST trajectories** with available frozen ASRF bundles. Two plug trajectories are excluded because their frozen prediction bundles are unavailable.

| Metric                   | Original ASRF | Compression | Compression + Merge |
| ------------------------ | ------------: | ----------: | ------------------: |
| Mean F1@50               |         0.840 |   **0.899** |           **0.928** |
| Boundary Recall ±20      |         0.372 |   **0.706** |               0.676 |
| Missed-Boundary Rate ±20 |         0.628 |   **0.294** |               0.324 |

Physics-guided compression increases mean F1@50 from **0.840 to 0.899** and substantially improves boundary recall.

The subsequent merge further increases F1@50 to **0.928**. Boundary recall decreases slightly compared with compression alone, reflecting the expected trade-off from removing additional segment boundaries.

The improvement is also observed across both the seen PP family and unseen skill families:

| Family  | Original ASRF | Compression | Compression + Merge |
| ------- | ------------: | ----------: | ------------------: |
| PP      |         0.918 |   **0.954** |           **0.974** |
| plug    |         0.764 |   **0.856** |           **0.909** |
| wipe    |         0.752 |   **0.800** |           **0.829** |
| pour    |         0.887 |   **0.919** |           **0.945** |
| unscrew |         0.719 |   **0.930** |           **0.963** |

PP is the seen family used during model development, while plug, wipe, pour, and unscrew represent unseen-family transfer.

Overall, **Physics-Guided Timeline Compression** removes redundant temporal content while preserving important physical interaction and transition regions. The following similarity-based merge further reduces segmentation fragmentation while protecting manipulation-critical events.

The complete pipeline operates entirely as post-processing and requires no ASRF retraining.
