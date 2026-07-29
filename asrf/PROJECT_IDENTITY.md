# ASRF project identity

- Project root: `/home/yue/Documents/zsc_Franka/asrf`
- Project/model name: ASRF — Action Segment Refinement Framework
- Python package: `asrf`
- Purpose: independent implementation and adaptation of ASRF for CITR-based
  robotic skill segmentation.
- Reference implementation: official `yiskw713/asrf` repository.
- Existing comparison baseline:
  `/home/yue/Documents/zsc_Franka/mstcn`
- Shared data root:
  `/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data`
- Shared Python interpreter:
  `/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python`

MSTCN is read-only from ASRF's perspective. ASRF must read the existing data
in place; raw data and annotations must never be modified by this project.

This project is intentionally separate from MSTCN and does not import runtime
code from the sibling checkout.
