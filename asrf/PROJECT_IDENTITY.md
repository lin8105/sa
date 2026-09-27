# ASRF project identity

- Project root: this repository's `asrf/` directory
- Project/model name: ASRF — Action Segment Refinement Framework
- Python package: `asrf`
- Purpose: independent implementation and adaptation of ASRF for CITR-based
  robotic skill segmentation.
- Reference implementation: official `yiskw713/asrf` repository.
- Existing comparison baseline: a separate MSTCN checkout (optional; not a runtime dependency)
- Shared data root: external project data, not included in this repository
- Python interpreter: use the configured project environment; machine-specific paths are omitted

MSTCN is read-only from ASRF's perspective. ASRF must read the existing data
in place; raw data and annotations must never be modified by this project.

This project is intentionally separate from MSTCN and does not import runtime
code from the sibling checkout.
