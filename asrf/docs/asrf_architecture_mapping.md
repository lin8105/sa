# ASRF architecture mapping

Reference repository: `https://github.com/yiskw713/asrf`  
Reference commit: `9623f1e8d9a1171333a4eeb65d190997b6c44a95`  
License: MIT

| ASRF file | Local class/function | Official file and symbol | Paper section | Preserved behavior | CITR adaptation and mask behavior | Tests |
|---|---|---|---|---|---|---|
| `src/asrf/models/heatmap_encoder.py` | `HeatmapEncoder` | No official counterpart | Project input adapter | N/A; official ASRF consumes I3D features | Independent MSTCN-derived RGB encoder, `[B,3,88,T] -> [B,128,T]`; height-only pooling and temporal-width assertions | `test_heatmap_encoder.py`, architecture tests |
| `src/asrf/models/layers.py` | `DilatedResidualLayer` | `libs/models/tcn.py: DilatedResidualLayer` | 3.1, 3.2 | k=3 symmetric Conv1d, ReLU, 1x1 Conv1d, dropout 0.5, residual add | Optional `[B,T]` mask is applied before and after the block; invalid outputs are zero | `test_temporal_layers.py` |
| `src/asrf/models/layers.py` | `RefinementStage` | `libs/models/tcn.py: SingleStageTCN` | 3.2, 3.3 | 1x1 input projection, ten residual layers, 1x1 output projection | Input is the previous branch probability tensor; mask is propagated | `test_temporal_layers.py`, architecture tests |
| `src/asrf/models/feature_extractor.py` | `LongTermFeatureExtractor` | `libs/models/tcn.py: ActionSegmentRefinementFramework` shared layers | 3.1 | 64-channel projection and ten dilated residual layers | Input is HeatmapEncoder output rather than I3D; no pooling; mask preserved | architecture tests |
| `src/asrf/models/asb.py` | `ActionSegmentationBranch` | `ActionSegmentRefinementFramework` ASB construction | 3.2 | Initial class logits plus three probability-fed refinement stages; softmax probabilities | Seven canonical pour classes; padded probabilities are zeroed | architecture tests |
| `src/asrf/models/brb.py` | `BoundaryRegressionBranch` | `ActionSegmentRefinementFramework` BRB construction | 3.3 | Initial boundary logits plus three sigmoid probability-fed refinement stages | One class-agnostic boundary channel; padded probabilities are zeroed | architecture tests |
| `src/asrf/models/model.py` | `ASRFModel`, `ASRFOutput` | `ActionSegmentRefinementFramework` | 3.1–3.3 | Shared features feed both branches; all four stage records retained | No losses or majority voting in `forward`; valid mask returned | `test_asrf_architecture.py`, round-trip tests |

No MSTCN temporal model, official checkpoint, or MSTCN weights were copied.
The HeatmapEncoder is project-specific; official ASRF's precomputed I3D input
is replaced by the existing CITR RGB heatmap interface. The official shared
extractor/ASB/BRB design is ported without changing its probability-fed stage
representation.

The official module returns only the final branch tensor in evaluation mode,
whereas the round-2 `ASRFOutput` intentionally retains all four stage tensors
in both modes for architecture diagnostics and future per-stage losses. The
official final inference representation is exactly the last element of each
stage list (`stage[-1]`); no refinement or postprocessing is performed here.
