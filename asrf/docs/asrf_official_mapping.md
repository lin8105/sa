# Official ASRF mapping

Reference: `https://github.com/yiskw713/asrf`, branch `main`, inspected
commit `9623f1e8d9a1171333a4eeb65d190997b6c44a95` (MIT License).

| Official file | Official symbol | Paper section | ASRF destination | Adaptation | Test | Round |
|---|---|---|---|---|---|---|
| `libs/models/tcn.py` | `DilatedResidualLayer` | 3.1, 3.2 | `models/layers.py` | Preserve dilation `2**i`, 3-tap convolution, 1x1 projection, dropout, residual | Layer shape/residual test | 2 |
| `libs/models/tcn.py` | `SingleStageTCN` | 3.2, 3.3 | `models/layers.py` | Preserve per-stage input projection, 10 residual layers, output projection | Stage shape test | 2 |
| `libs/models/tcn.py` | `ActionSegmentRefinementFramework` | 3.1–3.3 | `models/feature_extractor.py`, `models/asb.py`, `models/brb.py`, `models/model.py` | Split shared extractor, ASB, and BRB into explicit modules; keep four outputs in training | Architecture/interface tests | 2 |
| `libs/loss_fn/__init__.py` | `ActionSegmentationLoss` | 3.5.1 | `losses/classification.py`, `losses/combined.py` | Implement per-stage averaging and configured CE/TMSE/GS-TMSE terms | Loss reference tests | 3 |
| `libs/loss_fn/tmse.py` | `TMSE`, `GaussianSimilarityTMSE` | 3.5.1 | `losses/smoothing.py` | Preserve truncation at threshold 4 and feature-similarity weighting | Numerical loss tests | 3 |
| `libs/loss_fn/__init__.py` | `BoundaryRegressionLoss` | 3.5.2 | `losses/boundary.py` | Preserve masked BCE-with-logits and positive weighting | BCE/mask tests | 3 |
| `utils/generate_boundary_array.py` | `main` boundary generation | 3.3, 3.5.2 | `data/annotations.py`, `losses/boundary.py` | Generate first-frame and label-transition targets in memory; never write external data | Boundary-target tests | 3 |
| `libs/metric.py` | `argrelmax` | 3.4, 4.2 | `refinement/peaks.py` | Preserve thresholding, strict interior maxima, and first-frame boundary | Peak-selection tests | 3 |
| `libs/postprocess.py` | `PostProcessor._refinement_with_boundary` | 3.4 | `refinement/segments.py`, `refinement/majority_vote.py` | Preserve valid-mask selection, terminal `T`, and majority-vote tie policy | Refinement tests | 3 |
| `libs/metric.py` | `ScoreMeter` | 4, evaluation | `evaluation/metrics.py` | Adapt frame accuracy, normalized edit, segmental F1, and confusion matrix | Metric parity tests | 5 |
| `libs/metric.py` | `BoundaryScoreMeter` | 4, boundary evaluation | `evaluation/boundary_metrics.py` | Adapt thresholded local maxima and tolerance-5 matching | Boundary metric tests | 5 |
| `save_pred.py` | `predict` | 4.2 | `scripts/predict.py` | Export raw/refined labels and boundary plots without writing external data | Export tests | 5 |

Round-3 destinations are `data/boundary_targets.py`,
`losses/classification.py`, `losses/smoothing.py`, `losses/boundary.py`,
`losses/combined.py`, and `refinement/{peaks,segments,majority_vote,refine}.py`.
They retain explicit valid masks and right-only padding as the CITR adaptation
of the official ignore-index workflow.

The official class is named `ActionSegmentRefinementFramework`; the new
package will use the unambiguous project name ASRF and namespace `asrf`.

## Paper versus official code

- The paper describes a long-term TCN with ten 64-channel dilated residual
  layers, doubled dilation, and dropout 0.5. The official
  `DilatedResidualLayer` uses `nn.Dropout()` (PyTorch's default `p=0.5`), so
  this matches.
- The paper describes an initial prediction plus three refinement stages for
  both branches. The official constructor receives `n_stages_* = 4` and
  creates `n_stages_* - 1` subsequent stages, returning four predictions in
  training mode but only the final prediction in evaluation mode.
- The paper uses I3D features with 2048 channels. The official loader reads
  precomputed `.npy` arrays shaped `[C,T]`; the shipped configs set
  `in_channel: 2048`. ASRF will replace that feature input with its declared
  RGB heatmap encoder while preserving the temporal axis.
- The paper defines a boundary at the first frame and at every action start.
  `utils/generate_boundary_array.py` implements exactly that binary target and
  writes it as an external-dataset artifact in the official workflow. ASRF
  will generate the target in memory so raw data is never modified.
- The paper describes a local maximum above threshold `theta_p`. The code
  names this parameter `boundary_th`, defaults the helper to `0.7`, uses
  `0.5` in the checked-in experiment configs, zeroes values below threshold in
  place, treats frame zero as a boundary, accepts strict interior maxima only,
  and does not select the last frame as a local maximum.
- The paper says to majority-vote ASB labels between selected boundaries. The
  code appends the valid sequence length `T` as the terminal boundary. On a
  tie, it chooses the mode with the largest sum of the original ASB output
  values when outputs are class scores; its 2-D oracle path chooses the first
  mode. ASRF will document and test this tie policy rather than silently
  inventing a different one.
- The paper uses class-weighted CE, TMSE/GS-TMSE alternatives, and BCE for
  BRB. The official default experiment config enables median-frequency class
  weighting, CE, GS-TMSE, and BCE; `tmse` and focal loss are disabled. It
  computes BRB positive weight as the reciprocal positive-boundary ratio.
- The paper reports `lambda_b=0.2` for GTEA and `0.1` for 50 Salads and
  Breakfast. The checked-in official configs use the dataclass default `0.1`;
  ASRF will make this dataset-specific choice explicit in its own config.
- The official README calls the evaluator `eval.py` in one directory-listing
  sentence, while the repository entry point is `evaluate.py`; ASRF follows
  the executable file rather than the README typo.

All MSTCN source paths in the reuse audit are relative to the absolute root
`/home/yue/Documents/zsc_Franka/mstcn`; all ASRF destinations are relative to
`/home/yue/Documents/zsc_Franka/asrf`.
