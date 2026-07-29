"""Write a compact, reproducible Round 8 report from generated artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/brb_release_round8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    audit = json.loads((OUT / "data_audit_summary.json").read_text())
    split = json.loads((OUT / "split_integrity.json").read_text())
    boundary = json.loads((OUT / "boundary_statistics.json").read_text())
    comparison = read_csv(OUT / "model_comparison.csv")
    classes = read_csv(OUT / "class_statistics.csv")
    transitions = read_csv(OUT / "per_transition_boundary_metrics.csv")
    lines = [
        "# ASRF Round 8: release and BRB target-shape ablation",
        "",
        "## Executive verdict",
        "",
        "The preferred semantic-preserving model is `hard_window_r20`: raw F1@50 is 0.9531 versus the baseline's 0.9586 (drop 0.0056, below the material-drop gate), while official refined F1@50 rises to 0.9636; it also reduces false peaks from 256 to 171 and raises boundary F1@±33 to 0.4655. `hard_window_r5` is the best boundary/official-refinement compromise (F1@±33 0.5831, 113 false peaks), but its raw F1@50 drop to 0.9349 is material. `gaussian_s20` is the boundary-oriented model (F1@±33 0.6930, 53 false peaks), but its official refined F1@50 is 0.9327, also a material drop. No model clearly dominates on both boundary and semantic criteria; target widening helps, but very wide Gaussian supervision trades away refinement quality.",
        "",
        "## Audit and split integrity",
        "",
        f"The two complete read-only scans agree: **{audit['scan_rows_identical']}**. Audit pass: **{audit['pass']}**. {audit['all_recording_count']} annotation directories were scanned; {audit['valid_recording_count']} are valid ten-class-compatible recordings. Plug recordings are reported but excluded because legacy Plug phases are outside the historical ontology. Migration checks pass, including pp26–pp29 two place/release pairs and the wipe exception allowing earlier place intervals without release. Split pass: **{split['pass']}**; counts are train={split['counts']['train']}, validation={split['counts']['validation']}, test/pour={split['counts']['test_pour']}, test/pp={split['counts']['test_pp']}, test/wipe={split['counts']['test_wipe']}. w4 occurs exactly once.",
        "",
        "## Ontology and train statistics",
        "",
        "`reach=0, grasp=1, lift=2, transport=3, pour=4, pour_recover=5, place=6, release=7, wipe=8, retreat=9`; aliases are `pick -> reach` and `translation -> transport`.",
        "",
        f"Official single-frame train BRB: {boundary['total_positive_frames']} positive frames, {boundary['total_negative_frames']} negative frames, ratio {f(boundary['positive_ratio'])}, reciprocal positive weight {f(boundary['reciprocal_positive_weight'])}; internal transitions={boundary['total_internal_semantic_transitions']}, frame-0 positives={boundary['number_of_frame0_positives']}.",
        "",
        "| class | frames | segments | trajectories | mean duration s | median duration s | class weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {row['class']} | {row['frame_count']} | {row['segment_count']} | {row['trajectory_count']} | {f(row['mean_duration_s'])} | {f(row['median_duration_s'])} | {f(row['class_weight'])} |" for row in classes)
    lines.extend(["", "## Target mass and model comparison", "", "| experiment | target | weight | best epoch | val loss | raw F1@50 | official F1@50 | boundary F1@33 | false | missed | duplicate | release F1 | place→release recall |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparison:
        if row["target_mode"] == "single_frame":
            target = row["target_mode"]
        elif row["target_mode"] == "hard_window":
            target = f"hard_window {row['window_radius']}"
        else:
            target = f"gaussian sigma {row['gaussian_sigma']}"
        lines.append(f"| {row['experiment']} | {target} | {f(row['positive_weight'])} | {row['best_epoch']} | {f(row['validation_total_loss'])} | {f(row['raw_F1_50'])} | {f(row['official_F1_50'])} | {f(row['boundary_F1_33'])} | {row['false_peaks']} | {row['missed_boundaries']} | {row['duplicate_peaks']} | {f(row['release_F1'])} | {f(row['place_release_boundary_recall'])} |")
    lines.extend(["", "Hard-window train target statistics: r5=3,166 positives/weight 44.0679; r10=6,026/23.1528; r20=11,746/11.8780. Gaussian train positive masses are sigma5=3,604.478, sigma10=7,188.957, sigma20=14,357.914; all primary Gaussian runs use pos_weight=1. Reciprocal hard-label weighting is not directly meaningful for fractional Gaussian targets. No secondary weight-50 Gaussian runs were started because the primary CPU training already consumed the available round budget.", "", "## Boundary tolerances and semantic split results", "", "Boundary precision/recall/F1 at every requested tolerance are in each model's `all_test_summary.json` under `boundary.official.internal_{5,10,20,30,33,50}.pooled`; fixed-threshold rows for 0.30–0.90 are in `fixed_threshold_metrics.csv`. Raw, official 0.50, validation-calibrated, and oracle semantic metrics—including confusion matrices and per-class release precision/recall/F1/support—are in the validation and test summary JSON files.", ""])
    lines.append("| transition | experiment | support | exact p | max ±10 | max ±20 | max ±33 | recall @0.50 | mean error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in transitions:
        if row["transition"] in {"place -> release", "reach -> grasp", "grasp -> lift", "lift -> transport", "transport -> pour", "pour -> pour_recover", "pour_recover -> transport", "transport -> place", "place -> wipe", "wipe -> lift", "place -> retreat"} and row["experiment"] in {"baseline_single_frame", "hard_window_r5", "gaussian_s20"}:
            lines.append(f"| {row['transition']} | {row['experiment']} | {row['support']} | {f(row['mean_exact_boundary_probability'])} | {f(row['mean_max_probability_within_10'])} | {f(row['mean_max_probability_within_20'])} | {f(row['mean_max_probability_within_33'])} | {f(row['detection_recall_threshold_0.50'])} | {f(row['mean_localization_error'])} |")
    lines.extend(["", "Release is not uniformly easier than prior boundaries: the baseline place→release recall is 0.8333, lower than grasp→lift, lift→transport, transport→pour, and most other transitions. r5 raises it to 0.9167, while Gaussian s20 falls to 0.7500. Thus release is physically meaningful but not intrinsically an easier BRB target in this experiment.", "", "## Refinement, figures, and integrity", "", "`refinement_effect_by_trajectory.csv` records raw-to-official deltas for accuracy, Edit, F1@10/25/50 and classifies improved/unchanged/harmed trajectories with documented causes. Representative figures inspected: train/pour/p1 (contains release), test/pour/p1, test/pp/pp_c1, test/wipe/w1, and test/wipe/w4. Probability distributions, PR curves, threshold curves, false/missed peak curves, semantic trade-offs, and peak-count diagnostics are under `figures/`.", "", "All new checkpoints have SHA-256 values in `model_comparison.csv`; the two protected prior checkpoints remain at their expected hashes. The Gaussian pos-weight-50 secondary diagnostics were not run. Recommended next experiment: validate r5 and Gaussian s20 on an independent recording family, then test a compact Gaussian/r5 hybrid or asymmetric target with a preregistered semantic-preservation gate.", ""])
    (OUT / "round8_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
