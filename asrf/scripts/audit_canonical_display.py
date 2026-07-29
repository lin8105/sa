#!/usr/bin/env python
"""Audit repository-local user-facing label occurrences."""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs/round6_diagnostics/canonical_label_audit.csv"


def main() -> int:
    rows: list[dict[str, object]] = []
    roots = [REPO_ROOT / name for name in ("src", "scripts", "configs", "tests", "docs", "splits", "outputs")]
    pattern = re.compile(r"\b(pick|translation)\b", re.IGNORECASE)
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            # Historical audit records are evidence, not current model-facing output.
            relative = path.relative_to(REPO_ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                for match in pattern.finditer(line):
                    token = match.group(1).lower()
                    if relative.startswith("outputs/") and "round6_diagnostics" not in relative:
                        category = "historical_or_existing_export"
                        action = "preserve historical evidence; regenerated outputs must be canonical"
                    elif relative.startswith("docs/"):
                        category = "historical_or_explanatory_text"
                        action = "preserve as explanatory alias/history"
                    elif relative.startswith("splits/") or "pick and place" in line or "pick_and_place" in line:
                        category = "task_directory_or_path"
                        action = "not a class display"
                    elif relative in {"configs/labels_pour.yaml", "configs/labels_multitask.yaml"} or relative == "src/asrf/data/labels.py":
                        category = "parser_backward_compatibility"
                        action = "retain alias support; never emit as canonical class"
                    elif "audit" in relative or "verify_manual_correction" in relative or "build_multitask_metadata" in relative or "inspect_pour_split" in relative or "preflight" in relative or relative.startswith("tests/"):
                        category = "raw_label_audit"
                        action = "retain for raw-label verification"
                    else:
                        category = "review_required"
                        action = "must not be user-facing model class"
                    rows.append({"path": relative, "line": line_number, "token": token, "category": category, "action": action, "text": line.strip()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "line", "token", "category", "action", "text"])
        writer.writeheader()
        writer.writerows(rows)
    violations = [row for row in rows if row["category"] == "review_required"]
    print(f"alias_occurrences={len(rows)}")
    print(f"review_required={len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
