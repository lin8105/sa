"""Small CSV and text loggers used by the training entry point."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


class CSVMetricLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str], *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        existing = append and self.path.is_file() and self.path.stat().st_size > 0
        self._handle = self.path.open("a" if existing else "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if not existing:
            self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CSVMetricLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def append_log(path: str | Path, message: str) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


__all__ = ["CSVMetricLogger", "append_log"]
