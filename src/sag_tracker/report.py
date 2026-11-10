"""Writing the two output artefacts: metrics.json and trace.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

TRACE_COLUMNS = [
    "frame",
    "t_s",
    "separation_px",
    "separation_mm",
    "compression_mm",
    "velocity_mm_s",
    "upper_ok",
    "lower_ok",
    "valid",
]


def write_metrics(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_trace(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in TRACE_COLUMNS if c not in columns]
    if missing:
        raise KeyError(f"trace is missing columns: {', '.join(missing)}")

    n = len(columns["frame"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACE_COLUMNS)
        for i in range(n):
            writer.writerow([_fmt(columns[c][i]) for c in TRACE_COLUMNS])
    return path


def _fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return "" if np.isnan(value) else f"{value:.4f}"
