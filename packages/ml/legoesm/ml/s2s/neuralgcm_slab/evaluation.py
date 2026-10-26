"""Evaluation-table helpers for forecast metrics."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LeadTimeWindow:
    """Inclusive lead-time window defined in forecast days."""

    name: str
    start_day: int
    end_day: int


DEFAULT_S2S_WINDOWS = (
    LeadTimeWindow("wk3_4", 15, 28),
    LeadTimeWindow("wk5_6", 29, 42),
)


def load_metric_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compact metric table written by ``save_metric_table_csv``."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    fieldnames = [name for name in reader.fieldnames or () if name]
    columns: dict[str, list[float]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            columns[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def load_long_record_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_long_records(
    records: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    value_key: str = "value",
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    for record in records:
        key = tuple(record[name] for name in group_keys)
        grouped.setdefault(key, []).append(float(record[value_key]))

    aggregated: list[dict[str, object]] = []
    for key in sorted(grouped):
        row = {name: value for name, value in zip(group_keys, key, strict=True)}
        row[value_key] = float(np.nanmean(np.asarray(grouped[key], dtype=float)))
        aggregated.append(row)
    return aggregated


def aggregate_metric_table(
    metric_table: Mapping[str, Sequence[float]],
    *,
    lead_days: Sequence[int] | None,
    windows: Sequence[LeadTimeWindow],
) -> dict[str, dict[str, float]]:
    lengths = {len(values) for values in metric_table.values()}
    if len(lengths) > 1:
        raise ValueError("All metric columns must have the same number of rows")
    n_rows = lengths.pop() if lengths else 0
    lead = (
        np.arange(1, n_rows + 1, dtype=int)
        if lead_days is None
        else np.asarray(lead_days)
    )
    if lead.shape != (n_rows,):
        raise ValueError("lead_days must match the metric row count")

    summarized: dict[str, dict[str, float]] = {}
    for window in windows:
        mask = (lead >= window.start_day) & (lead <= window.end_day)
        if not np.any(mask):
            raise ValueError(f"Window {window.name!r} has no rows in the supplied table")
        summarized[window.name] = {
            name: float(np.nanmean(np.asarray(values, dtype=float)[mask]))
            for name, values in metric_table.items()
        }
    return summarized


def save_metric_summary_csv(
    summary_table: Mapping[str, Mapping[str, float]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_names = list(summary_table)
    all_columns = sorted(
        {name for values in summary_table.values() for name in values}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("window", *all_columns))
        writer.writeheader()
        for name in row_names:
            row = {"window": name}
            row.update(summary_table[name])
            writer.writerow(row)
    return path


def summarize_metric_csv(
    input_path: str | Path,
    output_path: str | Path | None,
    *,
    lead_days: Sequence[int] | None,
    windows: Sequence[LeadTimeWindow],
) -> Path:
    summary = aggregate_metric_table(
        load_metric_csv(input_path),
        lead_days=lead_days,
        windows=windows,
    )
    target = Path(output_path) if output_path is not None else Path(input_path).with_name(
        f"{Path(input_path).stem}_summary.csv"
    )
    return save_metric_summary_csv(summary, target)


__all__ = [
    "DEFAULT_S2S_WINDOWS",
    "LeadTimeWindow",
    "aggregate_long_records",
    "aggregate_metric_table",
    "load_long_record_csv",
    "load_metric_csv",
    "save_metric_summary_csv",
    "summarize_metric_csv",
]
