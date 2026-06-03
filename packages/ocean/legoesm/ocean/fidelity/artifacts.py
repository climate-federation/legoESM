"""Loader for per-case artifacts emitted by ``scripts/run_ocean_test_matrix.py``.

Each test-matrix runner writes a per-case directory at::

    results/ocean/<case>/<grid>/<resolution>/
        results.txt          # key:value scalar dump (status, days, dt, notes,
                             # plus any case-specific scalars)
        mean_timeseries.csv  # header: step,time_days,key1,key2,... ; rows = snapshots
        mean_timeseries.png  # rendered plot (not parsed)
        snapshots/*.npz      # per-snapshot field bundles (optional)
        diagnostics.txt      # extended _save_case_diagnostics output (optional)

This module loads the parsed forms into an :class:`ArtifactBundle` so the
Layer A fidelity tests and the Layer B report builder consume them through
one entry point. The schema is intentionally permissive — any of the files
above may be absent; the bundle just reports empty for that field and the
caller decides what is required.

The companion :func:`write_synthetic` writes the same on-disk schema so
Layer A tests can build fixtures without invoking the model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArtifactBundle:
    """Bundle of one case's matrix artifacts (every field may be empty)."""

    case: str
    grid: str
    resolution: str
    root: Path
    scalars: dict[str, Any] = field(default_factory=dict)
    timeseries: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[str, np.ndarray] = field(default_factory=dict)


def _coerce_scalar(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_results_txt(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = _coerce_scalar(v.strip())
    return out


def _parse_timeseries_csv(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return out
        cols: dict[str, list[str]] = {k: [] for k in header}
        for row in reader:
            for k, v in zip(header, row):
                cols[k].append(v)
    for k, values in cols.items():
        try:
            out[k] = np.asarray([float(x) for x in values], dtype=float)
        except ValueError:
            out[k] = np.asarray(values, dtype=object)
    return out


def _load_snapshots(snap_dir: Path) -> dict[str, np.ndarray]:
    """Load every ``.npz`` under ``snap_dir`` keyed as ``<stem>/<array_name>``."""
    out: dict[str, np.ndarray] = {}
    if not snap_dir.is_dir():
        return out
    for npz_path in sorted(snap_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            for arr_name in data.files:
                out[f"{npz_path.stem}/{arr_name}"] = np.asarray(data[arr_name])
    return out


def load(case_dir: Path | str) -> ArtifactBundle:
    """Load all artifacts under ``results/ocean/<case>/<grid>/<resolution>/``.

    The three trailing path components are interpreted as ``case``, ``grid``,
    and ``resolution`` (e.g. ``rest_state/latlon/36x72``). Missing files are
    silently treated as empty in the returned bundle.
    """
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case dir does not exist: {case_dir}")
    parts = case_dir.resolve().parts
    if len(parts) < 3:
        raise ValueError(
            f"case_dir {case_dir} must end in <case>/<grid>/<resolution>/"
        )
    resolution, grid, case = parts[-1], parts[-2], parts[-3]
    return ArtifactBundle(
        case=case,
        grid=grid,
        resolution=resolution,
        root=case_dir,
        scalars=_parse_results_txt(case_dir / "results.txt"),
        timeseries=_parse_timeseries_csv(case_dir / "mean_timeseries.csv"),
        snapshots=_load_snapshots(case_dir / "snapshots"),
    )


def write_synthetic(
    case_dir: Path,
    *,
    scalars: dict[str, Any] | None = None,
    timeseries: dict[str, np.ndarray] | None = None,
    snapshots: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    """Write a synthetic artifact bundle in the on-disk schema runners emit.

    ``snapshots`` is ``dict[snapshot_name -> dict[array_name -> array]]``;
    one ``.npz`` is written per snapshot.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    if scalars:
        with open(case_dir / "results.txt", "w", encoding="utf-8") as f:
            for k, v in scalars.items():
                f.write(f"{k}: {v}\n")
    if timeseries:
        non_axis_keys = [k for k in timeseries if k not in ("step", "time_days")]
        n = len(next(iter(timeseries.values())))
        step = timeseries.get("step", np.arange(n))
        time_days = timeseries.get("time_days", np.arange(n, dtype=float))
        with open(case_dir / "mean_timeseries.csv", "w", encoding="utf-8") as f:
            f.write(",".join(["step", "time_days", *non_axis_keys]) + "\n")
            for i in range(n):
                row = [
                    str(int(step[i])),
                    f"{float(time_days[i]):.8f}",
                    *(f"{float(timeseries[k][i]):.12e}" for k in non_axis_keys),
                ]
                f.write(",".join(row) + "\n")
    if snapshots:
        snap_dir = case_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for name, arrs in snapshots.items():
            np.savez(snap_dir / f"{name}.npz", **arrs)
