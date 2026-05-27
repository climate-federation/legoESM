"""Summarize a plane CRM RCE run's per-day trajectory.

Reads ``<output>/snapshots/snap_day_NNNN.npz`` (written every
``--snapshot-hours`` ~ 24 hr by ``run_rce_mpi_long.py``) and prints a
fixed-width table + writes ``<output>/trajectory.csv`` with one row
per snapshot day.

Optionally folds in ``<output>/profiles/prof_day_NNNN.npz`` columns
when the days match — adds horizontally-averaged column maxes of
``qc``, ``qr``, ``cloud_fraction`` and ``w_variance`` (key signals
for RCE convection spin-up that the surface-only snapshots miss).

Usage
-----
.. code-block:: bash

   .venv/bin/python scripts/summarize_rce_trajectory.py \\
       /tmp/iter98_crm32x32_rad10d

Designed as the iter-98 follow-up to ``plot_rce_surface_snapshots``:
that script renders per-day spatial fields; this script aggregates
the time-series across days for the iteration-log entry.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Sentinel emitted in BOTH the printed table and the CSV when a
# profile-derived column is unavailable for that snapshot day. iter-99
# Codex review LOW#6 flagged the previous CSV-empty / table-dash
# asymmetry — one sentinel keeps human + downstream readers in sync.
MISSING_SENTINEL = "NA"

# Anchored filename pattern for snapshot + profile files. iter-99 Codex
# MEDIUM#3 fix: the pre-iter-99 loose ``snap_day_*.npz`` glob silently
# admitted ``snap_day_backup.npz`` / ``snap_day_0001.old.npz`` etc.
_SNAP_FILE_RE = re.compile(r"^snap_day_(\d{4})\.npz$")
_PROF_FILE_RE = re.compile(r"^prof_day_(\d{4})\.npz$")

# Tolerance for cross-checking the ``day`` scalar stored INSIDE a
# profile npz against the day stored in the matching snapshot. iter-99
# Codex MEDIUM#2 fix: filename suffix alone is not enough — a renamed
# or stale profile file with the right name but wrong contents would
# silently corrupt the trajectory.
_PROF_DAY_MATCH_TOL_S = 60.0 / 86400.0  # 1 minute in fractional days

# CSV column order — single source of truth for both the printed
# table and the CSV file. Each entry: (column_key, header, fmt).
_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("day",            "day",             "{:6.2f}"),
    ("cwv_mean",       "CWV_mean[mm]",    "{:11.4f}"),
    ("cwv_min",        "CWV_min[mm]",     "{:10.4f}"),
    ("cwv_max",        "CWV_max[mm]",     "{:10.4f}"),
    ("cwv_std",        "CWV_std[mm]",     "{:10.4f}"),
    ("mse_mean",       "MSE_mean[J/m2]",  "{:13.4e}"),
    ("precip_mean",    "prec_mean",       "{:11.4e}"),
    ("precip_max",     "prec_max",        "{:10.4e}"),
    ("T_sfc_mean",     "T_sfc_mean[K]",   "{:12.3f}"),
    ("qv_sfc_mean",    "qv_sfc[kg/kg]",   "{:13.4e}"),
    ("qc_sfc_max",     "qc_sfc[kg/kg]",   "{:13.4e}"),
    ("qr_sfc_max",     "qr_sfc[kg/kg]",   "{:13.4e}"),
    ("wind_sfc_mean",  "U_sfc_mean",      "{:10.3f}"),
    ("wind_sfc_max",   "U_sfc_max",       "{:10.3f}"),
    # Profile-derived (None if profile file missing for that day).
    ("qc_col_max",     "qc_col[kg/kg]",   "{:13.4e}"),
    ("qr_col_max",     "qr_col[kg/kg]",   "{:13.4e}"),
    ("cf_col_max",     "cf_max",          "{:7.4f}"),
    ("w_var_col_max",  "w_var[m2/s2]",    "{:12.4e}"),
)


@dataclass
class DayRow:
    """Per-day diagnostic row. ``profile_*`` fields are ``None`` if
    no profile snapshot exists for that day."""

    day: float
    cwv_mean: float
    cwv_min: float
    cwv_max: float
    cwv_std: float
    mse_mean: float
    precip_mean: float
    precip_max: float
    T_sfc_mean: float
    qv_sfc_mean: float
    qc_sfc_max: float
    qr_sfc_max: float
    wind_sfc_mean: float
    wind_sfc_max: float
    qc_col_max: float | None = None
    qr_col_max: float | None = None
    cf_col_max: float | None = None
    w_var_col_max: float | None = None


def _snap_row(npz_path: Path) -> DayRow:
    d = np.load(npz_path)
    cwv = np.asarray(d["cwv"])
    return DayRow(
        day=float(d["day"]),
        cwv_mean=float(np.mean(cwv)),
        cwv_min=float(np.min(cwv)),
        cwv_max=float(np.max(cwv)),
        cwv_std=float(np.std(cwv)),
        mse_mean=float(np.mean(d["mse"])),
        precip_mean=float(np.mean(d["precip"])),
        precip_max=float(np.max(d["precip"])),
        T_sfc_mean=float(np.mean(d["T_sfc"])),
        qv_sfc_mean=float(np.mean(d["qv_sfc"])),
        qc_sfc_max=float(np.max(d["qc_sfc"])),
        qr_sfc_max=float(np.max(d["qr_sfc"])),
        wind_sfc_mean=float(np.mean(d["wind_sfc"])),
        wind_sfc_max=float(np.max(d["wind_sfc"])),
    )


def _attach_profile(row: DayRow, prof_path: Path) -> None:
    """Populate the profile-derived fields on ``row`` from
    ``prof_path``. Raises ``ValueError`` if the day scalar inside the
    profile npz disagrees with the snapshot's day by more than
    ``_PROF_DAY_MATCH_TOL_S`` (iter-99 Codex MEDIUM#2)."""
    d = np.load(prof_path)
    prof_day = float(d["day"])
    if not math.isfinite(prof_day) or abs(prof_day - row.day) > _PROF_DAY_MATCH_TOL_S:
        raise ValueError(
            f"profile day mismatch: snapshot {row.day:.6f} vs "
            f"profile {prof_day:.6f} in {prof_path.name} "
            f"(tolerance {_PROF_DAY_MATCH_TOL_S * 86400:.0f} s). "
            "Did a profile file get renamed / overwritten?"
        )
    row.qc_col_max = float(np.max(d["qc"]))
    row.qr_col_max = float(np.max(d["qr"]))
    row.cf_col_max = float(np.max(d["cloud_fraction"]))
    row.w_var_col_max = float(np.max(d["w_variance"]))


def collect_trajectory(out_dir: Path) -> list[DayRow]:
    """Read every ``snap_day_NNNN.npz`` under ``out_dir/snapshots``
    and optionally attach matching profile data.

    Returns rows sorted by ``day``. Raises:

    * ``FileNotFoundError`` if ``snapshots/`` is missing or contains
      no anchored-name snapshot files.
    * ``ValueError`` if any snapshot reports a non-finite or duplicate
      ``day`` value (iter-99 Codex LOW#5).
    * ``ValueError`` from ``_attach_profile`` if a profile file's
      stored day disagrees with its filename's snapshot day (iter-99
      Codex MEDIUM#2).
    """
    snap_dir = out_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"no snapshots dir at {snap_dir}")
    # iter-99 Codex MEDIUM#3: anchored regex rejects stray files.
    snap_files = sorted(
        p for p in snap_dir.iterdir() if _SNAP_FILE_RE.match(p.name)
    )
    if not snap_files:
        raise FileNotFoundError(
            f"no snap_day_NNNN.npz under {snap_dir} (4-digit index)"
        )
    prof_dir = out_dir / "profiles"
    rows: list[DayRow] = []
    for snap_path in snap_files:
        m = _SNAP_FILE_RE.match(snap_path.name)
        assert m is not None  # filtered above; mypy hint
        idx = m.group(1)
        row = _snap_row(snap_path)
        prof_path = prof_dir / f"prof_day_{idx}.npz"
        if prof_path.exists():
            _attach_profile(row, prof_path)
        rows.append(row)
    # iter-99 Codex LOW#5: reject NaN / duplicate day values BEFORE
    # sorting so corrupted snapshots cannot produce an undefined or
    # non-monotonic trajectory.
    days = [r.day for r in rows]
    for d_val in days:
        if not math.isfinite(d_val):
            raise ValueError(
                f"non-finite snapshot day {d_val!r}; one of "
                f"{[p.name for p in snap_files]!r} has corrupted "
                f"day metadata."
            )
    if len(set(days)) != len(days):
        raise ValueError(
            f"duplicate snapshot days {sorted(days)}; the snapshot "
            f"writer should emit one file per day index. Did a run "
            f"get re-started over an existing output dir?"
        )
    rows.sort(key=lambda r: r.day)
    return rows


def format_table(rows: list[DayRow]) -> str:
    """Format ``rows`` as a fixed-width table (header + per-day
    lines). Missing profile columns render as ``NA`` (matches the CSV
    serialisation — iter-99 Codex LOW#6 fix)."""
    headers = "  ".join(h for _, h, _ in _COLUMNS)
    lines = [headers]
    for row in rows:
        cells = []
        for key, _, fmt in _COLUMNS:
            value = getattr(row, key)
            if value is None:
                width_str = fmt.split(":")[1].split(".")[0]
                width = int(width_str) if width_str else 6
                cells.append(f"{MISSING_SENTINEL:>{width}}")
            else:
                cells.append(fmt.format(value))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def write_csv(rows: list[DayRow], csv_path: Path) -> None:
    """Serialise ``rows`` as ``trajectory.csv``. Missing profile
    fields use ``NA`` (matches the printed table — iter-99 Codex
    LOW#6 fix)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([key for key, _, _ in _COLUMNS])
        for row in rows:
            writer.writerow([
                MISSING_SENTINEL if getattr(row, key) is None
                else getattr(row, key)
                for key, _, _ in _COLUMNS
            ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out_dir",
        type=Path,
        help="Run output directory containing snapshots/ + profiles/.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <out_dir>/trajectory.csv.",
    )
    args = p.parse_args()

    rows = collect_trajectory(args.out_dir)
    csv_path = args.csv or args.out_dir / "trajectory.csv"
    write_csv(rows, csv_path)
    print(format_table(rows))
    print()
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
