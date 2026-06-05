"""Diff two plane CRM RCE runs' per-day trajectories.

Reads snapshot files from two output directories via
``summarize_rce_trajectory.collect_trajectory`` (reusing iter-99 /
iter-100 hardening — anchored ``snap_day_NNNN.npz`` regex, profile
day cross-check, NaN/duplicate-day rejection), aligns rows by
``day``, and prints a per-day delta table for CWV (mean/max), MSE,
T_sfc, qc_sfc / qr_sfc max, and max wind speed. Optionally writes
the deltas to ``--csv <path>`` (default ``<dir_a>/diff_vs_<dir_b>.csv``).

iter-110: the deterministic-reproducibility check that surfaced the
iter-98 vs iter-105 bit-equal trajectory through step 7400 was just
``awk`` against two log.txt files. This script makes the same check
first-class at the snapshot level.

iter-111 (Codex review of iter-110):

* MEDIUM #1 — config-shape validation: before aligning trajectories
  the script now compares the ``cwv`` array shape from the first
  snapshot of each run; mismatched shapes (e.g. 132×132 vs 32×32)
  raise ``ValueError`` unless ``--force-shape-mismatch`` is set.
* MEDIUM #2 — non-finite reporting: NaN / inf values in either run
  are now surfaced in the summary as a ``nonfinite_count`` per
  column instead of being collapsed silently to ``None``.
* MEDIUM #3 — CSV output: ``--csv PATH`` (default
  ``<dir_a>/diff_vs_<basename>.csv``) writes the per-day delta
  rows for downstream programmatic consumption.
* LOW #4 — one-to-one alignment: ``_align_rows`` now tracks used B
  rows so near-duplicate A days within ``day_tol`` cannot both pair
  to the same B row.
* LOW #5 — ASCII labels: column labels use ``d`` and ``J/m2`` (not
  ``Δ`` and ``J/m²``) so the table renders cleanly under any stdout
  encoding (Windows cp1252, ASCII-only terminals, etc.).

Usage
-----
.. code-block:: bash

   .venv/bin/python scripts/compare_rce_trajectories.py \\
       /tmp/iter98_crm32x32_rad10d \\
       /tmp/iter105_crm32x32_rad30d

   .venv/bin/python scripts/compare_rce_trajectories.py \\
       /tmp/iter98_crm32x32_rad10d \\
       /tmp/iter105_crm32x32_rad30d \\
       --csv /tmp/iter98_vs_105_delta.csv --quiet
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent

# Load the summarizer module so we can reuse collect_trajectory.
# Loading by file path (rather than ``import summarize_rce_trajectory``)
# is the pattern the integration tests already use; ``scripts/`` is
# not on ``sys.path`` by default.
_spec = importlib.util.spec_from_file_location(
    "summarize_rce_trajectory_for_compare",
    _THIS_DIR / "summarize_rce_trajectory.py",
)
assert _spec is not None and _spec.loader is not None
summary_mod = importlib.util.module_from_spec(_spec)
sys.modules["summarize_rce_trajectory_for_compare"] = summary_mod
_spec.loader.exec_module(summary_mod)


# Columns to diff (subset of the summarizer columns — only the
# numeric scalars that make sense as deltas). iter-111 LOW#5:
# ASCII-only labels so the table renders cleanly under any stdout
# encoding (Windows cp1252, ASCII-only terminals, etc.).
_DIFF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cwv_mean",       "dCWV_mean[mm]"),
    ("cwv_max",        "dCWV_max[mm]"),
    ("mse_mean",       "dMSE[J/m2]"),
    ("T_sfc_mean",     "dT_sfc[K]"),
    ("qc_sfc_max",     "dqc_sfc"),
    ("qr_sfc_max",     "dqr_sfc"),
    ("wind_sfc_max",   "dU_sfc"),
)


# iter-111 MEDIUM#2: distinct sentinel values so the summary can
# report finite-vs-NaN deltas separately.
_DELTA_MISSING = "MISSING"   # at least one row had None for the col
_DELTA_NONFINITE = "NONFINITE"  # at least one row had NaN / inf


def _diff_value(a: float | None, b: float | None) -> "float | str":
    """Return ``b - a`` or a sentinel string:

    * ``_DELTA_MISSING`` if either operand is ``None`` (e.g. a
      profile column on a day without a profile file).
    * ``_DELTA_NONFINITE`` if either operand is NaN / inf — these
      are surfaced in the summary's ``nonfinite_count`` (Codex
      MEDIUM#2 fix).
    """
    if a is None or b is None:
        return _DELTA_MISSING
    if not (math.isfinite(a) and math.isfinite(b)):
        return _DELTA_NONFINITE
    return b - a


def _ic_snapshot_shape(out_dir: "Path | str") -> tuple[int, ...]:
    """Read the IC snapshot's ``cwv`` array shape — a cheap proxy
    for the run's horizontal grid. Used by ``diff_trajectories`` to
    catch a same-day pair across MISMATCHED CONFIGS (iter-111
    MEDIUM#1).

    iter-119: accepts ``str`` as well as ``Path`` (same convenience
    as the summarizer's ``collect_trajectory``).
    """
    out_dir = Path(out_dir)
    snap_dir = out_dir / "snapshots"
    snap_files = sorted(p for p in snap_dir.iterdir()
                        if summary_mod._SNAP_FILE_RE.match(p.name))
    if not snap_files:
        raise FileNotFoundError(
            f"no anchored snap_day_NNNN.npz under {snap_dir}"
        )
    data = np.load(snap_files[0])
    return tuple(int(s) for s in data["cwv"].shape)


def _align_rows(
    rows_a: list, rows_b: list, *, day_tol: float = 1e-6,
) -> list[tuple[float, object, object]]:
    """Pair rows by ``day`` within ``day_tol`` (fractional days).
    Days present in only one trajectory are skipped — the caller
    sees only the intersection so per-day deltas are well-defined.

    iter-111 LOW#4: matched B rows are tracked in ``used_b`` so two
    A rows whose days are both within ``day_tol`` of the SAME B row
    cannot both pair to it. The first A row claims the B row; the
    second is dropped.

    Returns a list of ``(day, row_a, row_b)`` tuples sorted by day.
    """
    out: list[tuple[float, object, object]] = []
    used_b: set[int] = set()
    sorted_a = sorted(rows_a, key=lambda r: r.day)
    sorted_b = sorted(enumerate(rows_b), key=lambda kv: kv[1].day)
    for ra in sorted_a:
        for idx, rb in sorted_b:
            if idx in used_b:
                continue
            if abs(ra.day - rb.day) <= day_tol:
                used_b.add(idx)
                out.append((ra.day, ra, rb))
                break
    return out


def diff_trajectories(
    out_dir_a: "Path | str",
    out_dir_b: "Path | str",
    *,
    day_tol: float = 1e-6,
    force_shape_mismatch: bool = False,
) -> list[dict]:
    """Compute the per-day delta between two run output dirs.

    Validates that the IC snapshot's ``cwv`` array has the same
    shape in both runs (iter-111 MEDIUM#1) — a 132×132 vs 32×32
    diff produces meaningless numbers but matching days, so the
    check refuses by default. Pass ``force_shape_mismatch=True``
    to override (used by the CLI's ``--force-shape-mismatch`` flag).

    Returns a list of dicts (one per matched day) with keys ``day``,
    ``a``, ``b`` (raw DayRow refs), and one ``delta_<col>`` entry per
    diffable column. Skips days that appear in only one trajectory.

    iter-119: accepts ``str`` as well as ``Path`` for both args.
    """
    out_dir_a = Path(out_dir_a)
    out_dir_b = Path(out_dir_b)
    shape_a = _ic_snapshot_shape(out_dir_a)
    shape_b = _ic_snapshot_shape(out_dir_b)
    if shape_a != shape_b and not force_shape_mismatch:
        raise ValueError(
            f"snapshot shape mismatch: {out_dir_a.name} has "
            f"cwv.shape = {shape_a}, {out_dir_b.name} has {shape_b}. "
            f"Per-day domain-mean deltas between different grids are "
            f"meaningless. Pass ``--force-shape-mismatch`` if you "
            f"know what you're doing (e.g. comparing a refined run "
            f"to a coarse baseline as a sanity check)."
        )
    rows_a = summary_mod.collect_trajectory(out_dir_a)
    rows_b = summary_mod.collect_trajectory(out_dir_b)
    paired = _align_rows(rows_a, rows_b, day_tol=day_tol)
    diffs: list[dict] = []
    for day, ra, rb in paired:
        entry: dict = {"day": day, "a": ra, "b": rb}
        for key, _label in _DIFF_COLUMNS:
            entry[f"delta_{key}"] = _diff_value(
                getattr(ra, key), getattr(rb, key),
            )
        diffs.append(entry)
    return diffs


def format_diff_table(diffs: list[dict]) -> str:
    """Format ``diffs`` as a fixed-width table.

    iter-111 MEDIUM#2: distinct sentinels for None (``MISSING_SENTINEL``,
    i.e. ``NA``) vs NaN/inf (``NONFINITE``) so the table tells humans
    apart a missing profile column from a corrupt diagnostic.
    """
    header = (
        f"{'day':>6}  "
        + "  ".join(f"{label:>15}" for _key, label in _DIFF_COLUMNS)
    )
    lines = [header]
    for entry in diffs:
        cells = [f"{entry['day']:6.2f}"]
        for key, _label in _DIFF_COLUMNS:
            v = entry[f"delta_{key}"]
            if v == _DELTA_MISSING:
                cells.append(f"{summary_mod.MISSING_SENTINEL:>15}")
            elif v == _DELTA_NONFINITE:
                cells.append(f"{_DELTA_NONFINITE:>15}")
            else:
                cells.append(f"{v:15.6e}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def write_csv(diffs: list[dict], csv_path: Path) -> None:
    """Serialise diff rows as CSV. iter-111 MEDIUM#3 fix:
    documented-but-unimplemented output is now wired up."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day"] + [k for k, _ in _DIFF_COLUMNS])
        for entry in diffs:
            row = [entry["day"]]
            for key, _label in _DIFF_COLUMNS:
                v = entry[f"delta_{key}"]
                if v == _DELTA_MISSING:
                    row.append(summary_mod.MISSING_SENTINEL)
                elif v == _DELTA_NONFINITE:
                    row.append(_DELTA_NONFINITE)
                else:
                    row.append(v)
            writer.writerow(row)


def _column_summary(diffs: list[dict]) -> dict[str, dict[str, object]]:
    """Per-column summary: max |finite delta|, count of NONFINITE
    entries, count of MISSING entries. iter-111 MEDIUM#2: NaN /
    inf surfaces as a count rather than being collapsed silently."""
    out: dict[str, dict[str, object]] = {}
    for key, _label in _DIFF_COLUMNS:
        finite_vals = []
        nonfinite = 0
        missing = 0
        for e in diffs:
            v = e[f"delta_{key}"]
            if v == _DELTA_MISSING:
                missing += 1
            elif v == _DELTA_NONFINITE:
                nonfinite += 1
            else:
                finite_vals.append(abs(v))
        out[key] = {
            "max_abs": max(finite_vals) if finite_vals else None,
            "nonfinite_count": nonfinite,
            "missing_count": missing,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir_a", type=Path, help="Baseline run output dir.")
    p.add_argument("dir_b", type=Path, help="Comparison run output dir.")
    p.add_argument(
        "--day-tol",
        type=float,
        default=1e-6,
        help="Fractional-day tolerance for row alignment "
             "(default 1e-6 — bit-equal day labels).",
    )
    p.add_argument(
        "--force-shape-mismatch",
        action="store_true",
        default=False,
        help="Allow diffing two runs with different snapshot shapes "
             "(default: refuse — domain-mean deltas across grids are "
             "meaningless).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <dir_a>/diff_vs_<basename>.csv.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Skip printing the fixed-width table (still prints the "
             "summary line + writes CSV).",
    )
    args = p.parse_args()

    diffs = diff_trajectories(
        args.dir_a,
        args.dir_b,
        day_tol=args.day_tol,
        force_shape_mismatch=args.force_shape_mismatch,
    )
    if not args.quiet:
        print(format_diff_table(diffs))
        print()

    csv_path = args.csv or (
        args.dir_a / f"diff_vs_{args.dir_b.name}.csv"
    )
    write_csv(diffs, csv_path)

    summary = _column_summary(diffs)
    print(f"matched {len(diffs)} day(s); wrote {csv_path}")
    for key, info in summary.items():
        nf = info["nonfinite_count"]
        miss = info["missing_count"]
        if info["max_abs"] is None:
            print(
                f"  max |d {key}| = (no finite samples); "
                f"nonfinite={nf}, missing={miss}"
            )
        else:
            print(
                f"  max |d {key}| = {info['max_abs']:.6e}; "
                f"nonfinite={nf}, missing={miss}"
            )


if __name__ == "__main__":
    main()
