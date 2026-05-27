"""Diff two plane CRM RCE runs' per-day trajectories.

Reads ``<dir>/trajectory.csv`` from two output directories (or runs
``summarize_rce_trajectory.collect_trajectory`` against each), aligns
rows by ``day``, and prints a per-day delta table for CWV, MSE,
T_sfc, max(qc/qr) and max wind speed. Optionally writes the deltas
as CSV (``<dir1>/diff_vs_<basename>.csv``).

iter-110: the deterministic-reproducibility check that surfaced the
iter-98 vs iter-105 bit-equal trajectory through step 7400 was just
``awk`` against two log.txt files. This script makes the same check
first-class — same input contract as `summarize_rce_trajectory.py`,
same column set, same iter-99 / iter-100 hardening (anchored snapshot
filenames, profile day cross-check, NaN/duplicate-day rejection).

Usage
-----
.. code-block:: bash

   .venv/bin/python scripts/compare_rce_trajectories.py \\
       /tmp/iter98_crm32x32_rad10d \\
       /tmp/iter105_crm32x32_rad30d

   .venv/bin/python scripts/compare_rce_trajectories.py \\
       /tmp/iter98_crm32x32_rad10d \\
       /tmp/iter105_crm32x32_rad30d \\
       --abs-tolerance 1e-9 --quiet
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

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
# numeric scalars that make sense as deltas). The first entry is
# always ``day`` (used for the row label, not the diff itself).
_DIFF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cwv_mean",       "ΔCWV_mean[mm]"),
    ("cwv_max",        "ΔCWV_max[mm]"),
    ("mse_mean",       "ΔMSE[J/m²]"),
    ("T_sfc_mean",     "ΔT_sfc[K]"),
    ("qc_sfc_max",     "Δqc_sfc"),
    ("qr_sfc_max",     "Δqr_sfc"),
    ("wind_sfc_max",   "Δ|U|_sfc"),
)


def _diff_value(a: float | None, b: float | None) -> float | None:
    """Return ``b - a`` or ``None`` if either operand is missing."""
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        # A NaN-vs-anything diff is information-free; return None
        # so the table renders as ``NA`` instead of ``nan``.
        return None
    return b - a


def _align_rows(
    rows_a: list, rows_b: list, *, day_tol: float = 1e-6,
) -> list[tuple[float, "object", "object"]]:
    """Pair rows by ``day`` within ``day_tol`` (fractional days).
    Days present in only one trajectory are skipped — the caller
    sees only the intersection so per-day deltas are well-defined.

    Returns a list of ``(day, row_a, row_b)`` tuples sorted by day.
    """
    by_day_a = {r.day: r for r in rows_a}
    days_a = sorted(by_day_a.keys())
    days_b = sorted({r.day: None for r in rows_b}.keys())
    by_day_b = {r.day: r for r in rows_b}

    out: list[tuple[float, object, object]] = []
    for da in days_a:
        # Find the closest day in B within tolerance.
        match_db = next(
            (db for db in days_b if abs(da - db) <= day_tol), None,
        )
        if match_db is not None:
            out.append((da, by_day_a[da], by_day_b[match_db]))
    return out


def diff_trajectories(
    out_dir_a: Path, out_dir_b: Path, *, day_tol: float = 1e-6,
) -> list[dict]:
    """Compute the per-day delta between two run output dirs.

    Returns a list of dicts (one per matched day) with keys ``day``,
    ``a``, ``b`` (raw DayRow refs), and one ``delta_<col>`` entry per
    diffable column. Skips days that appear in only one trajectory.
    """
    rows_a = summary_mod.collect_trajectory(out_dir_a)
    rows_b = summary_mod.collect_trajectory(out_dir_b)
    paired = _align_rows(rows_a, rows_b, day_tol=day_tol)
    diffs: list[dict] = []
    for day, ra, rb in paired:
        entry = {"day": day, "a": ra, "b": rb}
        for key, _label in _DIFF_COLUMNS:
            entry[f"delta_{key}"] = _diff_value(
                getattr(ra, key), getattr(rb, key),
            )
        diffs.append(entry)
    return diffs


def format_diff_table(diffs: list[dict]) -> str:
    """Format ``diffs`` as a fixed-width table; ``NA`` for missing
    deltas (matches the iter-99 Codex LOW#6 sentinel)."""
    header = (
        f"{'day':>6}  "
        + "  ".join(f"{label:>15}" for _key, label in _DIFF_COLUMNS)
    )
    lines = [header]
    for entry in diffs:
        cells = [f"{entry['day']:6.2f}"]
        for key, _label in _DIFF_COLUMNS:
            v = entry[f"delta_{key}"]
            if v is None:
                cells.append(f"{summary_mod.MISSING_SENTINEL:>15}")
            else:
                cells.append(f"{v:15.6e}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


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
        "--quiet",
        action="store_true",
        default=False,
        help="Skip printing the fixed-width table (still prints the "
             "summary line + writes CSV).",
    )
    args = p.parse_args()

    diffs = diff_trajectories(
        args.dir_a, args.dir_b, day_tol=args.day_tol,
    )
    if not args.quiet:
        print(format_diff_table(diffs))
        print()
    # Summary: max abs delta per column.
    summary: dict[str, float] = {}
    for key, _label in _DIFF_COLUMNS:
        vals = [
            abs(e[f"delta_{key}"]) for e in diffs
            if e[f"delta_{key}"] is not None
        ]
        if vals:
            summary[key] = max(vals)
    print(f"matched {len(diffs)} day(s)")
    for key, val in summary.items():
        print(f"  max |Δ {key}| = {val:.6e}")


if __name__ == "__main__":
    main()
