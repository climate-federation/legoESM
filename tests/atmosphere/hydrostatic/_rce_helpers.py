"""Shared helpers for the cross-grid RCE smoke + nightly tests.

iter-78: extracted from ``test_rce_cross_grid_smoke.py`` (where they
lived since iter-19 + iter-44 + iter-46 + iter-66) into a dedicated
sibling module. Underscore prefix marks the module as a non-test
helper so pytest's ``test_*.py`` glob skips it.

Previous setup: ``test_rce_helpers_unit.py`` cross-imported the
helpers via
``from tests.atmosphere.hydrostatic.test_rce_cross_grid_smoke import
...`` which triggered pytest's collection of the source file as an
import side effect. iter-78 cleans this up — both consumers now
``from tests.atmosphere.hydrostatic._rce_helpers import ...``.

Helpers:
    _run_rce(grid_type, discretization, resolution, days, output_dir,
             timeout_s=600)
        Invoke ``scripts/run_rce.py`` with the iter-12-validated CLI.

    _parse_results(output_dir) -> dict | None
        Read ``results.txt`` and return its key:value fields.

    _parse_notes(notes: str) -> dict
        Parse a ``key=val, key=val`` notes string into floats.

    _assert_rce_pass(out_dir, label, temp_tol=1.0, max_v_cap=50.0)
        Shared post-run envelope assertion (status PASS,
        mean_T_sfc within tol of 300 K, max|v| < cap).

    _assert_dt_used(out_dir, label, expected_dt, abs_tol=None)
        Pin the dt actually used against the iter-13/26 ladder.

    _assert_max_wind_peak_below(out_dir, label, cap)
        Scan ``mean_timeseries.csv`` for the peak ``max_wind``
        across all logged days and assert below cap.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_RCE = REPO_ROOT / "scripts" / "run_rce.py"


def _run_rce(grid_type, discretization, resolution, days, output_dir,
             timeout_s=600):
    """Invoke scripts/run_rce.py with the iter-12-validated CLI.

    iter-44 Codex HIGH fix: hard-coded ``timeout=600`` was killing
    the C72 30-day nightly (iter-26 measured 2373 s wall) and the
    C48 30-day nightly (iter-15 measured 500 s, occasionally
    spilling past 600 on a busy box). ``timeout_s`` parameter lets
    each test pick its own ceiling.
    """
    env = os.environ.copy()
    # FORCE JAX_PLATFORMS=cpu (override exported value); iter-7
    # documented Metal MLIR crashes on spectral/voronoi/latlon-cgrid.
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(RUN_RCE),
        "--grid-type", grid_type,
        "--discretization", discretization,
        "--resolution", str(resolution),
        "--days", str(days),
        "--diag-days", "1",
        "--nlev", "20",
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
    )
    return result


def _parse_results(output_dir):
    """Read ``results.txt`` and return a dict of fields."""
    results_txt = output_dir / "results.txt"
    if not results_txt.exists():
        return None
    fields = {}
    for line in results_txt.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def _parse_notes(notes: str) -> dict:
    """Parse `mean_T_sfc=..., mean_T=..., max|v|=...` into a dict."""
    out = {}
    for part in notes.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            pass
    return out


def _assert_rce_pass(out_dir, label, temp_tol=1.0, max_v_cap=50.0):
    """Shared post-run assertion for a 2-day or longer RCE smoke.

    Asserts (1) results.txt exists, (2) status == PASS,
    (3) mean_T_sfc within `temp_tol` of 300 K, (4) max|v| < `max_v_cap`.
    Used by the parametrise smoke and the slow C96/C48 variants
    to avoid duplicating the envelope checks (Codex iter-19 MEDIUM).
    """
    fields = _parse_results(out_dir)
    assert fields is not None, (
        f"{label}: did not produce results.txt — script failed silently?"
    )
    assert fields.get("status") == "PASS", (
        f"{label}: status={fields.get('status')} (want PASS). "
        f"notes={fields.get('notes')}"
    )
    notes_dict = _parse_notes(fields.get("notes", ""))
    t_sfc = notes_dict.get("mean_T_sfc")
    assert t_sfc is not None, (
        f"{label}: results.txt notes line missing mean_T_sfc — "
        f"unexpected format: {fields.get('notes')!r}"
    )
    assert abs(t_sfc - 300.0) < temp_tol, (
        f"{label}: mean_T_sfc={t_sfc:.2f} drifted >{temp_tol} K from "
        f"IC 300 K — slab-ocean coupling or radiation regression "
        f"suspected. See CRM_implementation.md iter-12 for expected values."
    )
    max_v = notes_dict.get("max|v|")
    assert max_v is not None, (
        f"{label}: notes line missing max|v|."
    )
    assert max_v < max_v_cap, (
        f"{label}: max|v|={max_v:.2f} m/s exceeds {max_v_cap} m/s cap. "
        f"Production envelope is 2-12 m/s; >{max_v_cap} m/s means a "
        f"CFL crash in flight (even before the 200 m/s BLOWUP gate)."
    )


def _assert_dt_used(out_dir, label, expected_dt, abs_tol=None):
    """Pin the dt actually used in ``results.txt`` against the
    iter-13/26 ladder. Catches a silent auto_dt_rce ladder drift
    that would still pass the wider envelope checks.

    iter-46: factored out of the iter-44 C72 nightly so the same
    pattern can be applied to C48 + future N-day nightly tests
    without duplication.

    iter-51: added optional ``abs_tol`` for grids where the
    effective dt is post-clamped by another rule (latlon pole-cell
    CFL clamp uses ``pole_cell_dx(grid)`` + ``cfl_max_dt`` which
    returns a non-integer-divisible value depending on grid math —
    LL32 lands at ~81.844 s, not the ladder's 150 s). ``abs_tol``
    means "this dt should match the documented pin within this
    absolute tolerance"; ``None`` (default) means strict ``==``
    (suitable for the ladder values 600/300/150/75/37 which are
    integer-clean).
    """
    results_txt = out_dir / "results.txt"
    fields = _parse_results(out_dir)
    assert fields is not None, (
        f"{label}: did not produce results.txt at {results_txt}"
    )
    dt_used = float(fields.get("dt", "nan"))
    if abs_tol is None:
        ok = dt_used == expected_dt
    else:
        ok = abs(dt_used - expected_dt) <= abs_tol
    assert ok, (
        f"{label}: dt={dt_used} != {expected_dt}"
        f"{f' ± {abs_tol}' if abs_tol is not None else ''} "
        f"(the iter-13/26 ladder branch / post-clamp this test "
        f"pins). ``auto_dt_rce`` or the pole-cell-CFL clamp may "
        f"have drifted; update the ladder + this test together. "
        f"Source: {results_txt}"
    )


def _assert_max_wind_peak_below(out_dir, label, cap):
    """Scan ``mean_timeseries.csv`` for the peak ``max_wind`` across
    ALL logged days (not just the final) and assert below cap.

    iter-46: factored out of the iter-44 C72 nightly so the same
    Codex iter-44 MEDIUM#2 fix applies to C48 + future N-day
    nightly tests without duplication. ``_assert_rce_pass`` reads
    ``notes`` (final-day-only); a mid-run CFL spike that recovered
    by the last log day would silently pass it.

    Pins the exact column name ``max_wind`` written by
    ``run_rce.py:668-676`` (Codex iter-45 hardening — was a
    substring match in iter-44 that could false-match
    ``max_dvdt``).

    Codex iter-46 MEDIUM: tracks ``seen_max_wind`` so an empty
    timeseries (header-only, no data rows) or all-unparseable
    values can't pass the cap vacuously (initial ``peak_v=0.0``
    would otherwise satisfy ``< cap`` even with zero real data).

    Codex iter-66 HIGH: ``float("nan")`` parses successfully but
    ``max(0.0, nan)`` returns ``0.0`` in CPython (NaN-naive
    comparison). Reject NaN explicitly via ``math.isfinite``.
    """
    mean_csv = out_dir / "mean_timeseries.csv"
    assert mean_csv.exists(), (
        f"{label}: mean_timeseries.csv missing — run_rce.py "
        f"diagnostic emitter regressed. Expected at {mean_csv}"
    )
    import csv
    import math
    peak_v = 0.0
    seen_max_wind = False
    with open(mean_csv) as fh:
        reader = csv.DictReader(fh)
        assert (
            reader.fieldnames is not None
            and "max_wind" in reader.fieldnames
        ), (
            f"{label}: mean_timeseries.csv at {mean_csv} missing "
            f"``max_wind`` column. Headers: {reader.fieldnames!r}. "
            f"run_rce.py:668-676 schema may have changed; if "
            f"intentional, update this test."
        )
        for row in reader:
            try:
                val = abs(float(row["max_wind"]))
                if not math.isfinite(val):
                    continue
                peak_v = max(peak_v, val)
                seen_max_wind = True
            except (TypeError, ValueError):
                pass
    assert seen_max_wind, (
        f"{label}: mean_timeseries.csv at {mean_csv} had no "
        f"parseable finite ``max_wind`` rows (header-only, all "
        f"unparseable, or all NaN). The cap check would otherwise "
        f"pass vacuously against peak_v=0.0."
    )
    assert peak_v < cap, (
        f"{label}: peak max|v|={peak_v:.2f} across the full "
        f"timeseries exceeds {cap} m/s cap. A mid-run CFL spike "
        f"that recovered by the final log day would slip past the "
        f"notes-line (last-day-only) check in ``_assert_rce_pass``; "
        f"this assertion catches it. Source: {mean_csv}"
    )
