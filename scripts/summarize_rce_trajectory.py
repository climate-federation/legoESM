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


def collect_trajectory(out_dir: "Path | str") -> list[DayRow]:
    """Read every ``snap_day_NNNN.npz`` under ``out_dir/snapshots``
    and optionally attach matching profile data.

    iter-119: accepts ``str`` as well as ``Path`` for ergonomics
    (Path conversion at entry). Pre-iter-119 the function silently
    raised a confusing ``TypeError: unsupported operand type(s) for
    /`` when called with a string path.

    Returns rows sorted by ``day``. Raises:

    * ``FileNotFoundError`` if ``snapshots/`` is missing or contains
      no anchored-name snapshot files.
    * ``ValueError`` if any snapshot reports a non-finite or duplicate
      ``day`` value (iter-99 Codex LOW#5).
    * ``ValueError`` from ``_attach_profile`` if a profile file's
      stored day disagrees with its filename's snapshot day (iter-99
      Codex MEDIUM#2).
    """
    out_dir = Path(out_dir)
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


@dataclass
class QualityVerdict:
    """Result of ``evaluate_rce_quality``.

    ``passed`` is the boolean AND of every individual criterion;
    ``reasons`` lists one line per failed criterion so callers can
    surface specifics in CI output.

    ``evaluated`` distinguishes a real PASS (plateau check fired and
    every criterion was met) from an INSUFFICIENT verdict where the
    trajectory was too short to evaluate the plateau (iter-104
    Codex MEDIUM#3). Smokes / DAYS<10 land in INSUFFICIENT; the CLI
    surfaces it as a distinct exit code (4) so automation can tell
    "PASS but only finite + wind sanity ran" apart from a real
    plateau-PASS.
    """

    passed: bool
    reasons: list[str]
    evaluated: bool = True


# iter-104 Codex MEDIUM#7 fix: distinct CLI exit codes so automation
# can distinguish a DOD FAIL verdict from an IO / parse error.
# iter-114 LOW#2: argparse uses exit 2 for misuse (CLI usage error);
# don't reuse it. EXIT_USAGE is just an alias for that fact, kept here
# so the constants table is the single source of truth.
EXIT_OK = 0
EXIT_IO_ERROR = 1            # raised by Python on uncaught
                              # FileNotFoundError / ValueError
EXIT_USAGE = 2               # argparse.error / mutually-exclusive
                              # arg violations
EXIT_DOD_FAIL = 3            # --evaluate ran and at least one
                              # criterion failed
EXIT_DOD_INSUFFICIENT = 4    # --evaluate ran but trajectory too
                              # short for plateau check


# iter-102 / iter-109: default thresholds drawn from Wing et al.
# 2018 RCEMIP1 multi-model statistics at SST = 300 K. Reference:
# Wing, Reed, Satoh, Stevens, Bony, Ohno 2018, "Radiative-Convective
# Equilibrium Model Intercomparison Project", Geoscientific Model
# Development 11(2):793-813, doi:10.5194/gmd-11-793-2018. Fig. 5b
# PWV-vs-SST envelope at SST = 300 K shows inter-model spread
# ~45-60 mm (median ~50 mm, range across 16 cloud-resolving + 5
# global models).
#
# DEFAULT_CWV_RANGE_MM is intentionally asymmetric around the Wing
# band (45-60 mm):
#   - Lower bound 35 mm: Wing band lower + 10 mm safety margin to
#     absorb the IC dip. iter-98 IC = 49.94 mm; a symmetric ±5 mm
#     band would put the lower edge above the IC and false-FAIL
#     trajectories whose plateau mean is pulled down by the early
#     pre-spinup days included in the rolling window.
#   - Upper bound 65 mm: Wing band upper + 5 mm tolerance for the
#     iter-98 day-4 CWV overshoot to 57.18 mm.
# Tuning history captured in the iteration log; do NOT silently
# widen these — drift here MUST trigger a CRM_implementation.md
# update (locked by tests/unit/test_dod_doc_code_consistency.py).
DEFAULT_CWV_RANGE_MM: tuple[float, float] = (35.0, 65.0)
DEFAULT_MAX_W_THRESHOLD_MS: float = 50.0
# 5 % is the 10-day SPINUP stability gate, not the final 30-day DOD
# requirement (which is < 1 %). See CRM_implementation.md Definition
# of done criterion 2 for the split. iter-98 measured 0.6 % over 10
# days; iter-104 Codex MEDIUM flagged the 1 %/5 % conflation.
DEFAULT_MSE_RELATIVE_DRIFT: float = 0.05
DEFAULT_LAST_N_DAYS_FOR_PLATEAU: int = 10


def evaluate_rce_quality(
    rows: list[DayRow],
    *,
    cwv_range_mm: tuple[float, float] = DEFAULT_CWV_RANGE_MM,
    max_w_threshold_ms: float = DEFAULT_MAX_W_THRESHOLD_MS,
    mse_relative_drift: float = DEFAULT_MSE_RELATIVE_DRIFT,
    last_n_days_for_plateau: int = DEFAULT_LAST_N_DAYS_FOR_PLATEAU,
    check_stuck: bool = True,
) -> QualityVerdict:
    """Evaluate an RCE trajectory against the production DOD criteria.

    Criteria checked (matches ``CRM_implementation.md`` ``Definition of
    done``):

    1. Every CWV value is finite (no NaN / inf in the trajectory).
    2. Plateau CWV (mean of the last ``last_n_days_for_plateau`` rows)
       sits inside ``cwv_range_mm``. Default range
       ``(35, 65)`` mm covers the Wing 2018 RCEMIP1 multi-model
       spread at SST = 300 K plus a 5 mm tolerance on either side.
    3. ``|U|_sfc`` stays under ``max_w_threshold_ms`` everywhere.
       Production wrapper is gravity-wave / cumulus-scale; surface
       wind > 50 m/s is a blow-up.
    4. MSE relative drift across the plateau window is below
       ``mse_relative_drift`` (default 5 % — DOD criterion 2's
       ``< 1 %`` is too tight for the 10-day spinup, so this gate
       is the *stability* check; full DOD compliance is checked
       separately at 30-day end-state).

    ``rows`` shorter than ``last_n_days_for_plateau`` evaluates only
    criteria 1 + 3 (not enough data for a plateau measurement).
    """
    reasons: list[str] = []
    # Criterion 1: finite CWV everywhere.
    nonfinite_days = [
        r.day for r in rows
        if not math.isfinite(r.cwv_mean) or not math.isfinite(r.cwv_max)
    ]
    if nonfinite_days:
        reasons.append(
            f"non-finite CWV on day(s) {nonfinite_days!r}"
        )
    # Criterion 3: surface wind sanity (cheap NaN-catcher too).
    over_w = [
        r.day for r in rows
        if math.isfinite(r.wind_sfc_max)
        and r.wind_sfc_max > max_w_threshold_ms
    ]
    if over_w:
        reasons.append(
            f"|U|_sfc exceeded {max_w_threshold_ms} m/s on day(s) "
            f"{over_w!r} — production blow-up signature."
        )
    nonfinite_w = [r.day for r in rows if not math.isfinite(r.wind_sfc_max)]
    if nonfinite_w:
        reasons.append(
            f"non-finite |U|_sfc on day(s) {nonfinite_w!r}"
        )
    # iter-117 / iter-118 / iter-120: stuck-trajectory detectors.
    # detect_stuck_trajectory: pre-iter-95 Bug 2 (leading window).
    # detect_sustained_stuck_trajectory: mid-run regression that
    # would otherwise slip past the leading-window check (Codex
    # iter-118 MEDIUM). iter-118 opt-out for callers that only want
    # the plateau gates.
    if check_stuck:
        stuck, stuck_reason = detect_stuck_trajectory(rows)
        if stuck:
            reasons.append(stuck_reason)  # type: ignore[arg-type]
        sustained, sustained_reason = (
            detect_sustained_stuck_trajectory(rows)
        )
        if sustained:
            reasons.append(sustained_reason)  # type: ignore[arg-type]
    # Criteria 2 + 4: need at least last_n_days_for_plateau rows.
    evaluated_plateau = len(rows) >= last_n_days_for_plateau
    if evaluated_plateau:
        plateau = rows[-last_n_days_for_plateau:]
        plateau_cwv_mean = sum(r.cwv_mean for r in plateau) / len(plateau)
        if not (cwv_range_mm[0] <= plateau_cwv_mean <= cwv_range_mm[1]):
            reasons.append(
                f"plateau CWV mean {plateau_cwv_mean:.3f} mm outside "
                f"target range {cwv_range_mm} (last "
                f"{last_n_days_for_plateau} days)."
            )
        # iter-104 Codex MEDIUM#1: the mean alone misses a runaway
        # that starts low and ends high (e.g. 30 → 80 mm averages
        # 55 mm — passes). Also assert plateau CWV max stays inside
        # the upper bound (the lower bound is checked by the mean —
        # a slow drain dips the mean before any value goes below).
        plateau_cwv_max = max(r.cwv_max for r in plateau)
        if plateau_cwv_max > cwv_range_mm[1]:
            reasons.append(
                f"plateau CWV max {plateau_cwv_max:.3f} mm exceeded "
                f"upper bound {cwv_range_mm[1]} mm — runaway "
                f"evaporation signature."
            )
        mse_vals = [r.mse_mean for r in plateau if math.isfinite(r.mse_mean)]
        if mse_vals:
            mse_min, mse_max = min(mse_vals), max(mse_vals)
            # iter-104 Codex LOW#2: denominator should be
            # max(abs(min), abs(max), 1e-30) so all-negative MSE
            # arrays (defensive — MSE is non-negative in practice)
            # don't overstate drift.
            denom = max(abs(mse_max), abs(mse_min), 1e-30)
            relative = abs(mse_max - mse_min) / denom
            if relative > mse_relative_drift:
                reasons.append(
                    f"MSE relative drift {relative:.4e} > "
                    f"{mse_relative_drift} over last "
                    f"{last_n_days_for_plateau} days."
                )
    return QualityVerdict(
        passed=(not reasons),
        reasons=reasons,
        evaluated=evaluated_plateau,
    )


# iter-117: a "stuck trajectory" detector. Pre-iter-95 Bug 2
# (unconditional fix_moist_mass_plane rescaling) pinned CWV at the
# IC value (49.941 mm) for 11+ sim-hours because the mass fixer
# undid every surface-flux moisture gain. The bug was diagnosed by
# eyeballing the log; this detector formalises the check so a
# regression surfaces programmatically.
DEFAULT_STUCK_CONSECUTIVE_DAYS: int = 3
DEFAULT_STUCK_CWV_TOL_MM: float = 0.001

# iter-120 Codex MEDIUM: a mass-fixer that REGRESSES mid-run (after
# initial spinup) would not be caught by the leading-window check.
# Add a separate "sustained-stuck" detector that uses a TIGHTER
# tolerance so that legitimate late-equilibrium plateaus (which
# oscillate at ~1e-3 mm or larger) do not false-positive. Bit-equal
# CWV across consecutive days can only mean the run is not actually
# integrating moisture forward — distinct from "oscillating around
# an equilibrium value."
DEFAULT_SUSTAINED_STUCK_CWV_TOL_MM: float = 1e-7


def detect_stuck_trajectory(
    rows: list[DayRow],
    *,
    consecutive_days: int = DEFAULT_STUCK_CONSECUTIVE_DAYS,
    cwv_tol_mm: float = DEFAULT_STUCK_CWV_TOL_MM,
) -> tuple[bool, str | None]:
    """Detect a Bug-2-signature pinned-IC trajectory.

    Returns ``(True, reason)`` if the FIRST ``consecutive_days``
    rows all sit within ``cwv_tol_mm`` of each other, ``(False,
    None)`` otherwise.

    iter-118 Codex HIGH: the iter-117 version scanned the FULL
    trajectory with a sliding window, which false-positived on
    legitimate late-equilibrium plateaus where CWV drift falls
    below 0.001 mm/day after full equilibration. That misled
    users into chasing a mass-fixer bug on perfectly correct runs.

    iter-118 fix: scope the check to the LEADING window only.

    Pre-iter-95 Bug 2 signature: ``cwv_mean`` reported the IC value
    (49.941 mm) on every snapshot. Surface flux WAS adding qv at
    the lowest model level but the unconditional
    ``fix_moist_mass_plane`` rescaled total water back to IC every
    outer step — and the failure mode is at the START of the run
    (the IC pin persists from day 0). Real spinup like iter-98
    drifts CWV 49.94 → 53.63 mm in day 0-to-day-1 alone, more
    than 3,000x the 0.001 mm tolerance.

    Trajectories shorter than ``consecutive_days`` return
    ``(False, None)`` (cannot decide yet — caller can re-check
    later).
    """
    if len(rows) < consecutive_days:
        return False, None
    finite_rows = [r for r in rows if math.isfinite(r.cwv_mean)]
    if len(finite_rows) < consecutive_days:
        return False, None
    # iter-118 fix: ONLY the leading window. A late equilibrium
    # plateau with sub-tolerance drift is not Bug 2.
    window = finite_rows[:consecutive_days]
    cwvs = [r.cwv_mean for r in window]
    if max(cwvs) - min(cwvs) <= cwv_tol_mm:
        return True, (
            f"CWV pinned within {cwv_tol_mm} mm across the FIRST "
            f"{consecutive_days} snapshots "
            f"(days {window[0].day:.2f}..{window[-1].day:.2f}, "
            f"CWV range {min(cwvs):.6f}..{max(cwvs):.6f} mm). "
            f"This matches the pre-iter-95 Bug 2 signature: surface "
            f"flux gains undone by an unconditional mass fixer. "
            f"Check ``--no-mass-fixer`` is set in the driver call / "
            f"``fix_moist_mass_plane`` is not being applied "
            f"unconditionally."
        )
    return False, None


def detect_sustained_stuck_trajectory(
    rows: list[DayRow],
    *,
    consecutive_days: int = DEFAULT_STUCK_CONSECUTIVE_DAYS,
    cwv_tol_mm: float = DEFAULT_SUSTAINED_STUCK_CWV_TOL_MM,
) -> tuple[bool, str | None]:
    """Detect a mid-run "stuck" regression — CWV bit-equal across
    a sliding window AFTER some prior dynamic drift.

    iter-120 Codex MEDIUM: pairs with ``detect_stuck_trajectory``
    (leading-window only) to catch the hypothetical delayed-stuck
    case where a mass-fixer regression kicks in mid-run, AFTER the
    initial spinup that ``detect_stuck_trajectory`` would otherwise
    rule out.

    To avoid false-positives on legitimate late-equilibrium
    plateaus (which oscillate at ~1e-3 mm or larger from
    convection / radiation balance), this detector uses a much
    tighter tolerance (1e-7 mm by default — effectively bit-equal).
    A real run never sits THAT tightly; only a stuck-rescaling bug
    can produce bit-equal CWV across multiple snapshots.

    Returns ``(True, reason)`` if any sliding window of
    ``consecutive_days`` rows has CWV range < ``cwv_tol_mm``,
    EXCLUDING the leading window (which is already checked by
    ``detect_stuck_trajectory``). ``(False, None)`` otherwise.
    """
    if len(rows) < 2 * consecutive_days:
        # Too short for a meaningful "post-spinup" sustained check.
        return False, None
    finite_rows = [r for r in rows if math.isfinite(r.cwv_mean)]
    if len(finite_rows) < 2 * consecutive_days:
        return False, None
    # Sliding windows starting at index 1 onward (skip the leading
    # window — that's the other detector's job, with its own looser
    # tolerance + actionable Bug 2 message).
    for i in range(1, len(finite_rows) - consecutive_days + 1):
        window = finite_rows[i:i + consecutive_days]
        cwvs = [r.cwv_mean for r in window]
        if max(cwvs) - min(cwvs) <= cwv_tol_mm:
            return True, (
                f"CWV bit-equal within {cwv_tol_mm} mm across "
                f"{consecutive_days} consecutive snapshots "
                f"AFTER spinup (days {window[0].day:.2f}.."
                f"{window[-1].day:.2f}, range {min(cwvs):.9f}.."
                f"{max(cwvs):.9f} mm). Legitimate equilibrium "
                f"oscillates at ~1e-3 mm or larger; bit-equal "
                f"across multiple days suggests the moisture loop "
                f"stopped integrating mid-run — possibly a "
                f"reintroduced mass-fixer call or a state-update "
                f"regression."
            )
    return False, None


# iter-112: full 30-day DOD criterion 2 evaluator. Distinct from
# evaluate_rce_quality (which uses the 5 % spinup gate) because the
# DOD production target tightens MSE drift to < 1 % over the last 10
# days of a 30-day window — the equilibration check applied at the
# END of a full production run.
DOD_FINAL_MSE_DRIFT: float = 0.01
DOD_FINAL_MIN_DAYS: int = 30


def evaluate_rce_final_dod(
    rows: list[DayRow],
    *,
    cwv_range_mm: tuple[float, float] = DEFAULT_CWV_RANGE_MM,
    max_w_threshold_ms: float = DEFAULT_MAX_W_THRESHOLD_MS,
    final_mse_drift: float = DOD_FINAL_MSE_DRIFT,
    last_n_days_for_plateau: int = DEFAULT_LAST_N_DAYS_FOR_PLATEAU,
    min_days: int = DOD_FINAL_MIN_DAYS,
) -> QualityVerdict:
    """Evaluate a trajectory against the FULL ``CRM_implementation.md``
    Definition-of-done criterion 2 — the 30-day equilibration target.

    Differs from ``evaluate_rce_quality`` (the spinup gate) in three
    ways:

    1. Requires ``len(rows) >= min_days`` (default 30). Anything
       shorter returns ``evaluated=False``.
    2. MSE relative drift tolerance defaults to **1 %** (the
       production DOD requirement) instead of the 5 % spinup gate.
    3. All the spinup gate's checks (finite CWV, max|U|_sfc < 50
       m/s, plateau CWV range, runaway-evaporation max) still run.

    Use this once a 30-day production run finishes (iter-105 is the
    first such target). ``--evaluate`` continues to use the spinup
    gate so 10-day runs like iter-98 still get a verdict.

    iter-114 Codex LOW#6 — plateau-window scoping: ``min_days`` is a
    *data sufficiency* gate, not a full-window certification. The
    plateau CWV + MSE drift checks only inspect the LAST
    ``last_n_days_for_plateau`` rows (default 10). A 30-day run with a
    transient blow-up at day 5 that subsequently recovers WILL pass
    the final-DOD gate because the unstable window is not in the
    final 10 days. Callers who need full-window certification should
    additionally check ``max_w_threshold_ms`` (already covered) +
    eyeball the trajectory.csv for the missing window.
    """
    reasons: list[str] = []
    # If the trajectory is too short, fail INSUFFICIENT without
    # running any check (different semantics from spinup gate's
    # passed-but-not-evaluated).
    if len(rows) < min_days:
        return QualityVerdict(
            passed=False,
            reasons=[
                f"trajectory has {len(rows)} day(s); the 30-day DOD "
                f"check needs ≥ {min_days} days. Use --evaluate "
                f"(spinup gate) on shorter runs."
            ],
            evaluated=False,
        )

    # Reuse the spinup-gate checks but pass the tighter MSE drift +
    # the same plateau window length. Every reason on the spinup
    # gate is also a reason on the final DOD gate.
    spinup_verdict = evaluate_rce_quality(
        rows,
        cwv_range_mm=cwv_range_mm,
        max_w_threshold_ms=max_w_threshold_ms,
        mse_relative_drift=final_mse_drift,
        last_n_days_for_plateau=last_n_days_for_plateau,
    )
    return QualityVerdict(
        passed=spinup_verdict.passed,
        reasons=list(spinup_verdict.reasons),
        evaluated=True,
    )


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
    p.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate the trajectory against the DOD SPINUP gate "
             "(10-day window, 5 %% MSE drift). Prints PASS/FAIL + "
             "reasons; exit code non-zero on FAIL so the wrapper "
             "can gate. Mutually exclusive with --final-dod.",
    )
    p.add_argument(
        "--final-dod",
        action="store_true",
        default=False,
        help="Evaluate the trajectory against the FULL 30-day DOD "
             "criterion 2 (>=30 day rows, 1 %% MSE drift over last "
             "10 days). Use on a finished 30-day production run.",
    )
    # iter-127: --no-plateau-check skips the plateau gates and the
    # INSUFFICIENT branch. The remaining finite + max|U|_sfc + stuck
    # detectors still run. Useful for mid-flight progress monitoring
    # where the trajectory is too short for the plateau check but
    # the user wants a stability verdict (PASS/FAIL only).
    p.add_argument(
        "--no-plateau-check",
        action="store_true",
        default=False,
        help="Skip the plateau / MSE-drift gates and the "
             "INSUFFICIENT branch. Returns PASS unless stability "
             "checks (finite, max|U|_sfc, stuck detectors) fail. "
             "Only valid with --evaluate (not --final-dod).",
    )
    args = p.parse_args()
    if args.evaluate and args.final_dod:
        # iter-114 Codex LOW#2: ``raise SystemExit("msg")`` exits 1,
        # colliding with EXIT_IO_ERROR. ``argparse.error`` exits
        # EXIT_USAGE (2), which is the POSIX convention for CLI
        # misuse and distinct from both IO and DOD-FAIL exits.
        p.error(
            "--evaluate (spinup gate) and --final-dod (30-day DOD "
            "gate) are mutually exclusive; pick one."
        )
    if args.no_plateau_check and args.final_dod:
        p.error(
            "--no-plateau-check is only valid with --evaluate; "
            "--final-dod's plateau check IS the DOD criterion 2 "
            "verdict — skipping it would defeat the purpose."
        )

    rows = collect_trajectory(args.out_dir)
    csv_path = args.csv or args.out_dir / "trajectory.csv"
    write_csv(rows, csv_path)
    print(format_table(rows))
    print()
    print(f"wrote {csv_path} ({len(rows)} rows)")
    if args.evaluate or args.final_dod:
        if args.final_dod:
            verdict = evaluate_rce_final_dod(rows)
            label = "DOD FINAL"
        else:
            # iter-127: --no-plateau-check passes a sentinel
            # ``last_n_days_for_plateau`` larger than any practical
            # trajectory, so the plateau branch never fires but the
            # stability / stuck-detector branches still run.
            n_plateau = (
                10**6 if args.no_plateau_check
                else DEFAULT_LAST_N_DAYS_FOR_PLATEAU
            )
            verdict = evaluate_rce_quality(
                rows, last_n_days_for_plateau=n_plateau,
            )
            label = "DOD"
        # iter-104 Codex MEDIUM#3 + MEDIUM#7: three-way verdict +
        # distinct exit codes so automation can tell PASS from
        # INSUFFICIENT (too short for plateau check) and from a
        # crash on IO / parse error (exit 1 from uncaught Python).
        # iter-127: --no-plateau-check folds INSUFFICIENT into
        # PASS (only stability checks matter).
        if not verdict.evaluated and not args.no_plateau_check:
            print(f"{label} verdict: INSUFFICIENT")
            if not verdict.reasons:
                print(
                    f"  - trajectory has {len(rows)} day(s), need "
                    f"{DEFAULT_LAST_N_DAYS_FOR_PLATEAU} for the plateau check."
                )
            for r in verdict.reasons:
                print(f"  - {r}")
            raise SystemExit(EXIT_DOD_INSUFFICIENT)
        if verdict.passed:
            print(f"{label} verdict: PASS")
        else:
            print(f"{label} verdict: FAIL")
            for r in verdict.reasons:
                print(f"  - {r}")
            raise SystemExit(EXIT_DOD_FAIL)


if __name__ == "__main__":
    main()
