"""Unit tests for ``scripts/validate/summarize_rce_trajectory.py``.

iter-98: a tool tracking per-day RCE diagnostics is only useful if its
field extraction matches ``run_rce_mpi_long.py``'s snapshot writer.
These tests build synthetic ``snap_day_*.npz`` + ``prof_day_*.npz``
files with the exact key set the production driver writes (verified
against ``scripts/run/run_rce_mpi_long.py:save_snapshot_2d`` and
``save_profile``) and assert that:

1. Every column in the printed table + the CSV is finite and tracks
   the synthetic input values.
2. Days that have a matching profile file get profile-derived
   columns populated; days that don't get ``None`` placeholders
   that print as ``-``.
3. Snapshot day-order is preserved even when input files are listed
   out of order.
4. Missing ``snapshots/`` dir raises ``FileNotFoundError`` with a
   helpful message (catches misnamed argv).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the script module by path — it lives under ``scripts/`` which
# is not on ``sys.path`` in the repo. Same pattern used by other
# script-test modules in the integration suite.
_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "validate"
    / "summarize_rce_trajectory.py"
)
_spec = importlib.util.spec_from_file_location(
    "summarize_rce_trajectory", _SCRIPT,
)
assert _spec is not None and _spec.loader is not None
summary_mod = importlib.util.module_from_spec(_spec)
sys.modules["summarize_rce_trajectory"] = summary_mod
_spec.loader.exec_module(summary_mod)


SNAP_KEYS = (
    "t_sim", "day", "cwv", "mse", "precip",
    "T_sfc", "qv_sfc", "qc_sfc", "qr_sfc",
    "u_sfc", "v_sfc", "wind_sfc",
)
PROF_KEYS = (
    "t_sim", "day", "z", "z_half",
    "T", "qv", "qc", "qr", "cloud_fraction", "w_variance",
)


def _write_snapshot(path: Path, day: float, cwv_value: float) -> None:
    # iter-126: mirror iter-123's NaN-safety fix in
    # test_compare_rce_trajectories.py — ``arr * 0.0`` propagates
    # NaN. No current test passes NaN cwv_value through this helper,
    # but the failure mode is structurally identical to the iter-123
    # patched footgun; pre-empt before a future NaN-fixture lands.
    ny, nx = 4, 4
    arr = np.full((ny, nx), cwv_value, dtype=np.float64)
    zeros = np.zeros((ny, nx), dtype=np.float64)
    np.savez_compressed(
        path,
        t_sim=day * 86400.0,
        day=day,
        cwv=arr,
        mse=np.full((ny, nx), cwv_value * 1e7, dtype=np.float64),
        precip=zeros,
        T_sfc=np.full((ny, nx), 300.0 - day * 0.1, dtype=np.float64),
        qv_sfc=np.full((ny, nx), 0.02, dtype=np.float64),
        qc_sfc=np.full((ny, nx), day * 1e-5, dtype=np.float64),
        qr_sfc=np.full((ny, nx), day * 1e-6, dtype=np.float64),
        u_sfc=zeros,
        v_sfc=zeros,
        wind_sfc=np.full((ny, nx), day * 0.01, dtype=np.float64),
    )


def _write_profile(path: Path, day: float) -> None:
    nlev = 8
    z = np.linspace(0.0, 15_000.0, nlev, dtype=np.float64)
    np.savez_compressed(
        path,
        t_sim=day * 86400.0,
        day=day,
        z=z,
        z_half=z[:-1] + 0.5 * np.diff(z),
        T=np.linspace(298.0, 200.0, nlev),
        qv=np.linspace(0.02, 1e-6, nlev),
        qc=np.full(nlev, day * 1e-4),
        qr=np.full(nlev, day * 2e-5),
        cloud_fraction=np.linspace(0.0, day * 0.1, nlev),
        w_variance=np.full(nlev, day * 1e-3),
    )


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> Path:
    """3-day synthetic run with profiles at day 0 + 2 (day 1 has no
    profile — exercises the missing-profile branch)."""
    out_dir = tmp_path / "synthetic_run"
    snaps = out_dir / "snapshots"
    profs = out_dir / "profiles"
    snaps.mkdir(parents=True)
    profs.mkdir(parents=True)
    # Write out of order to exercise the day-sort in collect_trajectory.
    for day, idx in [(2.0, 2), (0.0, 0), (1.0, 1)]:
        _write_snapshot(
            snaps / f"snap_day_{idx:04d}.npz",
            day=day,
            cwv_value=50.0 + day * 1.5,
        )
    _write_profile(profs / "prof_day_0000.npz", day=0.0)
    _write_profile(profs / "prof_day_0002.npz", day=2.0)
    return out_dir


def test_snapshot_key_contract_matches_driver():
    """``_snap_row`` reads keys ``cwv``, ``mse``, ``precip``,
    ``T_sfc``, ``qv_sfc``, ``qc_sfc``, ``qr_sfc``, ``wind_sfc``,
    ``day``. If ``run_rce_mpi_long.py:save_snapshot_2d`` ever drops
    or renames one of those fields, the summarizer + this test break
    in lockstep, surfacing the contract drift."""
    driver_path = (
        Path(__file__).resolve().parents[4]
        / "scripts" / "run" / "run_rce_mpi_long.py"
    )
    text = driver_path.read_text()
    for key in SNAP_KEYS:
        assert f"{key}=" in text, (
            f"snapshot writer dropped key {key!r}; summarizer + "
            f"plot_rce_surface_snapshots both read this name."
        )


def test_profile_key_contract_matches_driver():
    """Same lock for profile keys consumed by ``_attach_profile``:
    ``qc``, ``qr``, ``cloud_fraction``, ``w_variance``."""
    driver_path = (
        Path(__file__).resolve().parents[4]
        / "scripts" / "run" / "run_rce_mpi_long.py"
    )
    text = driver_path.read_text()
    for key in ("qc", "qr", "cloud_fraction", "w_variance"):
        assert f"{key}=" in text, (
            f"profile writer dropped key {key!r}; summarizer reads "
            f"this for the column-max diagnostics."
        )


def test_collect_trajectory_sorts_by_day(synthetic_run: Path):
    rows = summary_mod.collect_trajectory(synthetic_run)
    assert [r.day for r in rows] == [0.0, 1.0, 2.0]


def test_snap_row_values(synthetic_run: Path):
    rows = summary_mod.collect_trajectory(synthetic_run)
    # Day 2: cwv_value = 50.0 + 2*1.5 = 53.0
    day2 = rows[2]
    assert day2.cwv_mean == pytest.approx(53.0)
    assert day2.cwv_min == pytest.approx(53.0)
    assert day2.cwv_max == pytest.approx(53.0)
    assert day2.cwv_std == pytest.approx(0.0)
    assert day2.qc_sfc_max == pytest.approx(2e-5)
    assert day2.qr_sfc_max == pytest.approx(2e-6)
    assert day2.wind_sfc_max == pytest.approx(0.02)
    assert day2.T_sfc_mean == pytest.approx(299.8)


def test_profile_attachment_present_and_missing(synthetic_run: Path):
    rows = summary_mod.collect_trajectory(synthetic_run)
    # Day 0 has profile → all four profile cols populated.
    assert rows[0].qc_col_max == pytest.approx(0.0)
    assert rows[0].qr_col_max == pytest.approx(0.0)
    assert rows[0].cf_col_max == pytest.approx(0.0)
    assert rows[0].w_var_col_max == pytest.approx(0.0)
    # Day 1 has no profile → all four are None.
    assert rows[1].qc_col_max is None
    assert rows[1].qr_col_max is None
    assert rows[1].cf_col_max is None
    assert rows[1].w_var_col_max is None
    # Day 2 has profile → populated with day=2 multiplier.
    assert rows[2].qc_col_max == pytest.approx(2e-4)
    assert rows[2].qr_col_max == pytest.approx(4e-5)
    # cloud_fraction = linspace(0, 0.2) → max = 0.2
    assert rows[2].cf_col_max == pytest.approx(0.2)
    assert rows[2].w_var_col_max == pytest.approx(2e-3)


def test_format_table_renders_na_sentinel_for_none(synthetic_run: Path):
    """iter-99 Codex LOW#6: table + CSV share the ``NA`` sentinel."""
    rows = summary_mod.collect_trajectory(synthetic_run)
    text = summary_mod.format_table(rows)
    day1_line = text.splitlines()[2]  # header + day0 + day1
    # Right-aligned NA appears at least 4 times (one per profile col).
    assert day1_line.count(summary_mod.MISSING_SENTINEL) >= 4, (
        f"expected at least 4 {summary_mod.MISSING_SENTINEL!r} "
        f"placeholders for missing profile cols on day 1, "
        f"got: {day1_line!r}"
    )


def test_write_csv_round_trip(synthetic_run: Path, tmp_path: Path):
    rows = summary_mod.collect_trajectory(synthetic_run)
    csv_path = tmp_path / "out.csv"
    summary_mod.write_csv(rows, csv_path)
    text = csv_path.read_text()
    lines = text.splitlines()
    # Header + 3 data rows.
    assert len(lines) == 4
    # Header includes both surface and profile columns.
    assert "cwv_mean" in lines[0]
    assert "qc_col_max" in lines[0]
    # Day 1 row's last 4 fields are NA (None → MISSING_SENTINEL).
    # iter-99 Codex LOW#6: CSV + table share the sentinel.
    fields = lines[2].split(",")
    assert fields[-4:] == [summary_mod.MISSING_SENTINEL] * 4, (
        f"day 1 profile cols should serialise as "
        f"{summary_mod.MISSING_SENTINEL!r}; got {fields[-4:]!r}"
    )


def test_write_csv_creates_parent_directories(
    synthetic_run: Path, tmp_path: Path,
):
    """iter-162: write_csv must create non-existent parent
    directories (mkdir(parents=True, exist_ok=True) at
    summarize_rce_trajectory.py:232). A regression that dropped
    the ``parents=True`` flag would crash on a deeply-nested CSV
    path. Realistic case: a wrapper that runs the summarizer
    inside a per-run output dir that hasn't been pre-created.
    """
    rows = summary_mod.collect_trajectory(synthetic_run)
    csv_path = tmp_path / "a" / "b" / "c" / "out.csv"
    assert not csv_path.parent.exists()
    summary_mod.write_csv(rows, csv_path)
    assert csv_path.exists()
    # exist_ok=True branch: re-writing to the same nested path
    # must NOT raise FileExistsError on the existing parent dir.
    summary_mod.write_csv(rows, csv_path)
    assert csv_path.exists()


def test_missing_snapshots_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no snapshots dir"):
        summary_mod.collect_trajectory(tmp_path)


def test_empty_snapshots_dir_raises(tmp_path: Path):
    (tmp_path / "snapshots").mkdir()
    with pytest.raises(FileNotFoundError, match="no snap_day_NNNN"):
        summary_mod.collect_trajectory(tmp_path)


# iter-99 Codex MEDIUM#3 + LOW#5 + MEDIUM#2 regression tests.


def test_stray_filename_in_snapshots_dir_is_ignored(tmp_path: Path):
    """iter-99 MEDIUM#3: a file named like ``snap_day_backup.npz`` or
    ``snap_day_0001.old.npz`` must NOT be admitted as a snapshot —
    the anchored regex requires exactly 4 digits + ``.npz``."""
    out_dir = tmp_path / "stray"
    snaps = out_dir / "snapshots"
    snaps.mkdir(parents=True)
    # One valid snapshot + two impostors.
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    _write_snapshot(snaps / "snap_day_backup.npz", day=99.0, cwv_value=999.0)
    _write_snapshot(snaps / "snap_day_0001.old.npz", day=99.0, cwv_value=999.0)
    rows = summary_mod.collect_trajectory(out_dir)
    assert len(rows) == 1, (
        f"expected only the anchored snap_day_0000.npz to be admitted; "
        f"got {len(rows)} rows: {[r.day for r in rows]}"
    )
    assert rows[0].day == pytest.approx(0.0)


def test_profile_day_value_mismatch_raises(tmp_path: Path):
    """iter-99 MEDIUM#2: if a profile file's stored ``day`` scalar
    disagrees with the snapshot's day by more than the 1-minute
    tolerance, ``collect_trajectory`` must raise — not silently
    attach a wrong-day profile."""
    out_dir = tmp_path / "profile_mismatch"
    snaps = out_dir / "snapshots"
    profs = out_dir / "profiles"
    snaps.mkdir(parents=True)
    profs.mkdir(parents=True)
    _write_snapshot(snaps / "snap_day_0005.npz", day=5.0, cwv_value=55.0)
    # Profile file matches by filename but stores a different day.
    _write_profile(profs / "prof_day_0005.npz", day=10.0)
    with pytest.raises(ValueError, match="profile day mismatch"):
        summary_mod.collect_trajectory(out_dir)


def test_profile_non_finite_day_raises(tmp_path: Path):
    """iter-165: _attach_profile's ``not math.isfinite(prof_day)``
    guard (summarize_rce_trajectory.py:132) must reject a profile
    npz whose ``day`` scalar is NaN. Pre-iter-165 only the
    finite-but-mismatched branch was exercised
    (test_profile_day_value_mismatch_raises); the non-finite
    branch fell through to the abs() comparison which would
    return NaN > tol -> True by accident but with the WRONG
    error message. Realistic corruption signature: a profile
    writer that hit a divide-by-zero before serialising.
    """
    out_dir = tmp_path / "profile_nan"
    snaps = out_dir / "snapshots"
    profs = out_dir / "profiles"
    snaps.mkdir(parents=True)
    profs.mkdir(parents=True)
    _write_snapshot(snaps / "snap_day_0005.npz", day=5.0, cwv_value=55.0)
    _write_profile(profs / "prof_day_0005.npz", day=float("nan"))
    with pytest.raises(ValueError, match="profile day mismatch"):
        summary_mod.collect_trajectory(out_dir)


def test_non_finite_snapshot_day_raises(tmp_path: Path):
    """iter-99 LOW#5: NaN / inf in a snapshot's ``day`` scalar must
    raise before sorting (sort order on NaN is undefined)."""
    out_dir = tmp_path / "nan_day"
    snaps = out_dir / "snapshots"
    snaps.mkdir(parents=True)
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    _write_snapshot(snaps / "snap_day_0001.npz", day=float("nan"),
                    cwv_value=51.0)
    with pytest.raises(ValueError, match="non-finite snapshot day"):
        summary_mod.collect_trajectory(out_dir)


def test_duplicate_snapshot_days_raises(tmp_path: Path):
    """iter-99 LOW#5: two snapshots reporting the same day value
    indicate a re-started run over an existing dir; sort would
    be order-stable but the trajectory loses meaning."""
    out_dir = tmp_path / "dup_day"
    snaps = out_dir / "snapshots"
    snaps.mkdir(parents=True)
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    _write_snapshot(snaps / "snap_day_0001.npz", day=0.0, cwv_value=51.0)
    with pytest.raises(ValueError, match="duplicate snapshot days"):
        summary_mod.collect_trajectory(out_dir)


def test_missing_sentinel_value():
    """iter-99 LOW#6: the sentinel is a documented constant, not
    magic string. Lock it so a future rename surfaces here."""
    assert summary_mod.MISSING_SENTINEL == "NA"


# iter-102: evaluate_rce_quality regression tests.


def _row(day: float, **overrides) -> "summary_mod.DayRow":
    """Build a minimal DayRow with sensible defaults that pass every
    criterion.

    iter-117 — default CWV varies linearly with ``day`` (50.0 +
    0.01 * day) so a series of ``_row(float(i))`` calls does NOT
    trip the new ``detect_stuck_trajectory`` gate (which would
    otherwise mark every "uniform 50.0 mm" synthetic fixture as a
    Bug 2 regression). Tests that want the stuck signature pass
    ``cwv_mean=49.9413`` (or any explicit constant) to override.
    """
    defaults = dict(
        day=day,
        cwv_mean=50.0 + 0.01 * day,
        cwv_min=50.0 + 0.01 * day,
        cwv_max=50.0 + 0.01 * day,
        cwv_std=0.0,
        mse_mean=3.5e9,
        precip_mean=0.0,
        precip_max=0.0,
        T_sfc_mean=300.0,
        qv_sfc_mean=0.02,
        qc_sfc_max=0.0,
        qr_sfc_max=0.0,
        wind_sfc_mean=0.0,
        wind_sfc_max=0.01,
    )
    defaults.update(overrides)
    return summary_mod.DayRow(**defaults)


def test_evaluate_passes_on_plateau_trajectory():
    rows = [_row(float(i)) for i in range(12)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert verdict.passed, verdict.reasons


def test_evaluate_flags_cwv_out_of_range():
    rows = [_row(float(i), cwv_mean=10.0, cwv_max=10.0) for i in range(12)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("plateau CWV" in r for r in verdict.reasons)


def test_evaluate_flags_max_w_blowup():
    rows = [_row(float(i)) for i in range(12)]
    rows[7] = _row(7.0, wind_sfc_max=999.0)
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("|U|_sfc exceeded" in r for r in verdict.reasons)


def test_evaluate_flags_nan_wind_via_nonfinite_branch():
    """iter-166: a NaN wind_sfc_max must be flagged via the
    ``non-finite |U|_sfc`` branch (summarize_rce_trajectory.py:365),
    NOT via the ``|U|_sfc exceeded`` branch — the latter short-
    circuits at math.isfinite() and would silently drop the NaN
    row from the over_w list.

    Pre iter-166 only the finite blow-up branch was exercised
    (test_evaluate_flags_max_w_blowup uses 999.0); the NaN
    branch fell through to plateau checks below which could
    pass or fail for unrelated reasons, masking the actual
    failure mode in the verdict.
    """
    rows = [_row(float(i)) for i in range(12)]
    rows[5] = _row(5.0, wind_sfc_max=float("nan"))
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("non-finite |U|_sfc" in r for r in verdict.reasons), (
        f"NaN wind must surface via the non-finite branch; "
        f"reasons: {verdict.reasons!r}"
    )
    # Sanity: must NOT be reported as a finite blow-up.
    assert not any("|U|_sfc exceeded" in r for r in verdict.reasons), (
        f"NaN should not be reported as a finite blow-up; "
        f"reasons: {verdict.reasons!r}"
    )


def test_evaluate_flags_nan_cwv():
    rows = [_row(float(i)) for i in range(12)]
    rows[5] = _row(5.0, cwv_mean=float("nan"))
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("non-finite CWV" in r for r in verdict.reasons)


def test_evaluate_flags_mse_drift():
    rows = []
    for i in range(12):
        # Drift MSE by 20% across the last 10 days — well above the
        # 5% default threshold.
        mse = 1.0e9 if i < 2 else 1.0e9 * (1.0 + 0.05 * (i - 2))
        rows.append(_row(float(i), mse_mean=mse))
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("MSE relative drift" in r for r in verdict.reasons)


def test_evaluate_short_trajectory_marks_insufficient():
    """iter-104 Codex MEDIUM#3: fewer than
    ``last_n_days_for_plateau`` rows → plateau checks skipped AND
    ``verdict.evaluated`` is False. ``passed`` may still be True
    (finite + wind checks ran), but callers can distinguish a real
    plateau-PASS from an INSUFFICIENT verdict."""
    # iter-117: vary CWV by 0.1 mm/day so the new stuck-trajectory
    # gate doesn't trip on this fixture (3 days of constant CWV
    # would be a stuck signature). The plateau check is what we
    # want to verify is skipped, not the stuck check.
    rows = [_row(float(i), cwv_mean=10.0 + 0.1 * i) for i in range(3)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert verdict.evaluated is False, (
        "3-day trajectory should be marked evaluated=False so "
        "callers can distinguish INSUFFICIENT from real PASS."
    )
    # Finite + wind sanity passed; plateau never ran → passed=True
    # but evaluated=False is the tri-state.
    assert verdict.passed
    assert verdict.reasons == []


def test_evaluate_full_trajectory_marks_evaluated():
    """Symmetric to test_evaluate_short_trajectory_marks_insufficient:
    when the trajectory IS long enough, ``evaluated=True`` so the
    CLI exit code differentiation (PASS → 0, FAIL → 3, INSUFFICIENT
    → 4) works."""
    rows = [_row(float(i)) for i in range(12)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert verdict.evaluated is True
    assert verdict.passed


def test_evaluate_iter98_inflight_trajectory_passes():
    """Smoke against the actual iter-98 trajectory shape: CWV
    overshoot then settle in [55, 58] mm. The defaults must accept
    this — if they don't, the defaults are too tight.

    iter-104 Codex LOW#6: the iter-98 trajectory is exactly 10 days,
    matching ``DEFAULT_LAST_N_DAYS_FOR_PLATEAU``. Extend the fixture
    by 1 spin-up day so any future raise of the plateau window
    surfaces here in lockstep (instead of silently skipping the
    plateau check and still printing PASS)."""
    cwv = [44.40,  # synthetic spin-up day so len(rows) > DEFAULT_LAST_N
           49.94, 53.63, 55.67, 56.77, 57.18, 57.12, 56.85, 56.54,
           56.20, 55.87]
    assert len(cwv) > summary_mod.DEFAULT_LAST_N_DAYS_FOR_PLATEAU, (
        "fixture too short — would hollow out plateau check if "
        "DEFAULT_LAST_N_DAYS_FOR_PLATEAU is ever raised."
    )
    rows = [_row(float(i), cwv_mean=v, cwv_max=v) for i, v in enumerate(cwv)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert verdict.evaluated is True
    assert verdict.passed, (
        "iter-98 in-flight 32x32 + radiation trajectory should pass "
        f"the DOD evaluator with default thresholds. Reasons: "
        f"{verdict.reasons!r}"
    )


def test_evaluate_flags_runaway_evaporation_via_max(tmp_path):
    """iter-104 Codex MEDIUM#1: a runaway evaporation that climbs
    from 30 mm to 80 mm has plateau MEAN ≈ 55 mm (inside default
    window) but plateau MAX = 80 mm > upper bound. The max-gate
    must catch this."""
    cwv = [30.0 + i * 5.0 for i in range(11)]  # 30, 35, ..., 80
    rows = [_row(float(i), cwv_mean=v, cwv_max=v) for i, v in enumerate(cwv)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("runaway evaporation" in r for r in verdict.reasons), (
        f"runaway evaporation should be flagged via the max-gate; "
        f"reasons: {verdict.reasons!r}"
    )


def test_evaluate_exit_code_constants():
    """iter-104 Codex MEDIUM#7: lock the distinct exit codes so a
    future refactor cannot silently re-collide them."""
    assert summary_mod.EXIT_OK == 0
    assert summary_mod.EXIT_IO_ERROR == 1
    assert summary_mod.EXIT_DOD_FAIL == 3
    assert summary_mod.EXIT_DOD_INSUFFICIENT == 4
    # 1 is reserved for uncaught Python errors; 2 is reserved by
    # argparse for misuse.
    codes = {
        summary_mod.EXIT_OK,
        summary_mod.EXIT_IO_ERROR,
        summary_mod.EXIT_DOD_FAIL,
        summary_mod.EXIT_DOD_INSUFFICIENT,
    }
    assert len(codes) == 4, "exit codes must be pairwise distinct"


def test_evaluate_mse_drift_denominator_uses_max_abs():
    """iter-104 Codex LOW#2: the relative-drift denominator should
    be ``max(abs(min), abs(max), 1e-30)`` so a (synthetic) negative
    MSE array doesn't overstate the drift."""
    rows = [
        _row(
            float(i),
            mse_mean=-2.0e9 + i * 5.0e6,   # -2e9 → -1.95e9, drift = 5e7
        )
        for i in range(12)
    ]
    verdict = summary_mod.evaluate_rce_quality(rows)
    # |max - min| = 5e7; |max| = 1.95e9, |min| = 2e9.
    # Correct denom = max(1.95e9, 2e9) = 2e9; drift = 0.025 < 5 %.
    # Pre-fix denom (just abs(max)) would have been 1.95e9 → drift
    # 0.0256 — still < 5 % so the *test* doesn't trip the threshold
    # either way. So we ALSO assert no MSE drift reason was
    # appended, locking the "no false positive" branch.
    assert not any("MSE relative drift" in r for r in verdict.reasons), (
        f"MSE drift should not fire on this synthetic; reasons: "
        f"{verdict.reasons!r}"
    )


# iter-112: evaluate_rce_final_dod regression tests.


def test_final_dod_insufficient_under_30_days():
    """iter-112: trajectories shorter than min_days (default 30)
    return ``passed=False, evaluated=False`` with a reason
    explaining why. Distinct from the spinup-gate INSUFFICIENT
    (which returns passed=True for finite-checks-pass)."""
    rows = [_row(float(i)) for i in range(15)]
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert verdict.evaluated is False
    assert verdict.passed is False
    assert any("30-day DOD" in r for r in verdict.reasons), (
        f"INSUFFICIENT should cite the 30-day requirement; got: "
        f"{verdict.reasons!r}"
    )


def test_final_dod_passes_on_quiet_30_day_run():
    """A 30-day trajectory with steady CWV around 50 mm and MSE
    drift below 1 % should pass the FINAL DOD gate (tighter than
    the spinup 5 %). Drift across the last 10 days = 0.057 % —
    well under the 1 % gate."""
    rows = []
    for i in range(30):
        # Linear MSE drift = 2e6 per day on a 3.5e9 base → 0.057 %
        # over the last 10 days. Inside the 1 % gate.
        mse = 3.5e9 + i * 2.0e6
        rows.append(_row(float(i), mse_mean=mse))
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert verdict.evaluated is True
    assert verdict.passed, (
        f"30-day quiet trajectory should pass FINAL DOD; reasons: "
        f"{verdict.reasons!r}"
    )


def test_final_dod_tighter_mse_gate_than_spinup():
    """A 30-day trajectory whose MSE drift over the last 10 days
    sits BETWEEN the final-DOD gate (1 %) and the spinup gate (5 %)
    passes spinup but FAILS final-DOD.

    iter-114 Codex LOW#3 — pin the synthetic drift to the active
    constants so a future widening / tightening of either gate
    forces the test to be re-evaluated (currently target drift =
    geometric mean of the two gates ≈ 2.24 %, well inside the 1 %
    vs 5 % band)."""
    final = summary_mod.DOD_FINAL_MSE_DRIFT
    spinup = summary_mod.DEFAULT_MSE_RELATIVE_DRIFT
    # Geometric mean keeps the synthetic comfortably inside both
    # gates' band regardless of how either gate moves.
    target_drift_per_step = (final * spinup) ** 0.5 / 9.0
    rows = []
    for i in range(30):
        if i < 20:
            mse = 1.0e9
        else:
            mse = 1.0e9 * (1.0 + target_drift_per_step * (i - 20))
        rows.append(_row(float(i), mse_mean=mse))
    spinup_verdict = summary_mod.evaluate_rce_quality(rows)
    final_verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert spinup_verdict.passed, (
        f"{spinup * 100:g}% spinup gate should pass on this "
        f"trajectory; reasons: {spinup_verdict.reasons!r}"
    )
    assert not final_verdict.passed, (
        f"{final * 100:g}% final-DOD gate should FAIL on the same "
        f"trajectory; reasons: {final_verdict.reasons!r}"
    )
    assert any("MSE relative drift" in r for r in final_verdict.reasons)


def test_final_dod_constants_locked():
    """iter-112: the 1 % final-DOD MSE drift and 30-day minimum
    are constants so a future widening is visible in code review."""
    assert summary_mod.DOD_FINAL_MSE_DRIFT == 0.01
    assert summary_mod.DOD_FINAL_MIN_DAYS == 30


def test_final_dod_fails_on_runaway_evaporation_in_plateau_window():
    """iter-162 companion: the runaway-evaporation max-gate
    (iter-104 Codex MEDIUM#1) must fire for final-DOD too when
    the climb happens INSIDE the plateau window. A 30-day
    trajectory with days 0-19 stable at 50 mm then climbing
    monotonically 50 → 80 mm over days 20-29 has plateau mean
    around 65 mm (still inside (35, 65)) but plateau max =
    80 mm (above upper bound).

    Pre iter-162 only the spinup-gate variant
    (test_evaluate_flags_runaway_evaporation_via_max) was
    tested; the final-DOD wrapper was uncovered.
    """
    rows = []
    # Days 0-19: stable 50 mm with sub-mm oscillation (avoids
    # detect_sustained_stuck false-positive).
    for i in range(20):
        cwv = 50.0 + 0.01 * (i % 3 - 1)
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    # Days 20-29: monotonic runaway 50 -> 80 mm.
    for i in range(20, 30):
        cwv = 50.0 + (i - 20) * (30.0 / 9.0)
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert not verdict.passed, (
        f"final-DOD must FAIL on a runaway evaporation inside the "
        f"plateau window; reasons: {verdict.reasons!r}"
    )
    assert any("runaway evaporation" in r for r in verdict.reasons)


def test_final_dod_fails_on_transient_nan_cwv_outside_plateau_window():
    """iter-161 companion: criterion 1 (finite CWV everywhere) is
    full-window scoped. A 30-day trajectory whose last 10 days are
    a perfect plateau but with a single NaN CWV at day 5 MUST FAIL
    the final-DOD gate. Closes the third scoping case (CWV plateau
    vs CWV finite vs |U|_sfc) so all three docstring claims at
    summarize_rce_trajectory.py:695-703 are independently locked.
    """
    rows = []
    for i in range(30):
        cwv = 55.0 + 0.01 * (i % 3 - 1)
        if i == 5:
            cwv = float("nan")
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert not verdict.passed, (
        f"final-DOD must FAIL on a day-5 NaN CWV even with a "
        f"perfect last-10-day plateau. reasons: {verdict.reasons!r}"
    )
    assert any("non-finite CWV" in r for r in verdict.reasons)
    assert verdict.evaluated


def test_final_dod_fails_on_transient_wind_blowup_outside_plateau_window():
    """iter-160 companion: evaluate_rce_final_dod's docstring
    (summarize_rce_trajectory.py:700-702) promises the
    ``max_w_threshold_ms`` check is **full-window** scoped
    (``already covered``). A 30-day trajectory whose CWV plateau
    recovery is perfect (last 10 days inside Wing range) but with
    a single day-5 surface-wind blow-up to 999 m/s MUST FAIL the
    final-DOD gate. This is the inverse of the iter-160 test:

    * iter-160 (CWV check): transient excursion *outside* the
      plateau window → PASS (plateau-window scoped).
    * iter-160 companion (|U|_sfc check): transient blow-up at
      *any* row → FAIL (full-window scoped).

    Pre iter-160 only the spinup-gate |U|_sfc blow-up was tested
    (test_evaluate_flags_max_w_blowup); the final-DOD wrapper was
    uncovered for the same scenario.
    """
    rows = []
    # 30 days, all CWV inside plateau, mostly low surface wind.
    for i in range(30):
        cwv = 55.0 + 0.01 * (i % 3 - 1)
        wind = 0.01 if i != 5 else 999.0  # day-5 spike
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv,
                         wind_sfc_max=wind))
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert not verdict.passed, (
        f"final-DOD must FAIL on a day-5 |U|_sfc blow-up even "
        f"with a perfect last-10-day CWV plateau. reasons: "
        f"{verdict.reasons!r}"
    )
    assert any("|U|_sfc exceeded" in r for r in verdict.reasons)
    assert verdict.evaluated


def test_final_dod_passes_with_early_cwv_excursion_outside_plateau_window():
    """iter-159: evaluate_rce_final_dod's docstring (lines 695-703
    of summarize_rce_trajectory.py) promises that the plateau CWV
    check inspects only the LAST ``last_n_days_for_plateau`` rows
    (default 10). A 30-day trajectory whose CWV is OUTSIDE the
    Wing 2018 plateau range for days 0-19 but recovers to the
    plateau over days 20-29 must therefore PASS the final-DOD
    gate.

    This behaviour was uncovered pre-iter-159 — only the strict-
    pass case (entire trajectory in range) was exercised. A
    regression that changed the plateau check to scan all rows
    (instead of just the trailing window) would silently break
    real recovery-from-overshoot trajectories like iter-98's
    day-4 spike to 57.18 mm.
    """
    # iter-159: drift the leading rows monotonically to evade the
    # detect_stuck_trajectory leading-window check (range > 0.001 mm
    # over the first 3 rows). Hold at 70 mm for the body, then drop
    # to a 55 mm plateau over the final 10 days.
    rows = []
    # Days 0-2: monotonic drift 70.0 -> 69.9 -> 69.8 (range 0.2 mm >
    # detect_stuck tol so not flagged as Bug 2).
    for i in range(3):
        cwv = 70.0 - 0.1 * i
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    # Days 3-19: hold around 70 mm (well above Wing upper bound 65 mm)
    # with sub-mm oscillation to evade detect_sustained_stuck_trajectory
    # (which flags bit-equal CWV across 3 consecutive post-spinup days).
    for i in range(3, 20):
        cwv = 70.0 + 0.01 * (i % 3 - 1)  # 69.99..70.01
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    # Days 20-29: plateau at 55 mm (inside Wing 45-60 mm band) with
    # sub-1 % MSE drift.
    for i in range(20, 30):
        cwv = 55.0 + 0.02 * (i % 3 - 1)  # 54.98..55.02 oscillation
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    verdict = summary_mod.evaluate_rce_final_dod(rows)
    assert verdict.passed, (
        f"final-DOD must pass on a recovery-from-overshoot "
        f"trajectory: early days outside plateau, last 10 days "
        f"inside (45, 60). reasons: {verdict.reasons!r}"
    )
    assert verdict.evaluated


def test_exit_usage_constant():
    """iter-114 Codex LOW#2: EXIT_USAGE=2 (argparse convention) is a
    distinct constant from EXIT_IO_ERROR=1; CLI misuse no longer
    collides with IO errors."""
    assert summary_mod.EXIT_USAGE == 2
    assert summary_mod.EXIT_USAGE != summary_mod.EXIT_IO_ERROR
    assert summary_mod.EXIT_USAGE != summary_mod.EXIT_DOD_FAIL
    assert summary_mod.EXIT_USAGE != summary_mod.EXIT_DOD_INSUFFICIENT


def test_evaluate_and_final_dod_mutually_exclusive_exit_code(tmp_path):
    """iter-114 Codex LOW#2: passing both --evaluate AND --final-dod
    must exit with EXIT_USAGE (2), not the SystemExit(str) default
    of 1 (which collides with EXIT_IO_ERROR)."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--evaluate",
            "--final-dod",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_USAGE, (
        f"Mutually-exclusive arg violation should exit "
        f"EXIT_USAGE={summary_mod.EXIT_USAGE}, got "
        f"{res.returncode}. stderr={res.stderr!r}"
    )


# iter-117: detect_stuck_trajectory regression tests.


def test_detect_stuck_trajectory_flags_pinned_cwv():
    """Pre-iter-95 Bug 2 signature: CWV pinned at IC for 11+ hours
    by the unconditional fix_moist_mass_plane rescaling. Synthetic
    fixture mirrors this with 3 consecutive snapshots all at
    49.9413 mm."""
    rows = [
        _row(0.0, cwv_mean=49.9413, cwv_max=49.9413),
        _row(1.0, cwv_mean=49.9413, cwv_max=49.9413),
        _row(2.0, cwv_mean=49.9413, cwv_max=49.9413),
    ]
    stuck, reason = summary_mod.detect_stuck_trajectory(rows)
    assert stuck is True
    assert reason is not None
    assert "pinned" in reason
    assert "Bug 2" in reason, (
        f"reason should reference Bug 2 for future debuggers; "
        f"got {reason!r}"
    )


def test_detect_stuck_trajectory_late_equilibrium_not_flagged():
    """iter-118 Codex HIGH fix: a real late-equilibrium plateau
    where CWV drift falls below 0.001 mm/day in the LATE part of a
    30-day run must NOT trip the stuck gate. The pre-iter-118
    sliding-window implementation false-positived here. The fix
    scopes the check to the LEADING window only — Bug 2 always
    pins CWV from day 0 onward."""
    # Days 0-9: spinup (49.94 → 56.55 mm, ~0.7 mm/day average drift).
    # Days 10-29: late equilibrium plateau with sub-tolerance drift.
    spinup = [49.94, 50.5, 51.3, 52.2, 53.6, 55.0, 56.0, 56.3, 56.4, 56.5]
    late_equilibrium = [56.5000 + 0.0001 * (i % 3 - 1) for i in range(20)]
    cwv = spinup + late_equilibrium
    rows = [_row(float(i), cwv_mean=v, cwv_max=v) for i, v in enumerate(cwv)]
    stuck, reason = summary_mod.detect_stuck_trajectory(rows)
    assert stuck is False, (
        f"Late equilibrium with sub-tolerance drift must NOT trip "
        f"the stuck gate (false positive would chase a non-bug). "
        f"reason={reason!r}"
    )


def test_detect_stuck_trajectory_passes_iter98_shape():
    """iter-98 in-flight 32x32 + radiation trajectory has CWV
    49.94 -> 53.63 -> 55.67 ... — far from stuck. Must NOT trip."""
    cwv = [49.9413, 53.6327, 55.6672, 56.7736, 57.1817]
    rows = [_row(float(i), cwv_mean=v, cwv_max=v) for i, v in enumerate(cwv)]
    stuck, _ = summary_mod.detect_stuck_trajectory(rows)
    assert not stuck


def test_detect_stuck_trajectory_undecided_on_short_trajectory():
    """Fewer rows than ``consecutive_days`` returns (False, None) —
    cannot decide yet, NOT a false negative."""
    rows = [_row(0.0, cwv_mean=49.9413), _row(1.0, cwv_mean=49.9413)]
    stuck, reason = summary_mod.detect_stuck_trajectory(rows)
    assert stuck is False
    assert reason is None


def test_detect_stuck_trajectory_tolerates_micro_drift():
    """A trajectory that drifts by ~1e-5 mm/day (rounding noise)
    is still effectively stuck — the default 0.001 mm tolerance
    catches it. Real spin-up is 3-4 mm/day so the tolerance has
    3+ orders of magnitude margin."""
    rows = [
        _row(0.0, cwv_mean=49.9413000),
        _row(1.0, cwv_mean=49.9413001),
        _row(2.0, cwv_mean=49.9413002),
    ]
    stuck, _ = summary_mod.detect_stuck_trajectory(rows)
    assert stuck is True


def test_stuck_detector_constants_locked():
    """Lock the defaults so a future widen is visible in code review."""
    assert summary_mod.DEFAULT_STUCK_CONSECUTIVE_DAYS == 3
    assert summary_mod.DEFAULT_STUCK_CWV_TOL_MM == 0.001


def test_evaluate_quality_wires_stuck_detector():
    """iter-117: evaluate_rce_quality must flag a stuck trajectory
    as a FAIL reason. The full evaluator should catch Bug 2
    regressions END-TO-END, not just via a separate diagnostic.

    iter-118: stuck reason now says ``pinned`` (was ``stuck``)."""
    rows = [
        _row(float(i), cwv_mean=49.9413, cwv_max=49.9413)
        for i in range(12)
    ]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("pinned" in r for r in verdict.reasons), (
        f"evaluate_rce_quality should surface the stuck detector's "
        f"reason; got {verdict.reasons!r}"
    )


def test_evaluate_quality_check_stuck_opt_out():
    """iter-118 Codex MEDIUM#1 fix: ``check_stuck=False`` lets
    callers run only the plateau gates without the leading-window
    stuck check. Useful when comparing two known-equilibrated
    trajectories where the leading window is intentionally
    constant (e.g. a continuation from the same restart)."""
    rows = [
        _row(0.0, cwv_mean=49.9413, cwv_max=49.9413),
        _row(1.0, cwv_mean=49.9413, cwv_max=49.9413),
        _row(2.0, cwv_mean=49.9413, cwv_max=49.9413),
    ]
    # Default: stuck-detect fires.
    on_verdict = summary_mod.evaluate_rce_quality(rows)
    assert any("pinned" in r for r in on_verdict.reasons)
    # Opt-out: stuck-detect skipped; reasons may still list other
    # failures (CWV out of range etc.) but NOT the pinned reason.
    off_verdict = summary_mod.evaluate_rce_quality(
        rows, check_stuck=False,
    )
    assert not any("pinned" in r for r in off_verdict.reasons), (
        f"check_stuck=False must suppress the pinned-CWV reason; "
        f"got {off_verdict.reasons!r}"
    )


def test_collect_trajectory_accepts_string_path(synthetic_run: Path):
    """iter-119: ``collect_trajectory`` accepts ``str`` as well as
    ``Path`` (an ergonomic fix for CLI / shell callers). Pre-iter-119
    a string path raised ``TypeError: unsupported operand type(s)
    for /`` deep inside the function."""
    rows = summary_mod.collect_trajectory(str(synthetic_run))
    assert [r.day for r in rows] == [0.0, 1.0, 2.0]


# iter-120 Codex MEDIUM: sustained-stuck detector tests.


def test_sustained_stuck_flags_mid_run_pin():
    """A trajectory that drifts normally for days 0-4 then bit-pins
    for days 5-8 (CWV literally identical) must be flagged as
    sustained-stuck. Models a mass-fixer regression that kicks in
    mid-run."""
    spinup = [49.94, 50.5, 51.3, 52.2, 53.6]
    stuck = [53.6, 53.6, 53.6, 53.6]  # bit-equal post-spinup
    cwv = spinup + stuck
    rows = [_row(float(i), cwv_mean=v, cwv_max=v)
            for i, v in enumerate(cwv)]
    flagged, reason = summary_mod.detect_sustained_stuck_trajectory(rows)
    assert flagged is True
    assert reason is not None
    assert "AFTER spinup" in reason


def test_sustained_stuck_passes_late_equilibrium_oscillation():
    """Late equilibrium oscillates at ~1e-3 mm (well above the
    sustained 1e-7 mm tolerance). Must NOT trip."""
    rows = []
    for i in range(15):
        # Sub-day oscillation at 0.01 mm scale — legitimate
        # convection / radiation balance noise.
        cwv = 50.0 + 0.01 * ((i * 7) % 5 - 2)  # 49.98..50.02
        rows.append(_row(float(i), cwv_mean=cwv, cwv_max=cwv))
    flagged, _ = summary_mod.detect_sustained_stuck_trajectory(rows)
    assert flagged is False


def test_sustained_stuck_undecided_on_short_trajectory():
    """Need at least 2 * consecutive_days rows to distinguish
    leading window from sustained tail. Shorter returns
    (False, None)."""
    rows = [_row(float(i), cwv_mean=50.0 + 0.1 * i) for i in range(4)]
    flagged, reason = summary_mod.detect_sustained_stuck_trajectory(rows)
    assert flagged is False
    assert reason is None


def test_sustained_stuck_returns_undecided_when_finite_rows_short():
    """iter-151: NaN cwv_mean rows must be filtered out, and if the
    survivor count drops below ``2 * consecutive_days`` the detector
    must return ``(False, None)`` — undecided, not False-Positive.

    Constructs 6 total rows (passes the first len check at
    ``2 * 3 = 6``) of which 4 are NaN — only 2 finite rows, well
    below the 6-row finite threshold. Exercises the second guard at
    summarize_rce_trajectory.py:636.
    """
    finite = [_row(0.0, cwv_mean=50.0, cwv_max=50.0),
              _row(5.0, cwv_mean=53.0, cwv_max=53.0)]
    nan_rows = [_row(float(i), cwv_mean=float("nan"),
                     cwv_max=float("nan"))
                for i in (1, 2, 3, 4)]
    rows = [finite[0]] + nan_rows + [finite[1]]
    flagged, reason = summary_mod.detect_sustained_stuck_trajectory(rows)
    assert flagged is False
    assert reason is None


def test_stuck_returns_undecided_when_finite_rows_short():
    """Companion test for ``detect_stuck_trajectory``: NaN rows
    filtered out, survivor count below ``consecutive_days`` returns
    ``(False, None)`` rather than tripping on the leading-window
    check. Exercises summarize_rce_trajectory.py:584.
    """
    rows = [_row(0.0, cwv_mean=50.0, cwv_max=50.0),
            _row(1.0, cwv_mean=float("nan"), cwv_max=float("nan")),
            _row(2.0, cwv_mean=float("nan"), cwv_max=float("nan"))]
    flagged, reason = summary_mod.detect_stuck_trajectory(rows)
    assert flagged is False
    assert reason is None


def test_sustained_stuck_constants_locked():
    """Lock the 1e-7 mm tight tolerance so a future widen surfaces
    in code review (a 1e-3 mm value would catch real equilibrium
    oscillations as false positives)."""
    assert summary_mod.DEFAULT_SUSTAINED_STUCK_CWV_TOL_MM == 1e-7


def test_evaluate_quality_wires_sustained_stuck_detector():
    """evaluate_rce_quality must surface sustained-stuck reasons
    on the verdict.reasons list."""
    spinup = [49.94, 50.5, 51.3, 52.2, 53.6]
    stuck = [53.6, 53.6, 53.6, 53.6]
    cwv = spinup + stuck
    rows = [_row(float(i), cwv_mean=v, cwv_max=v)
            for i, v in enumerate(cwv)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert not verdict.passed
    assert any("AFTER spinup" in r for r in verdict.reasons)


# iter-127: --no-plateau-check CLI behaviour.


def test_no_plateau_check_folds_insufficient_into_pass(tmp_path):
    """iter-127: a 3-day trajectory normally returns INSUFFICIENT
    (exit 4) from --evaluate. With --no-plateau-check the plateau
    branch is skipped and only stability checks run; PASS exits 0."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(3):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.5 * i,
        )
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--evaluate", "--no-plateau-check",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_OK, (
        f"--no-plateau-check should fold INSUFFICIENT into PASS on "
        f"a 3-day finite trajectory; got exit {res.returncode}, "
        f"stderr={res.stderr!r}"
    )
    # iter-130 Codex MEDIUM: --no-plateau-check uses a distinct
    # ``DOD STABILITY`` label so the verdict is not confused with a
    # full plateau-validated PASS.
    assert "DOD STABILITY verdict: PASS" in res.stdout


def test_no_plateau_check_still_fails_on_blowup(tmp_path):
    """iter-127: --no-plateau-check skips the plateau branch but
    leaves max|U|_sfc / NaN / stuck-detector gates intact. A 3-day
    trajectory with max|U|_sfc > 50 m/s must STILL FAIL."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(3):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.5 * i,
        )
    # Mutate the day-1 snapshot to plant a blow-up wind value via
    # the snapshot writer's wind_sfc factor: the iter-126 helper
    # uses ``day * 0.01`` so day=5000 → wind=50 m/s. Replace day 1.
    import numpy as np
    ny, nx = 4, 4
    zeros = np.zeros((ny, nx), dtype=np.float64)
    np.savez_compressed(
        snaps / "snap_day_0001.npz",
        t_sim=86400.0, day=1.0,
        cwv=np.full((ny, nx), 50.5), mse=np.full((ny, nx), 3.5e9),
        precip=zeros,
        T_sfc=np.full((ny, nx), 300.0),
        qv_sfc=np.full((ny, nx), 0.02),
        qc_sfc=zeros, qr_sfc=zeros,
        u_sfc=zeros, v_sfc=zeros,
        wind_sfc=np.full((ny, nx), 99.0),  # > 50 m/s blow-up.
    )
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--evaluate", "--no-plateau-check",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_DOD_FAIL, (
        f"--no-plateau-check should still FAIL on max|U|_sfc "
        f"blow-up; got exit {res.returncode}, "
        f"stdout={res.stdout!r}"
    )
    # iter-130 Codex MEDIUM: --no-plateau-check uses the DOD
    # STABILITY label on both PASS and FAIL branches.
    assert "DOD STABILITY verdict: FAIL" in res.stdout
    assert "|U|_sfc exceeded" in res.stdout


def test_no_plateau_check_rejected_with_final_dod(tmp_path):
    """iter-127: --no-plateau-check + --final-dod is rejected by
    argparse (EXIT_USAGE=2) because --final-dod's plateau check IS
    the DOD criterion 2 verdict — skipping it would defeat the
    purpose."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--final-dod", "--no-plateau-check",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_USAGE


def test_summarize_quiet_suppresses_table(tmp_path):
    """iter-128: --quiet flag skips the per-day fixed-width table
    but still emits trajectory.csv + the ``wrote ...`` summary
    line."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(3):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.5 * i,
        )
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    # Summary line still prints.
    assert "wrote" in res.stdout
    assert "(3 rows)" in res.stdout
    # Table header column (e.g. ``CWV_mean[mm]``) should NOT be in
    # stdout because the table was suppressed.
    assert "CWV_mean[mm]" not in res.stdout, (
        f"--quiet should suppress the table; stdout={res.stdout!r}"
    )
    # CSV still written.
    csv = tmp_path / "trajectory.csv"
    assert csv.exists()


def test_quiet_with_evaluate_still_prints_verdict(tmp_path):
    """iter-134: --quiet + --evaluate must still print the DOD
    verdict line on stdout (the verdict is the actionable signal
    even when the table is suppressed). The "wrote N rows" line +
    verdict ARE the entire stdout under --quiet."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(12):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.1 * i,
        )
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--quiet", "--evaluate",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_OK
    # Table column header (e.g. ``CWV_mean[mm]``) suppressed.
    assert "CWV_mean[mm]" not in res.stdout
    # Wrote-line still emits.
    assert "wrote" in res.stdout
    # DOD verdict still emits.
    assert "DOD verdict: PASS" in res.stdout


def test_quiet_with_no_plateau_check_prints_stability_verdict(tmp_path):
    """iter-134: --quiet + --evaluate + --no-plateau-check must
    still print the iter-130 ``DOD STABILITY`` label (table
    suppressed, verdict line preserved). Locks the verdict-print
    flow across all three flags."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(3):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.5 * i,
        )
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--quiet", "--evaluate", "--no-plateau-check",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_OK
    assert "CWV_mean[mm]" not in res.stdout
    assert "DOD STABILITY verdict: PASS" in res.stdout


# iter-137: --check-log-max-w + parse_log_max_w tests.


def _write_log_txt(path: Path, max_w_values: list[float]) -> None:
    """Build a minimal log.txt matching run_rce_mpi_long.py's
    schema. ``max_w_values`` populates the max|w| column row by
    row."""
    lines = [
        "# RCE MPI LONG  n_ranks=1 grid=4x4 nlev=10 dx=4000.0 dt=10.0 days=1.0 total_steps=8640",
        "# physics: gray radiation + Kessler microphysics",
        "# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep",
    ]
    for i, val in enumerate(max_w_values):
        step = (i + 1) * 100
        day = step / 8640.0
        lines.append(
            f"{step},{day:.6f},50.0,50.0,3.5e+09,{val:.4e},"
            f"0.0,0.0,0.0,0.4370"
        )
    path.write_text("\n".join(lines) + "\n")


def test_parse_log_max_w_returns_max_across_all_rows(tmp_path):
    """iter-137: parse_log_max_w scans every data row in log.txt
    and returns the maximum |w| seen."""
    log_path = tmp_path / "log.txt"
    _write_log_txt(log_path, [1e-3, 5e-3, 2e-3, 8e-3, 4e-3])
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(8e-3)
    assert n_rows == 5


def test_parse_log_max_w_missing_log_returns_zero(tmp_path):
    """No log.txt → (0.0, 0). Caller can check ``n_rows == 0`` to
    distinguish from "all rows were 0"."""
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == 0.0
    assert n_rows == 0


def test_parse_log_max_w_rejects_nan(tmp_path):
    """NaN in max|w| column raises ValueError — silent NaN-blow-up
    would otherwise pass through ``max(0.0, nan) == 0.0`` (iter-66
    helper bug pattern). iter-137 catches this at log-parse time."""
    log_path = tmp_path / "log.txt"
    _write_log_txt(log_path, [1e-3, float("nan"), 2e-3])
    with pytest.raises(ValueError, match="non-finite max"):
        summary_mod.parse_log_max_w(tmp_path)


def test_parse_log_max_w_rejects_pos_inf(tmp_path):
    """+Inf in max|w| column raises ValueError. The summarizer uses
    ``math.isfinite`` which rejects both NaN and +/-Inf; the NaN path
    is tested above, this covers the +Inf branch a runaway-w blowup
    would actually produce (e.g. an `1/0` divergence in the dycore)."""
    log_path = tmp_path / "log.txt"
    _write_log_txt(log_path, [1e-3, float("inf"), 2e-3])
    with pytest.raises(ValueError, match="non-finite max"):
        summary_mod.parse_log_max_w(tmp_path)


def test_parse_log_max_w_rejects_neg_inf(tmp_path):
    """-Inf in max|w| column also raises. The max|w| column is the
    absolute-value norm so -Inf should never occur physically, but
    ``math.isfinite`` is symmetric — guard against a bug where the
    driver prints a signed value by accident."""
    log_path = tmp_path / "log.txt"
    _write_log_txt(log_path, [1e-3, float("-inf"), 2e-3])
    with pytest.raises(ValueError, match="non-finite max"):
        summary_mod.parse_log_max_w(tmp_path)


def test_check_log_max_w_passes_on_quiet_run(tmp_path):
    """End-to-end CLI: --check-log-max-w on a finished run with
    max|w| < 50 m/s prints the log max + does NOT fail."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    _write_log_txt(tmp_path / "log.txt", [1e-3, 5e-3, 2e-3])
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--check-log-max-w", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "log max|w| = 5.0000e-03" in res.stdout


def test_check_log_max_w_fails_on_blowup(tmp_path):
    """--check-log-max-w on a run where log max|w| > 50 m/s
    exits EXIT_DOD_FAIL and prints the criterion-1 reason."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    _write_log_txt(tmp_path / "log.txt", [1e-3, 99.0, 2e-3])
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--check-log-max-w", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_DOD_FAIL
    assert "DOD criterion 1 FAIL" in res.stdout


# iter-139: HIGH/MEDIUM Codex iter-137/138 follow-up tests.


def test_parse_log_max_w_accepts_hashless_schema(tmp_path):
    """iter-139 MEDIUM#2: future driver schema drift to ``#step,``
    (no space after ``#``) must still parse."""
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        "# RCE MPI LONG header\n"
        "#step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        "100,0.01,50.0,50.0,3.5e+09,5.0e-03,0,0,0,0.4\n"
        "200,0.02,50.0,50.0,3.5e+09,8.0e-03,0,0,0,0.4\n"
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(8e-3)
    assert n_rows == 2


def test_parse_log_max_w_nan_message_includes_step(tmp_path):
    """iter-139 MEDIUM#6: NaN ValueError must include the file line
    number + the simulation ``step`` value so a debugger can jump
    straight to the offending point."""
    log_path = tmp_path / "log.txt"
    _write_log_txt(log_path, [1e-3, 5e-3, float("nan"), 2e-3])
    with pytest.raises(ValueError) as excinfo:
        summary_mod.parse_log_max_w(tmp_path)
    msg = str(excinfo.value)
    # The NaN row is index 2 of max_w_values → step (i+1)*100 = 300.
    assert "step 300" in msg, f"NaN reason missing step value; got: {msg!r}"
    assert "line " in msg, f"NaN reason missing line number; got: {msg!r}"


def test_check_log_max_w_fails_on_missing_log(tmp_path):
    """iter-139 HIGH#3: missing log.txt + --check-log-max-w must
    fail loudly (cannot certify DOD criterion 1 without telemetry).
    Pre-iter-139 this branch printed ``log max|w| = 0.0 over 0
    log rows`` and silently exited 0."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps / "snap_day_0000.npz", day=0.0, cwv_value=50.0)
    # NO log.txt written.
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--check-log-max-w", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_DOD_FAIL, (
        f"missing log.txt + --check-log-max-w should exit "
        f"EXIT_DOD_FAIL; got {res.returncode}, stdout={res.stdout!r}"
    )
    assert "log.txt missing or empty" in res.stdout


def test_parse_log_max_w_streams_large_log(tmp_path):
    """iter-139 HIGH#1: parse_log_max_w must stream the file, not
    slurp via read_text(). This test writes a 5000-row log (small
    enough for the test runner but large enough to exercise the
    line-by-line iteration path).

    We can't truly assert ``read_text()`` is not used without
    monkeypatching ``Path.read_text``; instead the test verifies
    the function returns the correct max for a moderately large
    log without inflating memory enough to crash CI."""
    log_path = tmp_path / "log.txt"
    # Build 5000 rows; max|w| = 12.0 at row 2500.
    rows = []
    for i in range(5000):
        step = (i + 1) * 100
        day = step / 8640.0
        val = 12.0 if i == 2500 else (i % 100) * 1e-4
        rows.append(
            f"{step},{day:.6f},50.0,50.0,3.5e+09,{val:.4e},"
            f"0.0,0.0,0.0,0.4370"
        )
    log_path.write_text(
        "# RCE MPI LONG header\n"
        "# physics line\n"
        "# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        + "\n".join(rows) + "\n"
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(12.0)
    assert n_rows == 5000


def test_final_dod_cli_emits_dod_final_label_on_pass(tmp_path):
    """iter-167: a --final-dod CLI invocation on a 30-day PASS
    trajectory must print ``DOD FINAL verdict: PASS`` and exit 0.
    Pre iter-167 the ``label = "DOD FINAL"`` branch in main()
    (summarize_rce_trajectory.py:864) was uncovered at the CLI
    level — only the Python function evaluate_rce_final_dod was
    directly tested. A regression that swapped the label to
    something else (e.g. ``DOD``) would still pass at the function
    level but break log scrapers that grep for ``DOD FINAL``.
    """
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(30):
        # Sub-mm CWV oscillation around 55 mm avoids
        # detect_stuck / detect_sustained_stuck and keeps the
        # plateau check inside (35, 65) mm. MSE drift over the
        # last 10 days stays well under 1 % (driven by the
        # _write_snapshot mse = cwv * 1e7 mapping).
        cwv = 55.0 + 0.01 * (i % 3 - 1)
        _write_snapshot(snaps / f"snap_day_{i:04d}.npz",
                        day=float(i), cwv_value=cwv)
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--final-dod", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (
        f"30-day PASS trajectory should exit 0; got "
        f"{res.returncode}; stdout={res.stdout!r}; "
        f"stderr={res.stderr!r}"
    )
    assert "DOD FINAL verdict: PASS" in res.stdout, (
        f"CLI must print 'DOD FINAL' label, not 'DOD' or "
        f"'DOD STABILITY'; stdout={res.stdout!r}"
    )


def test_final_dod_cli_exits_dod_fail_on_runaway_evaporation(tmp_path):
    """iter-168: --final-dod CLI on a 30-day trajectory whose
    plateau window shows runaway evaporation (CWV climbing past
    the Wing upper bound) exits EXIT_DOD_FAIL (3) with
    ``DOD FINAL verdict: FAIL`` + the runaway-evaporation reason.

    Locks the FAIL exit path (summarize_rce_trajectory.py:907)
    for the --final-dod CLI wrapper. Pre iter-168 only PASS +
    INSUFFICIENT were CLI-tested.
    """
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(30):
        if i < 20:
            cwv = 55.0 + 0.01 * (i % 3 - 1)
        else:
            # Days 20-29: monotonic runaway 55 -> 80 mm.
            cwv = 55.0 + (i - 20) * (25.0 / 9.0)
        _write_snapshot(snaps / f"snap_day_{i:04d}.npz",
                        day=float(i), cwv_value=cwv)
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--final-dod", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_DOD_FAIL, (
        f"runaway evaporation should exit EXIT_DOD_FAIL "
        f"({summary_mod.EXIT_DOD_FAIL}); got {res.returncode}; "
        f"stdout={res.stdout!r}"
    )
    assert "DOD FINAL verdict: FAIL" in res.stdout, (
        f"CLI must print 'DOD FINAL verdict: FAIL'; "
        f"stdout={res.stdout!r}"
    )
    assert "runaway evaporation" in res.stdout


def test_final_dod_cli_exits_insufficient_on_short_run(tmp_path):
    """iter-167: --final-dod on a < 30-day trajectory exits
    EXIT_DOD_INSUFFICIENT (4) with ``DOD FINAL verdict:
    INSUFFICIENT``. Distinct from EXIT_DOD_FAIL (3) so automation
    can tell "too short to grade" apart from "graded and failed".
    """
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(15):  # 15 < DOD_FINAL_MIN_DAYS (30)
        _write_snapshot(snaps / f"snap_day_{i:04d}.npz",
                        day=float(i), cwv_value=55.0)
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path), "--final-dod", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == summary_mod.EXIT_DOD_INSUFFICIENT, (
        f"15-day trajectory + --final-dod should exit "
        f"EXIT_DOD_INSUFFICIENT ({summary_mod.EXIT_DOD_INSUFFICIENT}); "
        f"got {res.returncode}; stdout={res.stdout!r}"
    )
    assert "DOD FINAL verdict: INSUFFICIENT" in res.stdout, (
        f"CLI must print 'DOD FINAL verdict: INSUFFICIENT'; "
        f"stdout={res.stdout!r}"
    )


def test_combined_final_dod_and_check_log_max_w_passes_30day(tmp_path):
    """iter-169: ``--final-dod --check-log-max-w`` is the exact
    production-recommended combo (per the wrapper iter-124 hint
    + iter-156 wrapper-level regression). At the summarizer CLI
    level: both gates fire, log_max_w line precedes the DOD FINAL
    verdict, and a 30-day PASS trajectory exits 0.

    Pre iter-169 only ``--evaluate --check-log-max-w`` was tested
    at the summarizer CLI level. The wrapper test confirms argv
    threading; this confirms summarizer behaviour on the actual
    flag combination iter-105 reports against.
    """
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(30):
        cwv = 55.0 + 0.01 * (i % 3 - 1)
        _write_snapshot(snaps / f"snap_day_{i:04d}.npz",
                        day=float(i), cwv_value=cwv)
    _write_log_txt(tmp_path / "log.txt", [1e-3, 5e-3, 2e-3, 8e-3])
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--final-dod", "--check-log-max-w", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (
        f"30-day PASS with both gates should exit 0; got "
        f"{res.returncode}; stdout={res.stdout!r}; "
        f"stderr={res.stderr!r}"
    )
    lines = res.stdout.splitlines()
    log_line_idx = next(
        i for i, ln in enumerate(lines) if "log max|w|" in ln
    )
    verdict_line_idx = next(
        i for i, ln in enumerate(lines) if "DOD FINAL verdict:" in ln
    )
    assert log_line_idx < verdict_line_idx, (
        f"log max|w| must precede DOD FINAL verdict on the same "
        f"run; log line {log_line_idx} verdict line "
        f"{verdict_line_idx}; stdout={res.stdout!r}"
    )
    assert "DOD FINAL verdict: PASS" in res.stdout
    # max([1e-3, 5e-3, 2e-3, 8e-3]) = 8e-3
    assert "8.0000e-03" in res.stdout


def test_combined_evaluate_and_check_log_max_w(tmp_path):
    """iter-140: --evaluate + --check-log-max-w fire BOTH gates
    (criterion 1 via log + criterion 2 via plateau). Locks the
    order: log_max_w prints first, then DOD verdict. On PASS both
    branches print, exit 0."""
    import subprocess
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(12):
        _write_snapshot(
            snaps / f"snap_day_{i:04d}.npz",
            day=float(i), cwv_value=50.0 + 0.1 * i,
        )
    _write_log_txt(tmp_path / "log.txt", [1e-3, 5e-3, 2e-3, 8e-3])
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[4]
                / "scripts" / "validate" / "summarize_rce_trajectory.py"),
            str(tmp_path),
            "--evaluate", "--check-log-max-w", "--quiet",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    # iter-142 Codex LOW#3 fix: compare LINE indices, not character
    # offsets. A future refactor merging both tokens onto a single
    # line would otherwise pass the character-offset check
    # vacuously. Find the line each token lives on + assert
    # log-line-idx < verdict-line-idx + they're on different lines.
    lines = res.stdout.splitlines()
    log_line_idx = next(
        i for i, ln in enumerate(lines) if "log max|w|" in ln
    )
    verdict_line_idx = next(
        i for i, ln in enumerate(lines) if "DOD verdict:" in ln
    )
    assert log_line_idx < verdict_line_idx, (
        f"Expected log max|w| BEFORE DOD verdict; "
        f"log line {log_line_idx} verdict line {verdict_line_idx}; "
        f"stdout={res.stdout!r}"
    )
    assert log_line_idx != verdict_line_idx, (
        f"log max|w| and DOD verdict must be on SEPARATE lines; "
        f"both found on line {log_line_idx}: "
        f"{lines[log_line_idx]!r}"
    )
    assert "DOD verdict: PASS" in res.stdout
    # max([1e-3, 5e-3, 2e-3, 8e-3]) = 8e-3
    assert "8.0000e-03" in res.stdout


def test_parse_log_max_w_returns_sentinel_on_schema_free_log(tmp_path):
    """iter-142 Codex Q1 gap: a log.txt with no ``# step,`` schema
    row returns (0.0, 0) — same sentinel as missing log. Callers
    using --check-log-max-w then fail per the iter-139 HIGH#3 fix."""
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        "# header comment only\n"
        "100,0.01,50.0,50.0,3.5e+09,5.0e-03,0,0,0,0.4\n"
        "200,0.02,50.0,50.0,3.5e+09,8.0e-03,0,0,0,0.4\n"
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == 0.0
    assert n_rows == 0


def test_parse_log_max_w_handles_utf8_content(tmp_path):
    """iter-142 Codex MEDIUM#5: log.txt with non-ASCII characters
    must parse without UnicodeDecodeError on platforms where the
    default codec isn't UTF-8."""
    log_path = tmp_path / "log.txt"
    # Embed a degree sign ° (U+00B0) in a comment line.
    log_path.write_text(
        "# RCE MPI LONG  T_sfc = 300 °C target\n"
        "# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        "100,0.01,50.0,50.0,3.5e+09,5.0e-03,0,0,0,0.4\n",
        encoding="utf-8",
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(5e-3)
    assert n_rows == 1


# iter-154: parse_log_max_w robustness against corrupt rows. The
# implementation has two distinct "skip + keep going" branches at
# scripts/validate/summarize_rce_trajectory.py:504 (column-count mismatch)
# and 508 (non-numeric float parse). Both surface in real runs:
# - col-count mismatch when an MPI rank crash truncates a row mid-
#   write (the driver appends row-at-a-time but flushes after each
#   write, so a SIGKILL between separator and newline yields a row
#   with too few cells);
# - ValueError when the driver substitutes a sentinel string (e.g.
#   ``"NaN"`` from an older format) that ``float()`` parses fine
#   but the math.isfinite guard catches — only a genuinely non-
#   numeric token reaches the ValueError continue.


def test_parse_log_max_w_skips_truncated_row(tmp_path):
    """A row with fewer cells than the header is skipped — the
    parser keeps going and returns the max from the surviving
    valid rows. Models a partial-write from an MPI rank crash."""
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        "# RCE MPI LONG short\n"
        "# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        "100,0.01,50.0,50.0,3.5e+09,5.0e-03,0,0,0,0.4\n"
        "200,0.02,50.0,50.0,3.5e+09,\n"  # truncated
        "300,0.03,50.0,50.0,3.5e+09,9.0e-03,0,0,0,0.4\n"
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(9e-3)
    assert n_rows == 2


def test_parse_log_max_w_skips_non_numeric_cell(tmp_path):
    """A cell that ``float()`` rejects with ValueError (e.g. a
    stray ASCII word) is skipped silently — the parser keeps
    going and returns the max from valid rows. Distinct from the
    NaN / +Inf branches above (those parse but fail isfinite)."""
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        "# RCE MPI LONG corrupt\n"
        "# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        "100,0.01,50.0,50.0,3.5e+09,5.0e-03,0,0,0,0.4\n"
        "200,0.02,50.0,50.0,3.5e+09,corrupt,0,0,0,0.4\n"
        "300,0.03,50.0,50.0,3.5e+09,9.0e-03,0,0,0,0.4\n"
    )
    max_w, n_rows = summary_mod.parse_log_max_w(tmp_path)
    assert max_w == pytest.approx(9e-3)
    assert n_rows == 2
