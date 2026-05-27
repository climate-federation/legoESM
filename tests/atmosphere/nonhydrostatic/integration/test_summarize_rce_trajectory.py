"""Unit tests for ``scripts/summarize_rce_trajectory.py``.

iter-98: a tool tracking per-day RCE diagnostics is only useful if its
field extraction matches ``run_rce_mpi_long.py``'s snapshot writer.
These tests build synthetic ``snap_day_*.npz`` + ``prof_day_*.npz``
files with the exact key set the production driver writes (verified
against ``scripts/run_rce_mpi_long.py:save_snapshot_2d`` and
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
    ny, nx = 4, 4
    arr = np.full((ny, nx), cwv_value, dtype=np.float64)
    np.savez_compressed(
        path,
        t_sim=day * 86400.0,
        day=day,
        cwv=arr,
        mse=arr * 1e7,
        precip=arr * 0.0,
        T_sfc=np.full_like(arr, 300.0 - day * 0.1),
        qv_sfc=np.full_like(arr, 0.02),
        qc_sfc=np.full_like(arr, day * 1e-5),
        qr_sfc=np.full_like(arr, day * 1e-6),
        u_sfc=np.full_like(arr, 0.0),
        v_sfc=np.full_like(arr, 0.0),
        wind_sfc=np.full_like(arr, day * 0.01),
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
        / "scripts" / "run_rce_mpi_long.py"
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
        / "scripts" / "run_rce_mpi_long.py"
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
    criterion. Tests override only the fields they care about."""
    defaults = dict(
        day=day,
        cwv_mean=50.0,
        cwv_min=50.0,
        cwv_max=50.0,
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


def test_evaluate_skips_plateau_on_short_trajectory():
    """Fewer than ``last_n_days_for_plateau`` rows → only the
    finite-checks fire; plateau criteria are skipped to avoid
    false-FAIL on early spin-up."""
    rows = [_row(float(i), cwv_mean=10.0) for i in range(3)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    # CWV is in finite range; no plateau check fires; PASS.
    assert verdict.passed, verdict.reasons


def test_evaluate_iter98_inflight_trajectory_passes():
    """Smoke against the actual iter-98 trajectory shape: CWV
    overshoot then settle in [55, 58] mm. The defaults must accept
    this — if they don't, the defaults are too tight."""
    cwv = [49.94, 53.63, 55.67, 56.77, 57.18, 57.12, 56.85, 56.54, 56.20, 55.87]
    rows = [_row(float(i), cwv_mean=v, cwv_max=v) for i, v in enumerate(cwv)]
    verdict = summary_mod.evaluate_rce_quality(rows)
    assert verdict.passed, (
        "iter-98 in-flight 32x32 + radiation trajectory should pass "
        f"the DOD evaluator with default thresholds. Reasons: "
        f"{verdict.reasons!r}"
    )
