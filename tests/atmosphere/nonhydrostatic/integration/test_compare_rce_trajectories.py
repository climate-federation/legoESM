"""Unit tests for ``scripts/compare_rce_trajectories.py``.

iter-110: the deterministic-reproducibility check between
``/tmp/iter98_crm32x32_rad10d`` and ``/tmp/iter105_crm32x32_rad30d``
showed log.txt rows are bit-equal step-for-step. This script
formalises that check at the snapshot level. The tests below build
synthetic snapshot pairs with known deltas and assert the diff
function produces the expected per-day deltas + summary maxes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load(module_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(module_name, file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


compare_mod = _load(
    "compare_rce_trajectories",
    REPO_ROOT / "scripts" / "compare_rce_trajectories.py",
)
summary_mod = compare_mod.summary_mod


def _write_snapshot(path: Path, day: float, cwv_value: float,
                    mse_value: float = 3.5e9) -> None:
    """Mirror the synthetic snapshot writer the summarizer tests
    use — same key set as ``run_rce_mpi_long.py:save_snapshot_2d``."""
    ny, nx = 4, 4
    arr = np.full((ny, nx), cwv_value, dtype=np.float64)
    np.savez_compressed(
        path,
        t_sim=day * 86400.0,
        day=day,
        cwv=arr,
        mse=np.full_like(arr, mse_value),
        precip=arr * 0.0,
        T_sfc=np.full_like(arr, 300.0),
        qv_sfc=np.full_like(arr, 0.02),
        qc_sfc=np.full_like(arr, 0.0),
        qr_sfc=np.full_like(arr, 0.0),
        u_sfc=np.full_like(arr, 0.0),
        v_sfc=np.full_like(arr, 0.0),
        wind_sfc=np.full_like(arr, 0.0),
    )


def _make_run(tmp_path: Path, name: str, day_cwv_pairs):
    """Build an output dir with the given day-cwv pairs."""
    out_dir = tmp_path / name
    snaps = out_dir / "snapshots"
    snaps.mkdir(parents=True)
    for idx, (day, cwv) in enumerate(day_cwv_pairs):
        _write_snapshot(snaps / f"snap_day_{idx:04d}.npz", day=day,
                        cwv_value=cwv)
    return out_dir


def test_identical_runs_have_zero_delta(tmp_path):
    days_cwv = [(0.0, 50.0), (1.0, 51.0), (2.0, 52.0)]
    a = _make_run(tmp_path, "a", days_cwv)
    b = _make_run(tmp_path, "b", days_cwv)
    diffs = compare_mod.diff_trajectories(a, b)
    assert len(diffs) == 3
    for entry in diffs:
        assert entry["delta_cwv_mean"] == pytest.approx(0.0)
        assert entry["delta_mse_mean"] == pytest.approx(0.0)


def test_diff_picks_up_per_day_delta(tmp_path):
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    # Day 0: 50.5 - 50.0 = 0.5
    assert diffs[0]["delta_cwv_mean"] == pytest.approx(0.5)
    # Day 1: 52.0 - 51.0 = 1.0
    assert diffs[1]["delta_cwv_mean"] == pytest.approx(1.0)


def test_diff_intersection_only(tmp_path):
    """Days present in only one run are dropped — per-day deltas
    need both endpoints to be well-defined."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0), (2.0, 52.0)])
    # B only has 2 days; matches a[0] + a[1] by day value.
    b = _make_run(tmp_path, "b", [(0.0, 50.0), (1.0, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    assert len(diffs) == 2
    assert [d["day"] for d in diffs] == [0.0, 1.0]


def test_align_rows_respects_day_tolerance(tmp_path):
    """A small floating-point drift in the stored day metadata
    (e.g. 1.0 vs 1.000001) must still align within the default
    1e-6 tolerance."""
    a = _make_run(tmp_path, "a", [(1.0, 50.0)])
    b = _make_run(tmp_path, "b", [(1.0000005, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b, day_tol=1e-6)
    assert len(diffs) == 1
    assert diffs[0]["delta_cwv_mean"] == pytest.approx(1.0)


def test_align_rows_rejects_above_tolerance(tmp_path):
    """A large drift (1 hour) is NOT a same-day match; the diff
    intersection drops it."""
    a = _make_run(tmp_path, "a", [(1.0, 50.0)])
    b = _make_run(tmp_path, "b", [(1.04, 51.0)])  # ~1-hr drift
    diffs = compare_mod.diff_trajectories(a, b, day_tol=1e-6)
    assert diffs == []


def test_format_table_renders_missing_as_na(tmp_path):
    """If a column happens to be ``None`` on one side (synthetic
    edge — the summarizer never emits None for surface columns, but
    profile columns can be None for missing profile files), the
    diff renders ``NA`` to match the iter-99 sentinel."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0)])
    b = _make_run(tmp_path, "b", [(0.0, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    # Force-inject a None to exercise the NA branch.
    diffs[0]["delta_cwv_mean"] = None
    text = compare_mod.format_diff_table(diffs)
    assert summary_mod.MISSING_SENTINEL in text


def test_summary_reports_max_abs_delta(tmp_path, capsys):
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    # Run main() via argv injection so the print output is captured.
    sys.argv = ["compare_rce_trajectories.py", str(a), str(b),
                "--quiet"]
    compare_mod.main()
    out = capsys.readouterr().out
    assert "matched 2 day(s)" in out
    # Max |Δ cwv_mean| over (0.5, 1.0) = 1.0
    assert "max |Δ cwv_mean| = 1.000000e+00" in out


def test_diff_against_real_iter98_baseline(tmp_path):
    """Smoke: build a synthetic 'iter-98 lookalike' (a single-day
    50.0 mm snapshot) and a 'iter-105 lookalike' identical to it.
    Verifies the script can be invoked on real iter-output-shaped
    inputs without crashing on the integer-day rounding the
    production writer uses."""
    a = _make_run(tmp_path, "a", [(0.0, 49.9413)])
    b = _make_run(tmp_path, "b", [(0.0, 49.9413)])
    diffs = compare_mod.diff_trajectories(a, b)
    assert len(diffs) == 1
    assert diffs[0]["delta_cwv_mean"] == pytest.approx(0.0)
