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
    use — same key set as ``run_rce_mpi_long.py:save_snapshot_2d``.

    iter-123: replaced ``precip=arr * 0.0`` with
    ``np.zeros((ny, nx))``; ``nan * 0.0 = nan`` propagated to the
    precip field on the NONFINITE-test fixtures, surfacing a numpy
    RuntimeWarning that bloated test output. Constants don't need
    arr-shape inheritance."""
    ny, nx = 4, 4
    arr = np.full((ny, nx), cwv_value, dtype=np.float64)
    zeros = np.zeros((ny, nx), dtype=np.float64)
    np.savez_compressed(
        path,
        t_sim=day * 86400.0,
        day=day,
        cwv=arr,
        mse=np.full((ny, nx), mse_value, dtype=np.float64),
        precip=zeros,
        T_sfc=np.full((ny, nx), 300.0, dtype=np.float64),
        qv_sfc=np.full((ny, nx), 0.02, dtype=np.float64),
        qc_sfc=zeros,
        qr_sfc=zeros,
        u_sfc=zeros,
        v_sfc=zeros,
        wind_sfc=zeros,
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
    """If a column happens to be ``MISSING`` on one side (synthetic
    edge — the summarizer never emits None for surface columns, but
    profile columns can be None for missing profile files), the
    diff renders ``NA`` to match the iter-99 sentinel.

    iter-111 MEDIUM#2: separate path for ``NONFINITE`` covered in
    ``test_nonfinite_rendered_separately`` below."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0)])
    b = _make_run(tmp_path, "b", [(0.0, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    # Force-inject a MISSING sentinel to exercise the NA branch.
    diffs[0]["delta_cwv_mean"] = compare_mod._DELTA_MISSING
    text = compare_mod.format_diff_table(diffs)
    assert summary_mod.MISSING_SENTINEL in text


def test_summary_reports_max_abs_delta(tmp_path, capsys, monkeypatch):
    """iter-111 LOW#6 fix: use monkeypatch.setattr for sys.argv so
    test state does not leak into subsequent tests."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    monkeypatch.setattr(
        sys, "argv",
        ["compare_rce_trajectories.py", str(a), str(b), "--quiet"],
    )
    compare_mod.main()
    out = capsys.readouterr().out
    assert "matched 2 day(s)" in out
    # Max |d cwv_mean| over (0.5, 1.0) = 1.0 (ASCII label per LOW#5).
    assert "max |d cwv_mean| = 1.000000e+00" in out


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


# iter-111: regression tests for Codex MEDIUM #1 / #2 / #3 + LOW #4.


def _make_run_with_shape(tmp_path, name, day_cwv_pairs, *, shape=(4, 4)):
    """Variant of ``_make_run`` that lets the test choose the
    horizontal grid shape. iter-111 MEDIUM#1: required for the
    shape-mismatch regression."""
    out_dir = tmp_path / name
    snaps = out_dir / "snapshots"
    snaps.mkdir(parents=True)
    for idx, (day, cwv) in enumerate(day_cwv_pairs):
        ny, nx = shape
        arr = np.full((ny, nx), cwv, dtype=np.float64)
        np.savez_compressed(
            snaps / f"snap_day_{idx:04d}.npz",
            t_sim=day * 86400.0, day=day,
            cwv=arr, mse=arr * 1e7, precip=arr * 0.0,
            T_sfc=np.full_like(arr, 300.0),
            qv_sfc=np.full_like(arr, 0.02),
            qc_sfc=np.full_like(arr, 0.0),
            qr_sfc=np.full_like(arr, 0.0),
            u_sfc=np.full_like(arr, 0.0),
            v_sfc=np.full_like(arr, 0.0),
            wind_sfc=np.full_like(arr, 0.0),
        )
    return out_dir


def test_diff_refuses_shape_mismatch_by_default(tmp_path):
    """iter-111 MEDIUM#1: a 132×132 vs 32×32 diff produces
    meaningful-looking domain-mean deltas but the numbers are
    physically meaningless. Refuse by default."""
    a = _make_run_with_shape(tmp_path, "a", [(0.0, 50.0)], shape=(4, 4))
    b = _make_run_with_shape(tmp_path, "b", [(0.0, 50.0)], shape=(8, 8))
    with pytest.raises(ValueError, match="snapshot shape mismatch"):
        compare_mod.diff_trajectories(a, b)


def test_diff_force_shape_mismatch_overrides(tmp_path):
    """iter-111 MEDIUM#1: ``force_shape_mismatch=True`` allows
    overriding the shape check (e.g. a sanity-check across
    resolutions)."""
    a = _make_run_with_shape(tmp_path, "a", [(0.0, 50.0)], shape=(4, 4))
    b = _make_run_with_shape(tmp_path, "b", [(0.0, 51.0)], shape=(8, 8))
    diffs = compare_mod.diff_trajectories(
        a, b, force_shape_mismatch=True,
    )
    assert len(diffs) == 1
    assert diffs[0]["delta_cwv_mean"] == pytest.approx(1.0)


def test_nonfinite_rendered_separately(tmp_path):
    """iter-111 MEDIUM#2: NaN / inf on one side is surfaced as a
    distinct ``NONFINITE`` sentinel (not collapsed into MISSING /
    None) so users can tell a corrupt diagnostic apart from a
    missing-profile column."""
    a = _make_run(tmp_path, "a", [(0.0, float("nan"))])
    b = _make_run(tmp_path, "b", [(0.0, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    assert diffs[0]["delta_cwv_mean"] == compare_mod._DELTA_NONFINITE
    table = compare_mod.format_diff_table(diffs)
    assert "NONFINITE" in table


def test_summary_reports_nonfinite_count(tmp_path):
    """iter-111 MEDIUM#2: _column_summary tracks NaN/inf separately
    from finite max-abs so a corrupt diagnostic is visible."""
    a = _make_run(tmp_path, "a", [(0.0, float("inf"))])
    b = _make_run(tmp_path, "b", [(0.0, 51.0)])
    diffs = compare_mod.diff_trajectories(a, b)
    summary = compare_mod._column_summary(diffs)
    assert summary["cwv_mean"]["nonfinite_count"] == 1
    assert summary["cwv_mean"]["max_abs"] is None


def test_csv_output_written(tmp_path):
    """iter-111 MEDIUM#3: ``--csv`` (or the default
    ``<dir_a>/diff_vs_<dir_b>.csv``) is actually written; previously
    the docstring claimed CSV output but main() never wrote one."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    csv_path = tmp_path / "diff.csv"
    compare_mod.write_csv(compare_mod.diff_trajectories(a, b), csv_path)
    assert csv_path.exists()
    text = csv_path.read_text()
    lines = text.splitlines()
    assert lines[0].startswith("day,cwv_mean,cwv_max")
    # 2 data rows + header.
    assert len(lines) == 3


def test_align_rows_one_to_one(tmp_path):
    """iter-111 LOW#4: two A rows whose days are both within
    ``day_tol`` of the SAME B row must not both pair to it. The
    first claims the B row; the second is dropped (the run had a
    sub-tolerance intra-run day spacing, which is itself rare)."""
    # Manually craft DayRow lists since collect_trajectory's
    # duplicate-day rejection blocks this through the file path.
    ra1 = summary_mod.DayRow(
        day=1.0, cwv_mean=50.0, cwv_min=50.0, cwv_max=50.0,
        cwv_std=0.0, mse_mean=3.5e9, precip_mean=0.0, precip_max=0.0,
        T_sfc_mean=300.0, qv_sfc_mean=0.02, qc_sfc_max=0.0,
        qr_sfc_max=0.0, wind_sfc_mean=0.0, wind_sfc_max=0.0,
    )
    ra2 = summary_mod.DayRow(
        day=1.0000001, cwv_mean=60.0, cwv_min=60.0, cwv_max=60.0,
        cwv_std=0.0, mse_mean=3.5e9, precip_mean=0.0, precip_max=0.0,
        T_sfc_mean=300.0, qv_sfc_mean=0.02, qc_sfc_max=0.0,
        qr_sfc_max=0.0, wind_sfc_mean=0.0, wind_sfc_max=0.0,
    )
    rb = summary_mod.DayRow(
        day=1.0, cwv_mean=55.0, cwv_min=55.0, cwv_max=55.0,
        cwv_std=0.0, mse_mean=3.5e9, precip_mean=0.0, precip_max=0.0,
        T_sfc_mean=300.0, qv_sfc_mean=0.02, qc_sfc_max=0.0,
        qr_sfc_max=0.0, wind_sfc_mean=0.0, wind_sfc_max=0.0,
    )
    paired = compare_mod._align_rows([ra1, ra2], [rb], day_tol=1e-5)
    assert len(paired) == 1, (
        f"only the first A row should pair with B; got {len(paired)} "
        f"pairings"
    )


def test_diff_trajectories_accepts_string_paths(tmp_path):
    """iter-119: ``diff_trajectories`` accepts ``str`` as well as
    ``Path`` (matches summarize_rce_trajectory.collect_trajectory's
    iter-119 ergonomics fix)."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.0)])
    diffs = compare_mod.diff_trajectories(str(a), str(b))
    assert len(diffs) == 1
    assert diffs[0]["delta_cwv_mean"] == pytest.approx(0.0)


def test_csv_default_path_written_by_cli(tmp_path, monkeypatch):
    """iter-147: when ``--csv`` is not passed, main() writes the
    diff CSV to the default path ``<dir_a>/diff_vs_<dir_b>.csv``.
    Locks the iter-111 MEDIUM#3 default-path contract that
    test_csv_output_written only covered via direct write_csv
    call."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    monkeypatch.setattr(
        sys, "argv",
        ["compare_rce_trajectories.py", str(a), str(b), "--quiet"],
    )
    compare_mod.main()
    # Default path: <dir_a>/diff_vs_<dir_b>.csv (dir_b's basename).
    expected_csv = a / f"diff_vs_{b.name}.csv"
    assert expected_csv.exists(), (
        f"expected default CSV at {expected_csv}, "
        f"got {list(a.iterdir())}"
    )
    text = expected_csv.read_text()
    assert text.startswith("day,cwv_mean,cwv_max"), (
        f"default CSV missing expected header; "
        f"got first line: {text.splitlines()[0]!r}"
    )


def test_csv_explicit_path_overrides_default(tmp_path, monkeypatch):
    """iter-147: ``--csv PATH`` overrides the default; the default
    path is NOT written + the explicit path IS."""
    a = _make_run(tmp_path, "a", [(0.0, 50.0), (1.0, 51.0)])
    b = _make_run(tmp_path, "b", [(0.0, 50.5), (1.0, 52.0)])
    explicit_csv = tmp_path / "custom_diff.csv"
    monkeypatch.setattr(
        sys, "argv",
        ["compare_rce_trajectories.py", str(a), str(b),
         "--csv", str(explicit_csv), "--quiet"],
    )
    compare_mod.main()
    default_csv = a / f"diff_vs_{b.name}.csv"
    assert explicit_csv.exists()
    assert not default_csv.exists(), (
        f"explicit --csv should suppress the default path write; "
        f"default also written at {default_csv}"
    )
