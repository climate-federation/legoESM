"""Direct test for ``scripts/plot/plot_spectral_level_shard.py`` — the spectral
level-shard cliff plotter.  Exercises the pure ``compute_curves`` speedup math
(true ``none@N=1`` baseline + level-N=1 fallback) and that ``make_figure``
renders a PNG without error (incl. the empty-rows 'no data' path).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = Path(__file__).resolve().parents[2] / "scripts" / "plot" / \
    "plot_spectral_level_shard.py"
_mod_spec = importlib.util.spec_from_file_location("plot_spectral_level_shard",
                                                   _SPEC)
plot_mod = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(plot_mod)


def _rows(extra=None):
    base = [
        {"n_max": "42", "nlev": "24", "n_devices": "1", "shard": "none",
         "steps_per_s": "100.0"},
        {"n_max": "42", "nlev": "24", "n_devices": "1", "shard": "level",
         "steps_per_s": "90.0"},
        {"n_max": "42", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "60.0"},
        {"n_max": "42", "nlev": "24", "n_devices": "4", "shard": "level",
         "steps_per_s": "40.0"},
    ]
    if extra:
        base += extra
    return base


def test_compute_curves_speedup_vs_none_baseline():
    curves = plot_mod.compute_curves(_rows())
    assert set(curves) == {42}
    pts = curves[42]
    # sorted by device count
    assert [p[0] for p in pts] == [1, 2, 4]
    # speedup = level sps / none@N=1 baseline (100.0) -> the documented cliff
    # (sub-linear, here even sub-1 anti-scaling)
    assert pts[0][2] == pytest.approx(0.90)   # 90/100
    assert pts[1][2] == pytest.approx(0.60)   # 60/100
    assert pts[2][2] == pytest.approx(0.40)   # 40/100
    # raw steps/s preserved
    assert pts[2][1] == pytest.approx(40.0)


def test_compute_curves_falls_back_to_level_n1_baseline():
    # no shard=none row -> baseline is level@N=1 (self-relative speedup)
    rows = [r for r in _rows() if r["shard"] != "none"]
    curves = plot_mod.compute_curves(rows)
    pts = curves[42]
    assert pts[0][2] == pytest.approx(1.0)    # 90/90
    assert pts[1][2] == pytest.approx(60.0 / 90.0)


def test_compute_curves_skips_nonpositive_and_unparseable():
    rows = _rows(extra=[
        {"n_max": "85", "nlev": "24", "n_devices": "1", "shard": "level",
         "steps_per_s": "0.0"},               # nonpositive -> skipped
        {"n_max": "x", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "5.0"},               # unparseable n_max -> skipped
        {"n_max": "85", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "NaN"},               # NaN sps -> skipped
        {"n_max": "85", "nlev": "24", "n_devices": "4", "shard": "level",
         "steps_per_s": "inf"},               # inf sps -> skipped (codex MAJOR)
        {"n_max": "-1", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "5.0"},               # nonpositive n_max -> skipped
        {"n_max": "85", "nlev": "24", "n_devices": "0", "shard": "level",
         "steps_per_s": "5.0"},               # nonpositive n_devices -> skipped
    ])
    curves = plot_mod.compute_curves(rows)
    # T85 has no usable baseline or points -> absent; T42 intact
    assert set(curves) == {42}


def test_nan_inf_baseline_does_not_poison():
    # codex MAJOR: a NaN none@N=1 baseline must be skipped so the level@N=1
    # fallback is used instead of poisoning every speedup to NaN.
    rows = [
        {"n_max": "42", "nlev": "24", "n_devices": "1", "shard": "none",
         "steps_per_s": "NaN"},               # NaN baseline -> skipped
        {"n_max": "42", "nlev": "24", "n_devices": "1", "shard": "level",
         "steps_per_s": "80.0"},
        {"n_max": "42", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "50.0"},
    ]
    curves = plot_mod.compute_curves(rows)
    pts = curves[42]
    assert pts[0][2] == pytest.approx(1.0)        # fallback baseline 80/80
    assert pts[1][2] == pytest.approx(50.0 / 80.0)


def test_conflicting_baseline_raises():
    rows = _rows(extra=[
        {"n_max": "42", "nlev": "24", "n_devices": "1", "shard": "none",
         "steps_per_s": "111.0"},             # != the 100.0 baseline -> raise
    ])
    with pytest.raises(ValueError, match="conflicting none"):
        plot_mod.compute_curves(rows)


def test_conflicting_level_point_raises():
    rows = _rows(extra=[
        {"n_max": "42", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "61.0"},              # != the 60.0 N=2 point -> raise
    ])
    with pytest.raises(ValueError, match="conflicting level"):
        plot_mod.compute_curves(rows)


def test_identical_duplicate_level_point_ok():
    # an identical duplicate (same value) is idempotent, not a conflict
    rows = _rows(extra=[
        {"n_max": "42", "nlev": "24", "n_devices": "2", "shard": "level",
         "steps_per_s": "60.0"},
    ])
    curves = plot_mod.compute_curves(rows)
    assert [p[0] for p in curves[42]] == [1, 2, 4]


def test_missing_required_column_raises():
    rows = [{"n_max": "42", "n_devices": "1", "steps_per_s": "100.0"}]  # no shard
    with pytest.raises(ValueError, match="missing required columns"):
        plot_mod.compute_curves(rows)


def test_make_figure_writes_png(tmp_path):
    out = plot_mod.make_figure(_rows(), tmp_path)
    assert out.exists() and out.suffix == ".png"
    assert out.stat().st_size > 0


def test_make_figure_empty_rows_no_data(tmp_path):
    out = plot_mod.make_figure([], tmp_path)
    assert out.exists()
