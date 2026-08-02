"""Tests for the physics-toggle stability screen harness."""
import numpy as np

from scripts.validate.physics_toggle_screen import (
    _blowup_day,
    build_cases,
    extract_run_metrics,
    format_table,
)


def test_build_cases_tier1():
    cases = build_cases("tier1")
    labels = [c[0] for c in cases]
    assert "baseline" in labels and labels[0] == "baseline"
    assert cases[0][1] == []  # baseline = no overrides
    # every physics-off case opts into --allow-disabled-physics
    for label, args in cases:
        if label.endswith("_off"):
            assert "--allow-disabled-physics" in args
            assert "none" in args
    # radiation leg uses 'gray' (no 'none' for radiation)
    rad = dict(cases)["radiation_gray"]
    assert rad == ["--radiation", "gray"]


def test_build_cases_unknown_tier_raises():
    import pytest
    with pytest.raises(ValueError):
        build_cases("nope")


def test_blowup_day_parsing():
    assert _blowup_day("BLOWUP at day 10: max wind 2250 m/s") == 10.0
    assert np.isinf(_blowup_day("COMPLETED"))
    assert np.isnan(_blowup_day("FAIL: something"))


def test_extract_run_metrics(tmp_path):
    d = tmp_path / "convection_off"
    d.mkdir()
    (d / "results.txt").write_text(
        "Grid: latlon 24\nStatus: BLOWUP at day 3: max wind 600.0 m/s\nWall time: 42s\n")
    np.savez(
        d / "timeseries.npz",
        days=np.array([1.0, 2.0, 3.0]),
        max_wind=np.array([30.0, 120.0, 600.0]),
        moisture_residual=np.array([0.1, 0.5, -3.2]),
        energy_toa_net=np.array([-50.0, -80.0, -120.0]),
        sw_up_toa=np.array([200.0, 215.0, 228.0]),
        lw_up_toa=np.array([250.0, 220.0, 190.0]),
    )
    m = extract_run_metrics(d)
    assert m["label"] == "convection_off"
    assert m["blowup_day"] == 3.0
    assert m["max_wind"] == 600.0
    assert m["moisture_resid"] == 3.2          # max abs
    assert m["energy_toa_net"] == -120.0       # last
    assert m["rsut"] == 228.0                   # last sw_up_toa (albedo proxy)
    assert m["olr"] == 190.0                    # last lw_up_toa


def test_build_cases_tier2_cloud():
    cases = build_cases("tier2_cloud")
    labels = [c[0] for c in cases]
    assert labels[0] == "baseline"
    assert "cc_off" in labels
    assert dict(cases)["cc_off"] == ["--no-convective-cloud"]


def test_build_cases_tier2_precip():
    cases = build_cases("tier2_precip")
    labels = [c[0] for c in cases]
    assert labels[0] == "baseline"
    assert "subgrid_auto" in labels
    d = dict(cases)
    assert d["subgrid_auto"] == ["--subgrid-autoconversion"]
    # Pierre's #840 anvil-condensate knob is in the matrix
    assert "--conv-cloud-condensate" in d["ccond_low"]
    # only CLI-reachable levers (no unreachable --params scheme fields)
    assert not any("--params" in args for _, args in cases)


def test_build_cases_tier2_evap():
    cases = build_cases("tier2_evap")
    labels = [c[0] for c in cases]
    assert labels[0] == "baseline"
    d = dict(cases)
    # evaporation lever (gustiness) + the untried cloud-overcast lever (p_xr)
    assert d["gust_600"] == ["--gustiness-zi", "600"]
    assert "--cloud-p-xr" in d["pxr_hi"]
    assert d["gust_pxr"] == ["--gustiness-zi", "600", "--cloud-p-xr", "0.6"]


def test_extract_run_metrics_survived(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "results.txt").write_text("Status: COMPLETED\n")
    np.savez(d / "timeseries.npz", max_wind=np.array([40.0]),
             moisture_residual=np.array([0.2]), energy_toa_net=np.array([2.0]))
    m = extract_run_metrics(d)
    assert np.isinf(m["blowup_day"]) and m["status"] == "COMPLETED"


def test_format_table_orders_survivors_first():
    rows = [
        {"label": "convection_off", "status": "BLOWUP at day 2", "blowup_day": 2.0,
         "max_wind": 600.0, "moisture_resid": 3.0, "energy_toa_net": -100.0},
        {"label": "baseline", "status": "COMPLETED", "blowup_day": float("inf"),
         "max_wind": 40.0, "moisture_resid": 0.2, "energy_toa_net": 1.0},
    ]
    table = format_table(rows)
    # survivor (baseline) sorts before the early blow-up
    assert table.index("baseline") < table.index("convection_off")
    assert "surv" in table
