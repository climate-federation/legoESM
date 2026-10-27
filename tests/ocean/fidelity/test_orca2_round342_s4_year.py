"""Unit and known-answer tests for ORCA2 Round 342 (1-year standalone S4 comparison)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round342_s4_year_driver as r342_driver,
    nemo_testcase_l4_orca2_round342_s4_year_gate as r342_gate,
    nemo_testcase_l4_orca2_round342_s4_year_metrics as r342_metrics,
)


def test_calendar_month_day_ranges_full_noleap_year() -> None:
    windows = r342_metrics.calendar_month_day_ranges(365)
    assert len(windows) == 12
    lengths = [w["calendar_days"] for w in windows]
    assert lengths == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert sum(lengths) == 365
    assert all(w["complete_month"] for w in windows)
    assert windows[0]["start_day"] == 1 and windows[0]["end_day"] == 31
    assert windows[-1]["start_day"] == 335 and windows[-1]["end_day"] == 365
    all_days = [d for w in windows for d in w["available_days"]]
    assert all_days == list(range(1, 366))


def test_calendar_month_day_ranges_partial_window() -> None:
    windows = r342_metrics.calendar_month_day_ranges(45)
    assert windows[0]["complete_month"] is True
    assert windows[0]["available_days"] == list(range(1, 32))
    assert windows[1]["complete_month"] is False
    assert windows[1]["available_days"] == list(range(32, 46))
    for w in windows[2:]:
        assert w["complete_month"] is False
        assert w["available_days"] == []


def test_decision185_reference_bars_are_reference_only_without_verdict() -> None:
    bars = r342_metrics.DECISION_185_REFERENCE_BARS
    assert bars["SST"]["reference_bar"] == 0.01
    assert bars["T_0-100m"]["reference_bar"] == 0.01
    assert bars["T_100-1000m"]["reference_bar"] == 0.01
    assert bars["T_>1000m"]["reference_bar"] == 0.01
    assert bars["SSS"]["reference_bar"] == 0.002
    assert bars["S_0-100m"]["reference_bar"] == 0.002
    assert bars["S_100-1000m"]["reference_bar"] == 0.002
    assert bars["S_>1000m"]["reference_bar"] == 0.002
    assert bars["SSH"]["reference_bar"] == 0.001
    assert bars["KE"]["reference_bar_fraction"] == 0.01
    assert bars["PSI"]["reference_bar_rms_sv"] == 0.1
    assert bars["PSI"]["reference_bar_pattern_corr_min"] == 0.999
    assert bars["DRAKE"]["reference_bar"] == 0.5
    assert bars["MLD"]["reference_bar"] == 1.0

    dummy_scalar = {"bias": 0.0, "rms": 0.0}
    monthly_summary = {
        k: dict(dummy_scalar)
        for k in (
            "SST", "T_0-100m", "T_100-1000m", "T_>1000m",
            "SSS", "S_0-100m", "S_100-1000m", "S_>1000m",
            "SSH", "MLD", "ICE_SST", "ICE_SSS",
        )
    }
    monthly_summary["KE"] = {"bias": 0.0, "rms": 0.0}
    monthly_summary["PSI"] = {"bias": 0.0, "rms": 0.0, "pattern_corr": 1.0}
    monthly_summary["DRAKE"] = {
        "nemo_mean_sv": 150.0,
        "cand_mean_sv": 150.0,
        "nemo_std_sv": 0.0,
        "cand_std_sv": 0.0,
    }
    mean_field_row = {
        **monthly_summary,
        "KE": {"bias": 0.0, "rms": 0.0, "nemo_mean": 0.02, "cand_mean": 0.02},
    }
    attached = r342_metrics._attach_decision185_reference(monthly_summary, mean_field_row)
    assert set(attached) == set(bars)
    for entry in attached.values():
        assert entry["bar_mode"] == "REFERENCE_ONLY"
        assert entry["verdict"] is None


def test_driver_explicit_k0_and_full_year_fld_read_schedule() -> None:
    bulk.validate_nemo_standalone_k0(r342_driver.EXPLICIT_K0_CONFIG)
    dt_s = 10800
    synthetic_monthly = np.arange(12, dtype=np.float64)[:, None, None] * np.ones((12, 2, 2))
    # Verify steps 1..10 use year-start seam (-1339200, 1339200) between Dec (11) and Jan (0)
    for kt in range(1, 11):
        isecsbc = kt * dt_s
        pair = r342_driver._monthly_pair_for_clock(synthetic_monthly, isecsbc)
        assert float(pair.before[0, 0]) == 11.0
        assert float(pair.after[0, 0]) == 0.0
        assert (pair.nrec_sec_before, pair.nrec_sec_after) == (-1339200, 1339200)

    # Verify full year kt = 1..2920 indices stay within record bounds
    seen_pairs: set[tuple[int, int]] = set()
    for kt in range(1, r342_driver.YEAR_STEPS + 1):
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s
        idx_6h = retained_clock // 21600
        idx_daily = retained_clock // 86400
        idx_monthly = int(np.searchsorted(r342_driver.MONTH_END_SEC, retained_clock, side="right"))
        assert 0 <= idx_6h < 1460
        assert 0 <= idx_daily < 365
        assert 0 <= idx_monthly < 12
        pair = r342_driver._monthly_pair_for_clock(synthetic_monthly, isecsbc)
        assert pair.nrec_sec_before <= isecsbc <= pair.nrec_sec_after
        seen_pairs.add((int(pair.before[0, 0]), int(pair.after[0, 0])))
    # All 12 cyclic month transitions must be visited across the 365-day year
    assert len(seen_pairs) == 12

    if r342_driver.DEFAULT_DECK_ROOT.is_dir():
        wet = np.ones((148, 180), dtype=bool)
        chl_1 = ladder.chlorophyll_at_step(r342_driver.DEFAULT_DECK_ROOT, 1, wet, dt_s=10800.0)
        chl_2920 = ladder.chlorophyll_at_step(r342_driver.DEFAULT_DECK_ROOT, 2920, wet, dt_s=10800.0)
        assert chl_1.shape == (148, 180) and np.isfinite(chl_1).all()
        assert chl_2920.shape == (148, 180) and np.isfinite(chl_2920).all()


@pytest.mark.skipif(
    not r342_gate.R339_NAMELIST_CFG.is_file(),
    reason="Round-339 K0 namelist or ppsrc not present on host",
)
def test_gate_checklist_controls_and_planted_failures(tmp_path: Path) -> None:
    ppsrc_dir = r342_gate._ppsrc_dir()
    staged_namelist = tmp_path / "namelist_cfg"
    info = r342_gate.stage_s4_year_namelist_cfg(r342_gate.R339_NAMELIST_CFG, staged_namelist)
    assert info["nn_itend"] == 2920
    assert info["nn_stock"] == 8
    assert info["expected_daily_restarts"] == 365

    nml_check = r342_gate.verify_namelist_groups_against_ppsrc(
        staged_namelist, r342_gate.NAMELIST_REF, ppsrc_dir
    )
    assert nml_check["unknown_cfg_groups"] == []

    planted_nml = r342_gate.plant_namelist_error(ppsrc_dir)
    assert planted_nml["passed"] is True
    assert "undeclared" in planted_nml["undeclared_group_error"]
    assert "E R R O R" in planted_nml["ocean_output_error"]

    alloc = r342_gate.verify_allocate_extents(ppsrc_dir)
    assert set(alloc["verified_allocates"]) == {"oce", "dom_oce", "sbc_oce"}
    planted_alloc = r342_gate.plant_allocate_mismatch(ppsrc_dir)
    assert planted_alloc["passed"] is True

    fixture = r342_gate.verify_restart_fortran_fixture(ppsrc_dir)
    assert fixture["passed"] is True
    assert "FIXTURE_OK writes=365 first=8 last=2920" in fixture["stdout"]
    planted_fix = r342_gate.plant_fixture_wrong_time_level(ppsrc_dir)
    assert planted_fix["passed"] is True


@pytest.mark.skipif(
    not (
        r342_driver.DEFAULT_OUTPUT_DIR / "candidate_step00000008.npz"
    ).is_file()
    or not r342_gate.R340_RUN_A.is_dir(),
    reason="20-step legoESM smoke snapshot or R340_RUN_A not present",
)
def test_evaluate_month_by_month_on_day1_smoke(tmp_path: Path) -> None:
    stub_nemo = tmp_path / "stub_nemo"
    r342_gate.stage_s4_year_run_dir(
        stub_nemo, r342_gate.DEFAULT_ICE_COVER_PATH, nn_itend=8, nn_stock=8
    )
    r342_gate.write_stub_s4_year_outputs(stub_nemo)
    day1_check = r342_gate.verify_day1_against_admitted_s4_kt9(stub_nemo)
    assert all(v["max_ulp"] == 0 for v in day1_check.values())

    report = r342_metrics.evaluate_month_by_month(
        nemo_root=stub_nemo,
        candidate_root=r342_driver.DEFAULT_OUTPUT_DIR,
        days=1,
        skip_mld=True,
    )
    assert report["bar_mode"] == "DECISION_185_REFERENCE_ONLY"
    assert report["verdict"] is None
    assert report["pass_fail_assigned"] is False
    m01 = report["months"][0]
    assert m01["label"] == "M01_Jan"
    assert m01["status"] == "EVALUATED_PARTIAL_MONTH"
    assert m01["verdict"] is None
    assert m01["metrics"]["SST"]["daily_mean_rms"] == 0.0
    assert m01["metrics"]["SSS"]["daily_mean_rms"] == 0.0
    assert m01["metrics"]["SSH"]["daily_mean_rms"] == 0.0
    assert m01["metrics"]["PSI"]["daily_mean_rms_sv"] == 0.0
    assert m01["metrics"]["PSI"]["daily_mean_pattern_corr"] == pytest.approx(1.0)
    assert m01["metrics"]["DRAKE"]["abs_mean_diff_sv"] == 0.0
    for m_pending in report["months"][1:]:
        assert m_pending["status"] == "PENDING_ACQUISITION"
        assert m_pending["verdict"] is None

