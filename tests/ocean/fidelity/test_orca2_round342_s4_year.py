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
    nemo_testcase_l4_orca2x_round1_pilot_metrics as pilot_metrics,
)


def test_calendar_month_day_ranges_full_noleap_year() -> None:
    windows = r342_metrics.calendar_month_day_ranges(365)
    assert len(windows) == 12
    lengths = [w["calendar_days"] for w in windows]
    assert lengths == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert sum(lengths) == 365
    assert all(w["complete_month"] for w in windows)
    assert windows[0]["start_day"] == 1 and windows[0]["end_day"] == 31
    assert windows[0]["saved_kts"][0] == 8 and windows[0]["saved_kts"][-1] == 248
    assert windows[-1]["start_day"] == 335 and windows[-1]["end_day"] == 365
    assert windows[-1]["saved_kts"][0] == 2680 and windows[-1]["saved_kts"][-1] == 2920
    all_days = [d for w in windows for d in w["available_days"]]
    assert all_days == list(range(1, 366))


def test_calendar_month_day_ranges_partial_window() -> None:
    windows = r342_metrics.calendar_month_day_ranges(45)
    assert windows[0]["complete_month"] is True
    assert windows[0]["available_days"] == list(range(1, 32))
    assert windows[1]["complete_month"] is False
    assert windows[1]["available_days"] == list(range(32, 46))
    assert windows[1]["saved_kts"] == [d * 8 for d in range(32, 46)]
    for w in windows[2:]:
        assert w["complete_month"] is False
        assert w["available_days"] == []
        assert w["saved_kts"] == []


def test_decision185_reference_bars_and_no_state_averaging_for_nonlinear_metrics() -> None:
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
    assert not hasattr(r342_metrics, "_mean_fields")

    dummy_scalar = {"bias": 0.0, "rms": 0.0}
    day1_row = {
        k: dict(dummy_scalar)
        for k in (
            "SST", "T_0-100m", "T_100-1000m", "T_>1000m",
            "SSS", "S_0-100m", "S_100-1000m", "S_>1000m",
            "SSH", "ICE_SST", "ICE_SSS",
        )
    }
    day1_row["MLD"] = {"bias": 0.5, "rms": 0.5, "nemo_mean": 40.0, "cand_mean": 40.5}
    day1_row["KE"] = {"bias": 0.002, "rms": 0.004, "nemo_mean": 0.02, "cand_mean": 0.022}
    day1_row["PSI"] = {"bias": 0.01, "rms": 0.05, "pattern_corr": 0.9995}
    day1_row["DRAKE"] = {"nemo_sv": 150.0, "cand_sv": 150.2}

    day2_row = {
        k: dict(dummy_scalar)
        for k in (
            "SST", "T_0-100m", "T_100-1000m", "T_>1000m",
            "SSS", "S_0-100m", "S_100-1000m", "S_>1000m",
            "SSH", "ICE_SST", "ICE_SSS",
        )
    }
    day2_row["MLD"] = {"bias": 1.5, "rms": 1.5, "nemo_mean": 60.0, "cand_mean": 61.5}
    day2_row["KE"] = {"bias": 0.004, "rms": 0.006, "nemo_mean": 0.04, "cand_mean": 0.044}
    day2_row["PSI"] = {"bias": 0.03, "rms": 0.07, "pattern_corr": 0.9991}
    day2_row["DRAKE"] = {"nemo_sv": 152.0, "cand_sv": 152.6}

    daily_rows = [day1_row, day2_row]
    geo = {"area_t": np.ones((2, 2)), "psi_weight": np.ones((2, 2))}
    maps = [
        {k: np.zeros((2, 2)) for k in ("SST", "SSS", "SSH", "MLD", "PSI", "KE")},
        {k: np.zeros((2, 2)) for k in ("SST", "SSS", "SSH", "MLD", "PSI", "KE")},
    ]
    monthly_summary = pilot_metrics.monthly(daily_rows, maps, geo)
    attached = r342_metrics._attach_decision185_reference(monthly_summary, daily_rows)
    assert set(attached) == set(bars)
    for entry in attached.values():
        assert entry["bar_mode"] == "REFERENCE_ONLY"
        assert entry["verdict"] is None

    # Verify per-saved-state metric averaging matches pilot_metrics.monthly
    assert attached["KE"]["nemo_mean"] == pytest.approx(0.03)
    assert attached["KE"]["cand_mean"] == pytest.approx(0.033)
    assert attached["KE"]["daily_mean_rms"] == pytest.approx(0.005)
    assert attached["KE"]["relative_rms_fraction"] == pytest.approx(0.005 / 0.03)
    assert attached["MLD"]["daily_mean_rms"] == pytest.approx(1.0)
    assert attached["MLD"]["nemo_mean"] == pytest.approx(50.0)
    assert attached["PSI"]["daily_mean_rms_sv"] == pytest.approx(0.06)
    assert attached["PSI"]["daily_mean_pattern_corr"] == pytest.approx(0.9993)
    assert attached["DRAKE"]["abs_mean_diff_sv"] == pytest.approx(0.4)


def test_driver_explicit_k0_and_full_year_fld_read_schedule() -> None:
    bulk.validate_nemo_standalone_k0(r342_driver.EXPLICIT_K0_CONFIG)
    dt_s = 10800
    synthetic_monthly = np.arange(12, dtype=np.float64)[:, None, None] * np.ones((12, 2, 2))
    for kt in range(1, 11):
        isecsbc = kt * dt_s
        pair = r342_driver._monthly_pair_for_clock(synthetic_monthly, isecsbc)
        assert float(pair.before[0, 0]) == 11.0
        assert float(pair.after[0, 0]) == 0.0
        assert (pair.nrec_sec_before, pair.nrec_sec_after) == (-1339200, 1339200)

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
    run_sh = (
        Path(r342_gate.__file__).resolve().parent
        / "nemo_testcase_l4_orca2_round342_s4_year_acquisition"
        / "run.sh"
    )
    staged_namelist = tmp_path / "namelist_cfg"
    info = r342_gate.stage_s4_year_namelist_cfg(r342_gate.R339_NAMELIST_CFG, staged_namelist)
    assert info["nn_itend"] == 2920
    assert info["nn_stock"] == 8
    assert info["expected_daily_restarts"] == 365

    syntax = r342_gate.verify_run_sh_syntax_and_plant(run_sh)
    assert syntax["passed"] is True

    disk = r342_gate.verify_disk_space(r342_gate.REPO_ROOT)
    assert disk["free_gib"] >= r342_gate.REQUIRED_DISK_GIB
    planted_disk = r342_gate.plant_disk_space_exhaustion(r342_gate.REPO_ROOT, run_sh)
    assert planted_disk["passed"] is True

    ram = r342_gate.verify_available_ram()
    assert ram["available_gib"] >= r342_gate.REQUIRED_RAM_GIB
    planted_ram = r342_gate.plant_ram_exhaustion(run_sh)
    assert planted_ram["passed"] is True

    planted_dirty = r342_gate.plant_dirty_worktree(run_sh)
    assert planted_dirty["passed"] is True

    with r342_gate.tempfile.TemporaryDirectory(dir=r342_gate._scratch_root()) as scratch_tmp:
        stage_bad_root = Path(scratch_tmp) / "stub_stage_bad"
        missing_ice = Path(scratch_tmp) / "missing_ice_cover.nc"
        stage_bad = r342_gate.subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stage_bad_root),
                "--stub-steps",
                "16",
                "--ice-cover",
                str(missing_ice),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert stage_bad.returncode != 0
        assert "R342_REFUSED_ARM_QUARANTINED" in stage_bad.stderr
        refused_stage = list((stage_bad_root / "refused").glob("*"))
        assert len(refused_stage) == 1 and r342_gate.TWIN_A_NAME in refused_stage[0].name
        assert any(refused_stage[0].iterdir())
        assert not (stage_bad_root / r342_gate.TWIN_A_NAME).exists()

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


def test_real_arm_a_completion_signal_and_plants(tmp_path: Path) -> None:
    real_logs = r342_gate.REAL_ARM_A_LOGS_DIR
    assert real_logs.is_dir()
    assert (real_logs / "mpirun.stderr.log").is_file()
    assert (real_logs / "mpirun.stdout.log").is_file()
    assert (real_logs / "mpirun.stdout.log").stat().st_size == 0
    assert (real_logs / "time.step").is_file()
    assert (real_logs / "ocean.output").is_file()
    assert (real_logs / "namelist_cfg").is_file()

    signal = r342_gate.verify_nemo_completion_signal(real_logs, expected_steps=2920)
    assert signal["nn_itend"] == 2920
    assert signal["nn_stock"] == 8
    assert signal["time_step"] == 2920
    assert signal["stderr_stop_0"] is True
    assert signal["stderr_last_line"] == "STOP 0"
    assert signal["stdout_bytes"] == 0
    assert signal["ocean_output_error_free"] is True

    planted = r342_gate.plant_completion_signal_failures(real_logs)
    assert planted["passed"] is True
    assert "STOP 0" in planted["missing_stop_error"]
    assert "STOP 0" in planted["stdout_only_stop_error"]
    assert "nonzero STOP" in planted["stop123_error"]
    assert "time.step (2912) != nn_itend (2920)" in planted["short_timestep_error"]
    assert "missing time.step" in planted["missing_timestep_error"]
    assert "E R R O R" in planted["ocean_output_error"]

    # Also verify direct mutations on a temporary copy of the real Arm-A logs
    work = tmp_path / "real_arm_a_copy"
    r342_gate.shutil.copytree(real_logs, work)
    assert r342_gate.verify_nemo_completion_signal(work, expected_steps=2920)["time_step"] == 2920

    # Plant short time.step = 2919
    (work / "time.step").write_text("     2919\n", encoding="utf-8")
    with pytest.raises(r342_gate.GateError, match=r"time\.step \(2919\) != nn_itend \(2920\)"):
        r342_gate.verify_nemo_completion_signal(work, expected_steps=2920)
    (work / "time.step").write_text("     2920\n", encoding="utf-8")

    # Plant missing STOP 0 on stderr
    (work / "mpirun.stderr.log").write_text("", encoding="utf-8")
    with pytest.raises(r342_gate.GateError, match=r"mpirun\.stderr\.log lacks clean STOP 0"):
        r342_gate.verify_nemo_completion_signal(work, expected_steps=2920)


@pytest.mark.skipif(
    not (
        r342_driver.DEFAULT_OUTPUT_DIR / "candidate_step00000008.npz"
    ).is_file()
    or not r342_gate.R340_RUN_A.is_dir(),
    reason="20-step legoESM smoke snapshot or R340_RUN_A not present",
)
def test_evaluate_month_by_month_on_day1_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_nemo = tmp_path / "stub_nemo"
    r342_gate.stage_s4_year_run_dir(
        stub_nemo, r342_gate.DEFAULT_ICE_COVER_PATH, nn_itend=8, nn_stock=8
    )
    capsys.readouterr()
    r342_gate.write_stub_s4_year_outputs(stub_nemo)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "STOP 0"
    assert (stub_nemo / "time.step").read_text(encoding="utf-8").strip() == "8"
    (stub_nemo / "mpirun.stdout.log").write_text(captured.out, encoding="utf-8")
    (stub_nemo / "mpirun.stderr.log").write_text(captured.err, encoding="utf-8")
    stub_sig = r342_gate.verify_nemo_completion_signal(stub_nemo, expected_steps=8)
    assert stub_sig["time_step"] == 8 and stub_sig["stderr_stop_0"] is True
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
    assert report["state_averaging_before_nonlinear_metrics"] is False
    assert report["saved_kts_evaluated"] == [8]
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
