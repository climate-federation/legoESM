from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases" /
         "nemo_testcase_l1_swg_online_control_health.py")
_SPEC = importlib.util.spec_from_file_location("swg_online_control_health", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
health = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(health)
_COARSE_TRUTH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/"
    "conservative_truth_dg50_1095d/snapshots")


@pytest.mark.skipif(not (_COARSE_TRUTH / "manifest.json").is_file(),
                    reason="registered dg50 coarse truth not mounted")
def test_report_runs_online_health_on_a_real_year(tmp_path):
    """Review EV-04: report() refused at the tip (registered diagnostics).

    One real dg50 year (the restricted truth's year 1) is packaged as a
    complete one-year dg50 snapshot archive and run through the CLI.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST

    root = tmp_path / "control"
    root.mkdir()
    pins = {}
    for day in range(1, health.DAYS_PER_YEAR + 1):
        with np.load(_COARSE_TRUTH / f"day{day:04d}.npz") as saved:
            fields = {name: np.asarray(saved[name])
                      for name in ("ssh_m", "u_m_s", "v_m_s")}
        path = root / f"day{day:04d}.npz"
        np.savez(path, day=np.int64(day),
                 dt_s=np.float64(SWG_DG50_NAMELIST.rn_Dt_s), **fields)
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete", "days": health.DAYS_PER_YEAR,
        "completed_days": health.DAYS_PER_YEAR,
        "schema": "legoesm-swg-double-gyre-daily-v1",
        "sampling": "daily_snapshot",
        "namelist": SWG_DG50_NAMELIST._asdict(), "daily_sha256": pins}))
    output = tmp_path / "health.json"
    assert health.main(["--root", str(root), "--deck", "dg50",
                        "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["diagnostics"] == "online"
    assert result["sampling"] == "daily_snapshot"
    assert [row["days"] for row in result["years"]] == [[1, 365]]
    row = result["years"][0]
    assert 0.0 < row["grid_noise_index_mean"] <= row["grid_noise_index_max"]
    assert 0.0 < row["top_octave_enstrophy_fraction"] < 1.0
    assert row["minimum_total_layer_thickness_m"] > 0.0
    assert 0.0 < max(row["maximum_u_courant"], row["maximum_v_courant"]) < 1.0


def test_summarize_year_reports_extrema_and_refuses_nonfinite():
    arrays = {
        "basin_mean_ke_m2_s2": np.array([1.0, 3.0]),
        "basin_mean_eke_m2_s2": np.array([0.5, 1.5]),
        "top_octave_ke_fraction": np.array([0.2]),
        "top_octave_enstrophy_fraction": np.array([0.7]),
        "native_grid_noise_index": np.array([0.01, 0.03]),
        "minimum_total_layer_thickness_m": np.array([499.0, 498.0]),
        "maximum_u_courant": np.array([0.1, 0.2]),
        "maximum_v_courant": np.array([0.3, 0.2]),
    }
    result = health.summarize_year(arrays)
    assert result["basin_mean_ke_m2_s2"] == 2.0
    assert result["grid_noise_index_max"] == 0.03
    assert result["minimum_total_layer_thickness_m"] == 498.0
    assert result["maximum_v_courant"] == 0.3

    arrays["native_grid_noise_index"][0] = np.nan
    with pytest.raises(Exception, match="non-finite"):
        health.summarize_year(arrays)
