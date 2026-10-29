"""Direct tests for the ORCA2 round-345 Decision-188 tools."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_decision188_gate as decision188,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_sst_ulp_gate as ulp_gate,
)


def _pilot_report(*, ke_rms: float) -> dict:
    monthly = {
        key: {"bias": 0.0, "rms": 0.0}
        for key in decision188.FIXED_RMS_BARS
    }
    monthly.update({
        "KE": {"bias": 0.0, "rms": ke_rms},
        "PSI": {"bias": 0.0, "rms": 0.0, "pattern_corr": 1.0},
        "DRAKE": {
            "nemo_mean_sv": 100.0, "cand_mean_sv": 100.0,
            "nemo_std_sv": 0.0, "cand_std_sv": 0.0,
        },
    })
    return {
        "format": "orca2x-r1-pilot-metrics-v1",
        "monthly": monthly,
        "daily": [
            {"day": day, "KE": {"nemo_mean": 0.1}}
            for day in range(1, 31)
        ],
    }


def test_decision188_plants_fire() -> None:
    assert decision188.run_plants() == {
        "fixed_bar_plant": "PASS",
        "ke_inside_3x_envelope_plant": "PASS",
        "ke_floor_plant": "PASS",
    }


def test_decision188_runs_on_orca2x_30_day_pilot_schema() -> None:
    result = decision188.evaluate_decision188(
        _pilot_report(ke_rms=0.002),
        _pilot_report(ke_rms=0.0007),
    )
    assert result["verdict"] == "PASS"
    assert result["months_evaluated"] == 1
    assert result["months"][0]["label"] == "PILOT_DAYS_1_30"
    assert result["months"][0]["status"] == "EVALUATED_PARTIAL_MONTH"
    assert result["months"][0]["metrics"]["KE"]["bar_fraction"] == pytest.approx(0.021)


def test_decision188_refuses_different_month_windows() -> None:
    candidate = decision188.minimal_year_report(sst=0.0, ke=0.0)
    envelope = decision188.minimal_year_report(sst=0.0, ke=0.0)
    envelope["months"][0]["saved_kts"] = [16]
    with pytest.raises(decision188.GateError, match="saved steps differ"):
        decision188.evaluate_decision188(candidate, envelope)


def _write_temperature(path: Path) -> np.ndarray:
    values = np.arange(12 * 2 * 2 * 3, dtype=np.float32).reshape(12, 2, 2, 3)
    values -= np.float32(20.0)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 12)
        dataset.createDimension("z", 2)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        variable = dataset.createVariable("votemper", "f4", ("time", "z", "y", "x"))
        variable.long_name = "synthetic potential temperature"
        variable.units = "Celsius"
        variable[:] = values
        time = dataset.createVariable("time", "i4", ("time",))
        time[:] = np.arange(12)
    return values


def _write_domain(path: Path) -> np.ndarray:
    top = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.int32)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        variable = dataset.createVariable("top_level", "i4", ("y", "x"))
        variable[:] = top
    return top > 0


def test_initial_sst_file_gets_exact_uniform_one_ulp_pattern(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    original = _write_temperature(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = run_dir / ulp_gate.TEMPERATURE_FILE
    target.symlink_to(source)
    wet = _write_domain(run_dir / ulp_gate.DOMAIN_FILE)

    audit = ulp_gate.perturb_initial_sst(run_dir)
    assert audit["changed_values"] == 12 * int(wet.sum())
    assert audit["exact_ulp_distance"] == 1
    assert not target.is_symlink()
    assert ulp_gate.verify_perturbation_manifest(run_dir) == audit

    with netCDF4.Dataset(target) as dataset:
        variable = dataset.variables["votemper"]
        variable.set_auto_maskandscale(False)
        actual = np.asarray(variable[:], dtype=np.float32)
    expected = original.copy()
    for record in range(12):
        expected[record, 0][wet] = np.nextafter(
            expected[record, 0][wet], np.float32(np.inf), dtype=np.float32
        )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    with netCDF4.Dataset(target, "r+") as dataset:
        variable = dataset.variables["votemper"]
        variable.set_auto_maskandscale(False)
        surface = np.asarray(variable[0, 0], dtype=np.float32)
        j, i = np.argwhere(wet)[0]
        surface[j, i] = np.nextafter(surface[j, i], np.float32(np.inf), dtype=np.float32)
        variable[0, 0] = surface
    with pytest.raises(ulp_gate.GateError, match="one-ULP pattern"):
        ulp_gate.verify_perturbation_manifest(run_dir)
