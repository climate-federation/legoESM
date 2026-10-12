"""Direct synthetic tests for the SWG_DG climate comparison instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "scripts/validate/"
          "ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_climate.py")
SPEC = importlib.util.spec_from_file_location("swg_dg_climate", SCRIPT)
climate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(climate)


def test_coordinate_assembly_deduplicates_halos_and_refuses_bad_coverage():
    lon = np.tile(np.arange(3, dtype=np.float64), (2, 1))
    lat = np.tile(np.arange(2, dtype=np.float64)[:, None], (1, 3))
    values = 10.0 * lat + lon
    tiles = [
        (lon[:, :2], lat[:, :2], values[:, :2]),
        (lon[:, 1:], lat[:, 1:], values[:, 1:]),
    ]
    assert np.array_equal(
        climate.assemble_coordinate_tiles(tiles, lon, lat), values)
    with pytest.raises(Exception, match="missed"):
        climate.assemble_coordinate_tiles(tiles[:1], lon, lat)
    conflicting = list(tiles)
    conflicting[1] = tuple(a.copy() for a in conflicting[1])
    conflicting[1][2][0, 0] += 1.0
    with pytest.raises(Exception, match="conflicting"):
        climate.assemble_coordinate_tiles(conflicting, lon, lat)


def _synthetic_provider(name="synthetic", sampling="daily_mean"):
    rows = np.array([1.0, 2.0, 4.0, 3.0])[:, None]
    day1 = np.broadcast_to(rows, (4, 4)).copy()
    day2 = day1 + 2.0

    def iterate():
        yield climate.DailyFrame(1, np.zeros((4, 4)), day1, np.zeros((4, 4)))
        yield climate.DailyFrame(2, np.full((4, 4), 2.0), day2, np.zeros((4, 4)))

    return climate.DailyProvider(name, sampling, {"synthetic": True}, iterate)


def _geometry():
    jj, ii = np.indices((4, 4), dtype=np.float64)
    masks = {kind: np.ones((4, 4), dtype=bool) for kind in "TUVF"}
    return {"masks": masks, "area": np.ones((4, 4)),
            "x": ii, "y": jj, "spacing_m": 1.0}


def test_synthetic_metrics_have_known_answer():
    out = climate.diagnose(_synthetic_provider(), days=2, geometry=_geometry())
    expected_mean_u = np.broadcast_to(
        np.array([2.0, 3.0, 5.0, 4.0])[:, None], (4, 4)).copy()
    expected_mean_u[:, 0] *= 0.5
    assert np.array_equal(out["mean_ssh_m"], np.ones((4, 4)))
    assert np.array_equal(out["mean_u_m_s"], expected_mean_u)
    assert np.array_equal(out["mean_v_m_s"], np.zeros((4, 4)))
    assert np.array_equal(out["mean_speed_m_s"], expected_mean_u)
    expected_eke = np.full((4, 4), 0.5)
    expected_eke[:, 0] = 0.125
    assert np.array_equal(out["eke_map_m2_s2"], expected_eke)
    assert np.array_equal(out["basin_mean_eke_m2_s2"], [0.40625, 0.40625])
    day1_c = np.broadcast_to(
        np.array([1.0, 2.0, 4.0, 3.0])[:, None], (4, 4)).copy()
    day2_c = day1_c + 2.0
    day1_c[:, 0] *= 0.5
    day2_c[:, 0] *= 0.5
    expected_ke = [np.mean(0.5 * day1_c ** 2), np.mean(0.5 * day2_c ** 2)]
    assert np.array_equal(out["basin_mean_ke_m2_s2"], expected_ke)
    assert np.array_equal(out["jet_y_m"], np.full(4, 2.0))
    assert np.array_equal(out["jet_eastward_speed_m_s"], expected_mean_u[2])
    assert out["spectrum_wavenumber_m-1"].shape == (2,)
    assert np.all(np.isfinite(out["mean_kinetic_energy_spectrum_m3_s2"]))


def test_comparison_refuses_different_daily_sampling():
    left = _synthetic_provider("NEMO", "daily_mean")
    right = _synthetic_provider("legoESM", "daily_snapshot")
    with pytest.raises(Exception, match="sampling mismatch"):
        climate.compare((left, right), days=2)


def test_instrument_writes_json_arrays_and_side_by_side_figure(tmp_path):
    providers = (_synthetic_provider("NEMO"),
                 _synthetic_provider("legoESM"))
    diagnosed = {provider.name: climate.diagnose(
        provider, days=2, geometry=_geometry()) for provider in providers}
    report = climate.write_outputs(
        tmp_path / "result", diagnosed, providers, days=2)
    assert report["window_days"] == [1, 2]
    assert set(report["metrics"]) == {"NEMO", "legoESM"}
    assert (tmp_path / "result/swg_dg_climate_report.json").is_file()
    assert (tmp_path / "result/swg_dg_climate_arrays.npz").is_file()
    assert (tmp_path / "result/swg_dg_climate_side_by_side.png").is_file()
