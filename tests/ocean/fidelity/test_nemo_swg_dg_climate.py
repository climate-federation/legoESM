"""Direct synthetic tests for the SWG_DG climate comparison instrument."""
from __future__ import annotations

import importlib.util
import json
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
    assert 0.0 <= out["top_octave_ke_fraction"][0] <= 1.0
    assert 0.0 <= out["top_octave_enstrophy_fraction"][0] <= 1.0
    assert np.all(out["native_checkerboard_index"] >= 0.0)


def test_checkerboard_index_has_a_live_grid_noise_plant():
    jj, ii = np.indices((8, 8))
    plant = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    assert climate.checkerboard_index(plant, np.zeros_like(plant)) == 1.0
    smooth = np.broadcast_to(np.linspace(-1.0, 1.0, 8), (8, 8))
    assert climate.checkerboard_index(smooth, np.zeros_like(smooth)) < 0.2


def test_identical_inputs_have_identical_spectra_and_zero_comparisons():
    providers = (_synthetic_provider("NEMO"),
                 _synthetic_provider("legoESM"))
    diagnosed = {provider.name: climate.diagnose(
        provider, days=2, geometry=_geometry()) for provider in providers}
    assert np.array_equal(
        diagnosed["NEMO"]["mean_kinetic_energy_spectrum_m3_s2"],
        diagnosed["legoESM"]["mean_kinetic_energy_spectrum_m3_s2"])
    comparisons = climate._comparison_summary(
        diagnosed["NEMO"], diagnosed["legoESM"])
    assert all(row["max_absolute_difference"] == 0.0
               for row in comparisons.values())


def test_comparison_refuses_different_daily_sampling():
    left = _synthetic_provider("NEMO", "daily_mean")
    right = _synthetic_provider("legoESM", "daily_snapshot")
    with pytest.raises(Exception, match="sampling mismatch"):
        climate.compare((left, right), days=2)


def test_supplied_card_requires_its_resolved_namelist():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
    )

    with pytest.raises(Exception, match="requires its resolved namelist"):
        climate.diagnose(
            _synthetic_provider(), days=2,
            card=build_swg_card(SWG_DG50_NAMELIST))


def test_instrument_writes_json_arrays_and_side_by_side_figure(tmp_path):
    providers = (_synthetic_provider("NEMO"),
                 _synthetic_provider("legoESM"))
    diagnosed = {provider.name: climate.diagnose(
        provider, days=2, geometry=_geometry()) for provider in providers}
    report = climate.write_outputs(
        tmp_path / "result", diagnosed, providers, days=2, start_day=2)
    assert report["window_days"] == [2, 3]
    assert set(report["metrics"]) == {"NEMO", "legoESM"}
    assert report["comparison"]["left"] == "NEMO"
    assert all(row["root_mean_square_difference"] == 0.0
               for row in report["comparison"]["metrics"].values())
    assert (tmp_path / "result/swg_dg_climate_report.json").is_file()
    assert (tmp_path / "result/swg_dg_climate_arrays.npz").is_file()
    assert (tmp_path / "result/swg_dg_climate_side_by_side.png").is_file()


def _write_nemo_fixture(root: Path, monkeypatch):
    import netCDF4

    executable = root / "nemo"
    executable.write_bytes(b"synthetic SWG_DG executable")
    monkeypatch.setattr(
        climate, "NEMO_DG_EXECUTABLE_SHA256", climate.sha256(executable))
    cfg = root / "namelist_cfg"
    ref = root / "namelist_ref"
    cfg.write_text("""\
&namusr_def
  rn_dx = 100000,
  jpkglo = 2,
/
&namlbc
  rn_shlat = 0.0,
/
""")
    ref.write_text("synthetic reference namelist\n")
    monkeypatch.setattr(
        climate, "NEMO_DG_NAMELIST_CFG_SHA256", climate.sha256(cfg))
    monkeypatch.setattr(
        climate, "NEMO_DG_NAMELIST_REF_SHA256", climate.sha256(ref))
    (root / "manifest.sha256").write_text(
        f"{climate.sha256(executable)}  nemo\n"
        f"{climate.sha256(cfg)}  namelist_cfg\n"
        f"{climate.sha256(ref)}  namelist_ref\n")
    (root / "resolved_namelist_lines.txt").write_text("""\
1: nn_SWG = 16
2: rn_domsiz = 4000000
3: rn_tau = 0.10
4: rn_modified_grav = 9.80665
5: rn_rfr = 1.e-7
6: rn_beta = 2.e-11
7: rn_f0 = 0.7e-4
8: rn_Uv = 2.0
9: rn_Lv = 10
10: rn_Dt = 50.
11: nn_write = 1728
""")
    jj, ii = np.indices((2, 2), dtype=np.float64)
    monkeypatch.setattr(climate, "_expected_coordinates", lambda: {
        kind: (ii, jj) for kind in "TUV"})
    for kind, variable in climate.NEMO_FIELDS.items():
        with netCDF4.Dataset(root / f"SWG_test_grid_{kind}_0000.nc", "w") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("y", 2)
            ds.createDimension("x", 2)
            ds.createVariable("time_counter", "f8", ("time",))[:] = [
                0.5 * climate.SECONDS_PER_DAY, 1.5 * climate.SECONDS_PER_DAY]
            ds.createVariable("nav_lon", "f8", ("y", "x"))[:] = ii
            ds.createVariable("nav_lat", "f8", ("y", "x"))[:] = jj
            field = ds.createVariable(variable, "f4", ("time", "y", "x"))
            field[:] = 0.0
            field.online_operation = "ave(x)"
            field.interval_operation = 50.0
            field.interval_write = climate.SECONDS_PER_DAY


def test_nemo_provider_admits_window_and_refuses_shifted_clock_and_deck(
        tmp_path, monkeypatch):
    _write_nemo_fixture(tmp_path, monkeypatch)
    provider = climate.nemo_ioipsl_provider(tmp_path, 1, start_day=2)
    assert provider.provenance["window_days"] == [2, 2]
    assert [frame.day for frame in provider.iterate()] == [1]
    import netCDF4
    path = tmp_path / "SWG_test_grid_T_0000.nc"
    with netCDF4.Dataset(path, "r+") as ds:
        ds["sossheig"].online_operation = "inst(x)"
    with pytest.raises(Exception, match="not a NEMO daily mean"):
        climate.nemo_ioipsl_provider(tmp_path, 1, start_day=2)
    with netCDF4.Dataset(path, "r+") as ds:
        ds["sossheig"].online_operation = "ave(x)"
    with netCDF4.Dataset(path, "r+") as ds:
        ds["time_counter"][1] += 1.0
    with pytest.raises(Exception, match="daily clock/window differs"):
        climate.nemo_ioipsl_provider(tmp_path, 1, start_day=2)
    resolved = tmp_path / "resolved_namelist_lines.txt"
    resolved.write_text(resolved.read_text().replace("rn_tau = 0.10",
                                                      "rn_tau = 0.11"))
    with pytest.raises(Exception, match="differs at rn_tau"):
        climate.nemo_ioipsl_provider(tmp_path, 1, start_day=2)
    resolved.write_text(resolved.read_text().replace("rn_tau = 0.11",
                                                      "rn_tau = 0.10").replace(
                                                          "nn_SWG = 16",
                                                          "nn_SWG = 16.5"))
    with pytest.raises(Exception, match="differs at nn_SWG"):
        climate.nemo_ioipsl_provider(tmp_path, 1, start_day=2)


def test_legoesm_provider_selects_later_window_and_gates_deck(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    pins = {}
    for day in range(1, 4):
        path = tmp_path / f"day{day:04d}.npz"
        field = np.full((644, 644), float(day))
        np.savez_compressed(
            path, day=day, dt_s=50.0,
            daily_mean_sampling=np.str_(
                "arithmetic mean of every post-step state in model day"),
            daily_mean_sample_count=np.int64(1_728),
            ssh_daily_mean_m=field, u_daily_mean_m_s=field,
            v_daily_mean_m_s=field)
        pins[path.name] = climate.sha256(path)
    manifest = {
        "status": "complete", "completed_days": 3, "days": 3,
        "schema": "legoesm-swg-double-gyre-daily-v2",
        "sampling": "daily_mean", "daily_sha256": pins,
        "namelist": SWG_MADDISON_NAMELIST._asdict(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    provider = climate.legoesm_npz_provider(tmp_path, 2, start_day=2)
    frames = list(provider.iterate())
    assert [frame.day for frame in frames] == [1, 2]
    assert [frame.ssh[0, 0] for frame in frames] == [2.0, 3.0]
    manifest["namelist"]["rn_tau_pa"] = 0.11
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="differs at rn_tau_pa"):
        climate.legoesm_npz_provider(tmp_path, 2, start_day=2)
    manifest["namelist"]["rn_tau_pa"] = 0.10
    manifest["namelist"]["rn_shlat"] = 2.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="differs at rn_shlat"):
        climate.legoesm_npz_provider(tmp_path, 2, start_day=2)


def test_filtered_provider_reads_target_stagger_and_refuses_bad_digest(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST

    pins = {}
    shape = (84, 84)
    for day in range(1, 3):
        path = tmp_path / f"day{day:04d}.npz"
        field = np.full(shape, float(day))
        np.savez(path, day=day, ssh_m=field, u_m_s=field, v_m_s=field)
        pins[path.name] = climate.sha256(path)
    manifest = {
        "schema": "legoesm-swg-conservative-truth-v2", "status": "complete",
        "completed_days": 2, "days": 2, "sampling": "daily_snapshot",
        "output_daily_sha256": pins,
        "target_namelist": SWG_DG50_NAMELIST._asdict(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    provider = climate.filtered_npz_provider(tmp_path, 1, start_day=2)
    frames = list(provider.iterate())
    assert len(frames) == 1 and frames[0].ssh[0, 0] == 2.0
    (tmp_path / "day0002.npz").write_bytes(b"plant")
    with pytest.raises(Exception, match="digest mismatch"):
        list(provider.iterate())
