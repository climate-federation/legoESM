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


def test_registered_default_full_json_is_branch_point_pinned():
    summary = climate._summary(climate.diagnose(
        _synthetic_provider(), days=2, geometry=_geometry()))
    expected = json.loads(r'''{
      "basin_mean_eke_m2_s2": {"max": 0.40625, "mean": 0.40625,
        "min": 0.40625,
        "sha256": "384a0491b37cd8f232fcae8a8ab99fc08e1edef47c36fbda6dd6a97447711b2b",
        "shape": [2]},
      "basin_mean_ke_m2_s2": {"max": 8.734375, "mean": 5.890625,
        "min": 3.046875,
        "sha256": "c377d4bda7c6969af58351ff7e6900255206e9e6132df1225b6dcbdd63921a26",
        "shape": [2]},
      "eke_map_m2_s2": {"max": 0.5, "mean": 0.40625, "min": 0.125,
        "sha256": "3e6859241ea14bdda66d652316549b1e2eb4aea8bf47d3e59a949c0d1f280718",
        "shape": [4, 4]},
      "jet_eastward_speed_m_s": {"max": 5.0, "mean": 4.375, "min": 2.5,
        "sha256": "b2980b76ddc11c35bc2f9b03ae3bc1162f93c353c6aa281167a9ff2f840c7b9d",
        "shape": [4]},
      "jet_x_m": {"max": 3.0, "mean": 1.5, "min": 0.0,
        "sha256": "9392b85eaba90b4aa6f39e1f269927b4bd6bec47cd2e34a80cf3ed914c26dc7e",
        "shape": [4]},
      "jet_y_m": {"max": 2.0, "mean": 2.0, "min": 2.0,
        "sha256": "cd34cc32a2fbe9220600b249a3fe6562dc85f45cad9f1ddd32b535591fb78415",
        "shape": [4]},
      "mean_kinetic_energy_spectrum_m3_s2": {"max": 3.2734375,
        "mean": 1.63671875, "min": 0.0,
        "sha256": "12c55abc0051072d5a2d3c2b18b225376026f7ca7ddd2339ac9365c1d93d3bad",
        "shape": [2]},
      "mean_speed_m_s": {"max": 5.0, "mean": 3.0625, "min": 1.0,
        "sha256": "6fd3f618c0ec4a970ae65b06f8ba01004ce73925596f4739778076bf1e318108",
        "shape": [4, 4]},
      "mean_ssh_m": {"max": 1.0, "mean": 1.0, "min": 1.0,
        "sha256": "dbd8ebb4d364765882694170cedf5afc7702dd5338a7198705c45b613f6e1c9d",
        "shape": [4, 4]},
      "mean_u_m_s": {"max": 5.0, "mean": 3.0625, "min": 1.0,
        "sha256": "6fd3f618c0ec4a970ae65b06f8ba01004ce73925596f4739778076bf1e318108",
        "shape": [4, 4]},
      "mean_v_m_s": {"max": 0.0, "mean": 0.0, "min": 0.0,
        "sha256": "38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca",
        "shape": [4, 4]},
      "spectrum_wavenumber_m-1": {"max": 0.25, "mean": 0.125,
        "min": 0.0, "sha256": "bf881abdd02907ef646666ca7352c12628539f6c4ef7e95958c0a4de7446f2c5",
        "shape": [2]}
    }''')
    assert summary == expected


def _masked_case():
    """6x6 box, 4x4 wet interior; land carries garbage the masks must drop."""
    jj, ii = np.indices((6, 6), dtype=np.float64)
    wet = (jj >= 1) & (jj <= 4) & (ii >= 1) & (ii <= 4)
    umask = np.zeros((6, 6), dtype=bool)
    umask[1:5, 1:4] = True       # U[j,i] joins T[j,i] and T[j,i+1]
    vmask = np.zeros((6, 6), dtype=bool)
    vmask[1:4, 1:5] = True       # V[j,i] joins T[j,i] and T[j+1,i]
    geometry = {"masks": {"T": wet, "U": umask, "V": vmask, "F": wet},
                "area": 1.0 + 0.125 * ii, "x": 10.0 * ii, "y": 10.0 * jj,
                "spacing_m": 1.0}
    frames = []
    for day in range(1, 4):
        u = 0.25 * day * np.sin(jj + 0.5 * ii) + 0.5
        v = 0.75 * np.cos(ii - 0.25 * day * jj)
        ssh = 0.1 * day * jj - 0.05 * ii
        frames.append(climate.DailyFrame(
            day, np.where(wet, ssh, 55.0), np.where(umask, u, 99.0),
            np.where(vmask, v, -77.0)))

    def iterate():
        yield from frames

    return climate.DailyProvider("pin", "daily_snapshot", {}, iterate), geometry


_MASKED_PIN = r'''{
  "basin_mean_eke_m2_s2": {"max": 0.028424933772348143,
    "mean": 0.01896185962853167, "min": 0.0010505878329638626,
    "sha256": "4268cc7ae96eeac4a7cba88cb0c3c1c588a11d54b3cf6a1785b6bae038d2cb89",
    "shape": [3]},
  "basin_mean_ke_m2_s2": {"max": 0.21559932026309608,
    "mean": 0.19498477460505922, "min": 0.17659295450427445,
    "sha256": "117d10baaea7acab53b903c7bc02199a0c156fce0b4706488846e4f4039e395f",
    "shape": [3]},
  "eke_map_m2_s2": {"max": 0.0740494825835769,
    "mean": 0.008361312036650479, "min": 0.0,
    "sha256": "371421d263364ff83ae0e49b3a5f57f9ee53cf43bc4e426c0cd01f49cb5b93a5",
    "shape": [6, 6]},
  "jet_eastward_speed_m_s": {"max": 0.976698103357434,
    "mean": 0.6881580696917116, "min": 0.3996180360259891,
    "sha256": "bd2934496223c2fdaf3fa43557c6182ba7a288dbb4d783350571c0d5244b098c",
    "shape": [6]},
  "jet_x_m": {"max": 40.0, "mean": 25.0, "min": 10.0,
    "sha256": "2ac3f30fc76ecd10be081d75a559819ea5972d982d5fd2fe82adb81e61114518",
    "shape": [6]},
  "jet_y_m": {"max": 10.0, "mean": 10.0, "min": 10.0,
    "sha256": "68973682a61863f552227624948fb41e07631d200a8e8ea712d697dfb0f2b224",
    "shape": [6]},
  "mean_kinetic_energy_spectrum_m3_s2": {"max": 0.5665374676491437,
    "mean": 0.28326873382457185, "min": 7.16606433474084e-33,
    "sha256": "9475b7f8fc9a2fdc4c8b526127e1bbbb6a59964e089f0a6c8a268f7f33725653",
    "shape": [2]},
  "mean_speed_m_s": {"max": 0.979738016058215,
    "mean": 0.2554680002129197, "min": 0.0,
    "sha256": "82329b6919b8ccb282d7df8dd757940e2ca921eb22207d17b397074e530c3a01",
    "shape": [6, 6]},
  "mean_ssh_m": {"max": 0.75, "mean": 0.16666666666666666, "min": 0.0,
    "sha256": "0a9cf5f812b00a6ce2c458d35a4a3f52d4ce2871bb7ff72e3b7ce08c48894766",
    "shape": [6, 6]},
  "mean_u_m_s": {"max": 0.976698103357434,
    "mean": 0.1412078183812031, "min": 0.0,
    "sha256": "5d323e8501e6f3a4bf890a1572df14964daa0c1c87adb576d3c19b2b45f33757",
    "shape": [6, 6]},
  "mean_v_m_s": {"max": 0.6666686220319479,
    "mean": 0.003994728883000511, "min": -0.6848423150186349,
    "sha256": "b2f809f7becba9e14abfc995e5936a3d215d2f6adea9d6ba85582bab0546cb1c",
    "shape": [6, 6]},
  "spectrum_wavenumber_m-1": {"max": 0.25, "mean": 0.125, "min": 0.0,
    "sha256": "bf881abdd02907ef646666ca7352c12628539f6c4ef7e95958c0a4de7446f2c5",
    "shape": [2]}
}'''


def test_registered_default_pin_covers_dry_cells_masks_and_v_path():
    """Review EV-06: the 4x4 all-wet, v=0 pin could not see dry-cell NaNs,
    mask handling, or anything on the v path.  This case has land faces
    carrying garbage, a nonzero v, a non-uniform area and a jet column whose
    row changes if v leaks into it.  Values were produced by the branch-point
    file (8f8551c87) and by this file, and agree exactly.
    """
    provider, geometry = _masked_case()
    summary = climate._summary(climate.diagnose(provider, days=3,
                                                geometry=geometry))
    assert summary == json.loads(_MASKED_PIN)


def test_local_grid_noise_index_catches_phase_reversed_checkerboard():
    jj, ii = np.indices((8, 8))
    plant = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    first = climate.grid_noise_index(plant, np.zeros_like(plant))
    second = climate.grid_noise_index(-plant, np.zeros_like(plant))
    assert first > 2.8 and second == first
    smooth = np.broadcast_to(np.linspace(-1.0, 1.0, 8), (8, 8))
    assert climate.grid_noise_index(smooth, np.zeros_like(smooth)) < 1.0e-12


def test_dst_streamfunction_spectrum_resolves_a_closed_basin_mode():
    n = 24
    y = np.arange(1, n + 1)[:, None]
    x = np.arange(1, n + 1)[None, :]
    psi = np.sin(2 * np.pi * y / (n + 1)) * np.sin(3 * np.pi * x / (n + 1))
    u = -np.gradient(psi, axis=0, edge_order=2)
    v = np.gradient(psi, axis=1, edge_order=2)
    _, ke, enstrophy = climate.closed_basin_spectra(u, v, 1.0)
    expected_shell = int(round(np.sqrt(2**2 + 3**2)))
    assert np.argmax(ke) + 1 == expected_shell
    assert np.argmax(enstrophy) + 1 == expected_shell


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


def test_online_diagnostics_require_card_namelist_and_refuse_unknown():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
    )

    with pytest.raises(Exception, match="require a card and resolved namelist"):
        climate.diagnose(
            _synthetic_provider(), days=2,
            card=build_swg_card(SWG_DG50_NAMELIST), diagnostics="online")
    with pytest.raises(ValueError, match="unknown climate diagnostics"):
        climate.diagnose(
            _synthetic_provider(), days=2, geometry=_geometry(),
            diagnostics="post-hoc")


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


def test_online_control_namelist_selects_viscosity_and_refuses_unknown():
    assert climate.online_control_namelist("dg50").rn_Lv_m == 10.0
    assert climate.online_control_namelist("dg50-nu80").rn_Lv_m == 80.0
    with pytest.raises(Exception, match="unsupported online control"):
        climate.online_control_namelist("post-hoc")
