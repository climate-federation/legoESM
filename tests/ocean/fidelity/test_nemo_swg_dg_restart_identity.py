"""Direct synthetic tests for the SWG_DG yearly restart checker."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_l1_swg_dg_restart_identity.py")
_SPEC = importlib.util.spec_from_file_location("swg_dg_restart_identity", _PATH)
identity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(identity)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _fields(value=1.0):
    return {
        name: np.full((2, 2), value, dtype=np.float64)
        for name in identity.FIELD_MAPPING
    }


def _masks():
    return {kind: np.ones((2, 2), dtype=bool) for kind in "TUV"}


def test_every_declared_wet_word_is_exact_and_capture_mapping_is_explicit():
    rows = identity.compare_fields(_fields(), _fields(), _masks())
    report = identity.build_report(
        day=365, step=630_720, rows=rows,
        nemo_provenance={"synthetic": True},
        lego_provenance={"synthetic": True})
    assert report["status"] == "PASS"
    assert report["summary"] == {
        "fields": 3,
        "wet_words": 12,
        "bit_unequal": 0,
        "max_abs": 0.0,
        "all_bit_identical": True,
        "prognostic_signed_zero_words": 0,
    }
    assert report["field_mapping"]["u_m_s"]["nemo"] == "un"
    assert report["field_mapping"]["v_m_s"]["nemo_index"] == "[0, 0]"
    assert "uu_n/vv_n" in report["capture_instant"]["excluded"]


def test_prognostic_signed_zero_is_debt_not_a_forcing_exception():
    nemo = _fields()
    lego = _fields()
    lego["ssh_m"][0, 0] = -0.0
    nemo["ssh_m"][0, 0] = 0.0
    rows = identity.compare_fields(nemo, lego, _masks())
    assert rows["ssh_m"]["bit_unequal"] == 1
    assert rows["ssh_m"]["signed_zero_only"] == 1
    report = identity.build_report(
        day=1, step=1_728, rows=rows,
        nemo_provenance={}, lego_provenance={})
    assert report["status"] == "DEBT"
    assert report["summary"]["prognostic_signed_zero_words"] == 1


def test_mask_excludes_dry_words_but_one_ulp_wet_plant_binds():
    nemo = _fields()
    lego = _fields()
    masks = _masks()
    masks["T"][0, 0] = False
    lego["ssh_m"][0, 0] = np.nextafter(1.0, np.inf)
    lego["ssh_m"][0, 1] = np.nextafter(1.0, np.inf)
    rows = identity.compare_fields(nemo, lego, masks)
    assert rows["ssh_m"]["wet_words"] == 3
    assert rows["ssh_m"]["bit_unequal"] == 1
    assert rows["ssh_m"]["signed_ulp_histogram"] == {"0": 2, "1": 1}


def test_restart_tile_loader_maps_un_vn_level_zero_and_reports_workspace(
        tmp_path):
    import netCDF4

    lon = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    lat = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    for rank, column in enumerate((slice(0, 1), slice(1, 2))):
        path = tmp_path / f"SWG_00001728_restart_{rank:04d}.nc"
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("t", 1)
            ds.createDimension("z", 2)
            ds.createDimension("y", 2)
            ds.createDimension("x", 1)
            ds.createVariable("kt", "f8").assignValue(1728.0)
            ds.createVariable("rdt", "f8").assignValue(50.0)
            ds.createVariable("nav_lon", "f4", ("y", "x"))[:] = lon[:, column]
            ds.createVariable("nav_lat", "f4", ("y", "x"))[:] = lat[:, column]
            for name in ("sshn", "uu_n", "vv_n"):
                variable = ds.createVariable(name, "f8", ("t", "y", "x"))
                variable[:] = 0.0 if name != "sshn" else lon[:, column] + 1.0
            for name in ("un", "vn"):
                variable = ds.createVariable(name, "f8", ("t", "z", "y", "x"))
                variable[:] = 0.0
                variable[0, 0] = lon[:, column] + (2.0 if name == "un" else 3.0)
    coordinates = {kind: (lon, lat) for kind in "TUV"}
    fields, provenance = identity.load_nemo_restart(
        tmp_path, 1_728, coordinates)
    assert np.array_equal(fields["ssh_m"], lon + 1.0)
    assert np.array_equal(fields["u_m_s"], lon + 2.0)
    assert np.array_equal(fields["v_m_s"], lon + 3.0)
    assert provenance["rank_files"] == 2
    assert provenance["excluded_restart_fields"]["uu_n"]["nonzero_words"] == 0


def test_restart_loader_refuses_non_fp64_prognostic_storage(tmp_path):
    import netCDF4

    path = tmp_path / "SWG_00001728_restart.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("z", 2)
        ds.createDimension("y", 1)
        ds.createDimension("x", 1)
        ds.createVariable("kt", "f8").assignValue(1728.0)
        ds.createVariable("rdt", "f8").assignValue(50.0)
        ds.createVariable("nav_lon", "f4", ("y", "x"))[:] = 0.0
        ds.createVariable("nav_lat", "f4", ("y", "x"))[:] = 0.0
        ds.createVariable("sshn", "f4", ("t", "y", "x"))[:] = 0.0
        for name in ("uu_n", "vv_n"):
            ds.createVariable(name, "f8", ("t", "y", "x"))[:] = 0.0
        for name in ("un", "vn"):
            ds.createVariable(name, "f8", ("t", "z", "y", "x"))[:] = 0.0
    coordinates = {kind: (np.zeros((1, 1)), np.zeros((1, 1))) for kind in "TUV"}
    with pytest.raises(RuntimeError, match="sshn restart storage is not float64"):
        identity.load_nemo_restart(tmp_path, 1_728, coordinates)


def test_mesh_loader_assembles_all_three_masks_and_refuses_bad_values(tmp_path):
    import netCDF4

    path = tmp_path / "mesh_mask.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("z", 2)
        ds.createDimension("y", 1)
        ds.createDimension("x", 2)
        ds.createVariable("nav_lon", "f4", ("y", "x"))[:] = [[0.0, 1.0]]
        ds.createVariable("nav_lat", "f4", ("y", "x"))[:] = [[0.0, 0.0]]
        for name in ("tmask", "umask", "vmask"):
            variable = ds.createVariable(name, "i1", ("t", "z", "y", "x"))
            variable[:] = 0
            variable[0, 0] = [[1, 0]]
    coordinates = {
        kind: (np.asarray([[0.0, 1.0]]), np.asarray([[0.0, 0.0]]))
        for kind in "TUV"
    }
    masks, provenance = identity.load_nemo_mesh_masks(tmp_path, coordinates)
    assert all(np.array_equal(mask, [[True, False]]) for mask in masks.values())
    assert provenance["rank_files"] == 1
    with netCDF4.Dataset(path, "a") as ds:
        ds["vmask"][0, 0, 0, 0] = 2
    with pytest.raises(RuntimeError, match="other than zero or one"):
        identity.load_nemo_mesh_masks(tmp_path, coordinates)


def test_legoesm_loader_refuses_fp32_even_when_other_admission_checks_pass(
        tmp_path, monkeypatch):
    endpoint = tmp_path / "day0001.npz"
    np.savez(
        endpoint, day=np.asarray(1), step=np.asarray(1_728),
        dt_s=np.asarray(50.0),
        **{name: np.ones((1, 1), dtype=np.float32)
           for name in identity.FIELD_MAPPING})
    manifest = {
        "status": "complete", "mode": "gpu-native",
        "integration": "one-jitted-scan-per-day", "steps_per_day": 1_728,
        "daily_sha256": {endpoint.name: identity.sha256(endpoint)},
        "namelist": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(identity, "require_legoesm_deck", lambda _: None)
    with pytest.raises(RuntimeError, match="endpoint storage is not float64"):
        identity.load_legoesm_endpoint(endpoint, day=1, step=1_728)


@pytest.mark.parametrize(("year", "day", "step"), (
    (1, 365, 630_720),
    (2, 730, 1_261_440),
    (3, 1_095, 1_892_160),
))
def test_committed_annual_reports_are_citable_exact_results(year, day, step):
    path = (_REPO_ROOT / "docs/ocean/fidelity/testcases"
            / f"swg_dg_year{year}_restart_identity.json")
    report = json.loads(path.read_text())
    assert report["status"] == "PASS"
    assert (report["day"], report["step"]) == (day, step)
    assert report["summary"] == {
        "all_bit_identical": True,
        "bit_unequal": 0,
        "fields": 3,
        "max_abs": 0.0,
        "prognostic_signed_zero_words": 0,
        "wet_words": 1_227_520,
    }
