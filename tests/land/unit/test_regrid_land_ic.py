"""The land-IC regridder: column order fidelity and identity round-trip."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, _REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_column_order_matches_the_spinup_driver():
    """The regridder addresses columns exactly as the spin-up driver did.

    The source state's column order was DEFINED by run_lmip_biophys's
    grid_latlon_rad; the regridder duplicates that logic (run scripts are not
    importable at package level), so this pins the two against drift — a
    silent divergence would scramble every soil column geographically while
    keeping all shapes valid.
    """
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    lb = _load("scripts/run/run_lmip_biophys.py", "_lb")

    class FakeLatLon:
        lat2d = np.linspace(-1.4, 1.4, 12).reshape(3, 4)
        lon2d = np.linspace(0.0, 6.2, 12).reshape(3, 4)

    class FakeMesh:
        latCell = np.linspace(-1.5, 1.5, 7)
        lonCell = np.linspace(0.0, 6.0, 7)

    for g in (FakeLatLon(), FakeMesh()):
        la, lo = rg._cols_rad(g)
        lb_la, lb_lo = lb.grid_latlon_rad(g)
        np.testing.assert_array_equal(la, np.asarray(lb_la))
        np.testing.assert_array_equal(lo, np.asarray(lb_lo))


def test_nearest_neighbour_identity():
    """Each source point is its own nearest neighbour — regrid to the same
    points returns the identity map, so state passes through untouched."""
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    rng = np.random.default_rng(0)
    lat = rng.uniform(-1.4, 1.4, 40)
    lon = rng.uniform(0.0, 6.2, 40)
    idx = nn(lat, lon, lat, lon)
    np.testing.assert_array_equal(idx, np.arange(40))


def test_nearest_neighbour_ignores_non_land_sources():
    """A target next to an excluded source gets the nearest INCLUDED one."""
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    near = nn(np.array([0.0, 0.5]), np.array([0.0, 0.0]),
              np.array([0.01]), np.array([0.0]),
              src_valid=np.array([False, True]))
    assert near[0] == 1


def test_regridding_latitude_returns_latitude():
    """Regrid the latitude field itself; a registration error is visible.

    The order pin and identity test above share the reconstruction logic with
    the script, so a CONSISTENTLY wrong convention (longitude 0-360 versus
    +/-180, pole ordering) would pass both.  Regridding a field that IS the
    coordinate catches that: each target column must receive a source latitude
    within one source cell of its own.
    """
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    src_lat, src_lon = np.meshgrid(
        np.deg2rad(np.arange(-89.0, 90.0, 2.0)),
        np.deg2rad(np.arange(0.0, 360.0, 2.0)), indexing="ij")
    sl, so = src_lat.ravel(), src_lon.ravel()
    rng = np.random.default_rng(1)
    tl = rng.uniform(-1.4, 1.4, 500)
    to = rng.uniform(-3.1, 6.2, 500)      # spans BOTH longitude conventions
    idx = nn(sl, so, tl, to)
    err = np.abs(sl[idx] - tl)
    assert err.max() < np.deg2rad(2.0), (
        f"a target column received a latitude {np.rad2deg(err.max()):.1f} deg "
        "away: the regridder's geometry is mis-registered")


# --- soil-hydraulics stamp through the regridder -----------------------------

def _source(tmp_path, *, stamp=None, extra=None, ncol=32, nlay=10):
    """A tiny multilayer restart on the 4x8 lat-lon grid."""
    import json
    rng = np.random.default_rng(0)
    fields = {
        "restart_version": np.array(2, dtype=np.int32),
        "land_mode": np.array("multilayer", dtype="U16"),
        "t_end_s": np.array(0.0), "n_steps_completed": np.array(0),
        "metadata_json": np.array(json.dumps(
            {"grid_type": "latlon", "resolution": 4})),
        "T_soil": 270.0 + rng.random((ncol, nlay)),
        "psi_soil": -1.0 - rng.random((ncol, nlay)),
        "theta_soil": 0.2 + 0.1 * rng.random((ncol, nlay)),
        "runoff_surface": np.zeros(ncol), "runoff_subsurface": np.zeros(ncol),
        "snow_depth": np.zeros(ncol), "snow_age": np.zeros(ncol),
    }
    if stamp is not None:
        fields["soil_hydraulics_json"] = np.array(json.dumps(stamp))
    fields.update(extra or {})
    p = tmp_path / "src.npz"
    np.savez(p, **fields)
    return p


def _argv(src, tmp_path, *more):
    return ["--source", str(src), "--surfdata", "unused.nc",
            "--target-grid", "latlon", "--target-resolution", "4",
            "--out", str(tmp_path / "out.npz"),
            "--source-soil-column", "10,3.0,2.0", *more]


def test_regridder_refuses_fields_it_does_not_remap(tmp_path):
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    src = _source(tmp_path, extra={"surface_water": np.zeros(32)})
    with pytest.raises(SystemExit, match="does not remap"):
        mod.main(_argv(src, tmp_path))


def test_regridder_needs_the_hydraulics_of_an_unstamped_source(tmp_path):
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    with pytest.raises(SystemExit, match="records no soil hydraulics"):
        mod.main(_argv(_source(tmp_path), tmp_path))


def test_regridder_will_not_overrule_a_stamped_source(tmp_path):
    from legoesm.land.restart import (
        HYDRAULICS_SOURCE_SURFDATA_COSBY, soil_hydraulics_stamp)
    pf = tmp_path / "cosby.nc"
    pf.write_bytes(b"cosby")
    stamp = soil_hydraulics_stamp("clapp_hornberger",
                                  HYDRAULICS_SOURCE_SURFDATA_COSBY, pf)
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    with pytest.raises(SystemExit, match="already records"):
        mod.main(_argv(_source(tmp_path, stamp=stamp), tmp_path,
                       "--source-soil-hydraulics", "clapp_hornberger",
                       "surfdata_cosby", str(pf)))


def test_regridder_writes_the_attested_stamp(tmp_path, monkeypatch):
    from legoesm.land.restart import (
        HYDRAULICS_SOURCE_SURFDATA_COSBY, file_md5, load_land_restart,
        soil_hydraulics_stamp, soil_hydraulics_stamps_match)
    pf = tmp_path / "cosby.nc"
    pf.write_bytes(b"cosby")
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    monkeypatch.setattr(mod, "_source_land_mask",
                        lambda surfdata, grid, n: np.ones(n, dtype=bool))
    src = _source(tmp_path)
    assert mod.main(_argv(src, tmp_path, "--source-soil-hydraulics",
                          "clapp_hornberger", "surfdata_cosby", str(pf))) == 0
    _, meta = load_land_restart(tmp_path / "out.npz",
                                expected_land_mode="multilayer",
                                expected_ncol=32)
    got = meta["soil_hydraulics"]
    assert soil_hydraulics_stamps_match(got, soil_hydraulics_stamp(
        "clapp_hornberger", HYDRAULICS_SOURCE_SURFDATA_COSBY, pf))
    assert got["attested"] is True
    assert got["attested_source_md5"] == file_md5(src)
    # The regridded flag is what forces a full conversion on load, now that
    # the source grid's column signature is dropped.
    assert meta["metadata"]["regridded_from"] == src.name


def test_stamp_only_adds_the_stamp_and_changes_nothing_else(tmp_path):
    from legoesm.land.restart import load_land_restart
    pf = tmp_path / "cosby.nc"
    pf.write_bytes(b"cosby")
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    src = _source(tmp_path)
    out = tmp_path / "stamped.npz"
    assert mod.main(["--source", str(src), "--out", str(out), "--stamp-only",
                     "--source-soil-hydraulics", "clapp_hornberger",
                     "surfdata_cosby", str(pf)]) == 0
    a, b = np.load(src), np.load(out)
    assert set(b.files) == set(a.files) | {"soil_hydraulics_json"}
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k])
    _, meta = load_land_restart(out, expected_land_mode="multilayer",
                                expected_ncol=32)
    assert meta["soil_hydraulics"]["retention_curve"] == "clapp_hornberger"
    with pytest.raises(SystemExit, match="already"):
        mod.main(["--source", str(out), "--out", str(tmp_path / "x.npz"),
                  "--stamp-only"])


def test_regridder_drops_the_source_grids_column_signature(tmp_path, monkeypatch):
    from legoesm.land.restart import (
        HYDRAULICS_SOURCE_SURFDATA_COSBY, soil_hydraulics_stamp)
    pf = tmp_path / "cosby.nc"
    pf.write_bytes(b"cosby")
    stamp = soil_hydraulics_stamp("clapp_hornberger",
                                  HYDRAULICS_SOURCE_SURFDATA_COSBY, pf)
    mod = _load("scripts/data/regrid_land_ic.py", "regrid_land_ic")
    monkeypatch.setattr(mod, "_source_land_mask",
                        lambda surfdata, grid, n: np.ones(n, dtype=bool))
    src = _source(tmp_path, stamp=stamp, extra={
        "soil_hydraulics_column_sig": np.arange(32, dtype=np.uint64)})
    assert mod.main(_argv(src, tmp_path)) == 0
    out = np.load(tmp_path / "out.npz")
    assert "soil_hydraulics_column_sig" not in out.files
    assert "soil_hydraulics_json" in out.files
