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
# --- ERA5 soil-temperature mode ---------------------------------------------

_DZ = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])   # interfaces 0,.05,.15,.35,.85,1.85,3.85 m
# stl1..4 = 1,2,3,4 (+ an offset): the overlap-weighted value per layer, by hand.
_EXPECTED_OFFSETS = np.array([1.0, 1.8, 2.35, 3.0, 3.85, 4.0])


def test_overlap_weights_match_hand_computed_profile():
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    w = rg.overlap_weights(_DZ)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(w @ np.array([1.0, 2.0, 3.0, 4.0]),
                               _EXPECTED_OFFSETS, rtol=0, atol=1e-12)


def test_era5_soil_temperature_replaces_land_only_with_land_donors():
    """Replaced columns carry the depth-mapped ERA5 profile of their nearest
    ERA5 LAND point (not the nearer ocean point); other columns are untouched.
    Fails if the ERA5 field is not used."""
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    src_lat = np.deg2rad(np.array([10.0, 10.0]))
    src_lon = np.deg2rad(np.array([0.0, 1.0]))          # 0 = ocean (nearer), 1 = land
    src_land = np.array([False, True])
    stl = np.array([[281.0, 291.0], [282.0, 292.0], [283.0, 293.0], [284.0, 294.0]])
    dst_lat = np.deg2rad(np.array([10.0, 50.0, -70.0]))
    dst_lon = np.deg2rad(np.array([0.1, 20.0, 0.0]))
    replace = np.array([True, False, False])             # land / not-land / glacier
    T_ic = np.full((3, _DZ.size), 250.0)
    T, dist = rg.era5_soil_temperature(T_ic, _DZ, dst_lat, dst_lon, replace,
                                       src_lat, src_lon, stl, src_land)
    np.testing.assert_allclose(T[0], 290.0 + _EXPECTED_OFFSETS, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(T[1:], T_ic[1:])
    assert 95.0 < dist[0] < 105.0                        # 0.9 deg of longitude at 10N
    assert np.isnan(dist[1:]).all()


def test_nearest_column_map_chunked_equals_dense(monkeypatch):
    import legoesm.coupler.grid_remap as gr
    rng = np.random.default_rng(3)
    sl, so = rng.uniform(-1.5, 1.5, 300), rng.uniform(0.0, 6.28, 300)
    tl, to = rng.uniform(-1.5, 1.5, 257), rng.uniform(0.0, 6.28, 257)
    dense = gr.nearest_column_map(sl, so, tl, to)
    monkeypatch.setattr(gr, "_NN_CHUNK_ELEMS", 300 * 7)   # 7 targets per chunk
    np.testing.assert_array_equal(gr.nearest_column_map(sl, so, tl, to), dense)
    assert gr.nearest_column_map(sl, so, tl[:0], to[:0]).size == 0


def _write_era5_nc(path, var, values, lat_deg, lon_deg, time="1979-01-01"):
    import xarray as xr
    xr.Dataset(
        {var: (("time", "lat", "lon"), values[None])},
        coords={"time": np.array([time], dtype="datetime64[ns]"),
                "lat": lat_deg, "lon": lon_deg}).to_netcdf(path)


def _era5_test_stamp(tmp_path):
    from legoesm.land.restart import (
        HYDRAULICS_SOURCE_SURFDATA_COSBY, soil_hydraulics_stamp)
    pf = tmp_path / "cosby_era5.nc"
    pf.write_bytes(b"cosby")
    return soil_hydraulics_stamp("clapp_hornberger",
                                 HYDRAULICS_SOURCE_SURFDATA_COSBY, pf)


def test_era5_soil_t_refuses_an_unstamped_source(tmp_path):
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    src = tmp_path / "ic.npz"
    np.savez(src, soil_dz=np.ones(3), T_soil=np.zeros((2, 3)))
    with pytest.raises(SystemExit, match="soil-hydraulics stamp"):
        rg.main(["--source", str(src), "--era5-soil-t", "a", "b", "c", "d",
                 "--era5-lsm", "l", "--out", str(tmp_path / "o.npz")])


def test_cli_swaps_only_soil_temperature_and_round_trips(tmp_path, monkeypatch):
    """End to end: only T_soil of replaced columns changes, every other field is
    byte-identical, provenance is stamped, and the driver's restart loader reads
    back exactly the mapped profile, layer by layer (the t=0 check)."""
    import json
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    import legoesm.grids.factory as gf
    from legoesm.land.restart import load_land_restart, save_land_restart
    from legoesm.land.state import MultiLayerLandState

    ncol, dz = 3, np.array([0.0029, 0.0059, 0.0117, 0.0235, 0.047, 0.094,
                            0.187, 0.376, 0.751, 1.501])
    rng = np.random.default_rng(5)
    nl = dz.size

    class Mesh:
        latCell = np.deg2rad(np.array([10.0, 50.0, -70.0]))
        lonCell = np.deg2rad(np.array([0.1, 20.0, 0.0]))

    monkeypatch.setattr(gf, "create_grid", lambda *a, **k: Mesh())
    monkeypatch.setattr(rg, "_land_and_glacier", lambda s, g, n: (
        np.array([True, False, True]), np.array([False, False, True])))

    ic_path = tmp_path / "ic.npz"
    save_land_restart(
        ic_path,
        MultiLayerLandState(
            T_soil=250.0 + rng.uniform(0, 20, (ncol, nl)),
            psi_soil=-rng.uniform(0.1, 5, (ncol, nl)),
            theta_soil=rng.uniform(0.1, 0.4, (ncol, nl)),
            runoff_surface=rng.uniform(0, 1e-5, ncol),
            runoff_subsurface=rng.uniform(0, 1e-5, ncol),
            snow_depth=rng.uniform(0, 50, ncol),
            snow_age=rng.uniform(0, 1e6, ncol)),
        land_mode="multilayer", t_end_s=3.0e7, n_steps_completed=10,
        metadata={"origin": "synthetic"}, soil_dz=dz,
        soil_hydraulics=_era5_test_stamp(tmp_path))
    ic = dict(np.load(ic_path, allow_pickle=False))

    lat, lon = np.array([10.0, -10.0]), np.array([0.0, 1.0])
    files = []
    for n, off in enumerate((1.0, 2.0, 3.0, 4.0)):
        f = tmp_path / f"stl{n + 1}.nc"
        _write_era5_nc(f, f"var{139 + n}", np.full((2, 2), 280.0 + off), lat, lon)
        files.append(str(f))
    lsm = tmp_path / "lsm.nc"
    _write_era5_nc(lsm, "var172", np.array([[0.0, 1.0], [1.0, 1.0]]), lat, lon,
                   time="2000-01-01")
    out = tmp_path / "out.npz"
    assert rg.main(["--source", str(ic_path), "--surfdata", "unused",
                    "--target-grid", "mpas", "--target-resolution", "6",
                    "--era5-soil-t", *files, "--era5-lsm", str(lsm),
                    "--out", str(out)]) == 0

    new = np.load(out, allow_pickle=False)
    for k in ic:
        if k not in ("T_soil", "metadata_json"):
            np.testing.assert_array_equal(new[k], ic[k], err_msg=k)
    expected = 280.0 + rg.overlap_weights(ic["soil_dz"]) @ np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(new["T_soil"][0], expected, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(new["T_soil"][1:], ic["T_soil"][1:])   # ocean + glacier
    meta = json.loads(str(new["metadata_json"]))
    assert meta["soil_t_donor_km_max"] > 0 and "soil_t_git_sha" in meta
    assert set(meta["soil_t_files"]) == {pathlib.Path(f).name for f in files}

    state, lmeta = load_land_restart(out, expected_land_mode="multilayer",
                                     expected_ncol=ncol,
                                     expected_n_layers=ic["T_soil"].shape[1])
    assert lmeta["soil_hydraulics"] == json.loads(str(ic["soil_hydraulics_json"]))
    np.testing.assert_allclose(np.asarray(state.T_soil)[0], expected, rtol=0, atol=1e-4)


def test_warm_era5_soil_is_ice_consistent_with_freeze_thaw_on():
    """ERA5-warm soil over spin-up water: diagnosed liquid stays within
    [residual, total] and the apparent heat capacity stays finite and above the
    sensible floor, so the T-only swap cannot create an impossible ice state."""
    import jax.numpy as jnp
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    from legoesm.land.soil_thermal import (SoilThermalConfig,
                                           compute_apparent_heat_capacity,
                                           liquid_water_content)
    cfg = SoilThermalConfig(enable_freeze_thaw=True)
    hyd = SoilHydraulicsConfig()
    theta = jnp.linspace(0.05, hyd.theta_sat, 7)
    sensible_floor = (1.0 - hyd.theta_sat) * cfg.C_soil
    for T in (255.0, 270.0, 273.0, 276.0):
        Tc = jnp.full_like(theta, T)
        liq, _ = liquid_water_content(Tc, theta, cfg)
        assert bool(jnp.all(liq <= theta + 1e-12))
        assert bool(jnp.all(liq >= cfg.theta_liq_residual_frac * theta - 1e-12))
        C = compute_apparent_heat_capacity(Tc, theta, hyd, cfg)
        assert bool(jnp.all(jnp.isfinite(C))) and bool(jnp.all(C >= sensible_floor))
