"""--tripole-strip-north-rows / --tripole-fold-pivot on the CORE-II driver.

eORCA1.2's top mesh row (331) is NEMO's duplicated fold-halo row: the mirror
of row 330.  With the row stripped the model grid is 331 x 362, and every
input the tripole lane reads on the eORCA1 mesh must drop the SAME row --
nothing else.  Each loader test below asserts exactly that: strip=1 returns
331 rows and equals the unstripped result minus its last row.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run" / "run_omip_core2.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_omip_core2_strip", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Stop(Exception):
    def __init__(self, kw):
        super().__init__("stop")
        self.kw = kw


# --- CLI ---------------------------------------------------------------------

def test_flags_default_off_and_round_trip():
    p = _mod()._build_arg_parser()
    a = p.parse_args([])
    assert a.tripole_strip_north_rows == 0
    assert a.tripole_fold_pivot == "legacy"
    b = p.parse_args(["--tripole-strip-north-rows", "1",
                      "--tripole-fold-pivot", "F"])
    assert b.tripole_strip_north_rows == 1
    assert b.tripole_fold_pivot == "F"


def _ns(**kw):
    base = dict(grid="tripole", tripole_strip_north_rows=0,
                tripole_fold_pivot="legacy", forcing_remap="bilinear")
    base.update(kw)
    return argparse.Namespace(**base)


def test_validator_accepts_defaults_and_strip_with_F():
    m = _mod()
    m.validate_tripole_strip_args(_ns())
    m.validate_tripole_strip_args(_ns(tripole_strip_north_rows=1))
    m.validate_tripole_strip_args(_ns(tripole_strip_north_rows=1,
                                      tripole_fold_pivot="F"))


@pytest.mark.parametrize("strip", [0, 2])
def test_F_pivot_without_strip_one_is_refused(strip):
    with pytest.raises(SystemExit, match="needs --tripole-strip-north-rows 1"):
        _mod().validate_tripole_strip_args(
            _ns(tripole_strip_north_rows=strip, tripole_fold_pivot="F"))


@pytest.mark.parametrize("grid", ["latlon_bathy", "mpas", "cubed_sphere",
                                  "fesom"])
@pytest.mark.parametrize("kw", [dict(tripole_strip_north_rows=1),
                                dict(tripole_strip_north_rows=1,
                                     tripole_fold_pivot="F")])
def test_non_tripole_grid_is_refused(grid, kw):
    with pytest.raises(SystemExit, match="--grid tripole only"):
        _mod().validate_tripole_strip_args(_ns(grid=grid, **kw))


def test_negative_strip_and_scrip_forcing_are_refused():
    m = _mod()
    with pytest.raises(SystemExit, match=">= 0"):
        m.validate_tripole_strip_args(_ns(tripole_strip_north_rows=-1))
    with pytest.raises(SystemExit, match="forcing-remap nemo_scrip"):
        m.validate_tripole_strip_args(
            _ns(tripole_strip_north_rows=1, forcing_remap="nemo_scrip"))


def test_main_forwards_both_flags_to_the_builder(monkeypatch, tmp_path):
    """CLI -> main -> build_tripole(strip_north_rows=, fold_pivot=)."""
    import sys
    m = _mod()

    def _fake(*a, **kw):
        raise _Stop(kw)
    monkeypatch.setattr(m, "build_tripole", _fake)
    monkeypatch.setattr(sys, "argv", [
        "run_omip_core2.py", "--grid", "tripole",
        "--tripole-strip-north-rows", "1", "--tripole-fold-pivot", "F",
        "--output", str(tmp_path)])
    with pytest.raises(_Stop) as e:
        m.main()
    assert e.value.kw["strip_north_rows"] == 1
    assert e.value.kw["fold_pivot"] == "F"


def test_manifest_mesh_string_unchanged_by_default():
    m = _mod()
    a = argparse.Namespace(mesh="x.nc", tripole_strip_north_rows=0,
                           tripole_fold_pivot="legacy")
    assert m._manifest_mesh(a) == "x.nc"
    b = argparse.Namespace(mesh="x.nc", tripole_strip_north_rows=1,
                           tripole_fold_pivot="F")
    assert m._manifest_mesh(b) == (
        "x.nc [tripole_strip_north_rows=1 tripole_fold_pivot=F]")


# --- builder plumbing ---------------------------------------------------------

def _mesh_path():
    from scripts.run import run_omip
    return run_omip._parse_resolution("tripole", "eorca1")["mesh_path"]


def test_build_tripole_forwards_to_create_setup(monkeypatch):
    from scripts.run import run_omip
    m = _mod()

    def _fake(*a, **kw):
        raise _Stop(kw)
    monkeypatch.setattr(run_omip, "_create_setup", _fake)
    with pytest.raises(_Stop) as e:
        m.build_tripole(3, 1000.0, _mesh_path(), strip_north_rows=1,
                        fold_pivot="F")
    assert e.value.kw["tripole_strip_north_rows"] == 1
    assert e.value.kw["tripole_fold_pivot"] == "F"


def test_create_setup_forwards_fold_pivot_to_the_grid_builder(monkeypatch):
    import legoesm.grids.tripole as tri
    from scripts.run import run_omip

    def _fake(*a, **kw):
        raise _Stop(kw)
    monkeypatch.setattr(tri, "create_tripole_grid", _fake)
    with pytest.raises(_Stop) as e:
        run_omip._create_setup("tripole", "eorca1", 3, 1000.0, "none", "II",
                               tripole_strip_north_rows=1,
                               tripole_fold_pivot="F")
    assert e.value.kw["strip_north_rows"] == 1
    assert e.value.kw["fold_pivot"] == "F"
    with pytest.raises(_Stop) as e0:
        run_omip._create_setup("tripole", "eorca1", 3, 1000.0, "none", "II")
    assert e0.value.kw["fold_pivot"] == "legacy"
    assert e0.value.kw["strip_north_rows"] == 0


def test_build_tripole_strips_the_mask_bathy_read(monkeypatch):
    """The builder's own mesh read (incl. the nemo_domain_cfg path) gets the
    strip count."""
    from scripts.run import run_omip
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omlc
    m = _mod()

    class _Cfg:
        def replace_flat(self, **kw):
            return self

        def flat_get(self, k):
            return None

    grid = types.SimpleNamespace(lat_T=np.zeros((331, 362)))
    monkeypatch.setattr(run_omip, "_create_setup",
                        lambda *a, **k: (grid, None, _Cfg(), None, None))
    monkeypatch.setattr(omlc, "LatLonCGridOceanModel", lambda *a, **k: None)

    def _fake(*a, **kw):
        raise _Stop(kw)
    monkeypatch.setattr(m, "read_mesh_mask_bathy", _fake)
    with pytest.raises(_Stop) as e:
        m.build_tripole(3, 1000.0, _mesh_path(), strip_north_rows=1,
                        fold_pivot="F", nemo_domain_cfg="/x/domain_cfg.nc")
    assert e.value.kw["strip_north_rows"] == 1
    assert e.value.kw["nemo_domain_cfg"] == "/x/domain_cfg.nc"


# --- synthetic eORCA-like grid (interior y x x; model y+1 x x+2) --------------

NY, NX = 6, 7


def _model_coords():
    """Full halo-carrying coords (NY+1, NX+2): interior at [:-1, 1:-1],
    cyclic columns copying the overlap, north halo row distinct."""
    lat_i = np.linspace(-60.0, 70.0, NY)[:, None] + np.zeros((1, NX))
    lon_i = np.linspace(10.0, 350.0, NX)[None, :] + np.zeros((NY, 1))
    lat = np.zeros((NY + 1, NX + 2))
    lon = np.zeros((NY + 1, NX + 2))
    lat[:-1, 1:-1], lon[:-1, 1:-1] = lat_i, lon_i
    lat[:-1, 0], lon[:-1, 0] = lat_i[:, -1], lon_i[:, -1]
    lat[:-1, -1], lon[:-1, -1] = lat_i[:, 0], lon_i[:, 0]
    lat[-1], lon[-1] = 80.0, lon[-2]
    return lat, lon, lat_i, lon_i


def _write_nc(path, dims, variables, lat_i, lon_i):
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        for d, n in dims.items():
            ds.createDimension(d, n)
        for name, (vdims, arr) in variables.items():
            v = ds.createVariable(name, "f8", vdims)
            v[:] = arr
        ds.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat_i
        ds.createVariable("nav_lon", "f8", ("y", "x"))[:] = lon_i


def _dry_tmask(shape):
    """A source wet mask with dry cells -- the nearest-wet REGRID honours it,
    the native embed ignores it, so a loader that silently fell off the
    embed path on the stripped grid returns different numbers."""
    tm = np.ones(shape, dtype=bool)
    tm[..., 1::2, ::3] = False
    return tm


def test_embed_orca_interior_strip():
    from legoesm.ocean.forcing.nemo_native_fields import embed_orca_interior
    src = np.random.default_rng(0).normal(size=(3, NY, NX))
    full = embed_orca_interior(src, NY + 1, NX + 2)
    s1 = embed_orca_interior(src, NY, NX + 2, strip_north_rows=1)
    assert s1.shape == (3, NY, NX + 2)
    np.testing.assert_array_equal(s1, full[:, :-1])


def test_sss_restoring_strip(tmp_path):
    from legoesm.ocean.forcing.nemo_native_fields import (
        load_nemo_sss_restoring_climatology)
    lat, lon, lat_i, lon_i = _model_coords()
    sss = 30.0 + np.random.default_rng(1).random((12, NY, NX))
    f = tmp_path / "sss.nc"
    _write_nc(f, {"time": 12, "y": NY, "x": NX},
              {"presalt": (("time", "y", "x"), sss)}, lat_i, lon_i)
    tm = _dry_tmask((NY, NX))
    wet = np.ones(lat.shape)
    full = load_nemo_sss_restoring_climatology(str(f), lat, lon, wet,
                                               src_tmask=tm)
    s1 = load_nemo_sss_restoring_climatology(
        str(f), lat[:-1], lon[:-1], wet[:-1], src_tmask=tm,
        strip_north_rows=1)
    assert s1.shape == (12, NY, NX + 2)
    np.testing.assert_array_equal(s1, full[:, :-1])


def test_monthly_init_strip(tmp_path):
    from legoesm.ocean.forcing.nemo_native_fields import (
        load_nemo_monthly_init_ts)
    lat, lon, lat_i, lon_i = _model_coords()
    rng = np.random.default_rng(2)
    nl = 2
    T = 5.0 + rng.random((12, nl, NY, NX))
    S = 34.0 + rng.random((12, nl, NY, NX))
    ft, fs = tmp_path / "t.nc", tmp_path / "s.nc"
    dims = {"time": 12, "z": nl, "y": NY, "x": NX}
    _write_nc(ft, dims, {"contemp": (("time", "z", "y", "x"), T)}, lat_i, lon_i)
    _write_nc(fs, dims, {"presalt": (("time", "z", "y", "x"), S)}, lat_i, lon_i)
    tm = _dry_tmask((nl, NY, NX))
    for tint in (False, True):
        Tf, Sf = load_nemo_monthly_init_ts(str(ft), str(fs), lat, lon, nl,
                                           src_tmask=tm, nemo_tint=tint)
        T1, S1 = load_nemo_monthly_init_ts(str(ft), str(fs), lat[:-1],
                                           lon[:-1], nl, src_tmask=tm,
                                           nemo_tint=tint,
                                           strip_north_rows=1)
        assert T1.shape == (NY, NX + 2, nl)
        np.testing.assert_array_equal(T1, Tf[:-1])
        np.testing.assert_array_equal(S1, Sf[:-1])


def test_ice_init_strip(tmp_path):
    from legoesm.ocean.forcing.nemo_native_fields import load_nemo_ice_init
    lat, lon, lat_i, lon_i = _model_coords()
    rng = np.random.default_rng(3)
    conc = rng.random((NY, NX))
    h = 0.5 + rng.random((NY, NX))
    # SI3 writes (0, 0) coords on land: mark some cells land so the regrid
    # path (coord-valid sources only) would differ from the embed.
    lat_s, lon_s = lat_i.copy(), lon_i.copy()
    lat_s[1::2, ::3] = 0.0
    lon_s[1::2, ::3] = 0.0
    f = tmp_path / "ice.nc"
    _write_nc(f, {"y": NY, "x": NX},
              {"at_i": (("y", "x"), conc), "ht_i": (("y", "x"), h)},
              lat_s, lon_s)
    wet = np.ones(lat.shape)
    full = load_nemo_ice_init(str(f), lat, lon, wet)
    s1 = load_nemo_ice_init(str(f), lat[:-1], lon[:-1], wet[:-1],
                            strip_north_rows=1)
    assert s1.concentration.shape == (NY, NX + 2)
    np.testing.assert_array_equal(s1.concentration, full.concentration[:-1])
    np.testing.assert_array_equal(s1.h_ice, full.h_ice[:-1])


# --- SCRIP (chl / geothermal): weights write the 331x360 interior --------------

def test_scrip_to_full_tripole_strip():
    m = _mod()
    interior = np.random.default_rng(4).normal(size=(331, 360)) + 10.0
    full = m._scrip_to_full_tripole(interior, 332, 362)
    s1 = m._scrip_to_full_tripole(interior, 331, 362, strip_north_rows=1)
    assert s1.shape == (331, 362)
    np.testing.assert_array_equal(s1, full[:-1])


def _patch_scrip(monkeypatch):
    import legoesm.ocean.coupler.omip2_applicator as app
    base = np.arange(331 * 360, dtype=float).reshape(331, 360)
    monkeypatch.setattr(app, "load_scrip_weights",
                        lambda path: (None, None, 4))
    monkeypatch.setattr(
        app, "apply_scrip_weights",
        lambda src, s0, w, nw, shp: base * 1e-3 + float(np.sum(src)) * 1e-6
        + 0.1)


def test_chl_scrip_strip(tmp_path, monkeypatch):
    m = _mod()
    _patch_scrip(monkeypatch)
    lat_i = np.linspace(-80, 80, 3)[:, None] + np.zeros((1, 4))
    lon_i = np.linspace(0, 270, 4)[None, :] + np.zeros((3, 1))
    chl = np.random.default_rng(5).random((12, 3, 4))
    f = tmp_path / "chl.nc"
    _write_nc(f, {"t": 12, "y": 3, "x": 4},
              {"CHLA": (("t", "y", "x"), chl)}, lat_i, lon_i)
    full = m.load_nemo_chl_monthly(None, "tripole", np.zeros((332, 362)),
                                   np.zeros((332, 362)), chl_file=str(f),
                                   chl_remap="nemo_scrip")
    s1 = m.load_nemo_chl_monthly(None, "tripole", np.zeros((331, 362)),
                                 np.zeros((331, 362)), chl_file=str(f),
                                 chl_remap="nemo_scrip", strip_north_rows=1)
    assert s1.shape == (12, 331, 362)
    np.testing.assert_array_equal(s1, full[:, :-1])
    with pytest.raises(SystemExit, match="eORCA1 tripole"):
        m.load_nemo_chl_monthly(None, "tripole", np.zeros((332, 362)),
                                np.zeros((332, 362)), chl_file=str(f),
                                chl_remap="nemo_scrip", strip_north_rows=1)


def test_geothermal_map_strip(tmp_path, monkeypatch):
    import netCDF4
    m = _mod()
    _patch_scrip(monkeypatch)
    f = tmp_path / "gh.nc"
    with netCDF4.Dataset(f, "w") as ds:
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        v = ds.createVariable("gh_flux", "f8", ("lat", "lon"))
        v.units = "mW m^{-2}"
        v[:] = 50.0 + np.random.default_rng(6).random((3, 4))
    monkeypatch.setattr(m, "_GHFLUX_NC", str(f))
    full = m.load_nemo_geothermal_flux("tripole", (332, 362))
    s1 = m.load_nemo_geothermal_flux("tripole", (331, 362),
                                     strip_north_rows=1)
    assert s1.shape == (331, 362)
    np.testing.assert_array_equal(s1, full[:-1])


# --- --nemo-ldf-file (331x360 inner-domain files) -------------------------------

def test_nemo_ldf_fields_strip(tmp_path, monkeypatch):
    import netCDF4
    import legoesm.ocean.dynamics.latlon_cgrid_operators as lco
    m = _mod()
    nk, ny, nx = 2, 331, 360
    rng = np.random.default_rng(7)
    ahmt = 1000.0 + rng.random((nk, ny, nx))
    ahmf = 2000.0 + rng.random((nk, ny, nx))
    e3f = 10.0 + rng.random((nk, ny, nx))
    ahmt[:, -3:] = 20000.0                     # uniform top rows (asserted)
    ahmf[:, -3:] = 20000.0
    lat_t = np.linspace(-78, 89, ny)[:, None] + 0.001 * np.arange(nx)[None]
    lat_f = lat_t + 0.25
    ldf, dcfg, mesh = tmp_path / "ldf.nc", tmp_path / "dcfg.nc", \
        tmp_path / "mesh.nc"
    with netCDF4.Dataset(ldf, "w") as ds:
        for d, n in (("z", nk), ("y", ny), ("x", nx)):
            ds.createDimension(d, n)
        ds.createVariable("ahmt_3d", "f8", ("z", "y", "x"))[:] = ahmt
        ds.createVariable("ahmf_3d", "f8", ("z", "y", "x"))[:] = ahmf
        ds.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat_t
    with netCDF4.Dataset(dcfg, "w") as ds:
        for d, n in (("z", nk), ("y", ny), ("x", nx)):
            ds.createDimension(d, n)
        ds.createVariable("e3f_0", "f8", ("z", "y", "x"))[:] = e3f
        ds.createVariable("gphif", "f8", ("y", "x"))[:] = lat_f
    ci = (np.arange(362) - 1) % 360
    gt = np.vstack([lat_t[:, ci], np.full((1, 362), 90.0)])
    gf = np.vstack([lat_f[:, ci], np.full((1, 362), 90.0)])
    with netCDF4.Dataset(mesh, "w") as ds:
        ds.createDimension("y", 332)
        ds.createDimension("x", 362)
        ds.createVariable("gphit", "f8", ("y", "x"))[:] = gt
        ds.createVariable("gphif", "f8", ("y", "x"))[:] = gf
    monkeypatch.setattr(
        lco, "nemo_fmask_shlat_3d",
        lambda act, grid, rn: np.ones((grid.n_lat + 1, grid.n_lon + 1, nk)))

    class _Z(types.SimpleNamespace):
        def _replace(self, **kw):
            return types.SimpleNamespace(**kw)

    def _run(n_lat, strip):
        z = _Z(n_levels=nk, h_partial=np.zeros(1), dz_ref=np.ones(nk),
               is_active=np.ones((n_lat, 362, nk)))
        g = types.SimpleNamespace(n_lat=n_lat, n_lon=362)
        return m.attach_nemo_ldf_fields(z, g, str(mesh), str(ldf), str(dcfg),
                                        rn_shlat=2.0, strip_north_rows=strip)

    full = _run(332, 0)
    s1 = _run(331, 1)
    for k in ("nemo_ahmt_3d", "nemo_ahmf_3d", "nemo_e3f_0"):
        a, b = np.asarray(getattr(full, k)), np.asarray(getattr(s1, k))
        assert b.shape[0] == a.shape[0] - 1, k
        np.testing.assert_array_equal(b, a[:-1], err_msg=k)
    assert np.asarray(s1.nemo_ahmt_3d).shape[0] == 331
    with pytest.raises(SystemExit, match="strip-north-rows 0 or 1"):
        _run(330, 2)


# --- runoff: IDW regrid + coastal spread + area renorm --------------------------

def test_runoff_strip(tmp_path, monkeypatch):
    """The coastal spread's north-edge stencil sees the stripped row, so a
    plain regrid onto the stripped coordinates is NOT the unstripped field
    minus its last row; the loader restores the row and drops it after."""
    import netCDF4
    m = _mod()
    lat, lon, lat_i, lon_i = _model_coords()
    lat[-1], lon[-1] = lat[-2], lon[-2][::-1]      # fold-duplicate halo row
    rng = np.random.default_rng(8)
    ro = np.where(rng.random((12, NY, NX)) > 0.4, rng.random((12, NY, NX)),
                  0.0) * 1e-3
    f = tmp_path / "runoff.nc"
    _write_nc(f, {"t": 12, "y": NY, "x": NX},
              {"sorunoff": (("t", "y", "x"), ro),
               "Icb_flux": (("t", "y", "x"), 0.5 * ro)}, lat_i, lon_i)
    mesh = tmp_path / "mesh.nc"
    with netCDF4.Dataset(mesh, "w") as ds:
        ds.createDimension("y", NY + 1)
        ds.createDimension("x", NX + 2)
        ds.createVariable("gphit", "f8", ("y", "x"))[:] = lat
        ds.createVariable("glamt", "f8", ("y", "x"))[:] = lon
    monkeypatch.setattr(m, "_RUNOFF_NC", str(f))
    monkeypatch.setattr(m, "_load_nemo_cell_area_m2",
                        lambda *a, **k: np.ones((NY, NX)))
    land = np.ones(lat.shape)
    land[-1] = 0.0
    land[0, 2] = 0.0

    def _grid(n):
        return types.SimpleNamespace(lat_T=np.deg2rad(lat[:n]),
                                     area=np.ones((n, NX + 2)))

    def _deg(n):
        return (np.rad2deg(np.deg2rad(lat[:n])),
                np.rad2deg(np.deg2rad(lon[:n])))

    full = m.load_runoff_monthly(_grid(NY + 1), "tripole", *_deg(NY + 1),
                                 str(mesh), land_mask=land, spread_passes=2)
    s1 = m.load_runoff_monthly(_grid(NY), "tripole", *_deg(NY), str(mesh),
                               land_mask=land[:-1], spread_passes=2,
                               strip_north_rows=1)
    assert s1.shape == (12, NY, NX + 2)
    np.testing.assert_allclose(s1, full[:, :-1], rtol=1e-13, atol=0.0)


def test_main_forwards_the_strip_count_to_every_mesh_shaped_loader():
    """Forwarding in main() is invisible to the direct loader tests: the
    builder, the NEMO monthly IC, the SSS target, the geothermal map, the
    chlorophyll SCRIP remap, the SI3 ice IC and the runoff must each receive
    the flag's row count.  Dropping any one fails this."""
    src = RUNNER.read_text()
    assert src.count("strip_north_rows=_tri_strip") == 7
    assert "_tri_strip = int(args.tripole_strip_north_rows)" in src
