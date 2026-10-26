"""nemo_native_fields loaders: eORCA halo embed + native/regrid paths."""

from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from legoesm.ocean.forcing.nemo_native_fields import (
    embed_orca_interior,
    load_nemo_monthly_init_ts,
    load_nemo_sss_restoring_climatology,
)


def _model_coords(ny_i=6, nx_i=8):
    """Model tripole-like coords (interior + ORCA halos)."""
    lat_i = np.linspace(-70, 89, ny_i)[:, None] * np.ones((1, nx_i))
    lon_i = np.ones((ny_i, 1)) * np.linspace(0, 350, nx_i)[None, :]
    lat = np.zeros((ny_i + 1, nx_i + 2))
    lon = np.zeros_like(lat)
    lat[:-1, 1:-1] = lat_i
    lon[:-1, 1:-1] = lon_i
    lat[:-1, 0] = lat_i[:, -1]; lon[:-1, 0] = lon_i[:, -1]
    lat[:-1, -1] = lat_i[:, 0]; lon[:-1, -1] = lon_i[:, 0]
    lat[-1] = lat[-2]; lon[-1] = lon[-2]
    return lat, lon, lat_i, lon_i


def test_embed_orca_interior_layout():
    src = np.arange(6 * 8, dtype=float).reshape(6, 8)
    out = embed_orca_interior(src, 7, 10)
    np.testing.assert_array_equal(out[:-1, 1:-1], src)
    np.testing.assert_array_equal(out[:-1, 0], src[:, -1])
    np.testing.assert_array_equal(out[:-1, -1], src[:, 0])
    np.testing.assert_array_equal(out[-1], out[-2])
    with pytest.raises(ValueError, match="does not match"):
        embed_orca_interior(src, 8, 10)


def _write_sss_nc(path, lat_i, lon_i, ny_i, nx_i):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time_counter", 12)
        ds.createDimension("y", ny_i)
        ds.createDimension("x", nx_i)
        v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
        v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
        v = ds.createVariable("presalt", "f8", ("time_counter", "y", "x"))
        v[:] = 30.0 + np.arange(12)[:, None, None] \
            + np.arange(ny_i)[None, :, None] * 0.1


def test_sss_climatology_native_embed(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    p = tmp_path / "sss.nc"
    _write_sss_nc(p, lat_i, lon_i, 6, 8)
    out = load_nemo_sss_restoring_climatology(
        str(p), lat, lon, np.ones(lat.shape, bool))
    assert out.shape == (12, 7, 10)
    # native embed: interior equals the file, months indexed correctly
    np.testing.assert_allclose(out[3][:-1, 1:-1][2, 4], 30 + 3 + 0.2)
    assert np.isfinite(out).all()


def test_sss_climatology_regrid_path(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    p = tmp_path / "sss.nc"
    # source on a DIFFERENT (coarser) grid -> NearestWetRegridder path
    _write_sss_nc(p, lat_i[::2, ::2], lon_i[::2, ::2], 3, 4)
    out = load_nemo_sss_restoring_climatology(
        str(p), lat, lon, np.ones(lat.shape, bool))
    assert out.shape == (12, 7, 10)
    assert np.isfinite(out).all()


def test_sss_climatology_unstructured_target(tmp_path):
    """MPAS Voronoi target: 1-D PAIRED cell centres -> nearest-wet, (12, nCells).

    The structured embed/meshgrid paths crash on a 1-D target
    (``n_lat, n_lon = lat.shape`` / nCells^2 outer product); the paired
    branch must regrid onto the cells directly.
    """
    _, _, lat_i, lon_i = _model_coords()
    p = tmp_path / "sss.nc"
    _write_sss_nc(p, lat_i, lon_i, 6, 8)
    # 1-D paired unstructured cell centres (nCells=5), inside the source hull
    cell_lat = np.array([-60.0, -10.0, 5.0, 40.0, 80.0])
    cell_lon = np.array([10.0, 120.0, 200.0, 300.0, 45.0])
    out = load_nemo_sss_restoring_climatology(
        str(p), cell_lat, cell_lon, np.ones(cell_lat.shape, bool))
    assert out.shape == (12, 5)              # (months, nCells) — NOT nCells^2
    assert np.isfinite(out).all()
    # month indexing survives the regrid: presalt = 30 + m + 0.1*row_i, so the
    # per-month field mean must step by exactly 1.0 between consecutive months.
    steps = np.diff(out.mean(axis=1))
    np.testing.assert_allclose(steps, np.ones(11), atol=1e-9)


def test_monthly_init_native(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    nlev = 5
    tp, sp = tmp_path / "t.nc", tmp_path / "s.nc"
    for path, var, base in ((tp, "contemp", 10.0), (sp, "presalt", 34.0)):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time_counter", 12)
            ds.createDimension("deptht", nlev)
            ds.createDimension("y", 6)
            ds.createDimension("x", 8)
            v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
            v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
            v = ds.createVariable(var, "f8",
                                  ("time_counter", "deptht", "y", "x"))
            fld = (base + np.arange(12)[:, None, None, None]
                   + np.arange(nlev)[None, :, None, None] * 0.01)
            fld = np.broadcast_to(fld, (12, nlev, 6, 8)).copy()
            fld[:, -1, 0, 0] = np.nan          # a below-bathy hole
            v[:] = fld
    T, S = load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                     n_levels=nlev, month=2)
    assert T.shape == (7, 10, nlev) and S.shape == T.shape
    assert np.isfinite(T).all() and np.isfinite(S).all()
    np.testing.assert_allclose(T[:-1, 1:-1][3, 3, 0], 10.0 + 1)   # month 2
    # column fill: the hole inherited the level above
    np.testing.assert_allclose(T[0, 1, nlev - 1], T[0, 1, nlev - 2])
    with pytest.raises(ValueError, match="levels"):
        load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                  n_levels=nlev + 1)
    with pytest.raises(ValueError, match="month"):
        load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                  n_levels=nlev, month=0)


def test_monthly_init_unstructured_target(tmp_path):
    """MPAS Voronoi target: 1-D paired cell centres -> (nCells, nlev).

    Exercises the exact path the run_omip_core2 MPAS gate removal exposes
    (2026-08-30): the loader must accept 1-D cell coords (NearestWetRegridder
    structured=False) so all grids can take the same NEMO monthly WOA IC.
    """
    nlev = 5
    tp, sp = tmp_path / "t.nc", tmp_path / "s.nc"
    lat_i = np.linspace(-70, 89, 6)[:, None] * np.ones((1, 8))
    lon_i = np.ones((6, 1)) * np.linspace(0, 350, 8)[None, :]
    for path, var, base in ((tp, "contemp", 10.0), (sp, "presalt", 34.0)):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time_counter", 12)
            ds.createDimension("deptht", nlev)
            ds.createDimension("y", 6)
            ds.createDimension("x", 8)
            v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
            v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
            v = ds.createVariable(var, "f8",
                                  ("time_counter", "deptht", "y", "x"))
            fld = (base + np.arange(12)[:, None, None, None]
                   + np.arange(nlev)[None, :, None, None] * 0.01)
            v[:] = np.broadcast_to(fld, (12, nlev, 6, 8)).copy()
    # 1-D paired MPAS cell centres (nCells=5) inside the source hull
    latc = np.array([-40.0, -10.0, 10.0, 40.0, 70.0])
    lonc = np.array([30.0, 100.0, 180.0, 250.0, 320.0])
    T, S = load_nemo_monthly_init_ts(str(tp), str(sp), latc, lonc,
                                     n_levels=nlev, month=2)
    assert T.shape == (5, nlev) and S.shape == (5, nlev)   # (nCells, nlev)
    assert np.isfinite(T).all() and np.isfinite(S).all()
    # month indexing survives: contemp = 10 + m at month 2 (index 1)
    np.testing.assert_allclose(T[:, 0], 10.0 + 1, atol=1e-6)


def test_regrid_skips_flood_filled_land_when_tmask_given(tmp_path):
    """NEMO's IC / restoring files are flood-filled: land cells hold finite
    junk.  With ``src_tmask`` the nearest-WET search must skip them; without
    it (the pre-fix behaviour) the same target samples the junk — so the
    test fails if the mask is ignored."""
    from legoesm.ocean.forcing.nemo_native_fields import nemo_src_tmask_for

    # 3x3 source: centre cell is LAND holding 5.0 (junk); wet ring holds 35.
    lat_i = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lon_i = np.array([[9.0, 10.0, 11.0]] * 3)
    p = tmp_path / "sss.nc"
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time_counter", 12)
        ds.createDimension("y", 3); ds.createDimension("x", 3)
        v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
        v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
        v = ds.createVariable("presalt", "f8", ("time_counter", "y", "x"))
        fld = np.full((3, 3), 35.0); fld[1, 1] = 5.0
        v[:] = np.broadcast_to(fld, (12, 3, 3))
    tmask = np.ones((1, 3, 3), bool); tmask[0, 1, 1] = False
    # unstructured target: one point ON the land cell centre
    tgt_lat = np.array([0.0]); tgt_lon = np.array([10.0])
    junk = load_nemo_sss_restoring_climatology(str(p), tgt_lat, tgt_lon,
                                               np.ones(1, bool))
    assert junk[0, 0] == 5.0            # pre-fix: samples the land fill
    good = load_nemo_sss_restoring_climatology(str(p), tgt_lat, tgt_lon,
                                               np.ones(1, bool),
                                               src_tmask=tmask)
    assert good[0, 0] == 35.0           # masked: nearest WET cell

    # mesh_mask with halos (1, nlev, y+1, x+2) -> interior mask, PROVEN on
    # the file's grid by its gphit/glamt interior matching nav_lat/nav_lon
    m = tmp_path / "mesh_mask.nc"
    glat = np.zeros((4, 5)); glon = np.zeros((4, 5))
    glat[:-1, 1:-1] = lat_i; glon[:-1, 1:-1] = lon_i
    with netCDF4.Dataset(m, "w") as ds:
        ds.createDimension("t", 1); ds.createDimension("z", 1)
        ds.createDimension("y", 4); ds.createDimension("x", 5)
        v = ds.createVariable("tmask", "i1", ("t", "z", "y", "x"))
        full = np.ones((1, 1, 4, 5), np.int8); full[0, 0, 1, 2] = 0
        v[:] = full
        v = ds.createVariable("gphit", "f8", ("t", "y", "x")); v[:] = glat
        v = ds.createVariable("glamt", "f8", ("t", "y", "x")); v[:] = glon
    tm = nemo_src_tmask_for(str(m), str(p))
    assert tm.shape == (1, 3, 3) and not tm[0, 1, 1]
    # a mesh of another resolution / offset grid must RAISE, never fall
    # back to "finite == wet" (that is the bug this guards against)
    lat_big = np.zeros((7, 10)); lon_big = np.zeros((7, 10))
    p2 = tmp_path / "sss_big.nc"
    _write_sss_nc(p2, lat_big[:-1, 1:-1], lon_big[:-1, 1:-1], 6, 8)
    with pytest.raises(ValueError, match="interior"):
        nemo_src_tmask_for(str(m), str(p2))
    p3 = tmp_path / "sss_shift.nc"
    with netCDF4.Dataset(p3, "w") as ds:
        ds.createDimension("time_counter", 12)
        ds.createDimension("y", 3); ds.createDimension("x", 3)
        v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i + 1.0
        v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
        v = ds.createVariable("presalt", "f8", ("time_counter", "y", "x"))
        v[:] = 35.0
    with pytest.raises(ValueError, match="coordinates do not match"):
        nemo_src_tmask_for(str(m), str(p3))


def test_monthly_init_nemo_tint_is_the_mid_month_blend(tmp_path):
    """NEMO (fldread ln_tint) starts 1 January from 0.5*Dec + 0.5*Jan, not
    from January's mid-month field. The fixture's field is base + month
    index, so the blend is exactly base + 5.5 for month=1 (Dec index 11,
    Jan index 0). Other start months are refused (the record centres need
    the run calendar). Without the keyword the January field is returned
    unchanged (non-vacuity)."""
    lat, lon, lat_i, lon_i = _model_coords()
    nlev = 3
    tp, sp = tmp_path / "t.nc", tmp_path / "s.nc"
    for path, var, base in ((tp, "contemp", 10.0), (sp, "presalt", 34.0)):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time_counter", 12)
            ds.createDimension("deptht", nlev)
            ds.createDimension("y", 6)
            ds.createDimension("x", 8)
            v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
            v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
            v = ds.createVariable(var, "f8",
                                  ("time_counter", "deptht", "y", "x"))
            v[:] = np.broadcast_to(
                base + np.arange(12)[:, None, None, None], (12, nlev, 6, 8))
    T0, S0 = load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                       n_levels=nlev, month=1)
    T1, S1 = load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                       n_levels=nlev, month=1, nemo_tint=True)
    np.testing.assert_allclose(T0[:-1, 1:-1][3, 3, 0], 10.0)         # Jan
    np.testing.assert_allclose(T1[:-1, 1:-1][3, 3, 0], 10.0 + 5.5)   # Dec/Jan
    np.testing.assert_allclose(S1[:-1, 1:-1][3, 3, 0], 34.0 + 5.5)
    with pytest.raises(ValueError, match="1 January"):
        load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                  n_levels=nlev, month=2, nemo_tint=True)
