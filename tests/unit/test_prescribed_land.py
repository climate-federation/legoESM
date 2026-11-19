"""Prescribed land boundary (atmosphere-only AMIP): file reading, units and
sign checks, regridding onto an MPAS mesh, and the periodic mid-month time
interpolation of ``legoesm.forcing.prescribed_land``."""
from __future__ import annotations

import jax
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.forcing.prescribed_land import (  # noqa: E402
    load_prescribed_land, prescribed_land_at_day,
)
from legoesm.forcing.time_utils import (  # noqa: E402
    NOLEAP_DAYS_PER_MONTH, NOLEAP_DAYS_PER_YEAR, NOLEAP_MONTH_STARTS,
)

# Mid-month days of the 365-day year, the builder's time axis.
_MID = np.asarray(NOLEAP_MONTH_STARTS[:12]) + 0.5 * np.asarray(NOLEAP_DAYS_PER_MONTH)
_LAT = np.arange(-89.5, 90.0, 1.0)
_LON = np.arange(0.5, 360.0, 1.0)


class _Grid:
    """Minimal grid: 4 columns with lat/lon in radians (what the regridder reads)."""
    grid_lat = np.deg2rad(np.array([-60.0, -10.0, 20.0, 70.0]))
    grid_lon = np.deg2rad(np.array([10.0, 100.0, 200.0, 300.0]))


def _fields(nt):
    month = np.arange(nt)[:, None, None]
    shape = (nt, _LAT.size, _LON.size)
    ts = np.broadcast_to(280.0 + 10.0 * np.cos(2 * np.pi * month / 12.0), shape)
    return {"ts": np.array(ts), "hfss": np.full(shape, 20.0),
            "hfls": np.full(shape, 60.0), "albedo": np.full(shape, 0.3)}


def _write(tmp_path, *, times=_MID, units=None, positive="up", drop=(),
           albedo=True, edit=None, lat=_LAT, name="presc.nc", lon=_LON):
    t = np.asarray(times, dtype=np.float64)
    f = _fields(t.size)
    if edit is not None:
        edit(f)
    u = {"ts": "K", "hfss": "W m-2", "hfls": "W m-2", "albedo": "1"}
    u.update(units or {})
    data = {}
    for v in ("ts", "hfss", "hfls") + (("albedo",) if albedo else ()):
        if v in drop:
            continue
        attrs = {"units": u[v]}
        if v in ("hfss", "hfls"):
            attrs["positive"] = positive
        arr = f[v] if lat[0] < lat[-1] else f[v][:, ::-1, :]
        data[v] = (("time", "lat", "lon"), arr, attrs)
    ds = xr.Dataset(data, coords={"time": t, "lat": lat, "lon": lon})
    p = tmp_path / name
    ds.to_netcdf(p)
    return str(p)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def test_load_regrids_every_record_onto_the_columns(tmp_path):
    pl = load_prescribed_land(_write(tmp_path), _Grid())
    for a in (pl.ts, pl.hfss, pl.hfls, pl.albedo):
        assert a.shape == (12, 4)
    np.testing.assert_allclose(pl.hfss, 20.0)
    np.testing.assert_allclose(pl.hfls, 60.0)
    np.testing.assert_allclose(pl.albedo, 0.3)
    np.testing.assert_allclose(pl.ts[0], 290.0, rtol=1e-12)
    np.testing.assert_allclose(pl.times_doy, _MID)


def test_descending_latitude_is_flipped_not_mirrored(tmp_path):
    """A north-to-south file reads the same as the south-to-north one."""
    def tilt(f):
        f["ts"] = f["ts"] + _LAT[None, :, None]   # ts grows northward
    up = load_prescribed_land(_write(tmp_path, edit=tilt), _Grid())
    dn = load_prescribed_land(
        _write(tmp_path, edit=tilt, lat=_LAT[::-1], name="dn.nc"), _Grid())
    np.testing.assert_allclose(dn.ts, up.ts, rtol=1e-12)
    np.testing.assert_allclose(up.ts[0] - 290.0, [-60.0, -10.0, 20.0, 70.0],
                               atol=1e-9)


def test_albedo_is_optional(tmp_path):
    pl = load_prescribed_land(_write(tmp_path, albedo=False), _Grid())
    assert pl.albedo is None
    assert prescribed_land_at_day(pl, 100.0)[3] is None


@pytest.mark.parametrize("kw, err, match", [
    (dict(drop=("hfls",)), KeyError, "missing variables"),
    (dict(drop=("ts",)), KeyError, "missing variables"),
    (dict(positive="down"), ValueError, "positive='up'"),
    (dict(units={"ts": "degC"}), ValueError, "units='K'"),
    (dict(units={"hfss": "W/m2"}), ValueError, "units='W m-2'"),
    (dict(units={"albedo": "%"}), ValueError, "units='1'"),
    (dict(times=_MID + 400.0), ValueError, "climatological year"),
    (dict(times=np.arange(13) * 28.0), ValueError, "2..12"),
    (dict(times=_MID[::-1]), ValueError, "strictly"),
    (dict(lon=_LON - 180.0), ValueError, r"within \[0, 360\)"),
])
def test_refuses_a_malformed_file(tmp_path, kw, err, match):
    with pytest.raises(err, match=match):
        load_prescribed_land(_write(tmp_path, **kw), _Grid())


def test_refuses_non_finite_and_out_of_range_values(tmp_path):
    def nan(f):
        f["hfss"][3, 10, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        load_prescribed_land(_write(tmp_path, edit=nan), _Grid())

    def bright(f):
        f["albedo"][0, 0, 0] = 1.2
    with pytest.raises(ValueError, match="albedo outside"):
        load_prescribed_land(_write(tmp_path, edit=bright, name="b.nc"), _Grid())


def test_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_prescribed_land(str(tmp_path / "absent.nc"), _Grid())


# --------------------------------------------------------------------------
# time interpolation
# --------------------------------------------------------------------------
def test_interpolation_hits_anchors_mid_month_and_wraps_the_year(tmp_path):
    pl = load_prescribed_land(_write(tmp_path), _Grid())
    for i, d in enumerate(_MID):
        np.testing.assert_allclose(prescribed_land_at_day(pl, d)[0], pl.ts[i],
                                   rtol=1e-12)
    # Mid-month value between two anchors: linear (Jan 15.5 -> Feb 45.0).
    w = (30.0 - _MID[0]) / (_MID[1] - _MID[0])
    np.testing.assert_allclose(prescribed_land_at_day(pl, 30.0)[0],
                               (1 - w) * pl.ts[0] + w * pl.ts[1], rtol=1e-12)
    # December -> January across the year boundary, both sides of 1 January.
    gap = _MID[0] + NOLEAP_DAYS_PER_YEAR - _MID[11]
    for day in (362.0, 3.0):
        w = ((day - _MID[11]) % NOLEAP_DAYS_PER_YEAR) / gap
        np.testing.assert_allclose(prescribed_land_at_day(pl, day)[0],
                                   (1 - w) * pl.ts[11] + w * pl.ts[0], rtol=1e-12)
    # Periodic over years (day counts since 1 January of the start year).
    for day in (3.0, 200.0, 362.0):
        a = prescribed_land_at_day(pl, day)
        b = prescribed_land_at_day(pl, day + 3 * NOLEAP_DAYS_PER_YEAR)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-12)


def test_interpolated_fluxes_keep_units_and_sign(tmp_path):
    """W m-2 positive up in, W m-2 positive up out: a constant field stays
    exactly that constant at every day (no scaling, no sign flip)."""
    pl = load_prescribed_land(_write(tmp_path), _Grid())
    for day in (0.0, 15.5, 100.25, 364.9):
        _, sh, lh, alb = prescribed_land_at_day(pl, day)
        np.testing.assert_allclose(sh, 20.0, rtol=1e-12)
        np.testing.assert_allclose(lh, 60.0, rtol=1e-12)
        np.testing.assert_allclose(alb, 0.3, rtol=1e-12)


# --------------------------------------------------------------------------
# regridding onto an MPAS mesh
# --------------------------------------------------------------------------
def test_regrid_to_mesh_keeps_the_land_mean_of_a_smooth_field(tmp_path):
    """Bilinear (the AMIP SST owner), not conservative: on a smooth field the
    area-weighted mean over a land-like mask of mesh cells matches the same
    mean of the exact field at the cell centres to 5e-3 K (and globally)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=4, lloyd_iterations=2)

    def exact(lat, lon):
        return 280.0 + 20.0 * np.sin(lat) ** 2 + 5.0 * np.cos(lon) * np.cos(lat)

    phi = np.deg2rad(_LAT)[None, :, None]
    lam = np.deg2rad(_LON)[None, None, :]

    def smooth(f):
        f["ts"] = np.broadcast_to(exact(phi, lam), f["ts"].shape).copy()
    pl = load_prescribed_land(_write(tmp_path, edit=smooth), mesh)
    a = np.asarray(mesh.areaCell)
    lat, lon = np.asarray(mesh.latCell), np.asarray(mesh.lonCell)
    ts = pl.ts[0]
    assert ts.shape == lat.shape
    want_cells = exact(lat, lon)
    # a "continent": 30-70 N, 0-120 E (crosses the lon seam of the source)
    land = ((lat > np.deg2rad(30.0)) & (lat < np.deg2rad(70.0))
            & (np.mod(lon, 2 * np.pi) < np.deg2rad(120.0)))
    assert land.sum() > 50
    for sel in (land, np.ones_like(land)):
        got = (ts[sel] * a[sel]).sum() / a[sel].sum()
        want = (want_cells[sel] * a[sel]).sum() / a[sel].sum()
        assert abs(got - want) < 5e-3, (got, want)
    assert np.abs(ts - want_cells).max() < 2e-2
