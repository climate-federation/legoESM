"""Multi-file NEMO mesh reading + north-halo-row stripping for the tripole lane.

Older NEMO runs (NOC ORCA0083 = ORCA12, 1/12 deg) ship the grid as THREE files
(``mesh_hgr.nc`` metrics, ``mesh_zgr.nc`` vertical scale factors, ``mask.nc``
masks) instead of one ``mesh_mask.nc``.  ``_read_nemo_mesh_mask`` and
``read_mesh_mask_bathy`` must read the split form identically to the merged
form, and ``strip_north_rows`` must drop exactly the requested DEAD halo rows
(NEMO jperio=4 T-pivot meshes end with one above the self-dual pivot row).
Synthetic files, no external data.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
xr = pytest.importorskip("xarray")

from legoesm.grids.tripole import (  # noqa: E402
    _read_nemo_mesh_mask,
    mesh_file_list,
    strip_north_rows_raw,
)
from legoesm.ocean.init_tripole import read_mesh_mask_bathy  # noqa: E402

NLAT, NLON, NLEV = 6, 8, 3


def _write(path, variables):
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("t", 1)
    ds.createDimension("z", NLEV)
    ds.createDimension("y", NLAT)
    ds.createDimension("x", NLON)
    for name, (dims, arr) in variables.items():
        v = ds.createVariable(name, arr.dtype, dims)
        v[:] = arr
    ds.close()


@pytest.fixture
def split_and_merged(tmp_path):
    rng = np.random.default_rng(3)
    twod = {n: (("t", "y", "x"), rng.uniform(1, 2, (1, NLAT, NLON)))
            for n in ("glamt", "gphit", "glamu", "gphiu", "glamv", "gphiv",
                      "glamf", "gphif", "e1t", "e2t", "e1u", "e2u", "e1v",
                      "e2v", "e1f", "e2f")}
    tmask = (rng.uniform(size=(1, NLEV, NLAT, NLON)) > 0.3).astype(np.int8)
    tmask[0, :, -1, :] = 0                       # dead north halo row
    masks = {n: (("t", "z", "y", "x"), tmask.copy())
             for n in ("tmask", "umask", "vmask", "fmask")}
    masks["tmaskutil"] = (("t", "y", "x"), tmask[:, 0].copy())
    e3t = rng.uniform(5, 50, (1, NLEV, NLAT, NLON))
    zgr = {"e3t_0": (("t", "z", "y", "x"), e3t)}
    hgr, zgr_p, msk, merged = (tmp_path / f for f in
                               ("mesh_hgr.nc", "mesh_zgr.nc", "mask.nc",
                                "mesh_mask.nc"))
    _write(hgr, twod)
    _write(zgr_p, zgr)
    _write(msk, masks)
    _write(merged, {**twod, **zgr, **masks})
    return [str(hgr), str(zgr_p), str(msk)], str(merged), tmask, e3t


def test_mesh_file_list_forms():
    assert mesh_file_list("a.nc") == ["a.nc"]
    assert mesh_file_list(["a.nc", "b.nc"]) == ["a.nc", "b.nc"]
    assert mesh_file_list(os.pathsep.join(["a.nc", "b.nc"])) == ["a.nc", "b.nc"]
    with pytest.raises(ValueError):
        mesh_file_list([])


def test_split_mesh_reads_like_merged(split_and_merged):
    files, merged, _, _ = split_and_merged
    a = _read_nemo_mesh_mask(files)
    b = _read_nemo_mesh_mask(merged)
    assert set(a) == set(b) and "tmask" in a and "e1t" in a
    for k in b:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    # pathsep-joined string form is the same read
    c = _read_nemo_mesh_mask(os.pathsep.join(files))
    for k in b:
        np.testing.assert_array_equal(np.asarray(c[k]), np.asarray(b[k]))


def test_strip_north_rows_drops_exactly_n():
    raw = {"glamt": np.arange(NLAT * NLON).reshape(NLAT, NLON)}
    assert strip_north_rows_raw(raw, 0) is raw
    out = strip_north_rows_raw(raw, 1)
    np.testing.assert_array_equal(out["glamt"], raw["glamt"][:-1])
    with pytest.raises(ValueError):
        strip_north_rows_raw(raw, -1)
    with pytest.raises(ValueError):
        strip_north_rows_raw(raw, NLAT)


def test_read_mesh_mask_bathy_split_equals_merged_and_strips(split_and_merged):
    files, merged, tmask, e3t = split_and_merged
    lm_a, hb_a = read_mesh_mask_bathy(files)
    lm_b, hb_b = read_mesh_mask_bathy(merged)
    np.testing.assert_array_equal(lm_a, lm_b)
    np.testing.assert_array_equal(hb_a, hb_b)
    np.testing.assert_array_equal(lm_a, tmask[0, 0].astype(np.float64))
    # depth is the masked column sum on surface-wet columns and exactly 0 on
    # surface-land columns (the random fixture has wet cells under a dry
    # surface; a real NEMO mask never does, so this only pins the land rule)
    expect = np.where(tmask[0, 0] > 0, (e3t[0] * tmask[0]).sum(axis=0), 0.0)
    np.testing.assert_allclose(hb_a, expect)
    lm_s, hb_s = read_mesh_mask_bathy(files, strip_north_rows=1)
    assert lm_s.shape == (NLAT - 1, NLON) and hb_s.shape == (NLAT - 1, NLON)
    np.testing.assert_array_equal(lm_s, lm_a[:-1])
    np.testing.assert_array_equal(hb_s, hb_a[:-1])


def test_read_mesh_mask_bathy_accepts_nemo3_e3t_name(tmp_path, split_and_merged):
    files, _, tmask, e3t = split_and_merged
    zgr_old = tmp_path / "mesh_zgr_old.nc"
    _write(zgr_old, {"e3t": (("t", "z", "y", "x"), e3t)})
    lm, hb = read_mesh_mask_bathy([files[0], str(zgr_old), files[2]])
    expect = np.where(tmask[0, 0] > 0, (e3t[0] * tmask[0]).sum(axis=0), 0.0)
    np.testing.assert_allclose(hb, expect)
    with pytest.raises(KeyError):
        read_mesh_mask_bathy([files[0], files[2]])       # no vertical file


def _write_tripole_like_mesh(path, n_lat, n_lon, *, dead_north_row):
    """Regular lat-lon metrics with a self-dual LAST row (constant latitude ->
    a NEMO T-pivot pivot row) and, optionally, one extra DEAD halo row above
    it whose latitude is NOT fold-symmetric (like ORCA12's stored last row).
    Returns the path list and the number of rows written."""
    rows = n_lat + (1 if dead_north_row else 0)
    lat = np.linspace(-60.0, 60.0, n_lat)
    lat_2d = np.repeat(lat[:, None], n_lon, axis=1)
    lat_2d[-1, :] = 66.0                                 # pivot row: constant
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    lon_2d = np.repeat(lon[None, :], n_lat, axis=0)
    if dead_north_row:
        lat_2d = np.concatenate(
            [lat_2d, 60.0 + 20.0 * np.arange(n_lon)[None, :] / n_lon], axis=0)
        lon_2d = np.concatenate([lon_2d, lon_2d[-1:]], axis=0)
    metric = np.full((rows, n_lon), 1.0e5)
    twod = {}
    for n in ("glamt", "glamu", "glamv", "glamf"):
        twod[n] = (("t", "y", "x"), lon_2d[None])
    for n in ("gphit", "gphiu", "gphiv", "gphif"):
        twod[n] = (("t", "y", "x"), lat_2d[None])
    for n in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"):
        twod[n] = (("t", "y", "x"), metric[None])
    tmask = np.ones((1, NLEV, rows, n_lon), dtype=np.int8)
    if dead_north_row:
        tmask[0, :, -1, :] = 0
    masks = {n: (("t", "z", "y", "x"), tmask.copy())
             for n in ("tmask", "umask", "vmask", "fmask")}
    masks["tmaskutil"] = (("t", "y", "x"), tmask[:, 0].copy())
    masks["e3t_0"] = (("t", "z", "y", "x"),
                      np.full((1, NLEV, rows, n_lon), 10.0))
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("t", 1)
    ds.createDimension("z", NLEV)
    ds.createDimension("y", rows)
    ds.createDimension("x", n_lon)
    for name, (dims, arr) in {**twod, **masks}.items():
        v = ds.createVariable(name, arr.dtype, dims)
        v[:] = arr
    ds.close()
    return rows


def test_create_tripole_grid_strip_north_rows_makes_pivot_row_last(tmp_path):
    """ORCA12-style mesh: a dead, fold-asymmetric halo row above the self-dual
    pivot row.  Without stripping the explicit fold convention is REJECTED (the
    last row is not self-symmetric); with ``strip_north_rows=1`` the pivot row
    becomes the last row, the fold validates, and the geometry has one row
    less.  Reverting the strip in ``create_tripole_grid`` fails this test."""
    from legoesm.grids.tripole import create_tripole_grid
    n_lat, n_lon = 12, 16
    p = tmp_path / "mesh_mask.nc"
    rows = _write_tripole_like_mesh(p, n_lat, n_lon, dead_north_row=True)
    assert rows == n_lat + 1
    with pytest.raises(ValueError, match="Fold symmetry check failed"):
        create_tripole_grid(str(p), fold_convention="(n_lon-i)%n_lon")
    geom = create_tripole_grid(str(p), fold_convention="(n_lon-i)%n_lon",
                               strip_north_rows=1)
    assert int(geom.n_lat) == n_lat and int(geom.n_lon) == n_lon
    assert bool(geom.fold.is_active) and int(geom.fold.fold_j) == n_lat - 1
    lm, hb = read_mesh_mask_bathy(str(p), strip_north_rows=1)
    assert lm.shape == (n_lat, n_lon) and lm[-1].min() == 1.0   # pivot row wet
    np.testing.assert_allclose(hb, 10.0 * NLEV)


def test_create_tripole_grid_ignores_unset_halo_columns_on_fold_row(tmp_path):
    """NOC ORCA0083 mesh_hgr stores (lat, lon) = (0, 0) in the two cyclic halo
    columns of its fold rows (never filled).  Those placeholders must not
    trip the fold-row self-symmetry check (they are land and no Arctic fold
    row can pass through (0 N, 0 E)); a genuinely asymmetric fold row must
    still be refused."""
    from legoesm.grids.tripole import create_tripole_grid
    n_lat, n_lon = 12, 16
    p = tmp_path / "mesh_mask.nc"
    _write_tripole_like_mesh(p, n_lat, n_lon, dead_north_row=False)
    ds = netCDF4.Dataset(p, "r+")
    for v in ("glamt", "gphit"):
        ds[v][0, -1, -2:] = 0.0                     # unset halo columns
    ds.close()
    geom = create_tripole_grid(str(p), fold_convention="(n_lon-i)%n_lon")
    assert int(geom.n_lat) == n_lat and bool(geom.fold.is_active)
    # control: a real asymmetry (not a (0,0) placeholder) is still refused
    ds = netCDF4.Dataset(p, "r+")
    ds["gphit"][0, -1, 3] = 20.0
    ds.close()
    with pytest.raises(ValueError, match="Fold symmetry check failed"):
        create_tripole_grid(str(p), fold_convention="(n_lon-i)%n_lon")


def test_read_mesh_mask_bathy_land_depth_is_zero_with_fillvalue_zero(tmp_path, split_and_merged):
    """NEMO writes e3t_0 with _FillValue = 0.0: xarray's default decoding
    turns land scale factors into NaN and the masked column sum into NaN on
    land.  The reader must return finite depths everywhere (0 on land)."""
    files, _, tmask, e3t = split_and_merged
    zgr_fill = tmp_path / "mesh_zgr_fill.nc"
    ds = netCDF4.Dataset(zgr_fill, "w")
    ds.createDimension("t", 1); ds.createDimension("z", NLEV)
    ds.createDimension("y", NLAT); ds.createDimension("x", NLON)
    v = ds.createVariable("e3t_0", "f8", ("t", "z", "y", "x"), fill_value=0.0)
    v[:] = e3t * tmask                        # land scale factors are exactly 0
    ds.close()
    lm, hb = read_mesh_mask_bathy([files[0], str(zgr_fill), files[2]])
    assert np.all(np.isfinite(hb))
    assert np.all(hb[lm < 0.5] == 0.0)
    np.testing.assert_allclose(hb[lm > 0.5], (e3t[0] * tmask[0]).sum(axis=0)[lm > 0.5])


def test_create_tripole_grid_refuses_mostly_unset_fold_row(tmp_path):
    """The placeholder rule must not turn an all-unset (0,0) fold row into a
    vacuous pass: more than a halo's worth of placeholders is refused."""
    from legoesm.grids.tripole import create_tripole_grid
    n_lat, n_lon = 12, 16
    p = tmp_path / "mesh_mask.nc"
    _write_tripole_like_mesh(p, n_lat, n_lon, dead_north_row=False)
    ds = netCDF4.Dataset(p, "r+")
    for v in ("glamt", "gphit"):
        ds[v][0, -1, :] = 0.0
    ds.close()
    with pytest.raises(ValueError, match="unset"):
        create_tripole_grid(str(p), fold_convention="(n_lon-i)%n_lon")
