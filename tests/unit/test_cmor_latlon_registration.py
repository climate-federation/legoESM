"""CMOR lat-lon output is SAMPLED at the latitudes it is LABELLED with.

The Voronoi and cube regridders default to pole-to-pole rows
(``linspace(-90, 90, n_lat)``) while the CMOR writer labels cell centres
(``-90 + dlat/2 ..``): every stored row sat up to half a cell off its label.
"""
from __future__ import annotations

import glob
import types

import numpy as np
import pytest
import xarray as xr

from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.factory import create_grid
from legoesm.io.cmor_output import LatSamplingMismatchError

RES = 5.0  # production CMOR resolution; half-cell misregistration = 2.5 deg


def _collector(output_dir=None):
    return DiagnosticCollector(
        nlev=4, sigma_full=np.linspace(0.1, 0.9, 4), dsigma=np.full(4, 0.25),
        experiment_id="amip", monthly_means=True, cmip_output=True, n_days=30,
        output_dir=output_dir, cmip_resolution_deg=RES, start_year=1979)


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 5, lloyd_iterations=10)  # ~2.4 deg cells


def test_fx_rows_hold_the_labelled_latitude(mesh, tmp_path):
    """A field linear in latitude, written through the real fx writer, stores
    in each row the value at that row's label."""
    dc = _collector(str(tmp_path))
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    dc.set_fixed_fields(land_fraction=(lat_deg + 90.0) / 180.0)
    dc.finalize_cmip_fixed()
    [f] = glob.glob(f"{tmp_path}/**/sftlf_*.nc", recursive=True)
    ds = xr.open_dataset(f)
    lat = ds["lat"].values
    stored_lat = ds["sftlf"].values.mean(axis=-1) / 100.0 * 180.0 - 90.0
    # Pole-to-pole sampling puts the edge rows 2.5 deg off; IDW error on a
    # linear field at this mesh is ~0.1 deg in the zonal mean.
    np.testing.assert_allclose(stored_lat, lat, atol=0.4)
    bnds = ds["lat_bnds"].values
    assert np.all((bnds[:, 0] < lat) & (lat < bnds[:, 1]))
    np.testing.assert_allclose(bnds.mean(axis=1), lat, atol=1e-9)


def test_sampled_rows_equal_file_labels_every_lane(mesh):
    """Weights sample exactly the labels every CMOR table is written with
    (Amon/day/fx all call ``_cmip_target_latlon``)."""
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    lat, lon = dc._cmip_target_latlon()
    np.testing.assert_array_equal(dc._voronoi_regrid_weights.lat_cent, lat)
    # Columns: regridder order rolled by _roll_to_cmip_lon == file labels.
    rolled = dc._roll_to_cmip_lon(dc._voronoi_regrid_weights.lon_cent[None, :])[0]
    np.testing.assert_allclose(rolled % 360.0, lon, atol=1e-9)

    cs = _collector()
    cs.set_cmip_grid_info(grid_type="cubed_sphere",
                          grid=types.SimpleNamespace(n=12), start_year=1979)
    np.testing.assert_array_equal(cs._cs_regrid_weights.lat_cent, lat)


def _fed(mesh, tmp_path):
    dc = _collector(str(tmp_path))
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    n = int(mesh.nCells)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=np.full((n, 4), 280.0), p_s=np.full(n, 1.0e5),
        lat_deg=np.degrees(np.asarray(mesh.latCell)))
    return dc


def test_resume_refuses_a_pre_fix_sidecar(mesh, tmp_path):
    """A sidecar holding pole-to-pole sums must not be merged into a month
    now sampled at the labels; a stamped one resumes."""
    dc = _fed(mesh, tmp_path)
    good = tmp_path / "cmor_accum_day_0015.npz"
    dc.save_cmor_accumulators(good)
    assert _fed(mesh, tmp_path).load_cmor_accumulators(good) is True
    with np.load(good) as z:
        old = {k: z[k] for k in z.files if not k.startswith("meta.")}
    stale = tmp_path / "old_sidecar.npz"
    np.savez(stale, **old)
    with pytest.raises(LatSamplingMismatchError, match="pole-to-pole"):
        _fed(mesh, tmp_path).load_cmor_accumulators(stale)


def test_append_refuses_an_unstamped_series(mesh, tmp_path):
    """A series file from before the fix is not extended with centre-sampled
    months; a series this code started is."""
    dc = _fed(mesh, tmp_path)
    dc.flush_cmip_monthly(40.0)
    [f] = glob.glob(f"{tmp_path}/**/tas_Amon_*.nc", recursive=True)
    with xr.open_dataset(f) as ds:
        assert ds.attrs["legoesm_lat_sampling"] == "labelled_cell_centres"
        ds = ds.load()
    del ds.attrs["legoesm_lat_sampling"]
    ds.to_netcdf(f)
    dc2 = _collector(str(tmp_path))
    dc2.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    n = int(mesh.nCells)
    dc2.feed_cmip_accumulators_native(
        day=45.0, T=np.full((n, 4), 280.0), p_s=np.full(n, 1.0e5),
        lat_deg=np.degrees(np.asarray(mesh.latCell)))
    with pytest.raises(LatSamplingMismatchError, match="pole-to-pole"):
        dc2.flush_cmip_monthly(70.0)


def test_regridder_rejects_bad_row_latitudes():
    from legoesm.grids.regridding import compute_voronoi_to_latlon_weights
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_voronoi_to_latlon_weights(
            np.zeros(3), np.zeros(3), n_lon=4, n_lat=3,
            lat_cent=np.array([10.0, 0.0, 20.0]))
    with pytest.raises(ValueError, match="finite"):
        compute_voronoi_to_latlon_weights(
            np.zeros(3), np.zeros(3), n_lon=4, n_lat=3,
            lat_cent=np.array([0.0, np.nan, 20.0]))


def _structured(lat_deg, lon_deg):
    grid = types.SimpleNamespace(lat=np.radians(lat_deg), lon=np.radians(lon_deg))
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="latlon", grid=grid, start_year=1979)
    return dc


def test_latlon_lane_interpolates_same_shape_grid_to_labels():
    """The lat-lon grid has the target SHAPE (5 deg: centred rows, lon
    0..355) but its columns sit half a cell west of the 2.5..357.5 labels:
    it must be interpolated, periodically across 360/0."""
    nlat, nlon = int(180 / RES), int(360 / RES)
    lat = np.linspace(-90 + RES / 2, 90 - RES / 2, nlat)
    lon = np.arange(nlon) * RES
    dc = _structured(lat, lon)
    f = lambda la, lo: la[:, None] + 10.0 * np.sin(np.radians(lo))[None, :]
    out = dc._regrid_to_latlon_2d(f(lat, lon))
    lat_l, lon_l = dc._cmip_target_latlon()
    # Bilinear error on 10*sin at 5 deg spacing <= 10*(0.0873**2)/8 ~ 0.0095;
    # the half-cell passthrough was off by up to 10*0.0436 = 0.436.
    np.testing.assert_allclose(out, f(lat_l, lon_l), atol=0.011)
    # Wrap column (label 357.5) interpolates 355 and 360==0, not clamps.
    np.testing.assert_allclose(out[:, -1], lat + 10.0 * np.sin(np.radians(357.5)),
                               atol=0.011)
    out3 = dc._regrid_to_latlon_3d(np.stack([f(lat, lon)] * 2, axis=-1))
    np.testing.assert_allclose(out3[..., 1], out, atol=1e-12)
    assert dc.cf_writer.require_centre_sampling_on_append


def test_latlon_lane_descending_rows_land_on_labels():
    """North-to-south native rows (either order is legal) interpolate onto
    the ascending labels; a field linear in latitude is exact."""
    lat = np.linspace(90.0, -90.0, 37)
    lon = np.arange(72) * 5.0 + 2.5          # already the labelled columns
    dc = _structured(lat, lon)
    out = dc._regrid_to_latlon_2d(np.broadcast_to(lat[:, None], (37, 72)))
    lat_l, _ = dc._cmip_target_latlon()
    np.testing.assert_allclose(out, np.broadcast_to(lat_l[:, None], out.shape),
                               atol=1e-9)


def test_latlon_lane_exact_label_grid_passes_through():
    nlat, nlon = int(180 / RES), int(360 / RES)
    lat = np.linspace(-90 + RES / 2, 90 - RES / 2, nlat)
    lon = np.arange(nlon) * RES + RES / 2
    dc = _structured(lat, lon)
    assert dc._structured_regrid is None
    field = np.arange(nlat * nlon, dtype=float).reshape(nlat, nlon)
    np.testing.assert_array_equal(dc._regrid_to_latlon_2d(field), field)
