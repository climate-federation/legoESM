"""Tests for the WB2 regridding primitive (Stage 0, Task 6). numpy/scipy only."""
import numpy as np

from evaluations.wb_regrid import wb2_grid, regrid_to_wb2


def test_wb2_grid_shape():
    lat, lon = wb2_grid()
    assert lat.shape == (121,) and lon.shape == (240,)
    assert lat[0] == -90.0 and lat[-1] == 90.0
    assert lon[0] == 0.0 and lon[-1] == 358.5


def test_regrid_constant_preserved():
    field = np.full((32, 64), 5.0)
    src_lat = np.linspace(-88.0, 88.0, 32)
    src_lon = np.linspace(0.0, 360.0, 64, endpoint=False)
    out, lat, lon = regrid_to_wb2(field, src_lat, src_lon)
    assert out.shape == (121, 240)
    assert np.allclose(out, 5.0, atol=1e-9)


def test_regrid_smooth_lat_field_small_interior_error():
    src_lat = np.linspace(-88.0, 88.0, 90)
    src_lon = np.linspace(0.0, 360.0, 180, endpoint=False)
    _, LAT = np.meshgrid(src_lon, src_lat)
    f = np.cos(np.deg2rad(LAT))
    out, tlat, tlon = regrid_to_wb2(f, src_lat, src_lon)
    TLON, TLAT = np.meshgrid(tlon, tlat)
    expect = np.cos(np.deg2rad(TLAT))
    interior = np.abs(TLAT) < 80.0            # exclude poleward extrapolation band
    assert np.max(np.abs(out - expect)[interior]) < 5e-3


def test_regrid_longitude_periodic_seam():
    # cos(lon) must regrid smoothly across the 360/0 seam (no discontinuity).
    src_lat = np.linspace(-88.0, 88.0, 40)
    src_lon = np.linspace(0.0, 360.0, 80, endpoint=False)
    LON, _ = np.meshgrid(src_lon, src_lat)
    f = np.cos(np.deg2rad(LON))
    out, tlat, tlon = regrid_to_wb2(f, src_lat, src_lon)
    TLON, _ = np.meshgrid(tlon, tlat)
    expect = np.cos(np.deg2rad(TLON))
    assert np.max(np.abs(out - expect)) < 5e-3


def test_regrid_mask_returns_bool():
    m = np.zeros((32, 64), dtype=bool)
    m[:, 10:20] = True
    src_lat = np.linspace(-88.0, 88.0, 32)
    src_lon = np.linspace(0.0, 360.0, 64, endpoint=False)
    out, lat, lon = regrid_to_wb2(m, src_lat, src_lon, mask=True)
    assert out.dtype == np.bool_
    assert out.any() and not out.all()


def test_regrid_handles_descending_latitude():
    # ERA5-style N->S latitude must be flipped internally, not mis-interpolated.
    src_lat = np.linspace(90.0, -90.0, 40)          # descending
    src_lon = np.linspace(0.0, 360.0, 80, endpoint=False)
    field = np.tile(src_lat[:, None], (1, 80))      # value == latitude
    out, tlat, tlon = regrid_to_wb2(field, src_lat, src_lon)
    i_eq = int(np.argmin(np.abs(tlat)))             # target row nearest equator
    assert abs(float(np.mean(out[i_eq]))) < 2.0     # latitude ~ 0 there
