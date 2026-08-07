"""Unit tests for the WOCE/Gouretski-IC → WOA18-format converter core fns.

Exercises the pure regrid + vertical-interp leaves directly (no large NEMO file):
a synthetic curvilinear patch must round-trip a smooth field to the 1° grid, and
the vertical interp must match a known linear column + degrade gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.init_woa import interp_column_to_depths
from scripts.data.build_woce_ic_latlon import (
    _xyz_unit_sphere, regrid_curv_to_latlon,
)


def test_xyz_unit_sphere_known_points():
    xyz = _xyz_unit_sphere(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    assert np.allclose(xyz[0], [1.0, 0.0, 0.0], atol=1e-12)   # equator/prime mer.
    assert np.allclose(xyz[1], [0.0, 0.0, 1.0], atol=1e-12)   # north pole
    assert np.allclose(np.linalg.norm(xyz, axis=-1), 1.0)


def test_regrid_recovers_smooth_field():
    # "curvilinear" source = a dense regular patch covering a region; field=lat.
    sy = np.linspace(-40.0, 40.0, 81)
    sx = np.linspace(0.0, 80.0, 81)
    LON, LAT = np.meshgrid(sx, sy)
    field = LAT.copy()                              # smooth, = latitude
    ocean = np.ones_like(field, dtype=bool)
    tgt_lat = np.arange(-30.5, 31.0, 1.0)
    tgt_lon = np.arange(10.5, 70.0, 1.0)
    out = regrid_curv_to_latlon(field, LAT, LON, ocean, tgt_lat, tgt_lon, k=4)
    # interior target latitudes should be recovered to well within a grid cell
    LO, LA = np.meshgrid(tgt_lon, tgt_lat)
    assert np.nanmax(np.abs(out - LA)) < 0.6
    assert np.isfinite(out).all()


def test_regrid_all_land_is_nan():
    sy = np.linspace(-10, 10, 11); sx = np.linspace(0, 10, 11)
    LON, LAT = np.meshgrid(sx, sy)
    ocean = np.zeros_like(LAT, dtype=bool)          # no ocean sources
    out = regrid_curv_to_latlon(LAT, LAT, LON, ocean,
                                np.array([0.0]), np.array([5.0]))
    assert np.isnan(out).all()


def test_regrid_ignores_nan_sources():
    sy = np.linspace(-20, 20, 41); sx = np.linspace(0, 40, 41)
    LON, LAT = np.meshgrid(sx, sy)
    field = LAT.copy()
    ocean = np.ones_like(field, dtype=bool)
    field[LAT < 0] = np.nan                          # mask south half as land
    ocean[LAT < 0] = False
    out = regrid_curv_to_latlon(field, LAT, LON, ocean,
                                np.array([10.5]), np.array([20.5]), k=4)
    assert np.isfinite(out).all()
    assert abs(float(out[0, 0]) - 10.5) < 1.5        # ~ nearest northern value


def test_regrid_locality_cutoff_leaves_far_targets_nan():
    # single ocean source near the equator/prime meridian
    src_lat = np.array([[0.0]]); src_lon = np.array([[0.0]])
    field = np.array([[15.0]]); ocean = np.array([[True]])
    tgt_lon = np.array([0.0])
    # near target (<3°) is filled; far target (>3°) stays NaN (no donor in reach)
    near = regrid_curv_to_latlon(field, src_lat, src_lon, ocean,
                                 np.array([1.0]), tgt_lon, k=1, max_deg=3.0)
    far = regrid_curv_to_latlon(field, src_lat, src_lon, ocean,
                                np.array([45.0]), tgt_lon, k=1, max_deg=3.0)
    assert np.isfinite(near).all() and abs(float(near[0, 0]) - 15.0) < 1e-6
    assert np.isnan(far).all()


def test_vertical_interp_below_bottom_holds_edge():
    # WOA depths deeper than the deepest valid native level hold the deepest value
    src_z = np.array([0.0, 100.0, 500.0])
    col = np.array([18.0, 12.0, 6.0])
    out = interp_column_to_depths(col, src_z, np.array([500.0, 1000.0, 5000.0]))
    assert np.isclose(out[0], 6.0)
    assert np.isclose(out[1], 6.0) and np.isclose(out[2], 6.0)   # edge-held


def test_vertical_interp_linear_column():
    src_z = np.array([0.0, 100.0, 1000.0])
    col = np.array([20.0, 10.0, 4.0])                # piecewise-linear in z
    woa = np.array([0.0, 50.0, 100.0, 500.0])
    out = interp_column_to_depths(col, src_z, woa)
    assert np.isclose(out[0], 20.0)
    assert np.isclose(out[1], 15.0)                  # midway 0-100 m
    assert np.isclose(out[2], 10.0)
    assert 4.0 < out[3] < 10.0


def test_vertical_interp_degenerate_is_nan():
    src_z = np.array([0.0, 100.0, 1000.0])
    col = np.array([np.nan, np.nan, 4.0])            # <2 valid
    out = interp_column_to_depths(col, src_z, np.array([0.0, 50.0]))
    assert np.isnan(out).all()
