"""Unit tests for ``scripts/data/prep_era5_reference_mon.py``.

Covers the pure helpers that decide whether the staged ERA5 reference files
are byte-compatible with the ones the AMIP pattern scorer already consumes:
the month-midpoint time axis, the CF cell bounds, the filename stamp, the
plev19 level set, and the CDO operator ORDER (which is what actually
determines the emitted grid and level direction).

The expected time values below are read off the already-staged
``ua_native6_ERA5_Amon_mon_19790116120000-...nc``: time[0..3] =
47131.5, 47161.0, 47190.5, 47221.0 with time_bnds[0] = (47116, 47147).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    here = Path(__file__).resolve().parents[2]
    path = here / "scripts" / "data" / "prep_era5_reference_mon.py"
    spec = importlib.util.spec_from_file_location("_prep_era5_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()


# --- time axis ------------------------------------------------------------

def test_time_axis_matches_staged_era5_values():
    centres, bnds = MOD.month_midpoint_time_axis(1979, 1979)
    assert centres.shape == (12,)
    assert bnds.shape == (12, 2)
    # exact values from the staged ua file
    np.testing.assert_allclose(
        centres[:4], [47131.5, 47161.0, 47190.5, 47221.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(bnds[0], [47116.0, 47147.0], rtol=0, atol=1e-9)


def test_time_centres_are_true_month_midpoints_not_fixed_offset():
    """February must NOT sit on the 16th -- a fixed offset would drift."""
    centres, bnds = MOD.month_midpoint_time_axis(1979, 1979)
    np.testing.assert_allclose(centres, bnds.mean(axis=1), atol=1e-12)
    # January (31 d) and February (28 d) have different half-widths
    assert not np.isclose(centres[1] - bnds[1, 0], centres[0] - bnds[0, 0])


def test_time_axis_is_contiguous_and_monotonic():
    centres, bnds = MOD.month_midpoint_time_axis(1979, 1982)
    assert centres.size == 4 * 12
    assert np.all(np.diff(centres) > 0)
    # each month's upper bound is the next month's lower bound
    np.testing.assert_allclose(bnds[:-1, 1], bnds[1:, 0], atol=1e-12)
    # leap year 1980 February spans 29 days
    assert np.isclose(bnds[13, 1] - bnds[13, 0], 29.0)


def test_stamp_matches_staged_filename_convention():
    assert MOD.stamp(47131.5) == "19790116120000"
    centres, _ = MOD.month_midpoint_time_axis(2014, 2014)
    assert MOD.stamp(centres[-1]) == "20141216120000"


# --- cell bounds ----------------------------------------------------------

def test_cell_bounds_are_contiguous_and_cover_the_poles():
    lat = np.arange(-90.0, 90.0 + 0.25, 0.25)
    b = MOD.cell_bounds(lat)
    assert b.shape == (lat.size, 2)
    np.testing.assert_allclose(b[:-1, 1], b[1:, 0], atol=1e-9)
    assert b[0, 0] < -90.0 < b[0, 1]
    assert b[-1, 0] < 90.0 < b[-1, 1]


def test_cell_bounds_rejects_degenerate_input():
    with pytest.raises(ValueError):
        MOD.cell_bounds(np.array([1.0]))


# --- levels ---------------------------------------------------------------

def test_plev19_matches_the_model_cmor_axis():
    """The staged levels must equal the axis the model's CMOR output uses.

    If these drift apart the scorer's ``ref.interp(plev=model.plev)`` stops
    being an identity and silently starts interpolating.
    """
    from legoesm.io.cmor_output import CMIP6_PLEV19

    np.testing.assert_allclose(
        np.asarray(MOD.PLEV19_PA, dtype=float),
        np.asarray([float(p) for p in CMIP6_PLEV19]),
    )


def test_plev19_is_descending():
    p = np.asarray(MOD.PLEV19_PA)
    assert np.all(np.diff(p) < 0)
    assert p[0] == 100000.0


# --- CDO pipeline ---------------------------------------------------------

def test_cdo_command_order_for_3d_var():
    """Operators apply right-to-left; the order decides grid and lev direction."""
    spec = MOD.VAR_SPECS["wap"]
    cmd = MOD.build_cdo_command(
        spec, Path("in.grb"), Path("out.nc"), Path("grid.txt"))
    joined = " ".join(cmd)
    # rename happens last (leftmost), regularisation first (rightmost)
    assert cmd[-3] == "-setgridtype,regular"
    assert "-chname,var135,wap" in cmd
    # sellevel must precede remapbil in EXECUTION, i.e. appear to its RIGHT
    assert joined.index("-remapbil") < joined.index("-sellevel")
    # invertlev must run after remapbil, i.e. appear to its LEFT
    assert joined.index("-invertlev") < joined.index("-remapbil")
    assert all(f"{p:g}" in joined for p in MOD.PLEV19_PA)


def test_cdo_command_for_2d_var_has_no_level_ops():
    spec = MOD.VAR_SPECS["psl"]
    joined = " ".join(MOD.build_cdo_command(
        spec, Path("in.grb"), Path("out.nc"), Path("grid.txt")))
    assert "-sellevel" not in joined
    assert "-invertlev" not in joined
    assert "-chname,var151,psl" in joined


def test_grib_paths_point_at_the_right_pool_trees():
    assert MOD.VAR_SPECS["wap"].grib(1979) == Path(
        "/pool/data/ERA5/E5/pl/an/1M/135/E5pl00_1M_1979_135.grb")
    assert MOD.VAR_SPECS["psl"].grib(2014) == Path(
        "/pool/data/ERA5/E5/sf/an/1M/151/E5sf00_1M_2014_151.grb")


# --- target grid ----------------------------------------------------------

def test_target_grid_matches_staged_reference_grid():
    g = dict(
        line.split("=", 1) for line in MOD.TARGET_GRID.strip().splitlines()
    )
    g = {k.strip(): v.strip() for k, v in g.items()}
    assert g["gridtype"] == "lonlat"
    assert (int(g["xsize"]), int(g["ysize"])) == (1440, 721)
    # lat ASCENDING from -90, matching the staged ua/ta/ps coordinate
    assert float(g["yfirst"]) == -90.0 and float(g["yinc"]) == 0.25
    assert float(g["xfirst"]) == 0.0 and float(g["xinc"]) == 0.25
    assert int(g["gridsize"]) == 1440 * 721


# --- verification helpers -------------------------------------------------

def test_area_weighted_mean_uses_cos_lat():
    """A raw mean on a lat-lon grid over-weights the poles; this must not."""
    lat = np.array([-60.0, 0.0, 60.0])
    field = np.array([[0.0], [1.0], [0.0]])
    got = MOD._area_weighted_mean(field, lat)
    w = np.cos(np.deg2rad(lat))
    assert np.isclose(got, w[1] / w.sum())
    assert not np.isclose(got, field.mean())


def test_area_weighted_mean_makes_nan_fatal():
    with pytest.raises(ValueError):
        MOD._area_weighted_mean(np.array([[np.nan], [1.0]]),
                                np.array([0.0, 10.0]))
