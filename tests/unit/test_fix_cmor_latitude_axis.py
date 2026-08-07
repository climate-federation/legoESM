"""Tests for scripts/data/fix_cmor_latitude_axis.py.

The corrector must (a) land a displaced field back on its labelled axis,
(b) leave the already-correct ``areacella`` alone in interpolate mode,
(c) refuse to run twice or on a file it does not target, and (d) preserve
every value in relabel mode.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from legoesm import constants

_SPEC = importlib.util.spec_from_file_location(
    "fix_cmor_latitude_axis",
    Path(__file__).resolve().parents[2]
    / "scripts" / "data" / "fix_cmor_latitude_axis.py",
)
fx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fx)

NLAT, NLON = 36, 72


def _broken_dataset(nlat: int = NLAT, nlon: int = NLON) -> xr.Dataset:
    """A CMOR file exactly as the broken lane wrote it.

    Values are the smooth field ``sin(lat)`` evaluated at the POLE-INCLUSIVE
    latitudes, stored under CELL-CENTRE labels — the defect, reproduced.
    """
    lab = fx.cell_centre_lat(nlat)
    src = fx.pole_inclusive_lat(nlat)
    lon = np.linspace(360.0 / nlon / 2, 360.0 - 360.0 / nlon / 2, nlon)
    data = np.broadcast_to(np.sin(np.deg2rad(src))[None, :, None],
                           (2, nlat, nlon)).copy()
    # A second field with the geometry that actually matters: symmetric and
    # peaked at the equator, like annual-mean insolation.  sin(lat) is
    # ANTISYMMETRIC, so its global mean is zero on BOTH axes and cannot
    # detect the displacement at all (the first draft of this test used it
    # and compared 1e-17 against 1e-17).
    insol = np.broadcast_to(np.cos(np.deg2rad(src))[None, :, None],
                            (2, nlat, nlon)).copy()
    return xr.Dataset(
        {
            "rsdt": (("time", "lat", "lon"), data, {"units": "W m-2"}),
            "insol": (("time", "lat", "lon"), insol, {"units": "W m-2"}),
            "areacella": (("lat", "lon"),
                          fx.cell_area(lab, nlon, float(constants.R_earth))),
            "lat_bnds": (("lat", "bnds"), fx.lat_bounds(lab)),
        },
        coords={"time": [0.0, 1.0], "lat": lab, "lon": lon},
        attrs={"history": "written by legoESM"},
    )


# --------------------------------------------------------------------------
# axis helpers
# --------------------------------------------------------------------------

def test_axes_differ_and_are_the_two_conventions():
    lab, pole = fx.cell_centre_lat(NLAT), fx.pole_inclusive_lat(NLAT)
    np.testing.assert_allclose(lab[[0, -1]], [-87.5, 87.5], atol=1e-12)
    np.testing.assert_allclose(pole[[0, -1]], [-90.0, 90.0], atol=1e-12)
    # The displacement this tool undoes: lat/(nlat-1), max dlat/2 at the pole.
    np.testing.assert_allclose(pole - lab, lab / (NLAT - 1), atol=1e-12)
    assert fx.is_cell_centre_axis(lab) and not fx.is_cell_centre_axis(pole)


def test_cell_area_closes_on_the_sphere():
    a = fx.cell_area(fx.cell_centre_lat(NLAT), NLON, float(constants.R_earth))
    np.testing.assert_allclose(
        a.sum(), 4.0 * np.pi * float(constants.R_earth) ** 2, rtol=1e-12)


# --------------------------------------------------------------------------
# interpolate mode: the data moves onto the labels
# --------------------------------------------------------------------------

def test_interpolate_lands_the_field_on_its_labelled_axis():
    ds = _broken_dataset()
    lab = np.asarray(ds["lat"].values)
    truth = np.sin(np.deg2rad(lab))

    before = np.abs(ds["rsdt"].values[0, :, 0] - truth).max()
    out = fx.correct_dataset(ds, mode="interpolate", source_grid="mpas")
    after = np.abs(out["rsdt"].values[0, :, 0] - truth).max()

    assert before > 0.015, before          # the defect is present to start
    assert after < 0.002, after            # and is removed
    assert after < before / 5


def test_interpolate_recovers_the_area_weighted_global_mean():
    """The number that started this: an area-weighted mean, low before, right after.

    Uses the equator-peaked ``insol`` field — the displacement biases it LOW,
    exactly as it biased the real ``rsdt`` by -0.94%.
    """
    ds = _broken_dataset()
    w = np.asarray(ds["areacella"].values)
    gm = lambda f: float((f * w).sum() / w.sum())  # noqa: E731
    truth = gm(np.broadcast_to(
        np.cos(np.deg2rad(np.asarray(ds["lat"].values)))[:, None],
        (NLAT, NLON)))

    before = gm(ds["insol"].values[0])
    after = gm(fx.correct_dataset(ds)["insol"].values[0])
    # The defect is a LOW bias of the right order (~0.5% here vs 0.94% for
    # the real rsdt), and the correction removes an order of magnitude of it.
    assert before < truth
    assert (truth - before) / truth > 2e-3
    assert abs(after - truth) < abs(before - truth) / 10


def test_interpolate_leaves_grid_metrics_untouched():
    ds = _broken_dataset()
    out = fx.correct_dataset(ds, mode="interpolate")
    for name in ("areacella", "lat_bnds"):
        np.testing.assert_array_equal(out[name].values, ds[name].values)
    np.testing.assert_array_equal(out["lat"].values, ds["lat"].values)


def test_interpolate_never_extrapolates():
    """Targets lie strictly inside the source range, so no value may exceed it."""
    ds = _broken_dataset()
    out = fx.correct_dataset(ds)
    src = ds["rsdt"].values
    assert out["rsdt"].values.min() >= src.min() - 1e-12
    assert out["rsdt"].values.max() <= src.max() + 1e-12


# --------------------------------------------------------------------------
# relabel mode: the labels move onto the data
# --------------------------------------------------------------------------

def test_relabel_preserves_every_value_and_moves_the_axis():
    ds = _broken_dataset()
    out = fx.correct_dataset(ds, mode="relabel")
    np.testing.assert_array_equal(out["rsdt"].values, ds["rsdt"].values)
    np.testing.assert_allclose(
        out["lat"].values, fx.pole_inclusive_lat(NLAT), atol=1e-12)
    # areacella must follow the axis, and still close on the sphere.
    np.testing.assert_allclose(
        out["areacella"].values.sum(),
        4.0 * np.pi * float(constants.R_earth) ** 2, rtol=1e-12)
    assert not np.array_equal(out["areacella"].values, ds["areacella"].values)


# --------------------------------------------------------------------------
# dispatch hardening + idempotence: refuse rather than silently mangle
# --------------------------------------------------------------------------

def test_refuses_a_second_pass():
    out = fx.correct_dataset(_broken_dataset())
    assert fx.FIX_ATTR in out.attrs
    with pytest.raises(ValueError, match="double-correct"):
        fx.correct_dataset(out)


def test_refuses_a_non_cell_centre_axis():
    ds = _broken_dataset()
    ds = ds.assign_coords(lat=("lat", fx.pole_inclusive_lat(NLAT)))
    with pytest.raises(ValueError, match="not the cell-centre axis"):
        fx.correct_dataset(ds)


@pytest.mark.parametrize("kwargs, match", [
    ({"mode": "shuffle"}, "unknown mode"),
    ({"source_grid": "latlon"}, "unknown source_grid"),
    ({"source_grid": "gaussian"}, "unknown source_grid"),
])
def test_unknown_selection_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fx.correct_dataset(_broken_dataset(), **kwargs)


def test_refuses_a_dataset_without_lat():
    ds = xr.Dataset({"x": ("time", [1.0, 2.0])}, coords={"time": [0.0, 1.0]})
    with pytest.raises(ValueError, match="no 'lat' dimension"):
        fx.correct_dataset(ds)


# --------------------------------------------------------------------------
# CLI round trip
# --------------------------------------------------------------------------

def test_cli_writes_a_corrected_tree(tmp_path):
    tree = tmp_path / "cmor" / "Amon"
    tree.mkdir(parents=True)
    _broken_dataset().to_netcdf(tree / "rsdt_Amon.nc")

    out = tmp_path / "fixed"
    assert fx.main([str(tmp_path / "cmor"), "--out", str(out),
                    "--source-grid", "mpas"]) == 0
    got = out / "Amon" / "rsdt_Amon.nc"
    assert got.exists()
    with xr.open_dataset(got, decode_times=False) as d:
        assert fx.FIX_ATTR in d.attrs
        truth = np.sin(np.deg2rad(np.asarray(d["lat"].values)))
        assert np.abs(d["rsdt"].values[0, :, 0] - truth).max() < 0.002


def test_cli_dry_run_writes_nothing(tmp_path):
    tree = tmp_path / "cmor"
    tree.mkdir()
    _broken_dataset().to_netcdf(tree / "rsdt_Amon.nc")
    out = tmp_path / "fixed"
    assert fx.main([str(tree), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_cli_missing_tree_returns_error(tmp_path):
    assert fx.main([str(tmp_path / "nope")]) == 2
