"""NEMO Dai-Trenberth runoff wiring in run_omip_core2 (--runoff).

Validates the loader + month index that ungate the SSS comparison:
  * `_runoff_month_idx` maps perpetual-year time to the right NOLEAP calendar month;
  * `load_runoff_monthly` reads NEMO's runoff file, regrids to a model lat-lon grid,
    and returns a finite, non-negative (12, n_lat, n_lon) field with real discharge
    (total > 0).
Skips gracefully if the (host-resolved) NEMO runoff file is absent.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from scripts.run import run_omip_core2 as R


def test_month_idx_noleap_boundaries():
    """Day-of-year -> month uses noleap month lengths (Jan 0 .. Dec 11)."""
    dt = 300.0
    sec = 86400.0
    # (day_of_year, expected month index)
    cases = [(0, 0), (30, 0), (31, 1), (58, 1), (59, 2), (90, 3),
             (151, 5), (180, 5), (334, 11), (364, 11)]
    for day, exp in cases:
        step = int(round(day * sec / dt))
        assert R._runoff_month_idx(step, dt) == exp, (day, exp,
                                                      R._runoff_month_idx(step, dt))
    # perpetual wrap: day 365 -> 0 (Jan)
    assert R._runoff_month_idx(int(365 * sec / dt), dt) == 0


@pytest.mark.skipif(not os.path.exists(R._RUNOFF_NC),
                    reason="NEMO runoff file not present on host")
def test_load_runoff_monthly_latlon():
    """load_runoff_monthly returns a sane (12, n_lat, n_lon) runoff field."""
    from legoesm.grids.latlon import create_latlon_grid
    n_lat, n_lon = 90, 180
    grid = create_latlon_grid(n_lat, n_lon)
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    R_m = R.load_runoff_monthly(grid, "latlon", lat2d, lon2d, mesh_path=None)
    assert R_m.shape == (12, n_lat, n_lon)
    assert np.all(np.isfinite(R_m))
    assert np.all(R_m >= 0.0)              # runoff is INTO the ocean (>=0)
    assert R_m.sum() > 0.0                 # real discharge present
    # seasonality: not every month identical (climatological variation)
    assert not np.allclose(R_m[0], R_m[6])


@pytest.mark.skipif(not os.path.exists(R._RUNOFF_NC),
                    reason="NEMO runoff file not present on host")
def test_coastal_spread_conserves_area_integral_and_reduces_peak():
    """The coastal-spread option preserves the per-month AREA-INTEGRAL of the
    runoff [kg/s] (the loader's contract: the area-conservative renorm pins the
    target integral to the source total on EVERY grid) while reducing the
    over-concentrated peak (better SSS).  NOTE: the plain CELL-sum is NOT
    conserved on a lat-lon grid — spreading moves mass across cos(lat)-varying
    cell areas (~1e-4 relative shift), which is exactly why the renorm is
    area-weighted; asserting the cell sum was the pre-renorm (stale) contract."""
    from legoesm.grids.latlon import create_latlon_grid
    n_lat, n_lon = 90, 180
    grid = create_latlon_grid(n_lat, n_lon)
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    land = np.ones((n_lat, n_lon))  # all ocean (isolate the spread operator)
    R0 = R.load_runoff_monthly(grid, "latlon", lat2d, lon2d, None,
                               land_mask=land, spread_passes=0)
    R2 = R.load_runoff_monthly(grid, "latlon", lat2d, lon2d, None,
                               land_mask=land, spread_passes=3)
    # area-integral conserved per month (renormalised): identical totals with
    # and without spread.
    A = np.asarray(grid.area)
    s0 = (R0 * A).sum(axis=(1, 2)); s2 = (R2 * A).sum(axis=(1, 2))
    assert np.allclose(s0, s2, rtol=1e-6), (s0, s2)
    # peak reduced (spread smears the concentrated river mouths)
    assert R2.max() < R0.max()


def test_arctic_salt_forcing_flags_registered(capsys):
    """The three faithful-Arctic-salt levers are registered in the CLI."""
    import sys

    saved = sys.argv
    sys.argv = ["run_omip_core2.py", "--help"]
    try:
        with pytest.raises(SystemExit):
            R.main()
    finally:
        sys.argv = saved
    out = capsys.readouterr().out
    for flag in ("--sss-restore-file", "--nemo-monthly-init",
                 "--nemo-init-month", "--runoff-depth-nemo-ini"):
        assert flag in out, flag


def _white_sea_totals(regrid):
    """Jan runoff [m3/s] in 63-68N 30-45E: (model grid, NEMO source file)."""
    import xarray as xr
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(90, 180)
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    lon2d = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    Rm = R.load_runoff_monthly(grid, "latlon", lat2d, lon2d, None,
                               land_mask=np.ones((90, 180)), spread_passes=0,
                               regrid=regrid)[0]
    box = (lat2d >= 63) & (lat2d <= 68) & (lon2d >= 30) & (lon2d <= 45)
    tgt = float((Rm * np.asarray(grid.area))[box].sum()) / 1000.0
    ds = xr.open_dataset(R._RUNOFF_NC, decode_times=False)
    src = sum(np.nan_to_num(np.asarray(ds[v].values[0], float))
              for v in R._runoff_component_vars(False) if v in ds)
    sl, so = R._squeeze2d(ds["nav_lat"].values), R._squeeze2d(ds["nav_lon"].values)
    so = np.where(so > 180.0, so - 360.0, so)
    sbox = (sl >= 63) & (sl <= 68) & (so >= 30) & (so <= 45)
    return tgt, float((src * R._load_nemo_cell_area_m2())[sbox].sum()) / 1000.0


def test_volume_nearest_conserves_each_river_locally():
    """volume_nearest keeps the White Sea's rivers in the White Sea (to the
    global renorm, ~1); idw4 is reported alongside as the defect it replaces."""
    tgt, src = _white_sea_totals("volume_nearest")
    tgt_idw, _ = _white_sea_totals("idw4")
    print(f"White Sea Jan runoff m3/s: source {src:.0f}, volume_nearest {tgt:.0f}, idw4 {tgt_idw:.0f}")
    assert src > 1000.0
    assert abs(tgt / src - 1.0) < 0.10, (tgt, src)


def test_unknown_runoff_regrid_raises():
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(10, 20)
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    with pytest.raises(ValueError, match="runoff regrid"):
        R.load_runoff_monthly(grid, "latlon", lat2d, lon2d, None,
                              land_mask=np.ones((10, 20)), regrid="bogus")
