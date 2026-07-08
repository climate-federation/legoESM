"""End-to-end test for the transient LUH2 surfdata producer (assemble side)."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_data.sources.luh2 import LUH2_STATE_NAMES

_IDX = {name: i for i, name in enumerate(CLM5_PFT_NAMES)}

pytest.importorskip("xarray")
pytest.importorskip("netCDF4")


def _write_base_surfdata(path, lat, lon):
    """A minimal static base harmonized surfdata (percent cover), for reuse as PNV."""
    ny, nx = lat.size, lon.size
    pft = np.zeros((1, N_PFT_CLM5, ny, nx))
    pft[0, _IDX["broadleaf_evergreen_tropical"]] = 30.0   # PNV natural shape (percent)
    pft[0, _IDX["c3_grass"]] = 10.0
    pft[0, _IDX["c4_grass"]] = 10.0                       # C4 grass fraction 0.5
    write_surfdata(
        path,
        lat=lat, lon=lon, soil_dz=np.array([1.0]),
        sand_pct=np.full((1, ny, nx), 40.0), clay_pct=np.full((1, ny, nx), 20.0),
        organic=np.full((1, ny, nx), 5.0), bulk_density=np.full((1, ny, nx), 1300.0),
        soil_color=np.full((ny, nx), 4.0), cell_area=np.full((ny, nx), 1.0e10),
        year=np.array([2000.0]),
        f_land=np.full((1, ny, nx), 50.0),
        f_lake=np.full((1, ny, nx), 3.0), f_glacier=np.full((1, ny, nx), 1.0),
        pft_frac=pft,
        monthly_lai=np.full((12, N_PFT_CLM5, ny, nx), 2.0),
        monthly_sai=np.full((12, N_PFT_CLM5, ny, nx), 0.5),
        monthly_height_top=np.full((12, N_PFT_CLM5, ny, nx), 10.0),
        monthly_height_bot=np.full((12, N_PFT_CLM5, ny, nx), 0.1),
    )


def _write_luh2_states(path, lat, lon, nyear=3):
    """A synthetic LUH2 states NetCDF on the same grid as the base (regrid = identity)."""
    import xarray as xr
    ny, nx = lat.size, lon.size
    rng = np.linspace(0.02, 0.05, len(LUH2_STATE_NAMES))
    data = {
        s: (("time", "lat", "lon"), np.full((nyear, ny, nx), v))
        for s, v in zip(LUH2_STATE_NAMES, rng)
    }
    xr.Dataset(
        data,
        coords={"time": np.arange(nyear, dtype=float), "lat": lat, "lon": lon},
    ).to_netcdf(path)


def test_build_luh2_transient_surfdata(tmp_path):
    import xarray as xr
    from legoesm.land.surface_data.assemble import build_luh2_transient_surfdata

    lat = np.linspace(-30.0, 30.0, 3)
    lon = np.linspace(0.0, 240.0, 4)
    base_nc = tmp_path / "base.nc"
    luh2_nc = tmp_path / "luh2.nc"
    out_nc = tmp_path / "transient.nc"
    _write_base_surfdata(base_nc, lat, lon)
    _write_luh2_states(luh2_nc, lat, lon, nyear=3)

    build_luh2_transient_surfdata(str(base_nc), str(luh2_nc), str(out_nc))

    ds = xr.open_dataset(out_nc, decode_times=False)
    try:
        # Transient pft_frac with a real year axis.
        assert ds["pft_frac"].dims == ("year", "npft", "lat", "lon")
        assert ds.sizes["year"] == 3 and ds.sizes["npft"] == N_PFT_CLM5
        assert ds["year"].values.tolist() == [850, 851, 852]   # default year_base

        pft = ds["pft_frac"].values          # percent
        f_land = ds["f_land"].values         # percent
        # f_land is the per-year PFT sum (percent), and stored cover is in percent.
        np.testing.assert_allclose(pft.sum(axis=1), f_land, rtol=1e-6)
        assert pft.max() <= 100.0 + 1e-6 and pft.min() >= -1e-9

        # Area conservation through the whole pipeline: output land (fraction) ==
        # regridded LUH2 12-state total.  On a shared grid the conservative regrid
        # is the identity, so the state total is the fixed per-cell sum of the 12
        # ramped state values.
        expected_state_total_frac = float(np.linspace(0.02, 0.05, len(LUH2_STATE_NAMES)).sum())
        np.testing.assert_allclose(f_land / 100.0, expected_state_total_frac, rtol=1e-6)

        # Static groups carried through from the base.
        np.testing.assert_allclose(ds["f_lake"].values, 3.0)
        np.testing.assert_allclose(ds["f_glacier"].values, 1.0)
        assert ds["f_lake"].dims == ("year", "lat", "lon")     # broadcast over years
        np.testing.assert_allclose(ds["sand_pct"].values, 40.0)
        assert ds.sizes.get("month", 12) == 12
    finally:
        ds.close()


def test_luh2_producer_cli_parser():
    from scripts.data.build_luh2_surfdata import build_arg_parser

    ap = build_arg_parser()
    args = ap.parse_args(
        ["--base-surfdata", "b.nc", "--luh2-states", "l.nc", "--out", "o.nc",
         "--year-start", "1850", "--year-end", "2014"])
    assert args.base_surfdata == "b.nc" and args.luh2_states == "l.nc"
    assert args.year_start == 1850 and args.year_end == 2014
    with pytest.raises(SystemExit):
        ap.parse_args(["--luh2-states", "l.nc"])   # missing required --base-surfdata/--out


def _write_hyde_states(path, lat, lon, nyear=3):
    """A synthetic HYDE states NetCDF on the same grid (regrid = identity)."""
    import xarray as xr
    ny, nx = lat.size, lon.size
    data = {
        "cropland": (("time", "lat", "lon"), np.full((nyear, ny, nx), 20.0)),   # km²
        "pasture": (("time", "lat", "lon"), np.full((nyear, ny, nx), 10.0)),
        "rangeland": (("time", "lat", "lon"), np.full((nyear, ny, nx), 5.0)),
        "built_up": (("time", "lat", "lon"), np.full((nyear, ny, nx), 2.0)),
        "garea": (("lat", "lon"), np.full((ny, nx), 100.0)),                    # km²/cell
    }
    xr.Dataset(
        data,
        coords={"time": np.array([1900.0, 1950.0, 2000.0]), "lat": lat, "lon": lon},
    ).to_netcdf(path)


def test_build_anthropogenic_transient_surfdata_hyde(tmp_path):
    import xarray as xr
    from legoesm.land.surface_data.assemble import build_anthropogenic_transient_surfdata

    lat = np.linspace(-30.0, 30.0, 3)
    lon = np.linspace(0.0, 240.0, 4)
    base_nc = tmp_path / "base.nc"
    hyde_nc = tmp_path / "hyde.nc"
    out_nc = tmp_path / "transient_hyde.nc"
    _write_base_surfdata(base_nc, lat, lon)
    _write_hyde_states(hyde_nc, lat, lon, nyear=3)

    build_anthropogenic_transient_surfdata(
        str(base_nc), str(hyde_nc), str(out_nc), dataset="hyde")

    ds = xr.open_dataset(out_nc, decode_times=False)
    try:
        assert ds["pft_frac"].dims == ("year", "npft", "lat", "lon")
        assert ds.sizes["year"] == 3 and ds.sizes["npft"] == N_PFT_CLM5
        assert ds["year"].values.tolist() == [1900, 1950, 2000]
        pft = ds["pft_frac"].values                    # percent
        f_land = ds["f_land"].values                   # percent
        # Area conservation WITHIN the base land fraction: base f_land is 50%, so
        # every cell/year cover sums to 50% (the ocean/lake/glacier mask is
        # preserved, not overwritten to 100%), and f_land == the PFT sum.
        np.testing.assert_allclose(pft.sum(axis=1), 50.0, atol=1e-6)
        np.testing.assert_allclose(f_land, 50.0, atol=1e-6)
        # Grid-cell anthropogenic fractions are preserved through the overlay:
        # crop 20 km²/100 = 0.2 -> crop_c3 (base C4-crop frac 0); PNV puts nothing
        # on crop rows, so crop_c3 is exactly the crop fraction (20%).
        np.testing.assert_allclose(pft[:, _IDX["crop_c3"]], 20.0, atol=1e-6)
        np.testing.assert_allclose(pft[:, _IDX["crop_c4"]], 0.0, atol=1e-6)
        # bare_soil: PNV shape has no bare weight, so bare == urban (built_up 2%).
        np.testing.assert_allclose(pft[:, _IDX["bare_soil"]], 2.0, atol=1e-6)
        # Grass rows carry BOTH the pasture overlay AND the natural residual routed
        # by the PNV shape (base grass is part of PNV): natural budget
        # 0.5-0.2-0.15-0.02=0.13, PNV c3/c4-grass weight 10/50=0.2 -> 0.026, plus
        # pasture 0.15 split 0.5 -> 0.075; total 0.101 each (base C4 ratio 0.5).
        np.testing.assert_allclose(pft[:, _IDX["c3_grass"]], 10.1, atol=1e-6)
        np.testing.assert_allclose(pft[:, _IDX["c4_grass"]], 10.1, atol=1e-6)
    finally:
        ds.close()


def test_build_anthropogenic_transient_surfdata_unknown_dataset(tmp_path):
    from legoesm.land.surface_data.assemble import build_anthropogenic_transient_surfdata

    lat = np.linspace(-30.0, 30.0, 3)
    lon = np.linspace(0.0, 240.0, 4)
    base_nc = tmp_path / "base.nc"
    _write_base_surfdata(base_nc, lat, lon)
    with pytest.raises(ValueError, match="unknown dataset"):
        build_anthropogenic_transient_surfdata(
            str(base_nc), "x.nc", str(tmp_path / "o.nc"), dataset="bogus")


def test_anthropogenic_producer_cli_parser():
    from scripts.data.build_anthropogenic_surfdata import build_arg_parser

    ap = build_arg_parser()
    args = ap.parse_args(
        ["--base-surfdata", "b.nc", "--anthro", "a.nc", "--dataset", "hyde",
         "--out", "o.nc", "--year-start", "1850", "--year-end", "2016",
         "--crop-share", "0.6"])
    assert args.dataset == "hyde" and args.anthro == "a.nc"
    assert args.year_start == 1850 and args.crop_share == 0.6
    with pytest.raises(SystemExit):
        ap.parse_args(["--dataset", "nope", "--base-surfdata", "b.nc",
                       "--anthro", "a.nc", "--out", "o.nc"])   # bad dataset choice
