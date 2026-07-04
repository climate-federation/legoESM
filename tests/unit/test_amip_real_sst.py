"""Verify the AMIP loader consumes a REAL input4MIPs AMIP II bcs SST file.

Real PCMDI/input4MIPs AMIP II boundary conditions use ``tosbcs`` in
Kelvin and ``siconcbcs`` in percent — different from the synthetic deck
(``sst`` in Celsius, ``sic`` fraction).  This fabricates an
input4MIPs-shaped file and checks that, with the right deck flags
(sst_offset=0, sic_scale=0.01), the loader + the new units guard produce
physical SST (Kelvin) and a [0,1] ice fraction, and that the WRONG flags
(the synthetic-Celsius default offset) fail loudly.
"""

import numpy as np
import pytest
import xarray as xr
from legoesm.forcing.amip import AMIPForcingConfig, load_amip_forcing

from legoesm import constants


def _write_input4mips_sst(path, *, sst_units="K", sic_units="%"):
    nlat, nlon, nt = 36, 72, 12
    lat = np.linspace(-89, 89, nlat)
    lon = np.linspace(0, 357.5, nlon)
    # tosbcs in Kelvin: warm tropics ~302 K, cold high lat ~271 K.
    tos = 271.0 + 31.0 * np.cos(np.deg2rad(lat))[None, :, None] * np.ones(
        (nt, nlat, nlon))
    # siconcbcs in percent: 0 in tropics, up to ~95% at poles.
    sic = np.clip(
        100.0 * (np.abs(lat)[None, :, None] - 60.0) / 30.0, 0.0, 95.0
    ) * np.ones((nt, nlat, nlon))
    ds = xr.Dataset(
        {
            "tosbcs": (("time", "lat", "lon"), tos),
            "siconcbcs": (("time", "lat", "lon"), sic),
        },
        coords={"time": np.arange(nt, dtype=float), "lat": lat, "lon": lon},
    )
    ds["tosbcs"].attrs["units"] = sst_units
    ds["siconcbcs"].attrs["units"] = sic_units
    ds.to_netcdf(path)


def _target_grid():
    # Lat-lon grid (no JAX x64 requirement, unlike Gaussian) — the SST
    # regrid is grid-agnostic so this exercises the same loader path.
    from legoesm.grids.factory import create_grid
    return create_grid("latlon", 24)


def test_real_input4mips_sst_correct_flags(tmp_path):
    p = tmp_path / "amip2_bcs.nc"
    _write_input4mips_sst(p)
    grid = _target_grid()
    cfg = AMIPForcingConfig(
        dataset="custom", path=str(p),
        sst_var="tosbcs", sic_var="siconcbcs",
        sst_offset=0.0,    # already Kelvin
        sic_scale=0.01,    # percent -> fraction
    )
    forcing = load_amip_forcing(cfg, grid)
    sst = np.asarray(forcing.sst)
    sic = np.asarray(forcing.sic)
    assert np.all(np.isfinite(sst)) and np.all(np.isfinite(sic))
    assert 240.0 < sst.min() and sst.max() < 340.0, "SST not physical Kelvin"
    assert 0.0 <= sic.min() and sic.max() <= 1.0, "SIC not a [0,1] fraction"


def test_real_kelvin_sst_with_wrong_celsius_offset_fails(tmp_path):
    # The synthetic-Celsius default offset (+273.15) on a Kelvin file
    # must be rejected by the units guard, not silently produce ~575 K.
    p = tmp_path / "amip2_bcs.nc"
    _write_input4mips_sst(p)
    grid = _target_grid()
    cfg = AMIPForcingConfig(
        dataset="custom", path=str(p),
        sst_var="tosbcs", sic_var="siconcbcs",
        sst_offset=constants.T_freeze,   # Celsius offset: WRONG for a Kelvin file
        sic_scale=0.01,
    )
    with pytest.raises(ValueError, match="(?i)kelvin|spurious|physical ocean"):
        load_amip_forcing(cfg, grid)


def test_real_percent_sic_with_wrong_fraction_scale_fails(tmp_path):
    # A percent SIC file at scale 1.0 reaches ~95 -> must be rejected.
    p = tmp_path / "amip2_bcs.nc"
    _write_input4mips_sst(p)
    grid = _target_grid()
    cfg = AMIPForcingConfig(
        dataset="custom", path=str(p),
        sst_var="tosbcs", sic_var="siconcbcs",
        sst_offset=0.0,
        sic_scale=1.0,   # WRONG for a percent file
    )
    with pytest.raises(ValueError, match="(?i)percent|fraction|<=1"):
        load_amip_forcing(cfg, grid)


def test_real_era5_fraction_sic_with_wrong_percent_scale_fails(tmp_path):
    # NCAR-RDA ERA5 sea-ice (ci) carries units "(0-1)" — an unambiguous [0,1] fraction.
    # Running it with sic_scale=0.01 (the percent scale) must be REJECTED: else all values
    # *0.01 land in [0,0.01], passing the <=1 value-sanity floor and SILENTLY zeroing the
    # ice (codex-review iter 420).
    p = tmp_path / "era5_ci.nc"
    nlat, nlon, nt = 36, 72, 12
    lat = np.linspace(-89, 89, nlat)
    lon = np.linspace(0, 357.5, nlon)
    sst = 290.0 * np.ones((nt, nlat, nlon))
    sic = np.clip(np.abs(lat)[None, :, None] / 90.0, 0.0, 1.0) * np.ones((nt, nlat, nlon))
    ds = xr.Dataset(
        {"SSTK": (("time", "lat", "lon"), sst), "CI": (("time", "lat", "lon"), sic)},
        coords={"time": np.arange(nt, dtype=float), "lat": lat, "lon": lon})
    ds["SSTK"].attrs["units"] = "K"
    ds["CI"].attrs["units"] = "(0-1)"
    ds.to_netcdf(p)
    cfg = AMIPForcingConfig(
        dataset="custom", path=str(p), sst_var="SSTK", sic_var="CI",
        sst_offset=0.0, sic_scale=0.01)   # WRONG: "(0-1)" is already a fraction
    with pytest.raises(ValueError, match="(?i)fraction"):
        load_amip_forcing(cfg, _target_grid())
