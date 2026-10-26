#!/usr/bin/env python
"""Generate synthetic CMIP6-shape AMIP forcing files for legoESM.

Produces a self-contained, **physically reasonable** mini AMIP deck on the
local filesystem that exercises every forcing channel of `run_amip.py`:

  - SST + SIC (lat-lon, monthly, 1979-2014)
  - GHG annual (CMIP6 `(time, lat=1, lon=1)` schema with CO2/CH4/N2O/CFCs)
  - Ozone monthly climatology with 30 pressure levels
  - Solar TSI + 14-band spectral fractions
  - Aerosol Kinne-style monthly zonal AOD (550 nm)
  - Volcanic stratospheric AOD time series

The fields are deliberately synthetic so the script can run without network
access, but they obey the **physical bounds and CMIP6 schemas** that the
loaders in ``src/legoesm/forcing/external.py`` and ``amip.py`` expect:

| Field            | Range                          | Reference            |
|------------------|--------------------------------|----------------------|
| Global-mean SST  | 287.5–289 K (warming trend)    | HadISST 1979–2014    |
| Sea-ice fraction | 0–1, polar caps only           | NSIDC                |
| CO2              | 336 → 397 ppm (1979 → 2014)    | NOAA Mauna Loa       |
| CH4              | 1550 → 1830 ppb                | NOAA AGGI            |
| N2O              | 301 → 327 ppb                  | NOAA AGGI            |
| Ozone column     | 250–400 DU (lat- and season-dep) | TOMS / SBUV         |
| TSI              | 1361 ± 0.7 W/m² (11-yr cycle)   | SOLARIS-HEPPA        |
| AOD (550 nm)     | 0.05–0.25, cosine-lat profile   | Kinne 2019           |
| Volcanic AOD     | small baseline + Pinatubo (1991)| GISS ATI             |

Usage::

    python scripts/generate_amip_forcing.py --out forcing/ \\
        --start-year 1979 --end-year 2014

Files produced (under ``--out``):

    sst_sic_amip_1979-2014.nc
    ghg_amip_1979-2014.nc
    ozone_amip_clim.nc
    solar_amip_1979-2014.nc
    aerosol_amip_clim.nc
    volcanic_amip_1979-2014.nc

The driver (``scripts/run/run_amip_smoke_deck.py``) consumes them through
the standard ``run_amip.py`` flags (``--forcing-path``, ``--ghg-file``,
``--ozone-file`` …).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import netCDF4
import numpy as np

# CLAUDE.md "Constant and Parameter Discipline": never hardcode 273.15 etc.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from legoesm import constants  # noqa: E402


# ============================================================================
# Time helpers
# ============================================================================

def monthly_midpoints_days(start_year: int, end_year: int) -> np.ndarray:
    """Days since start_year-01-01 of mid-month points for each month in
    [start_year, end_year]."""
    months_per_year = 12
    nyears = end_year - start_year + 1
    n = nyears * months_per_year
    out = np.empty(n, dtype=np.float64)
    # mid-month days within a 365-day year (no-leap to keep schema stable)
    mid_doy = np.array(
        [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]
    )
    for k in range(n):
        y = k // months_per_year
        m = k % months_per_year
        out[k] = y * 365.0 + mid_doy[m]
    return out


def fractional_years(start_year: int, end_year: int) -> np.ndarray:
    """Annual fractional-year axis matching the CMIP6 GHG file convention."""
    return np.arange(start_year, end_year + 1, dtype=np.float64)


# ============================================================================
# SST + SIC
# ============================================================================

def make_sst_sic(out_path: Path, start_year: int, end_year: int,
                 nlat: int = 73, nlon: int = 144, *,
                 warming_trend_K_per_decade: float = 0.18) -> None:
    """Generate a monthly SST + SIC NetCDF in HadISST-like format.

    Schema follows HadISST: ``sst`` (°C), ``sic`` (fraction), with
    coordinates ``lat`` (descending, +90 to -90), ``lon`` (0–359), ``time``
    (numeric days). The mean state is a Gaussian-in-latitude tropical
    warm pool (~302 K), polar minima (~273 K), with a smooth annual cycle
    and a linear historical warming trend.

    Sea-ice is set in the polar caps where SST < freezing, with hemispheric
    seasonal asymmetry.
    """
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    time = monthly_midpoints_days(start_year, end_year)
    ntime = time.size

    lat_rad = np.deg2rad(lat)

    # Climatological zonal-mean SST (Kelvin) — broad cosine² profile that gives
    # an area-weighted (cos lat) global mean of ~291 K, matching observations.
    # T(lat) = T_pole + (T_eq - T_pole) * cos²(lat) gives:
    #   integral_-π/2^π/2 T(lat) cos(lat) dlat / 2 = T_pole + (T_eq-T_pole)*2/3
    T_eq = 302.0
    T_pole = constants.T_freeze + 0.5  # ~273.65 K — slightly above sea-water freezing
    sst_clim = T_pole + (T_eq - T_pole) * np.cos(lat_rad) ** 2
    sst_clim = np.maximum(sst_clim, constants.T_freeze_ocean)  # 271.35 K

    # 2D climatology with weak zonal asymmetry (tropical Pacific cold tongue)
    lon_rad = np.deg2rad(lon)
    lat_factor = np.cos(lat_rad) ** 4
    lon_factor = np.cos(lon_rad - np.pi)
    cold_tongue = 1.0 * lat_factor[:, None] * lon_factor[None, :]
    sst_2d = sst_clim[:, None] - cold_tongue
    sst_2d = np.maximum(sst_2d, constants.T_freeze_ocean)

    # Annual cycle: phase shifts with hemisphere
    months = np.arange(ntime)
    annual_phase = 2 * np.pi * (months % 12) / 12.0

    rng = np.random.default_rng(42)
    sst_out = np.empty((ntime, nlat, nlon), dtype=np.float32)
    sic_out = np.zeros((ntime, nlat, nlon), dtype=np.float32)

    for t in range(ntime):
        # Hemispheric seasonal cycle: +amplitude in summer, -in winter
        hemi_amp = 8.0 * np.sin(lat_rad)
        seasonal = -hemi_amp * np.cos(annual_phase[t])
        # Tropics have small annual cycle; midlats have largest
        seasonal *= 1.0 - np.exp(-((lat_rad / np.deg2rad(40.0)) ** 2)) * 0.3

        # Linear warming trend: +0.18 K / decade since 1979
        years_since_start = t / 12.0
        warming = warming_trend_K_per_decade / 10.0 * years_since_start

        # Add small spatial-temporal noise for realism
        noise = rng.normal(0.0, 0.2, (nlat, nlon))

        sst_t = sst_2d + seasonal[:, None] + warming + noise
        sst_t = np.maximum(sst_t, constants.T_freeze_ocean)
        sst_out[t] = sst_t

        # Sea-ice: where SST < 274 K, with polar emphasis
        sic = np.where(sst_t < 274.0, np.clip((274.0 - sst_t) / 3.0, 0.0, 1.0), 0.0)
        # Suppress sea-ice in extra-polar regions
        sic *= np.where(np.abs(lat[:, None]) > 50.0, 1.0, 0.0)
        sic_out[t] = sic.astype(np.float32)

    # Convert to HadISST units: SST in Celsius, SIC as fraction
    sst_celsius = sst_out - constants.T_freeze

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", ntime)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = time
        t.units = f"days since {start_year}-01-01 00:00:00"
        t.calendar = "noleap"
        la = ds.createVariable("lat", "f4", ("lat",))
        la[:] = lat.astype(np.float32)
        la.units = "degrees_north"
        lo = ds.createVariable("lon", "f4", ("lon",))
        lo[:] = lon.astype(np.float32)
        lo.units = "degrees_east"
        sst_v = ds.createVariable("sst", "f4", ("time", "lat", "lon"),
                                   fill_value=-1.0e30)
        sst_v[:] = sst_celsius
        sst_v.units = "degC"
        sst_v.long_name = "Sea Surface Temperature (synthetic)"
        sic_v = ds.createVariable("sic", "f4", ("time", "lat", "lon"),
                                   fill_value=-1.0e30)
        sic_v[:] = sic_out
        sic_v.units = "1"
        sic_v.long_name = "Sea-ice concentration"
        ds.title = "Synthetic AMIP SST/SIC (legoESM)"
        ds.source = "scripts/generate_amip_forcing.py"
        ds.references = "Schema: HadISST"

    # Area-weighted global mean SST (cos-lat weighting)
    w_lat = np.cos(lat_rad)
    w_lat /= w_lat.sum()
    sst_K = sst_celsius + constants.T_freeze
    sst_global_mean = (sst_K.mean(axis=(0, 2)) * w_lat).sum()
    print(f"Wrote {out_path} ({ntime} months × {nlat} × {nlon}); "
          f"area-weighted global SST mean={sst_global_mean:.2f}K, "
          f"trend={warming_trend_K_per_decade:.2f} K/decade")


# ============================================================================
# GHG (annual_file, CMIP6 style)
# ============================================================================

# Historical observed mole fractions (NOAA AGGI / Mauna Loa).  The driver
# expects values in the file's native CMIP6 mass-fraction *factor* units:
# CO2 -> 1e-6, CH4/N2O -> 1e-9, CFCs -> 1e-12.
_GHG_HISTORICAL = {
    1979: dict(co2=336.78, ch4=1550.0, n2o=301.0, cfc11=170.0, cfc12=300.0),
    1985: dict(co2=346.04, ch4=1645.0, n2o=304.0, cfc11=220.0, cfc12=400.0),
    1990: dict(co2=354.39, ch4=1714.0, n2o=308.0, cfc11=259.0, cfc12=478.0),
    1995: dict(co2=360.91, ch4=1748.0, n2o=311.5, cfc11=265.0, cfc12=531.0),
    2000: dict(co2=369.55, ch4=1773.0, n2o=316.0, cfc11=261.0, cfc12=541.0),
    2005: dict(co2=379.80, ch4=1774.0, n2o=319.0, cfc11=251.0, cfc12=540.0),
    2010: dict(co2=389.85, ch4=1798.0, n2o=323.0, cfc11=240.0, cfc12=530.0),
    2014: dict(co2=397.55, ch4=1830.0, n2o=327.1, cfc11=233.0, cfc12=518.0),
    2021: dict(co2=414.72, ch4=1895.0, n2o=334.5, cfc11=224.0, cfc12=496.0),
}


def _interpolate_ghg(years: np.ndarray) -> dict:
    keys = np.array(sorted(_GHG_HISTORICAL.keys()), dtype=np.float64)
    out = {k: np.empty_like(years, dtype=np.float64)
           for k in ("co2", "ch4", "n2o", "cfc11", "cfc12")}
    for i, y in enumerate(years):
        if y <= keys[0]:
            row = _GHG_HISTORICAL[int(keys[0])]
        elif y >= keys[-1]:
            row = _GHG_HISTORICAL[int(keys[-1])]
        else:
            j = int(np.searchsorted(keys, y))
            y0, y1 = keys[j - 1], keys[j]
            r0 = _GHG_HISTORICAL[int(y0)]
            r1 = _GHG_HISTORICAL[int(y1)]
            f = (y - y0) / (y1 - y0)
            row = {k: r0[k] + f * (r1[k] - r0[k]) for k in r0}
        for k in out:
            out[k][i] = row[k]
    return out


def make_ghg_annual(out_path: Path, start_year: int, end_year: int) -> None:
    """Generate CMIP6-style annual GHG file with `(time, lat=1, lon=1)`
    schema and `time.units = "year as %Y.%f"`.

    Values are stored as raw ppmv/ppbv/pptv (matching the loader contract
    in ``_load_ghg_annual_file``). The ``units`` attribute records the
    CMIP6 multiplier convention (1e-6, 1e-9, 1e-12) for documentation.
    """
    years = fractional_years(start_year, end_year)
    n = years.size
    ghg = _interpolate_ghg(years)

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = years
        t.units = "year as %Y.%f"
        for name, key, units in [
            ("CO2", "co2", "1e-6"),
            ("CH4", "ch4", "1e-9"),
            ("N2O", "n2o", "1e-9"),
            ("CFC_11", "cfc11", "1e-12"),
            ("CFC_12", "cfc12", "1e-12"),
        ]:
            v = ds.createVariable(name, "f8", ("time", "lat", "lon"))
            v[:] = ghg[key].reshape(n, 1, 1)
            v.units = units
            v.long_name = f"{name} mole fraction (synthetic CMIP6 historical)"
        ds.title = "Synthetic CMIP6 historical GHG (legoESM)"
        ds.source = "scripts/generate_amip_forcing.py"

    print(f"Wrote {out_path} ({n} years; CO2 {ghg['co2'][0]:.1f} → "
          f"{ghg['co2'][-1]:.1f} ppm)")


# ============================================================================
# Ozone (monthly climatology with vertical levels)
# ============================================================================

def make_ozone_clim(out_path: Path, *, nlat: int = 36, nlev: int = 30,
                     start_year: int | None = None,
                     end_year: int | None = None) -> None:
    """Monthly zonal-mean ozone climatology or time-varying field.

    When ``start_year``/``end_year`` are *not* provided (default), writes
    a 12-month climatology that is sampled cyclically by the loader
    (``_interp_monthly_cyclic``).  This is the historical legoESM AMIP
    setup.

    When ``start_year`` and ``end_year`` are provided, writes an
    interannually-varying field with ``ntime = (end_year - start_year + 1) * 12``
    so the loader takes the non-cyclic dispatch
    (``_interp_monthly_noncyclic`` keyed on ``ntime > 12``) and the
    Antarctic ozone-hole season strengthens linearly from 1979→2014
    to mimic the CMIP6 input4MIPs reduced-ozone era.

    Schema:
      - var ``vmro3`` [mol/mol] dims ``(time, lat, plev)``
      - ``time``: mid-month days; ``"days since 0000-01-01"`` for the
        12-month clim, ``"days since <start_year>-01-01"`` and CF
        ``calendar="noleap"`` for the multi-year file (matches
        ``_extract_first_date`` requirements).
      - ``lat``: degrees_north
      - ``plev``: Pa, ascending (top of atmosphere → surface).
    """
    cyclic = start_year is None or end_year is None
    if cyclic:
        nyears = 1
        ntime = 12
    else:
        nyears = end_year - start_year + 1
        ntime = nyears * 12
    if cyclic:
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
    else:
        # Multi-year axis with mid-month days since start_year-01-01
        mid_days = monthly_midpoints_days(start_year, end_year)
    lat = np.linspace(-87.5, 87.5, nlat)
    # Ascending pressure axis (top of atmosphere → surface) so that
    # ``_interp_vertical``'s ``np.searchsorted(log_p_src, ...)`` operates on
    # a monotonically-increasing source.  CMIP6 files often store plev in
    # descending order; the loader (and the corresponding fix downstream)
    # canonicalises to ascending before vertical interpolation.
    plev = np.logspace(np.log10(0.1), np.log10(101325.0), nlev)  # Pa
    plev_hPa = plev / 100.0

    # Gaussian-in-log10(p) ozone profile centred at 30 hPa with FWHM ~ one
    # decade in pressure (i.e. 10–100 hPa). Tightened to land global-mean
    # column near 300 DU rather than the ~900 DU the broad e-fold yielded.
    log_p = np.log10(plev_hPa)
    log_p_peak = np.log10(30.0)
    sigma_logp = 0.40  # log10 units; FWHM ≈ 0.94 decades
    profile_p = np.exp(-((log_p - log_p_peak) / sigma_logp) ** 2)

    lat_rad = np.deg2rad(lat)
    lat_factor = 1.0 + 0.4 * np.sin(lat_rad) ** 2
    # Peak stratospheric ozone (~4.5 ppmv) chosen so a 0.40-decade Gaussian
    # integrates to a realistic ~300 DU global-mean column.
    o3_max_vmr = 4.5e-6

    # Base 3D field: (lat, plev)
    o3_clim = (o3_max_vmr * profile_p[None, :] * lat_factor[:, None]).astype(np.float64)

    # Antarctic ozone hole proxy: 60% reduction at SH polar lower stratosphere
    # (~50–200 hPa) in austral spring (Sep–Nov, months 8,9,10).  For
    # multi-year files the reduction strengthens linearly from
    # 0% in 1979 to 60% in 2014 so the non-cyclic dispatch produces a
    # visible secular trend.
    plev_hole_mask = (plev_hPa > 50.0) & (plev_hPa < 200.0)
    hole_lat_mask = lat < -55.0

    o3_3d = np.broadcast_to(o3_clim, (ntime, nlat, nlev)).copy()
    for t in range(ntime):
        m = t % 12
        if m in (8, 9, 10):
            if cyclic:
                hole_factor = 0.45  # 55% reduction
            else:
                year_index = t // 12
                year_frac = year_index / max(nyears - 1, 1)
                # ramp from 1.0 (no hole) to 0.4 (60% reduction) over the file span
                hole_factor = 1.0 - 0.6 * year_frac
            o3_3d[t][np.ix_(hole_lat_mask, plev_hole_mask)] *= hole_factor

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", ntime)
        ds.createDimension("lat", nlat)
        ds.createDimension("plev", nlev)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = mid_days
        t.units = ("days since 0000-01-01 00:00:00" if cyclic
                    else f"days since {start_year}-01-01 00:00:00")
        t.calendar = "noleap"
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = lat
        la.units = "degrees_north"
        pl = ds.createVariable("plev", "f8", ("plev",))
        pl[:] = plev
        pl.units = "Pa"
        pl.long_name = "Pressure level"
        v = ds.createVariable("vmro3", "f8", ("time", "lat", "plev"))
        v[:] = o3_3d
        v.units = "mol mol-1"
        v.long_name = ("Ozone volume mixing ratio (synthetic monthly clim)"
                        if cyclic else
                        "Ozone volume mixing ratio (synthetic interannual)")
        ds.title = ("Synthetic AMIP ozone monthly climatology" if cyclic
                     else "Synthetic AMIP ozone interannual time series")
        ds.source = "scripts/generate_amip_forcing.py"

    # Column ozone in Dobson Units. DU = molecules/cm² / 2.687e16.
    # Column molecules/m² = ∫ vmro3 * (n_air) dz, with n_air dz = -dp/(m_air g).
    # ⇒ N(O3) [molec/m²] = (N_A / m_air / g) * ∫ vmro3 * |dp|.
    N_A = constants.N_A
    m_air = constants.M_dry  # kg/mol
    g = constants.g
    # |dp| over levels from interface midpoints (assume cell-bottom = plev[i+1] etc.)
    dp = np.abs(np.diff(np.concatenate([[0.0], plev])))  # shape (nlev,)
    col_molec_m2 = (o3_3d * dp[None, None, :]).sum(axis=-1) * (N_A / m_air / g)
    col_du = col_molec_m2 / (2.687e16 * 1e4)  # 1e4 cm²/m²
    print(f"Wrote {out_path} ({ntime} months × {nlat} lat × {nlev} plev); "
          f"col O3 ~ {col_du.min():.1f}–{col_du.max():.1f} DU")


# ============================================================================
# Solar (TSI + 14-band spectral)
# ============================================================================

def make_solar(out_path: Path, start_year: int, end_year: int) -> None:
    """Daily solar TSI and 14-band spectral fractions.

    Schema follows MPI-M ``swflux_14band_cmip6_*.nc``:
      - ``time`` [days since 1850-01-01], daily resolution
      - ``TSI`` (time,) [W/m²] with 11-yr Schwabe cycle (~0.7 W/m² amplitude)
      - ``SSI_frac`` (time, numwl=14) — band fractions summing to 1
    """
    days_per_year = 365
    day0 = (start_year - 1850) * days_per_year
    day1 = (end_year - 1850 + 1) * days_per_year
    times = np.arange(day0, day1, dtype=np.float64)
    n = times.size

    # 11-year solar cycle, peak ~ 1361.5, trough ~ 1360.0
    years = times / days_per_year + 1850
    cycle = 0.6 * np.sin(2 * np.pi * (years - 1986.0) / 11.0)
    tsi = constants.S_0 + cycle  # W/m²

    # 14-band spectral fractions (RRTMG-SW order). Time-invariant baseline
    # plus ~0.5% UV variability across the cycle (UV varies ~10× more than VIS).
    # Approximate baseline fractions (sum to 1) from RRTMG-SW solar reference.
    base = np.array([
        0.001, 0.005, 0.012, 0.015, 0.030, 0.060, 0.115,
        0.205, 0.225, 0.180, 0.085, 0.040, 0.020, 0.007,
    ])
    base = base / base.sum()
    uv_idx = slice(0, 4)  # bands 1-4 are UV
    delta_uv = 0.05 * cycle  # +5% in solar max, -5% in min, in UV bands only
    spec = np.broadcast_to(base, (n, 14)).copy()
    for i in range(uv_idx.start, uv_idx.stop):
        spec[:, i] *= 1.0 + delta_uv
    # Renormalise to keep sum=1
    spec /= spec.sum(axis=1, keepdims=True)

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n)
        ds.createDimension("numwl", 14)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = times
        t.units = "days since 1850-01-01 00:00:00"
        t.calendar = "noleap"
        tsi_v = ds.createVariable("TSI", "f8", ("time",))
        tsi_v[:] = tsi
        tsi_v.units = "W m-2"
        ssi_v = ds.createVariable("SSI_frac", "f8", ("time", "numwl"))
        ssi_v[:] = spec
        ssi_v.units = "1"
        ssi_v.long_name = "Per-band fraction of TSI"
        ds.title = "Synthetic CMIP6 solar TSI + 14-band spectrum"
        ds.source = "scripts/generate_amip_forcing.py"

    print(f"Wrote {out_path} ({n} days; TSI {tsi.min():.2f}–{tsi.max():.2f} W/m²)")


# ============================================================================
# Aerosol (Kinne-style monthly zonal AOD)
# ============================================================================

def make_aerosol_clim(out_path: Path, *, nlat: int = 96) -> None:
    """Monthly zonal-mean AOD@550 nm climatology.

    Loader expects ``aod`` (time, lat); we write the simplified shape
    that ``_load_monthly_zonal_anchored`` consumes after the multi-dim collapse.
    """
    lat = np.linspace(-89.0, 89.0, nlat)
    months = np.arange(12)
    mid_days = np.array([15.5 + 30.4375 * m for m in months])

    lat_rad = np.deg2rad(lat)
    base = 0.05 + 0.20 * np.cos(lat_rad) ** 2  # tropical maximum
    # Annual cycle: NH summer dust+pollution peak, SH biomass burning peak
    nh_summer = np.maximum(np.sin(lat_rad), 0.0) * 0.05
    sh_burning = np.maximum(-np.sin(lat_rad), 0.0) * 0.04
    aod = np.empty((12, nlat))
    for m in months:
        nh_factor = np.cos(2 * np.pi * (m - 6) / 12.0)  # peak in July (m=6)
        sh_factor = np.cos(2 * np.pi * (m - 9) / 12.0)  # peak in October
        aod[m] = base + nh_factor * nh_summer + sh_factor * sh_burning
    aod = np.maximum(aod, 0.02)

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", nlat)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = mid_days
        t.units = "days since 0000-01-01 00:00:00"
        t.calendar = "noleap"
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = lat
        la.units = "degrees_north"
        v = ds.createVariable("aod", "f8", ("time", "lat"))
        v[:] = aod
        v.units = "1"
        v.long_name = "Aerosol optical depth at 550 nm"
        ds.title = "Synthetic AMIP aerosol climatology"
        ds.source = "scripts/generate_amip_forcing.py"

    print(f"Wrote {out_path} (12 months × {nlat} lat; AOD "
          f"{aod.min():.3f}–{aod.max():.3f})")


# ============================================================================
# Volcanic aerosol (column AOD time series)
# ============================================================================

def make_volcanic(out_path: Path, start_year: int, end_year: int, *,
                  nlat: int = 36) -> None:
    """Monthly zonal-mean volcanic stratospheric AOD time series.

    Includes the El Chichón (1982) and Pinatubo (1991) signals as
    Gaussian-in-time stratospheric AOD bumps with a tropics-bias profile.
    Saved with ``aod(time, lat)`` so the legacy ``_load_monthly_zonal_anchored``
    path consumes it directly.
    """
    nyears = end_year - start_year + 1
    n = nyears * 12
    times_days = monthly_midpoints_days(start_year, end_year)
    lat = np.linspace(-87.5, 87.5, nlat)
    lat_rad = np.deg2rad(lat)

    # Background stratospheric AOD ~ 0.005
    aod_bg = 0.005 * (1.0 + 0.5 * np.cos(lat_rad) ** 2)
    aod = np.broadcast_to(aod_bg, (n, nlat)).astype(np.float64).copy()

    # El Chichón (1982 April): peak 0.10, lifetime ~24 months
    el_chichon_t = (1982 - start_year) * 12 + 3 if start_year <= 1982 else None
    pinatubo_t = (1991 - start_year) * 12 + 5 if start_year <= 1991 else None

    for t_peak, peak_aod, hemi_lat0 in [
        (el_chichon_t, 0.10, 17.0),
        (pinatubo_t, 0.18, 15.0),
    ]:
        if t_peak is None or t_peak >= n:
            continue
        for t in range(n):
            dt = (t - t_peak)
            if -2 < dt < 36:  # 36-month lifetime
                temporal = peak_aod * np.exp(-((dt - 6) / 10.0) ** 2)
                # Spread from injection latitude
                spatial = np.exp(-((lat - hemi_lat0) / 30.0) ** 2)
                aod[t] += temporal * spatial

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n)
        ds.createDimension("lat", nlat)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = times_days
        t.units = f"days since {start_year}-01-01 00:00:00"
        t.calendar = "noleap"
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = lat
        la.units = "degrees_north"
        v = ds.createVariable("aod", "f8", ("time", "lat"))
        v[:] = aod
        v.units = "1"
        v.long_name = "Stratospheric volcanic AOD (550 nm)"
        ds.title = "Synthetic AMIP volcanic aerosol time series"
        ds.source = "scripts/generate_amip_forcing.py"

    print(f"Wrote {out_path} ({n} months × {nlat} lat; AOD max {aod.max():.3f})")


# ============================================================================
# Driver
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/forcing_amip"),
                        help="Output directory for forcing files")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--nlat-sst", type=int, default=73,
                        help="Latitude grid size for SST/SIC")
    parser.add_argument("--nlon-sst", type=int, default=144,
                        help="Longitude grid size for SST/SIC")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "sst", "ghg", "ozone", "solar",
                                 "aerosol", "volcanic"],
                        help="Generate a single component (debug)")
    parser.add_argument("--ozone-interannual", action="store_true", default=False,
                        help="Generate the ozone file as interannually-varying "
                             "(ntime = nyears*12) so the loader uses the "
                             "non-cyclic dispatch.  Default: 12-month climatology.")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    sy, ey = args.start_year, args.end_year
    if args.component in ("all", "sst"):
        make_sst_sic(args.out / f"sst_sic_amip_{sy}-{ey}.nc",
                     sy, ey, nlat=args.nlat_sst, nlon=args.nlon_sst)
    if args.component in ("all", "ghg"):
        make_ghg_annual(args.out / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
    if args.component in ("all", "ozone"):
        if args.ozone_interannual:
            make_ozone_clim(
                args.out / f"ozone_amip_{sy}-{ey}.nc",
                start_year=sy, end_year=ey,
            )
        else:
            make_ozone_clim(args.out / "ozone_amip_clim.nc")
    if args.component in ("all", "solar"):
        make_solar(args.out / f"solar_amip_{sy}-{ey}.nc", sy, ey)
    if args.component in ("all", "aerosol"):
        make_aerosol_clim(args.out / "aerosol_amip_clim.nc")
    if args.component in ("all", "volcanic"):
        make_volcanic(args.out / f"volcanic_amip_{sy}-{ey}.nc", sy, ey)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
