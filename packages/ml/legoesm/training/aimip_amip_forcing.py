"""Exact AIMIP-1 prescribed SST / sea-ice forcing loader.

Loads the official AIMIP Phase-1 forcing dataset
``ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc`` (Zenodo 10.5281/zenodo.17065758;
DOI record 17065758) and exposes it on the model's Gaussian grid with the
protocol's temporal convention: monthly-mean values centered at the start of
each month, **linearly interpolated** to model time (NOT the CMIP input4MIPs
mid-month algorithm, per the AIMIP spec).

Variables in the file: ``sea_surface_temperature`` [K], ``sea_ice_cover``
[fraction 0-1], ``land_sea_mask`` [land fraction]; grid 721x1440 (0.25 deg),
time 1978-10-01 .. 2025-01-01 (556 monthly steps).

Used by BOTH the AMIP inference driver and the stability fine-tune so the model
is trained and evaluated under identical, protocol-exact forcing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from legoesm.training.era5_to_state import regrid_2d_to_gaussian

# Default location the downloader (scripts) writes to; overridable.
DEFAULT_AIMIP_FORCING = (
    "results/aimip_forcing/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
)
_SST_VAR = "sea_surface_temperature"
_SIC_VAR = "sea_ice_cover"
_LSM_VAR = "land_sea_mask"


def _regrid_field_2d(field_2d, era5_lat_deg, era5_lon_deg, grid):
    """Regrid a (lat, lon) ERA5 field to the Gaussian grid (n_lat, n_lon).

    ``regrid_2d_to_gaussian`` needs ascending latitude in RADIANS; ERA5 is
    (degrees, N->S) so convert + flip. NaN (SST over land) is filled with the
    NEAREST finite ocean value first so the linear interpolation stays finite AND
    uncontaminated at coastal ocean cells (true land is masked out downstream by
    land_sea_mask).
    """
    f = np.asarray(field_2d, dtype=np.float64)
    lat = np.asarray(era5_lat_deg, dtype=np.float64)
    lon = np.asarray(era5_lon_deg, dtype=np.float64)
    if np.any(~np.isfinite(f)):
        # Nearest finite ocean SST (codex review): a global-mean fill would bleed
        # into coastal ocean target cells through the linear regrid.
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(
            ~np.isfinite(f), return_distances=False, return_indices=True,
        )
        f = f[tuple(idx)]
    if lat[0] > lat[-1]:  # RegularGridInterpolator requires ascending axes
        lat = lat[::-1]
        f = f[::-1, :]
    return np.asarray(regrid_2d_to_gaussian(f, np.deg2rad(lat), np.deg2rad(lon), grid))


def regrid_monthly_forcing_to_gaussian(forcing_path, grid, cache_path=None):
    """Regrid every monthly SST / sea-ice / land-mask field to the Gaussian grid.

    Returns ``(times_ns, sst, sic, land)`` where ``times_ns`` is an int64 array
    of nanosecond timestamps (556,), ``sst``/``sic`` are (556, ncol), and
    ``land`` is the static land fraction (ncol,). Result is cached to
    ``cache_path`` (``.npz``) because regridding 556 0.25-deg fields is slow and
    both the run and the fine-tune reuse it.
    """
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path)
        return d["times_ns"], d["sst"], d["sic"], d["land"]

    import xarray as xr
    ds = xr.open_dataset(forcing_path)
    lat_deg = ds.latitude.values
    lon_deg = ds.longitude.values
    times_ns = ds.time.values.astype("datetime64[ns]").astype(np.int64)
    n_t = ds.sizes["time"]
    ncol = len(grid.lat) * len(grid.lon)

    sst = np.empty((n_t, ncol), dtype=np.float64)
    sic = np.empty((n_t, ncol), dtype=np.float64)
    sst_da = ds[_SST_VAR].values
    sic_da = ds[_SIC_VAR].values
    for i in range(n_t):
        sst[i] = _regrid_field_2d(sst_da[i], lat_deg, lon_deg, grid).reshape(-1)
        sic[i] = np.clip(
            _regrid_field_2d(sic_da[i], lat_deg, lon_deg, grid).reshape(-1), 0.0, 1.0
        )
    land = _regrid_field_2d(ds[_LSM_VAR].values, lat_deg, lon_deg, grid).reshape(-1)
    ds.close()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, times_ns=times_ns, sst=sst, sic=sic, land=land)
    return times_ns, sst, sic, land


def interp_forcing_at(times_ns, field, target_ns):
    """Linear-interpolate a monthly Gaussian field (n_t, ncol) to ``target_ns``.

    ``target_ns`` is a scalar int64 ns timestamp. Values outside the monthly
    range clamp to the endpoints (np.interp semantics). Returns (ncol,).
    """
    t = np.asarray(times_ns, dtype=np.float64)
    tt = float(target_ns)
    if tt <= t[0]:
        return np.asarray(field[0], dtype=np.float64)
    if tt >= t[-1]:
        return np.asarray(field[-1], dtype=np.float64)
    j = int(np.searchsorted(t, tt))  # t[j-1] < tt <= t[j]
    w = (tt - t[j - 1]) / (t[j] - t[j - 1])
    return (1.0 - w) * np.asarray(field[j - 1]) + w * np.asarray(field[j])


def ghg_vmr_at_year(year, experiment: str = "amip") -> dict:
    """Well-mixed GHG volume mixing ratios for a calendar year.

    RRTMGP is a PHYSICAL radiation scheme: unlike a pure-ML emulator it cannot
    learn the anthropogenic radiative-forcing trend from data — it must be given
    the actual time-varying concentrations, or a multi-decade run produces no
    warming. The CMIP6 AMIP protocol mandates the same transient CO2/CH4/N2O as
    the historical run, so we read the built-in ``experiments`` GHG table
    (co2/ch4/n2o, 1900-2021) and convert to RRTMGP VMRs. CFC-11/12 (not in the
    table) use the ``GHGConfig`` defaults (roughly the post-Montreal plateau).

    Returns a dict ``{co2, ch4, n2o, cfc11, cfc12}`` of dimensionless VMRs, ready
    for ``forcing['ghg_vmr']`` / ``ghg_vmr_override``.
    """
    from legoesm.forcing.experiments import ghg_at_year
    from legoesm.forcing.external import GHGConfig, ghg_concentrations_to_vmr

    co2_ppmv, ch4_ppbv, n2o_ppbv = ghg_at_year(experiment, float(year))
    _c = GHGConfig()  # CFC defaults (table carries only co2/ch4/n2o)
    return ghg_concentrations_to_vmr({
        "co2_ppmv": float(co2_ppmv),
        "ch4_ppbv": float(ch4_ppbv),
        "n2o_ppbv": float(n2o_ppbv),
        "cfc11_pptv": _c.cfc11_pptv,
        "cfc12_pptv": _c.cfc12_pptv,
    })


# --- historical ozone (ERA5 ARCO) -------------------------------------------
# WB2 has no ozone; ARCO-ERA5 (full 37-level store) carries the transient
# historical ozone (stratospheric depletion + recovery) RRTMGP needs.
_ARCO_ERA5_STORE = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
_ARCO_O3_VARS = ("ozone_mass_mixing_ratio", "o3")


def open_arco_era5():
    """Open the public ARCO-ERA5 store (anon GCS) for historical ozone."""
    import xarray as xr
    return xr.open_zarr(
        _ARCO_ERA5_STORE, chunks=None, storage_options={"token": "anon"},
    )


def ozone_vmr_at_date(ds_arco, date, grid, sigma_full, p_s_col):
    """Historical ERA5 ozone (ARCO) at ``date`` -> O3 VMR on the model sigma grid.

    Loads ozone mass mixing ratio (plev, lat, lon) nearest ``date``, regrids each
    pressure level to the Gaussian grid, interpolates plev->sigma with the model
    surface pressure ``p_s_col`` (ncol,), and converts mass->volume mixing ratio
    (x M_dry/M_o3). Returns (ncol, nlev) for ``forcing['o3_vmr']`` — the
    transient historical ozone a physical RRTMGP run needs.
    """
    import jax.numpy as jnp
    import pandas as pd

    from legoesm import constants
    from legoesm.training.vertical_interp import interp_pressure_to_sigma

    var = next((v for v in _ARCO_O3_VARS if v in ds_arco), None)
    if var is None:
        raise KeyError(f"no ozone var {_ARCO_O3_VARS} in ARCO store")
    times = pd.DatetimeIndex(ds_arco.time.values)
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day)
    if target < times[0] or target > times[-1]:
        raise ValueError(f"{date} outside ARCO ozone range")
    da = ds_arco[var].sel(time=target, method="nearest")  # (level, lat, lon)
    o3 = np.asarray(da.values, dtype=np.float64)
    lev_hpa = np.asarray(da.level.values, dtype=np.float64)
    lat = np.asarray(ds_arco.latitude.values, dtype=np.float64)
    lon = np.asarray(ds_arco.longitude.values, dtype=np.float64)
    # Regrid each pressure level to the Gaussian grid -> (ncol, n_plev).
    o3_g = np.stack(
        [_regrid_field_2d(o3[k], lat, lon, grid).reshape(-1) for k in range(o3.shape[0])],
        axis=-1,
    )
    plev_pa = lev_hpa * 100.0
    order = np.argsort(plev_pa)  # interp expects ascending pressure
    plev_pa, o3_g = plev_pa[order], o3_g[:, order]
    o3_sigma = np.asarray(interp_pressure_to_sigma(
        jnp.asarray(o3_g), jnp.asarray(plev_pa),
        jnp.asarray(np.asarray(p_s_col).reshape(-1)), jnp.asarray(sigma_full),
    ))
    # mass mixing ratio [kg/kg] -> volume mixing ratio [mol/mol].
    return (o3_sigma * (float(constants.M_dry) / float(constants.M_o3))).astype(np.float64)
