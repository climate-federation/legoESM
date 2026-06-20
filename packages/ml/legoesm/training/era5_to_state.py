"""Convert ERA5 / WeatherBench2 data to legoESM model state.

Handles the full pipeline:
1. Load ERA5 from GCS or local Zarr cache
2. Horizontal regridding (lat-lon → spectral Gaussian or cubed-sphere)
3. Vertical interpolation (pressure levels → sigma/hybrid)
4. State assembly into SegmentCarry for the compiled dycore

Also provides a local Zarr cache to avoid repeated GCS downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm.ml.data.era5_loader import (
    ERA5Config,
    WB2_ERA5_ZARR,
    create_era5_dataset,
)
from legoesm.ml.channel_packing import WB2_PRESSURE_LEVELS
def resolve_var(ds, name):
    """Find a variable in the dataset, trying common aliases."""
    aliases = {
        'temperature': 't', 'u_component_of_wind': 'u',
        'v_component_of_wind': 'v', 'specific_humidity': 'q',
        'surface_pressure': 'sp', 'skin_temperature': 'skt',
        'geopotential': 'z',
        'geopotential_at_surface': 'z_sfc',
    }
    if name in ds:
        return name
    if name in aliases and aliases[name] in ds:
        return aliases[name]
    for k, v in aliases.items():
        if name == k and v in ds:
            return v
    return None


from legoesm.training.vertical_interp import interp_pressure_to_sigma

logger = logging.getLogger(__name__)

# Public ARCO-ERA5 store (Analysis-Ready Cloud-Optimized ERA5 on GCS,
# anon-readable).  Source of the radiation-flux TARGETS: the default WB2
# state store's ``mean_*_radiation_flux`` variables are NaN at every
# analysis time (probe 8533800 — 0/20 sampled times populated), but
# ARCO-ERA5 carries the same ERA5 fields with clean W/m² mean-rate fluxes
# (probe 8533818).  Same 0.25° 1440×721 grid as the WB2 state store.
ARCO_ERA5_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

# Module-level cache for regridding weights (expensive to recompute)
_CS_WEIGHT_CACHE: dict[tuple, object] = {}


def _get_cs_weights(n_lon_era5: int, grid):
    """Get or compute cached cubed-sphere regridding weights."""
    key = (n_lon_era5, id(grid))
    if key not in _CS_WEIGHT_CACHE:
        from legoesm.grids.regridding import compute_gauss_to_cs_weights
        from legoesm.grids.gaussian import create_gaussian_grid
        gauss_proxy = create_gaussian_grid(
            n_max=n_lon_era5 // 2 - 1, dealiasing="linear",
        )
        _CS_WEIGHT_CACHE[key] = compute_gauss_to_cs_weights(gauss_proxy, grid)
    return _CS_WEIGHT_CACHE[key]


def open_era5_zarr(zarr_path: str):
    """Open an ERA5 Zarr store with dimension normalization.

    Public wrapper around the ERA5 loader's internal helpers.
    """
    import xarray as xr
    if zarr_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(store, chunks=None)
    else:
        ds = xr.open_zarr(zarr_path, chunks=None)
    # Normalize dimension names
    rename = {}
    for short, long in [("lat", "latitude"), ("lon", "longitude")]:
        if long in ds.dims and short not in ds.dims:
            rename[long] = short
    if rename:
        ds = ds.rename(rename)
    return ds


# Extended config with surface variables needed for dycore IC + forcing
class TrainingERA5Config(NamedTuple):
    """ERA5 config extended with surface variables for dycore training."""
    zarr_store: str = WB2_ERA5_ZARR
    pressure_variables: tuple = (
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    )
    surface_variables: tuple = (
        "surface_pressure",
        "skin_temperature",           # SST proxy
        "geopotential_at_surface",    # surface geopotential for phis [m2/s2]
    )
    levels: tuple = WB2_PRESSURE_LEVELS
    time_range: tuple = ("1979-01-01", "2020-12-31")
    dt_hours: int = 6
    local_cache_dir: str = ""     # empty = no cache
    # Radiation-flux targets for AIMIP TOA + surface flux supervision.
    # When True, ``load_era5_slice`` also loads ERA5 TOA/surface radiation
    # and derives the four model-comparable fluxes (rsut, OLR, surface net
    # SW, surface net LW) so the target carry's ``held_*`` fields hold real
    # observations instead of zeros.  Default False → byte-identical legacy.
    load_radiation_fluxes: bool = False
    # Radiation-flux TARGETS come from a SEPARATE store: the WB2 state
    # store's flux vars are all-NaN, so fluxes are read from ARCO-ERA5
    # (clean ``mean_*_radiation_flux`` in W/m², same 0.25° grid).  Set to
    # "" to read fluxes from the state ``zarr_store`` instead (only valid
    # if that store actually populates them).
    flux_zarr: str = ARCO_ERA5_ZARR
    # ARCO ``mean_*_radiation_flux`` are W/m² mean rates → divide by 1.0
    # (no-op).  A store accumulating J/m² over the hour would need 3600.0.
    flux_accum_seconds: float = 1.0


# ---------------------------------------------------------------------------
# Local Zarr cache
# ---------------------------------------------------------------------------

def ensure_local_cache(
    config: TrainingERA5Config,
    cache_dir: str | Path,
    years: tuple[int, int] = (2015, 2020),
) -> Path:
    """Download a subset of ERA5 to a local Zarr store.

    Caches the specified year range with all configured variables
    and levels.  Subsequent calls skip download if the store exists.

    Parameters
    ----------
    config : TrainingERA5Config
    cache_dir : Path
        Directory for the local cache.
    years : tuple
        (start_year, end_year) to cache.

    Returns
    -------
    Path to the local Zarr store.
    """

    cache_path = Path(cache_dir) / "era5_training_cache.zarr"
    if cache_path.exists():
        logger.info(f"Using cached ERA5 at {cache_path}")
        return cache_path

    logger.info(f"Downloading ERA5 {years[0]}-{years[1]} to {cache_path}...")

    # Open remote
    era5_cfg = ERA5Config(
        zarr_store=config.zarr_store,
        variables=config.pressure_variables,
        levels=config.levels,
        time_range=(f"{years[0]}-01-01", f"{years[1]}-12-31"),
        dt_hours=config.dt_hours,
    )
    ds = create_era5_dataset(era5_cfg)

    # Also grab surface variables
    ds_full = open_era5_zarr(config.zarr_store)
    ds_full = ds_full.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))

    for svar in config.surface_variables:
        resolved = [v for v in [resolve_var(ds_full, svar)] if v]
        for r in resolved:
            if r in ds_full and r not in ds:
                ds[r] = ds_full[r]

    # Write to local Zarr
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(cache_path), mode="w")
    logger.info(f"Cached ERA5 to {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# ERA5 → model state conversion
# ---------------------------------------------------------------------------

class ERA5Slice(NamedTuple):
    """A single ERA5 time slice with all needed fields."""
    T: np.ndarray          # (n_lat, n_lon, n_plev) temperature [K]
    u: np.ndarray          # (n_lat, n_lon, n_plev) zonal wind [m/s]
    v: np.ndarray          # (n_lat, n_lon, n_plev) meridional wind [m/s]
    q: np.ndarray          # (n_lat, n_lon, n_plev) specific humidity [kg/kg]
    p_s: np.ndarray        # (n_lat, n_lon) surface pressure [Pa]
    sst: np.ndarray        # (n_lat, n_lon) skin/SST temperature [K]
    phis: np.ndarray       # (n_lat, n_lon) surface geopotential [m²/s²]
    lat: np.ndarray        # (n_lat,) latitude [rad]
    lon: np.ndarray        # (n_lon,) longitude [rad]
    plev_Pa: np.ndarray    # (n_plev,) pressure levels [Pa], ascending
    # Optional radiation-flux targets [W/m²] (None unless
    # ``config.load_radiation_fluxes``).  Conventions match the model's
    # ``SegmentCarry.held_*`` fields:
    #   rsut       = TOA outgoing (reflected) SW, positive up
    #   olr        = TOA outgoing LW (OLR/rlut), positive up
    #   sfc_net_sw = surface net SW (down − up), positive down
    #   sfc_net_lw = surface net LW (down − up), positive down (usually <0)
    rsut: np.ndarray = None        # (n_lat, n_lon)
    olr: np.ndarray = None         # (n_lat, n_lon)
    sfc_net_sw: np.ndarray = None  # (n_lat, n_lon)
    sfc_net_lw: np.ndarray = None  # (n_lat, n_lon)


def load_era5_slice(
    config: TrainingERA5Config, time_idx: int, ds=None, flux_ds=None
) -> ERA5Slice:
    """Load a single ERA5 time slice with all fields needed for IC + forcing.

    Parameters
    ----------
    config : TrainingERA5Config
    time_idx : int
        Time index into the dataset.
    ds : xarray.Dataset, optional
        Pre-opened ERA5 store.  Pass it when looping over many snapshots
        (the AIMIP window loader) so the GCS zarr is opened once instead
        of per slice.  ``None`` opens the store from ``config`` (legacy).

    Returns
    -------
    ERA5Slice with all fields on the native ERA5 lat-lon grid.
    """
    store = config.local_cache_dir if config.local_cache_dir else config.zarr_store
    if ds is None:
        ds = open_era5_zarr(store)

    # Select time
    ds_t = ds.isel(time=time_idx)

    # Extract lat/lon
    lat = np.deg2rad(ds_t.lat.values.astype(np.float64))
    lon = np.deg2rad(ds_t.lon.values.astype(np.float64))

    # Pressure levels (hPa → Pa, ensure ascending)
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    plev_hPa = np.array(config.levels, dtype=np.float64)
    plev_Pa = np.sort(plev_hPa * 100.0)  # ascending in Pa

    def _get_3d(name):
        """Extract a 3D variable as (lat, lon, level) with levels ascending in pressure."""
        resolved = [v for v in [resolve_var(ds_t, name)] if v]
        if not resolved:
            return np.zeros((len(lat), len(lon), len(plev_Pa)))
        data = ds_t[resolved[0]].sel({level_dim: list(config.levels)}).values
        if data.ndim == 3:
            dims = list(ds_t[resolved[0]].dims)
            spatial = {"lat", "lon", "latitude", "longitude"}
            level_axis = next((i for i, d in enumerate(dims) if d not in spatial), 0)
            if level_axis != 2:
                data = np.moveaxis(data, level_axis, -1)
        # Ensure levels are ascending in pressure
        if plev_hPa[0] > plev_hPa[-1]:
            data = data[..., ::-1]
        return data.astype(np.float32)

    def _get_2d(name):
        """Extract a 2D surface variable as (lat, lon)."""
        # Try time-selected dataset first, then full dataset for static fields
        resolved = resolve_var(ds_t, name)
        if resolved is None:
            resolved = resolve_var(ds, name)
            if resolved is None:
                return np.zeros((len(lat), len(lon)), dtype=np.float32)
            data = ds[resolved].values
        else:
            data = ds_t[resolved].values
        # Drop singleton dimensions and take first slice of any extra dims
        data = data.squeeze()
        while data.ndim > 2:
            data = data[0]
        return data.astype(np.float32)

    # Optional radiation-flux targets (TOA + surface) for AIMIP flux
    # supervision.  The WB2 store provides ``mean_*_radiation_flux`` vars
    # in W/m² (probe-verified).  Derive the four model-comparable fluxes:
    #   rsut       = top_downward_SW − top_net_SW   (reflected up, +up)
    #   OLR        = −top_net_LW                     (TOA net LW = −OLR)
    #   sfc_net_sw = surface_net_SW                  (down − up, +down)
    #   sfc_net_lw = surface_net_LW                  (down − up, +down)
    # ``flux_accum_seconds`` (default 1.0) converts an accumulated-J/m²
    # store to W/m²; it is a no-op for the W/m² WB2 store.
    rsut = olr = sfc_net_sw = sfc_net_lw = None
    if config.load_radiation_fluxes:
        fzarr = config.flux_zarr or store
        if flux_ds is None:
            flux_ds = open_era5_zarr(fzarr) if config.flux_zarr else ds
        # Select the flux-store snapshot at the SAME timestamp as the state
        # slice (ARCO is hourly; the WB2 6h analysis times are a subset,
        # matched exactly by datetime).
        fds_t = flux_ds.sel(time=ds_t.time.values, method="nearest")
        # Align the flux-store lat ordering to the state grid: the fluxes
        # are regridded later with era5.lat/era5.lon, so they must share
        # that ordering.  Same 0.25° ERA5 grid + same 0..360 lon origin, so
        # only the lat sense can differ (ARCO is N->S, WB2 may be S->N).
        flux_lat_deg = np.asarray(flux_ds.lat.values, dtype=np.float64)
        state_lat_deg = np.rad2deg(lat)
        flip_lat = (np.sign(flux_lat_deg[1] - flux_lat_deg[0])
                    != np.sign(state_lat_deg[1] - state_lat_deg[0]))

        def _flux_2d(name):
            r = resolve_var(fds_t, name)
            src = fds_t
            if r is None:
                r = resolve_var(flux_ds, name)
                src = flux_ds
            if r is None:
                raise ValueError(
                    f"load_radiation_fluxes=True but flux variable {name!r} "
                    f"is absent from {fzarr}.  Run "
                    f"scripts/tmp/_probe_arco_era5.py to list available "
                    f"radiation variables."
                )
            d = np.asarray(src[r].values).squeeze()
            while d.ndim > 2:
                d = d[0]
            if d.shape != (len(lat), len(lon)):
                raise ValueError(
                    f"flux field {name!r} grid {d.shape} != state grid "
                    f"{(len(lat), len(lon))}; flux_zarr must match the state "
                    f"store resolution (both 0.25° ERA5)."
                )
            if flip_lat:
                d = d[::-1]
            return d.astype(np.float32)

        acc = np.float32(config.flux_accum_seconds)
        toa_dn_sw = _flux_2d("mean_top_downward_short_wave_radiation_flux")
        toa_net_sw = _flux_2d("mean_top_net_short_wave_radiation_flux")
        toa_net_lw = _flux_2d("mean_top_net_long_wave_radiation_flux")
        sfc_net_sw_v = _flux_2d("mean_surface_net_short_wave_radiation_flux")
        sfc_net_lw_v = _flux_2d("mean_surface_net_long_wave_radiation_flux")
        rsut = (toa_dn_sw - toa_net_sw) / acc
        olr = (-toa_net_lw) / acc
        sfc_net_sw = sfc_net_sw_v / acc
        sfc_net_lw = sfc_net_lw_v / acc

    return ERA5Slice(
        T=_get_3d("temperature"),
        u=_get_3d("u_component_of_wind"),
        v=_get_3d("v_component_of_wind"),
        q=_get_3d("specific_humidity"),
        p_s=_get_2d("surface_pressure"),
        sst=_get_2d("skin_temperature"),
        phis=_get_2d("geopotential_at_surface"),  # already in m²/s²
        lat=lat,
        lon=lon,
        plev_Pa=plev_Pa,
        rsut=rsut,
        olr=olr,
        sfc_net_sw=sfc_net_sw,
        sfc_net_lw=sfc_net_lw,
    )


def _era5_held_fluxes(era5: ERA5Slice, regrid_2d_fn, shape_2d):
    """Regrid ERA5 radiation-flux targets onto the model grid for the carry.

    Returns ``(held_sw_up_toa, held_lw_up_toa, held_sw_net_sfc,
    held_lw_net_sfc)`` — i.e. (rsut, OLR, surface net SW, surface net LW) —
    each mapped to the model 2D layout by ``regrid_2d_fn`` (a grid-specific
    callback: Gaussian/lat-lon interpolation or cubed-sphere ``regrid_scalar``).
    When the slice carries no fluxes (``load_radiation_fluxes=False``) returns
    four ``jnp.zeros(shape_2d)`` — byte-identical to the legacy zero-fill.

    ``held_sw_down_toa`` (rsdt) is intentionally NOT set from ERA5: it is
    prescribed insolation that the dycore computes each step, and the flux
    loss does not penalize it.  One shared implementation so the spectral /
    lat-lon / cubed-sphere builders never re-derive flux-target packing.
    """
    if era5.rsut is None:
        z = jnp.zeros(shape_2d)
        return z, z, z, z
    return (
        jnp.asarray(regrid_2d_fn(era5.rsut)),
        jnp.asarray(regrid_2d_fn(era5.olr)),
        jnp.asarray(regrid_2d_fn(era5.sfc_net_sw)),
        jnp.asarray(regrid_2d_fn(era5.sfc_net_lw)),
    )


def era5_to_spectral_carry(
    era5: ERA5Slice,
    grid,
    sigma,
):
    """Convert ERA5 slice to SegmentCarry on a spectral (Gaussian) grid.

    The ERA5 lat-lon data is directly analyzed into spherical harmonics
    (since the Gaussian grid is also lat-lon), then vertically
    interpolated to model sigma levels.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : GaussianGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    SegmentCarry
    """
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    sigma_full = np.asarray(sigma.sigma_full)

    # ERA5 is on 1440x721 lat-lon; regrid to Gaussian grid via simple
    # nearest-neighbor or linear interpolation in lat-lon space
    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(
        era5, grid,
    )

    # Vertical interpolation: pressure levels → sigma levels
    p_s_jax = jnp.asarray(p_s_ll)
    plev = jnp.asarray(era5.plev_Pa)
    sigma_f = jnp.asarray(sigma_full)

    T_model = interp_pressure_to_sigma(jnp.asarray(T_ll), plev, p_s_jax, sigma_f)
    u_model = interp_pressure_to_sigma(jnp.asarray(u_ll), plev, p_s_jax, sigma_f)
    v_model = interp_pressure_to_sigma(jnp.asarray(v_ll), plev, p_s_jax, sigma_f)
    # ERA5 q is SPECIFIC HUMIDITY (mass vapor / mass moist air).  The
    # legoesm physics path treats q_v as MASS MIXING RATIO (mass vapor
    # / mass dry air) — saturation_mixing_ratio in thermo.py returns
    # the mixing-ratio convention, and atmosphere/physics modules
    # consume q_v under that convention.  Convert at the ERA5
    # boundary: r = q / (1 − q).  In the tropical PBL (q ≈ 0.025) the
    # bias from skipping this conversion is ~3% of q.  Clip to avoid
    # division blow-up at q = 1.
    q_specific = jnp.maximum(
        interp_pressure_to_sigma(jnp.asarray(q_ll), plev, p_s_jax, sigma_f),
        0.0,
    )
    q_specific = jnp.clip(q_specific, 0.0, 0.99)
    q_model = q_specific / (1.0 - q_specific)

    # Surface geopotential (regrid to Gaussian)
    phis_ll = regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid)
    phis_jax = jnp.asarray(phis_ll)

    # Build HydrostaticState
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_jax, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_jax, name="phis", dims=dims_2d, units="m2/s2"),
    )

    # Pack into SegmentCarry; held radiation fluxes hold ERA5 targets
    # when loaded (else zeros — legacy behaviour).
    shape_3d = T_model.shape
    shape_2d = p_s_jax.shape

    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_2d_to_gaussian(f, era5.lat, era5.lon, grid), shape_2d,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


def era5_to_cubedsphere_carry(
    era5: ERA5Slice,
    grid,
    sigma,
):
    """Convert ERA5 slice to SegmentCarry on a cubed-sphere grid.

    Uses KD-tree regridding from lat-lon to cubed-sphere, then
    vertical interpolation to model sigma levels.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : CubedSphereGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    SegmentCarry
    """
    from legoesm.grids.regridding import regrid_scalar
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    sigma_full = np.asarray(sigma.sigma_full)
    n_lon_era5 = era5.T.shape[1]
    weights = _get_cs_weights(n_lon_era5, grid)

    # Regrid 3D fields
    def _regrid_3d(field_ll):
        """Regrid (n_lat, n_lon, n_plev) → (6, n, n, n_plev)."""
        # Flatten spatial dims for regrid_scalar
        flat = field_ll.reshape(-1, field_ll.shape[-1])
        return regrid_scalar(jnp.asarray(flat), weights)

    T_cs = _regrid_3d(era5.T)
    u_cs = _regrid_3d(era5.u)
    v_cs = _regrid_3d(era5.v)
    q_cs = _regrid_3d(era5.q)

    # Regrid 2D fields
    p_s_cs = regrid_scalar(jnp.asarray(era5.p_s.ravel()), weights)
    phis_cs = regrid_scalar(jnp.asarray(era5.phis.ravel()), weights)

    # Vertical interpolation
    plev = jnp.asarray(era5.plev_Pa)
    sigma_f = jnp.asarray(sigma_full)
    T_model = interp_pressure_to_sigma(T_cs, plev, p_s_cs, sigma_f)
    u_model = interp_pressure_to_sigma(u_cs, plev, p_s_cs, sigma_f)
    v_model = interp_pressure_to_sigma(v_cs, plev, p_s_cs, sigma_f)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING
    # RATIO (see comment at the lat-lon path above).  Convert
    # r = q / (1 − q) at the boundary.
    q_specific = jnp.maximum(
        interp_pressure_to_sigma(q_cs, plev, p_s_cs, sigma_f), 0.0,
    )
    q_specific = jnp.clip(q_specific, 0.0, 0.99)
    q_model = q_specific / (1.0 - q_specific)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_cs, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_cs, name="phis", dims=dims_2d, units="m2/s2"),
    )

    shape_3d = T_model.shape
    shape_2d = p_s_cs.shape

    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_scalar(jnp.asarray(f.ravel()), weights), shape_2d,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


def era5_to_latlon_carry(
    era5: ERA5Slice,
    grid,
    sigma,
):
    """Convert ERA5 slice to a SegmentCarry on the lat-lon C-grid.

    The driver-level lat-lon state stores u, v, T at CELL CENTRES
    ``(n_lat, n_lon, nlev)`` — the Arakawa C-grid face staggering
    (u at lon interfaces, v at lat interfaces) is internal to the
    ``CGridLatLonPrimitiveEquationModel`` step, which re-staggers the
    cell-centred winds on the first integration.  So the IC carry is
    built exactly like the spectral path: regrid the ERA5 lat-lon
    fields onto the model lat-lon grid (a plain 2-D interpolation,
    since both are lat-lon), vertically interpolate to sigma, and
    convert ERA5 specific humidity to the model's mixing-ratio
    convention.  No face interpolation is applied here — doing so
    would double-stagger against the dycore's own re-staggering.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : LatLonGrid  (exposes ``.lat`` / ``.lon`` in radians)
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    SegmentCarry
    """
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    sigma_full = np.asarray(sigma.sigma_full)

    # ERA5 lat-lon → model lat-lon grid (reuses the generic
    # grid.lat/grid.lon interpolator shared with the Gaussian path).
    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(era5, grid)

    p_s_jax = jnp.asarray(p_s_ll)
    plev = jnp.asarray(era5.plev_Pa)
    sigma_f = jnp.asarray(sigma_full)

    T_model = interp_pressure_to_sigma(jnp.asarray(T_ll), plev, p_s_jax, sigma_f)
    u_model = interp_pressure_to_sigma(jnp.asarray(u_ll), plev, p_s_jax, sigma_f)
    v_model = interp_pressure_to_sigma(jnp.asarray(v_ll), plev, p_s_jax, sigma_f)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING
    # RATIO r = q / (1 − q) (see the spectral path for the rationale).
    q_specific = jnp.clip(
        jnp.maximum(
            interp_pressure_to_sigma(jnp.asarray(q_ll), plev, p_s_jax, sigma_f),
            0.0,
        ),
        0.0, 0.99,
    )
    q_model = q_specific / (1.0 - q_specific)

    phis_ll = regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid)
    phis_jax = jnp.asarray(phis_ll)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_jax, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_jax, name="phis", dims=dims_2d, units="m2/s2"),
    )

    shape_3d = T_model.shape
    shape_2d = p_s_jax.shape
    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_2d_to_gaussian(f, era5.lat, era5.lon, grid), shape_2d,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def regrid_latlon_to_gaussian(era5: ERA5Slice, grid):
    """Regrid ERA5 lat-lon fields to the model's Gaussian grid.

    Uses scipy linear interpolation for simplicity.  This runs at
    init time (outside JIT), so numpy/scipy are fine.

    Returns T, u, v, q (n_lat, n_lon, n_plev) and p_s (n_lat, n_lon)
    on the Gaussian grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    era5_lat = np.asarray(era5.lat)  # radians
    era5_lon = np.asarray(era5.lon)  # radians
    gauss_lat = np.asarray(grid.lat)  # radians
    gauss_lon = np.asarray(grid.lon)  # radians

    # Build target mesh
    lat_g, lon_g = np.meshgrid(gauss_lat, gauss_lon, indexing='ij')

    def _interp_3d(field):
        """Interpolate (n_lat, n_lon, n_plev) to Gaussian grid."""
        n_plev = field.shape[-1]
        result = np.zeros((len(gauss_lat), len(gauss_lon), n_plev), dtype=np.float32)
        for k in range(n_plev):
            interp = RegularGridInterpolator(
                (era5_lat, era5_lon), field[:, :, k],
                method='linear', bounds_error=False, fill_value=None,
            )
            result[:, :, k] = interp((lat_g, lon_g))
        return result

    def _interp_2d(field):
        interp = RegularGridInterpolator(
            (era5_lat, era5_lon), field,
            method='linear', bounds_error=False, fill_value=None,
        )
        return interp((lat_g, lon_g)).astype(np.float32)

    return (
        _interp_3d(era5.T),
        _interp_3d(era5.u),
        _interp_3d(era5.v),
        _interp_3d(era5.q),
        _interp_2d(era5.p_s),
    )


def regrid_2d_to_gaussian(field_2d, era5_lat, era5_lon, grid):
    """Regrid a 2D field from ERA5 lat-lon to Gaussian grid."""
    from scipy.interpolate import RegularGridInterpolator

    gauss_lat = np.asarray(grid.lat)
    gauss_lon = np.asarray(grid.lon)
    lat_g, lon_g = np.meshgrid(gauss_lat, gauss_lon, indexing='ij')

    interp = RegularGridInterpolator(
        (np.asarray(era5_lat), np.asarray(era5_lon)),
        np.asarray(field_2d),
        method='linear', bounds_error=False, fill_value=None,
    )
    return interp((lat_g, lon_g)).astype(np.float32)


def load_era5_ic(
    zarr_path: str,
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
) -> ERA5Slice:
    """Load a single ERA5 time slice for use as AMIP initial conditions.

    Unlike ``load_era5_slice``, this function does not require a
    ``TrainingERA5Config``.  It auto-detects available pressure levels
    from the Zarr store and selects the timestamp nearest to the
    requested date.

    Parameters
    ----------
    zarr_path : str
        Path to a local Zarr store or GCS URI containing ERA5 data.
    year, month, day, hour : int
        Target datetime for IC (default: 1 January of *year* at 00:00 UTC).

    Returns
    -------
    ERA5Slice
        Single time slice ready to pass to ``era5_to_cubedsphere_carry``
        or ``era5_to_spectral_carry``.
    """
    import pandas as pd

    ds = open_era5_zarr(zarr_path)

    # --- locate nearest time index ---
    times = ds.time.values
    try:
        target = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        time_series = pd.DatetimeIndex(times)
        time_idx = int(np.argmin(np.abs(time_series - target)))
    except Exception:
        # cftime objects (e.g. noleap calendar)
        import cftime
        target_cf = cftime.datetime(year, month, day, hour)
        diffs = np.array(
            [abs((t - target_cf).total_seconds()) for t in times],
            dtype=np.float64,
        )
        time_idx = int(np.argmin(diffs))

    # --- auto-detect pressure levels ---
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    levels_hPa = tuple(
        int(v) for v in sorted(ds[level_dim].values.tolist())
    )

    cfg = TrainingERA5Config(
        zarr_store=zarr_path,
        levels=levels_hPa,
        local_cache_dir="",
    )
    return load_era5_slice(cfg, time_idx)
