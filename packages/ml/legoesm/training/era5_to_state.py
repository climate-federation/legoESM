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

from legoesm import constants
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


from legoesm.training.vertical_interp import (
    interp_pressure_to_sigma,
    interp_pressure_to_hybrid,
)

logger = logging.getLogger(__name__)


def _hybrid_p_s_floor(sigma, dp_floor: float = 100.0) -> float:
    """Minimum surface pressure for all hybrid layer thicknesses >= dp_floor.

    The L40 hybrid coordinate develops near-zero or negative layer thicknesses
    (dp = dA*p_ref + dB*p_s < dp_floor) when p_s << p_ref.  This returns the
    smallest p_s that keeps every level's dp above dp_floor Pa.
    """
    import numpy as _np
    dA = _np.asarray(sigma.dA)
    dB = _np.asarray(sigma.dB)
    p_ref = float(sigma.p_ref)
    # dp(k) = dA[k]*p_ref + dB[k]*p_s >= dp_floor
    # → p_s >= (dp_floor - dA[k]*p_ref) / dB[k]  when dB[k] > 0
    p_s_per_level = _np.where(dB > 0, (dp_floor - dA * p_ref) / dB, 0.0)
    return float(_np.max(p_s_per_level))


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


def load_era5_slice(config: TrainingERA5Config, time_idx: int) -> ERA5Slice:
    """Load a single ERA5 time slice with all fields needed for IC + forcing.

    Parameters
    ----------
    config : TrainingERA5Config
    time_idx : int
        Time index into the dataset.

    Returns
    -------
    ERA5Slice with all fields on the native ERA5 lat-lon grid.
    """
    store = config.local_cache_dir if config.local_cache_dir else config.zarr_store
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

    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    sigma_full = np.asarray(sigma.sigma_full)

    # ERA5 is on 1440x721 lat-lon; regrid to Gaussian grid via simple
    # nearest-neighbor or linear interpolation in lat-lon space
    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(
        era5, grid,
    )

    # Vertical interpolation: pressure levels → model levels.
    # Use TRUE hybrid pressure p(k) = A(k)*p_ref + B(k)*p_s to avoid
    # the sigma approximation error over steep terrain (see cubed-sphere
    # path comment for details).
    p_s_jax = jnp.asarray(p_s_ll)
    plev = jnp.asarray(era5.plev_Pa)
    if _is_hybrid:
        _A = jnp.asarray(sigma.A_full)
        _B = jnp.asarray(sigma.B_full)
        _p_ref = float(sigma.p_ref)
        def _vinterp(f):
            return interp_pressure_to_hybrid(jnp.asarray(f), plev, p_s_jax, _A, _B, _p_ref)
    else:
        sigma_f = jnp.asarray(sigma_full)
        def _vinterp(f):
            return interp_pressure_to_sigma(jnp.asarray(f), plev, p_s_jax, sigma_f)

    T_model = _vinterp(T_ll)
    u_model = _vinterp(u_ll)
    v_model = _vinterp(v_ll)
    # ERA5 q is SPECIFIC HUMIDITY (mass vapor / mass moist air).  The
    # legoesm physics path treats q_v as MASS MIXING RATIO (mass vapor
    # / mass dry air) — saturation_mixing_ratio in thermo.py returns
    # the mixing-ratio convention, and atmosphere/physics modules
    # consume q_v under that convention.  Convert at the ERA5
    # boundary: r = q / (1 − q).  In the tropical PBL (q ≈ 0.025) the
    # bias from skipping this conversion is ~3% of q.  Clip to avoid
    # division blow-up at q = 1.
    q_specific = jnp.maximum(_vinterp(q_ll), 0.0)
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

    # Pack into SegmentCarry with zero held fields
    shape_3d = T_model.shape
    shape_2d = p_s_jax.shape

    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


def era5_to_cubedsphere_carry(
    era5: ERA5Slice,
    grid,
    sigma,
    target_phis=None,
):
    """Convert ERA5 slice to SegmentCarry on a cubed-sphere grid.

    Uses KD-tree regridding from lat-lon to cubed-sphere, then
    vertical interpolation to model sigma levels.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : CubedSphereGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    target_phis : array-like, optional
        Accepted for caller compatibility (the driver passes the model's
        ETOPO surface geopotential here).  CURRENTLY NOT APPLIED: the IC
        dynamics are initialised on the *smoothed ERA5* orography below
        (the validated behaviour — job 25918469), while the CMOR ``orog``
        field separately reports the ETOPO mountain mask
        (``model_driver._setup_diagnostics``).  Wiring this through to place
        the dynamics on ``target_phis`` (with a barometric p_s adjustment
        from ERA5 orography to ETOPO) would change the IC and is a
        deliberate, revalidation-gated change intentionally NOT made here.

    Returns
    -------
    SegmentCarry
    """
    # ``target_phis`` is intentionally unused — see the parameter docstring.
    del target_phis
    from legoesm.grids.regridding import regrid_scalar
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
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
    q_cs = _regrid_3d(era5.q)

    # Regrid winds then rotate from geographic (east, north) to local panel
    # (x, y) coordinates.  ERA5 u/v are in geographic frame; the cubed-sphere
    # dycore expects winds in the local panel frame.  On equatorial faces the
    # angle is ~0 so the rotation is a no-op; on polar faces it can be ±90°,
    # which is exactly where the ~100 m/s polar-vortex jet would otherwise be
    # placed in the wrong direction, triggering immediate numerical blowup.
    u_cs_geo = _regrid_3d(era5.u)  # (6, n, n, n_plev) in geographic frame
    v_cs_geo = _regrid_3d(era5.v)
    _angle = jnp.asarray(grid.angle)[..., None]  # (6, n, n, 1) → broadcasts
    _cos_a = jnp.cos(_angle)
    _sin_a = jnp.sin(_angle)
    u_cs = _cos_a * u_cs_geo + _sin_a * v_cs_geo
    v_cs = -_sin_a * u_cs_geo + _cos_a * v_cs_geo

    # Regrid 2D fields
    p_s_cs = regrid_scalar(jnp.asarray(era5.p_s.ravel()), weights)
    phis_cs_raw = regrid_scalar(jnp.asarray(era5.phis.ravel()), weights)

    # Apply topography smoothing to match load_real_topography defaults
    # (topo_smoothing=4, topo_edge_blend=0.3).  Without this, raw ERA5 phis
    # has steep gradients near cubed-sphere face boundaries: the northern
    # Tibet slope (~37°N) sits only 3-4 cells from face 1's polar edge, and
    # the Arakawa-Lamb gradient scheme amplifies face-boundary gradient errors
    # to O(dx^-1) magnitude.  With raw ERA5 phis differences of ~50 kJ/kg,
    # this creates spurious ~0.4 m/s² PGF that drives blowup in ~1–5 days
    # even from rest.
    from legoesm.grids.topography import smooth_phis_cubed_sphere
    phis_cs = smooth_phis_cubed_sphere(phis_cs_raw)

    # Barometric p_s correction for hydrostatic consistency with smoothed phis.
    # Smoothing phis lowers terrain gradients (dB_dx ↓), but without this
    # adjustment ln_ps is unchanged, so the split-PGF correction term
    # pg_corr_x = R_d × T × B × p_s/p × dln_ps/dx stays the same while
    # dB_dx decreases — WORSENING the cancellation residual over Tibet.
    # Barometric formula: p_s_new = p_s × exp[(phis_raw − phis_smooth) / (R_d × T_sfc)]
    # Derivation: hydrostatic dln_p = −dΦ / (R_d × T) → Δln_p = −ΔΦ / (R_d × T)
    # T_sfc proxy = ERA5 T at 1000 hPa (plev_Pa ascending → last index = 1000 hPa).
    _T_sfc_cs = T_cs[..., -1]  # (6, n, n) — 1000 hPa, nearest to surface
    _delta_phis = phis_cs_raw - phis_cs  # > 0 where terrain was lowered by smoothing
    p_s_corrected = p_s_cs * jnp.exp(_delta_phis / (constants.R_d * _T_sfc_cs))

    # Enforce a minimum surface pressure to prevent degenerate hybrid levels.
    # The L40 hybrid coordinate has p(k) = A(k)*p_ref + B(k)*p_s.  Near the
    # surface the A coefficients decrease toward 0 while B → 1; when p_s << p_ref
    # the A*p_ref term dominates and adjacent levels can have |dp| < 1 Pa or
    # even dp < 0 (inverted).  At p_s = 56703 Pa (Tibet, 4751 m), 19 of 40
    # levels are underground and the arch-peak at level 28–29 has dp = −1 Pa,
    # causing catastrophic vertical-velocity amplification in the continuity eq.
    # Fix: raise p_s to p_s_floor wherever needed; simultaneously lower phis
    # by the barometric-formula equivalent so the split-PGF cancellation is
    # maintained (phis consistent with raised p_s).
    if _is_hybrid:
        p_s_floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
        _ln_ratio = jnp.maximum(0.0, jnp.log(p_s_floor / p_s_corrected))
        phis_cs = phis_cs - constants.R_d * _T_sfc_cs * _ln_ratio
        p_s_cs = jnp.maximum(p_s_corrected, p_s_floor)
    else:
        p_s_cs = p_s_corrected

    # Vertical interpolation.
    # CRITICAL: for hybrid sigma-pressure coordinates, use the TRUE level
    # pressure p(k) = A(k)*p_ref + B(k)*p_s — NOT the sigma approximation
    # (A+B)*p_s.  Over steep terrain (Tibet, Andes) where p_s << p_ref, the
    # sigma approximation misplaces upper levels by 100-180 hPa, introducing
    # ~20 K temperature errors and ~70 m/s wind imbalances that cause
    # immediate numerical blowup.
    plev = jnp.asarray(era5.plev_Pa)
    if _is_hybrid:
        _A = jnp.asarray(sigma.A_full)
        _B = jnp.asarray(sigma.B_full)
        _p_ref = float(sigma.p_ref)
        def _vinterp(field_cs):
            return interp_pressure_to_hybrid(field_cs, plev, p_s_cs, _A, _B, _p_ref)
    else:
        sigma_f = jnp.asarray(sigma_full)
        def _vinterp(field_cs):
            return interp_pressure_to_sigma(field_cs, plev, p_s_cs, sigma_f)

    T_model = _vinterp(T_cs)
    u_model = _vinterp(u_cs)
    v_model = _vinterp(v_cs)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING
    # RATIO (see comment at the lat-lon path above).  Convert
    # r = q / (1 − q) at the boundary.
    q_specific = jnp.maximum(_vinterp(q_cs), 0.0)
    q_specific = jnp.clip(q_specific, 0.0, 0.99)
    q_model = q_specific / (1.0 - q_specific)

    if logger.isEnabledFor(logging.INFO):
        import jax as _jax
        _hT = _jax.device_get(T_model)
        _hu = _jax.device_get(u_model)
        _hv = _jax.device_get(v_model)
        _hps = _jax.device_get(p_s_cs)
        _hphis = _jax.device_get(phis_cs)
        _hphis_raw = _jax.device_get(phis_cs_raw)
        _p_s_floor_val = _hybrid_p_s_floor(sigma, dp_floor=100.0) if _is_hybrid else 0.0
        logger.info(
            f"  ERA5→CS IC: T=[{float(_hT.min()):.0f},{float(_hT.max()):.0f}]K "
            f"u=[{float(_hu.min()):.0f},{float(_hu.max()):.0f}]m/s "
            f"v=[{float(_hv.min()):.0f},{float(_hv.max()):.0f}]m/s "
            f"p_s=[{float(_hps.min()):.0f},{float(_hps.max()):.0f}]Pa "
            f"phis=[{float(_hphis.min()):.0f},{float(_hphis.max()):.0f}]m2/s2 "
            f"(raw phis peak={float(_hphis_raw.max()):.0f}, "
            f"p_s_floor={_p_s_floor_val:.0f}Pa)"
        )

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

    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
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
    return pack_carry(
        state,
        q_v=q_model,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


class MPASCarry(NamedTuple):
    """ERA5 initial condition on an MPAS/Voronoi mesh.

    Unlike the cube/lat-lon ``SegmentCarry`` (which stores cell-centred
    ``u``/``v``), the MPAS hydrostatic state carries the horizontal wind as
    the **edge-normal** component on mesh edges, and has no separate ``v``.
    """
    u: jnp.ndarray      # (nEdges, nlev) edge-normal velocity [m/s]
    T: jnp.ndarray      # (nCells, nlev) temperature [K]
    p_s: jnp.ndarray    # (nCells,) surface pressure [Pa]
    phis: jnp.ndarray   # (nCells,) surface geopotential [m^2/s^2]
    q_v: jnp.ndarray    # (nCells, nlev) water vapour mixing ratio [kg/kg]


def era5_to_mpas_carry(
    era5: ERA5Slice,
    mesh,
    sigma,
):
    """Convert an ERA5 slice to an initial condition on an MPAS/Voronoi mesh.

    The MPAS hydrostatic dycore stores the prognostic horizontal velocity as
    the **edge-normal** component ``u`` on mesh edges (no cell-centred ``v``),
    with scalars (``T``/``p_s``/``phis``/tracers) at cell centres.  This
    builds that state from ERA5:

    1. Horizontal regrid (KD-tree inverse-distance) of the regular ERA5
       lat-lon fields onto the mesh — scalars to cell centres
       (``latCell``/``lonCell``), winds to edge midpoints
       (``latEdge``/``lonEdge``).
    2. Edge-normal projection of the geographic ERA5 (east, north) winds
       via ``angleEdge`` (eastward = 0):
       ``u_n = u_east·cos(angleEdge) + v_north·sin(angleEdge)`` — the inverse
       of the ``reconstruct_cell_velocity`` the physics uses.
    3. Vertical interpolation from ERA5 pressure levels to the model
       sigma/hybrid levels (true level pressure ``A·p_ref + B·p_s`` for
       hybrid), with a hybrid ``p_s`` floor over high terrain.
    4. Specific humidity → mixing ratio (``r = q/(1−q)``), matching the
       cube/lat-lon convention.

    Parameters
    ----------
    era5 : ERA5Slice
    mesh : VoronoiMesh
        Exposes ``latCell``/``lonCell``/``latEdge``/``lonEdge``/``angleEdge``
        (radians) and ``nCells``/``nEdges``.
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    MPASCarry
    """
    from legoesm.grids.regridding import (
        compute_latlon_to_voronoi_weights, regrid_scalar,
    )
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate

    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    sigma_full = np.asarray(sigma.sigma_full)

    era5_lat = np.asarray(era5.lat)  # radians
    era5_lon = np.asarray(era5.lon)  # radians

    # Horizontal regrid weights: ERA5 lat-lon -> cell centres / edge midpoints.
    cell_w = compute_latlon_to_voronoi_weights(
        era5_lat, era5_lon,
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
    )
    edge_w = compute_latlon_to_voronoi_weights(
        era5_lat, era5_lon,
        np.asarray(mesh.latEdge), np.asarray(mesh.lonEdge),
    )

    # Scalars at cells (nCells, n_plev) / (nCells,).
    T_cell = regrid_scalar(jnp.asarray(era5.T), cell_w)
    q_cell = regrid_scalar(jnp.asarray(era5.q), cell_w)
    p_s_cell = regrid_scalar(jnp.asarray(era5.p_s), cell_w)
    phis_cell = regrid_scalar(jnp.asarray(era5.phis), cell_w)

    # Winds at edge midpoints (geographic east/north), then project to the
    # edge normal.  ERA5 u/v are in the geographic frame; angleEdge gives the
    # edge-normal direction relative to local east.
    u_east_edge = regrid_scalar(jnp.asarray(era5.u), edge_w)   # (nEdges, n_plev)
    v_north_edge = regrid_scalar(jnp.asarray(era5.v), edge_w)
    angle = jnp.asarray(mesh.angleEdge)[:, None]               # (nEdges, 1)
    u_n_edge = u_east_edge * jnp.cos(angle) + v_north_edge * jnp.sin(angle)

    # Surface pressure at edges (for placing the wind levels over terrain).
    p_s_edge = regrid_scalar(jnp.asarray(era5.p_s), edge_w)

    # Hybrid p_s floor over high terrain: raise p_s where the hybrid layers
    # would become degenerate (dp < dp_floor), and lower phis by the
    # barometric equivalent so the split-PGF cancellation is preserved.
    # No phis smoothing is applied (the coarse mesh is already smooth and
    # there is no mesh-native cube-edge artefact to blend), so unlike the
    # cubed-sphere path there is no smoothing-driven p_s correction.
    if _is_hybrid:
        p_s_floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
        _T_sfc = T_cell[..., -1]  # 1000 hPa (plev ascending -> last index)
        _ln_ratio = jnp.maximum(0.0, jnp.log(p_s_floor / p_s_cell))
        phis_cell = phis_cell - constants.R_d * _T_sfc * _ln_ratio
        p_s_cell = jnp.maximum(p_s_cell, p_s_floor)
        p_s_edge = jnp.maximum(p_s_edge, p_s_floor)

    # Vertical interpolation to model levels.
    plev = jnp.asarray(era5.plev_Pa)
    if _is_hybrid:
        _A = jnp.asarray(sigma.A_full)
        _B = jnp.asarray(sigma.B_full)
        _p_ref = float(sigma.p_ref)

        def _vinterp_cell(field):
            return interp_pressure_to_hybrid(field, plev, p_s_cell, _A, _B, _p_ref)

        def _vinterp_edge(field):
            return interp_pressure_to_hybrid(field, plev, p_s_edge, _A, _B, _p_ref)
    else:
        sigma_f = jnp.asarray(sigma_full)

        def _vinterp_cell(field):
            return interp_pressure_to_sigma(field, plev, p_s_cell, sigma_f)

        def _vinterp_edge(field):
            return interp_pressure_to_sigma(field, plev, p_s_edge, sigma_f)

    T_model = _vinterp_cell(T_cell)
    u_model = _vinterp_edge(u_n_edge)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING RATIO.
    q_specific = jnp.clip(jnp.maximum(_vinterp_cell(q_cell), 0.0), 0.0, 0.99)
    q_model = q_specific / (1.0 - q_specific)

    if logger.isEnabledFor(logging.INFO):
        import jax as _jax
        _hT = _jax.device_get(T_model)
        _hu = _jax.device_get(u_model)
        _hps = _jax.device_get(p_s_cell)
        _hphis = _jax.device_get(phis_cell)
        logger.info(
            f"  ERA5->MPAS IC: T=[{float(_hT.min()):.0f},{float(_hT.max()):.0f}]K "
            f"u_n=[{float(_hu.min()):.0f},{float(_hu.max()):.0f}]m/s "
            f"p_s=[{float(_hps.min()):.0f},{float(_hps.max()):.0f}]Pa "
            f"phis=[{float(_hphis.min()):.0f},{float(_hphis.max()):.0f}]m2/s2"
        )

    return MPASCarry(
        u=u_model,
        T=T_model,
        p_s=p_s_cell,
        phis=phis_cell,
        q_v=q_model,
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
