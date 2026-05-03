"""Model integration bridge for radiation.

Provides `make_radiation_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
- "mpas"         : MPASPrimitiveEquationModel (Voronoi mesh, sigma coords)
"""

from __future__ import annotations

import math
from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    TerrainMetric,
    pressure_from_sigma,
    pressure_from_hybrid,
)
from legoesm import constants

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.radiation.config import (
    OzoneProfileConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.core.precision import get_policy
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
    daylight_fraction,
    perpetual_equinox_insolation,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import sh_analysis_3d


def _make_time_state():
    """Create a mutable time-state dict and its ``set_time`` mutator.

    Returns ``(_time, set_time)`` where *_time* is the mutable dict and
    *set_time* is a function that updates it in-place.
    """
    _time = {"day_of_year": 80.0, "seconds_of_day": 43200.0}

    def set_time(day_of_year: float, seconds_of_day: float):
        _time["day_of_year"] = day_of_year
        _time["seconds_of_day"] = seconds_of_day

    return _time, set_time


def _get_radiation_fn(config: RadiationConfig):
    """Select the radiation backend based on config.scheme."""
    if config.scheme == "gray":
        return gray_radiation, config.gray
    elif config.scheme == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        return rrtmgp_radiation, config.rrtmgp
    else:
        raise ValueError(f"Unknown radiation scheme: {config.scheme!r}")


def _compute_insolation(
    lat: jnp.ndarray,
    config: RadiationConfig,
    lon: jnp.ndarray | None = None,
    day_of_year: float = 80.0,
    seconds_of_day: float = 43200.0,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Compute TOA insolation and (optionally) cosine zenith angle.

    Returns
    -------
    insolation : jnp.ndarray
        TOA downward SW flux per column [W/m^2].
    cos_sza : jnp.ndarray or None
        Cosine of solar zenith angle (clipped >=0) per column.
        Only returned when ``config.diurnal_cycle`` is True.
    f_day : jnp.ndarray or None
        Daylight fraction per column.  Returned for non-diurnal daily-mean
        insolation so that RRTMGP can use a daytime-effective cos(SZA)
        rather than a day+night average.
    """
    S_0 = config.rrtmgp.S_0 if config.scheme == "rrtmgp" else config.gray.S_0
    obliquity = config.gray.obliquity

    if config.diurnal_cycle and lon is not None:
        hour = seconds_of_day / 3600.0
        cos_sza = cos_zenith_angle(lat, lon, day_of_year, hour, obliquity)
        cos_sza_pos = jnp.maximum(cos_sza, 0.0)
        return S_0 * cos_sza_pos, cos_sza_pos, None

    # No diurnal cycle — daily-mean or perpetual-equinox insolation.
    gray_config = config.gray
    if gray_config.perpetual_equinox:
        # Equinox: f_day = 0.5 everywhere
        f_day = jnp.full_like(lat, 0.5)
        return perpetual_equinox_insolation(lat, S_0), None, f_day
    f_day = daylight_fraction(lat, day_of_year, obliquity)
    return daily_mean_insolation(lat, day_of_year, S_0, obliquity), None, f_day


def _compute_ozone_vmr(
    p_full: jnp.ndarray,
    lat: jnp.ndarray,
    ozone_config: OzoneProfileConfig,
) -> jnp.ndarray | None:
    """Compute ozone VMR for RRTMGP.

    Parameters
    ----------
    p_full : jnp.ndarray
        Pressure at full levels (ncol, nlev) [Pa].
    lat : jnp.ndarray
        Latitude (ncol,) [rad].
    ozone_config : OzoneProfileConfig

    Returns
    -------
    jnp.ndarray or None
        Ozone VMR (ncol, nlev), or None to use the built-in profile.
    """
    if ozone_config.source == "standard":
        return None  # rrtmgp_radiation uses its built-in _standard_o3_profile

    if ozone_config.source == "none":
        return jnp.full_like(p_full, 1.0e-10)

    # source == "analytical": latitude-dependent Gaussian profile.
    p_hPa = p_full / 100.0
    p_peak = ozone_config.p_peak_hPa
    sigma = ozone_config.sigma_logp
    o3 = ozone_config.o3_max_vmr * jnp.exp(
        -0.5 * ((jnp.log(p_hPa) - jnp.log(p_peak)) / sigma) ** 2
    )

    if ozone_config.lat_dependence:
        # Ozone is ~2x higher at poles than equator in the lower stratosphere.
        lat_factor = 1.0 + 0.5 * jnp.sin(lat) ** 2  # (ncol,)
        o3 = o3 * lat_factor[:, None]  # broadcast to (ncol, nlev)

    return jnp.clip(o3, 1.0e-10, None)


# ===========================================================================
# Shared column extraction for all hydrostatic grids
# ===========================================================================

def _extract_tracer_columns(state, ncol, nlev, dtype=None):
    """Extract water vapor and cloud condensate columns from state tracers.

    Works for any state type (HydrostaticState, SpectralHydrostaticState, etc.)
    Returns (q_v_col, q_cloud_col, q_ice_col) all shaped (ncol, nlev).
    """
    if dtype is None:
        # Try to infer dtype from state.T or state.T_hat.  Fall back to
        # the configured compute precision rather than hardcoding fp64
        # — the latter would force radiation to allocate fp64 zeros on
        # Metal/fp32 backends, which then upcast the whole RRTMGP
        # column to fp64 via dtype promotion.
        if hasattr(state, "T"):
            dtype = state.T.data.dtype
        elif hasattr(state, "T_hat"):
            # Spectral state stores complex T_hat; the corresponding
            # real-valued T column should match the spectral grid's
            # real dtype.
            dtype = state.T_hat.data.real.dtype
        else:
            dtype = get_policy().compute
    T_col_shape = (ncol, nlev)
    q_v_col = jnp.zeros(T_col_shape, dtype=dtype)
    q_cloud_col = None
    q_ice_col = None

    tracers = getattr(state, "tracers", None)
    if tracers is not None:
        if "q_v" in tracers:
            _qv_raw = tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        if "q_c" in tracers:
            _qc_raw = tracers["q_c"]
            _qc_data = _qc_raw.data if hasattr(_qc_raw, "data") else _qc_raw
            q_cloud_col = jnp.maximum(_qc_data.reshape(ncol, nlev), 0.0)
        if "q_i" in tracers:
            _qi_raw = tracers["q_i"]
            _qi_data = _qi_raw.data if hasattr(_qi_raw, "data") else _qi_raw
            q_ice_col = jnp.maximum(_qi_data.reshape(ncol, nlev), 0.0)

    return q_v_col, q_cloud_col, q_ice_col


def _get_grid_lat_lon(grid_or_mesh, shape_2d):
    """Get latitude/longitude arrays from any grid type.

    Handles cubed-sphere, lat-lon, and MPAS Voronoi grids uniformly.
    Returns (lat, lon) broadcast to shape_2d.
    """
    if hasattr(grid_or_mesh, 'latCell'):
        # MPAS Voronoi mesh: lat/lon already (nCells,) = shape_2d
        return jnp.asarray(grid_or_mesh.latCell), jnp.asarray(grid_or_mesh.lonCell)

    lat = jnp.asarray(grid_or_mesh.grid_lat)
    lon = jnp.asarray(grid_or_mesh.grid_lon)
    if lat.ndim < len(shape_2d):
        lat = jnp.broadcast_to(
            lat.reshape((*lat.shape, *([1] * (len(shape_2d) - lat.ndim)))),
            shape_2d,
        )
    if lon.ndim < len(shape_2d):
        if lon.ndim == 1 and len(shape_2d) == 2:
            lon = jnp.broadcast_to(lon[None, :], shape_2d)
        else:
            lon = jnp.broadcast_to(
                lon.reshape((*([1] * (len(shape_2d) - lon.ndim)), *lon.shape)),
                shape_2d,
            )
    return lat, lon


def _pack_hydrostatic_tendencies(dT_dt, state, shape_3d, shape_2d):
    """Pack column heating rate into a HydrostaticTendencies.

    Returns a HydrostaticTendencies with only dT_dt non-zero.
    Works for cubed-sphere, lat-lon, and MPAS (v fields are zero or None
    depending on whether state.v is present).
    """
    dims_3d = state.T.dims
    dims_2d = state.p_s.dims
    # Pin all zero-tendency placeholders to the upstream state precision
    # so we never silently promote a f32 column path to f64 just to
    # carry an unused-momentum tendency placeholder on the diff result.
    _u_dtype = state.u.data.dtype
    _ps_dtype = state.p_s.data.dtype

    has_v = state.v is not None
    du_shape = state.u.data.shape  # (6,n,n,nlev) or (nEdges,nlev)
    du_dims = state.u.dims

    dv_dt = None
    if has_v:
        dv_dt = Field(
            data=jnp.zeros(state.v.data.shape, dtype=state.v.data.dtype), name="dv_dt_rad",
            dims=state.v.dims, units="m/s^2",
        )

    return HydrostaticTendencies(
        du_dt=Field(
            data=jnp.zeros(du_shape, dtype=_u_dtype), name="du_dt_rad",
            dims=du_dims, units="m/s^2",
        ),
        dT_dt=Field(data=dT_dt, name="dT_dt_rad", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_rad",
            dims=dims_2d, units="Pa/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_rad",
            dims=dims_2d, units="m^2/s^3",
        ),
        dv_dt=dv_dt,
    )


def _call_radiation_backend(
    radiation_config: RadiationConfig,
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    cos_sza: jnp.ndarray | None = None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
    ghg_vmr_override: dict | None = None,
    f_day: jnp.ndarray | None = None,
    rrtmgp_solver=None,
):
    """Call configured radiation backend with a unified integration interface.

    Parameters
    ----------
    cos_sza : jnp.ndarray or None
        If provided (diurnal cycle), used directly as RRTMGP cos(zenith).
        Otherwise derived from ``insolation / S_0``.
    q_cloud : jnp.ndarray or None
        Cloud liquid water mixing ratio (ncol, nlev) [kg/kg].
    q_ice : jnp.ndarray or None
        Cloud ice mixing ratio (ncol, nlev) [kg/kg].
    ghg_vmr_override : dict or None
        Runtime GHG VMR overrides passed to RRTMGP (e.g. transient CO2).
    f_day : jnp.ndarray or None
        Daylight fraction per column for non-diurnal RRTMGP.  When provided,
        the two-stream solver uses the daytime-effective cos(SZA) instead of
        the day+night average, and SW fluxes/heating are rescaled by f_day
        to recover daily-mean energy balance.
    """
    if radiation_config.scheme == "gray":
        radiation_fn, scheme_config = _get_radiation_fn(radiation_config)
        return radiation_fn(
            T=T,
            p_full=p_full,
            p_half=p_half,
            sfc_temperature=sfc_temperature,
            lat=lat,
            q_v=q_v,
            insolation=insolation,
            config=scheme_config,
        )

    # RRTMGP: use actual cos_sza if available (diurnal cycle), else derive
    # from daily-mean insolation using the daytime-effective zenith angle.
    _sw_scale = None
    if cos_sza is None:
        S_0 = radiation_config.rrtmgp.S_0
        if f_day is not None:
            # Use daytime-effective cos(SZA): insol = S_0 * f_day * <cos_sza>_day
            # so <cos_sza>_day = insol / (S_0 * f_day).  The solver sees the
            # correct daytime optical path; we rescale SW output by f_day afterward.
            f_day_safe = jnp.maximum(f_day, 1.0e-6)
            cos_sza = jnp.clip(
                insolation / (S_0 * f_day_safe), 0.0, 1.0,
            )
            _sw_scale = f_day
        else:
            cos_sza = jnp.clip(
                insolation / jnp.clip(S_0, 1.0e-6, None),
                0.0,
                1.0,
            )
    q_v_safe = q_v if q_v is not None else jnp.zeros_like(T)

    # Compute ozone VMR based on config.
    o3_vmr = _compute_ozone_vmr(p_full, lat, radiation_config.ozone)

    # Compute cloud properties if cloud scheme is active.
    cloud_kwargs = {}
    if radiation_config.cloud_scheme != "none":
        cloud_config = CloudConfig(scheme=radiation_config.cloud_scheme)
        dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
        cloud_props = compute_cloud_properties(
            T=T,
            p_full=p_full,
            q_v=q_v_safe,
            dp=dp,
            config=cloud_config,
            q_cloud=q_cloud,
            q_ice=q_ice,
        )
        cloud_kwargs = {
            "cloud_path_liq": cloud_props.lwp,
            "cloud_path_ice": cloud_props.iwp,
            "cloud_r_eff_liq": cloud_props.r_eff_liq,
            "cloud_r_eff_ice": cloud_props.r_eff_ice,
            "cloud_fraction": cloud_props.cloud_fraction,
        }

    # RRTMGP path: use solver directly (config is baked in).
    if rrtmgp_solver is None:
        # Fallback: construct a solver on the fly (e.g. called without
        # the pre-built solver from make_radiation_physics).
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)

    result = rrtmgp_solver.solve_columns(
        T=T,
        p_full=p_full,
        p_half=p_half,
        sfc_temperature=sfc_temperature,
        q_v=q_v_safe,
        cos_zenith=cos_sza,
        sfc_albedo=sfc_albedo_override,
        sfc_emissivity=sfc_emissivity_override,
        o3_vmr=o3_vmr,
        ghg_vmr_override=ghg_vmr_override,
        **cloud_kwargs,
    )

    # When using daytime-effective cos(SZA), the solver computes SW fluxes at
    # the daytime level (1/f_day times too large).  Rescale to daily-mean.
    if _sw_scale is not None:
        s = _sw_scale[:, None]  # (ncol, 1) for broadcasting against (ncol, nlev)
        result = RadiationOutput(
            lw_flux_up=result.lw_flux_up,
            lw_flux_down=result.lw_flux_down,
            sw_flux_up=result.sw_flux_up * s,
            sw_flux_down=result.sw_flux_down * s,
            heating_rate=result.lw_heating_rate + result.sw_heating_rate * s,
            lw_heating_rate=result.lw_heating_rate,
            sw_heating_rate=result.sw_heating_rate * s,
        )

    return result


def make_radiation_physics(
    radiation_config: RadiationConfig,
    model_type: str = "hydrostatic",
) -> Callable:
    """Create a physics function for radiation matching a model's signature.

    Parameters
    ----------
    radiation_config : RadiationConfig
        Radiation configuration (selects gray or RRTMGP).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    # Load heavy/static RRTMGP optics once outside model JIT traces.
    rrtmgp_solver = None
    if radiation_config.scheme == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        RRTMGP.preload(radiation_config.rrtmgp)
        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)

    if model_type == "hydrostatic":
        return _make_hydrostatic_radiation(radiation_config, rrtmgp_solver)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_radiation(radiation_config, rrtmgp_solver)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_radiation(radiation_config, rrtmgp_solver)
    elif model_type == "mpas":
        return _make_mpas_radiation(radiation_config, rrtmgp_solver)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE (cubed-sphere, lat-lon, and MPAS Voronoi)
# ===========================================================================

def _make_hydrostatic_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
) -> Callable:
    """Create radiation physics_fn for any hydrostatic model.

    Handles cubed-sphere, lat-lon FV, and MPAS Voronoi grids via the
    shared ``_get_grid_lat_lon`` / ``_extract_tracer_columns`` /
    ``_pack_hydrostatic_tendencies`` helpers.

    Signature: (state, grid_or_mesh, sigma_coord) -> HydrostaticTendencies
    """
    _time, set_time = _make_time_state()

    def physics_fn(state, grid_or_mesh, sigma_coord) -> HydrostaticTendencies:
        T = state.T.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape
        ncol = int(math.prod(int(s) for s in shape_2d))

        lat, lon = _get_grid_lat_lon(grid_or_mesh, shape_2d)

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Surface temperature = lowest-level temperature
        T_sfc = T[..., -1]

        insol, cos_sza, f_day = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Flatten to column-major (ncol, nlev)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        q_v_col, q_cloud_col, q_ice_col = _extract_tracer_columns(
            state, ncol, nlev,
        )

        f_day_col = f_day.reshape(ncol) if f_day is not None else None
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
        )

        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        return _pack_hydrostatic_tendencies(dT_dt, state, shape_3d, shape_2d)

    physics_fn.set_time = set_time
    return physics_fn

# MPAS uses the same unified hydrostatic radiation function.
_make_mpas_radiation = _make_hydrostatic_radiation


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
) -> Callable:
    """Create radiation physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
    """
    _time, set_time = _make_time_state()

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        lat = grid.lat                     # (6, n, n)
        lon = grid.lon                     # (6, n, n)

        # Reference profiles (1D -> broadcast)
        theta_0 = height_coord.theta_ref   # (nlev,)
        rho_0 = height_coord.rho_ref       # (nlev,)

        # Total fields
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        # Temperature: T = theta * exner
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape  # (6, n, n, nlev)
        shape_w = state.w.data.shape  # (6, n, n, nlev+1)
        shape_2d = state.phis.data.shape  # (6, n, n)

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Surface temperature = lowest-level temperature
        T_sfc = T[..., -1]

        # Insolation (and optionally cos_sza for diurnal cycle).
        insol, cos_sza, f_day = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        # NH state stores water vapor in tracer slot 0 when moist tracers exist.
        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        # Cloud condensate from tracer slots 1 (q_c) and 3 (q_i).
        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None
            ).reshape(ncol, nlev)

        f_day_col = f_day.reshape(ncol) if f_day is not None else None
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
        )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")
        # Pin zero-tendency placeholders to the upstream state precision
        # so x64 default zeros do not leak into the f32 column path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        return NonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_rad",
                dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_rad",
                dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_rad",
                dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad",
                dims=dims_tr, units="1/s",
            ),
        )

    physics_fn.set_time = set_time
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
) -> Callable:
    """Create radiation physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord) -> SpectralHydrostaticState

    Transforms spectral state to Gaussian grid, computes radiation,
    then transforms temperature tendency back to spectral space.
    """
    _time, set_time = _make_time_state()

    def physics_fn(state, grid, sigma_coord, grid_fields=None):
        # 1. Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']         # (n_lat, n_lon, nlev)
        p_s = fields['p_s']     # (n_lat, n_lon)
        lat = grid.lat          # (n_lat,)

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Surface temperature = lowest level
        T_sfc = T[..., -1]  # (n_lat, n_lon)

        # Insolation (with diurnal cycle support).
        # For diurnal cycle we need 2-D lat/lon; otherwise lat is 1-D and
        # the result is broadcast to (n_lat, n_lon).
        if radiation_config.diurnal_cycle:
            lat_2d = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
            lon_2d = jnp.broadcast_to(grid.lon[None, :], (n_lat, n_lon))
            insol, cos_sza, f_day = _compute_insolation(
                lat_2d, radiation_config,
                lon=lon_2d,
                day_of_year=_time["day_of_year"],
                seconds_of_day=_time["seconds_of_day"],
            )
        else:
            insol_1d, _, f_day_1d = _compute_insolation(
                lat, radiation_config,
                day_of_year=_time["day_of_year"],
                seconds_of_day=_time["seconds_of_day"],
            )
            insol = jnp.broadcast_to(insol_1d[:, None], (n_lat, n_lon))
            cos_sza = None
            f_day = jnp.broadcast_to(f_day_1d[:, None], (n_lat, n_lon)) if f_day_1d is not None else None

        # Reshape to columns: (n_lat, n_lon, ...) -> (ncol, ...)
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = jnp.broadcast_to(lat[:, None], (n_lat, n_lon)).reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        q_v_col, q_cloud_col, q_ice_col = _extract_tracer_columns(
            state, ncol, nlev,
        )

        f_day_col = f_day.reshape(ncol) if f_day is not None else None
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
        )

        # Reshape heating rate back to (n_lat, n_lon, nlev)
        dT_dt = rad_out.heating_rate.reshape(n_lat, n_lon, nlev)

        # 3. Transform T tendency to spectral space
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # No wind or surface pressure tendencies from radiation
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )

    physics_fn.set_time = set_time
    return physics_fn


# _make_mpas_radiation is defined as an alias above (= _make_hydrostatic_radiation)
