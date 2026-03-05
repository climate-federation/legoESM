"""Model integration bridge for radiation.

Provides `make_radiation_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

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
    TerrainMetric,
    pressure_from_sigma,
)
from legoesm import constants

from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import (
    daily_mean_insolation,
    perpetual_equinox_insolation,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


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


def _compute_insolation(lat: jnp.ndarray, config: RadiationConfig) -> jnp.ndarray:
    """Compute TOA insolation using shared geometry settings."""
    gray_config = config.gray
    if gray_config.perpetual_equinox:
        return perpetual_equinox_insolation(lat, gray_config.S_0)
    return daily_mean_insolation(
        lat, 80.0, gray_config.S_0, gray_config.obliquity,
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
):
    """Call configured radiation backend with a unified integration interface."""
    radiation_fn, scheme_config = _get_radiation_fn(radiation_config)

    if radiation_config.scheme == "gray":
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

    # RRTMGP currently receives an effective daily-mean cosine zenith.
    cos_zenith = jnp.clip(
        insolation / jnp.clip(radiation_config.gray.S_0, 1.0e-6, None),
        0.0,
        1.0,
    )
    q_v_safe = q_v if q_v is not None else jnp.zeros_like(T)
    return radiation_fn(
        T=T,
        p_full=p_full,
        p_half=p_half,
        sfc_temperature=sfc_temperature,
        q_v=q_v_safe,
        cos_zenith=cos_zenith,
        config=scheme_config,
    )


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
    if model_type == "hydrostatic":
        return _make_hydrostatic_radiation(radiation_config)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_radiation(radiation_config)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_radiation(radiation_config)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_radiation(
    radiation_config: RadiationConfig,
) -> Callable:
    """Create radiation physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord) -> HydrostaticTendencies
    """
    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
    ) -> HydrostaticTendencies:
        T = state.T.data          # (6, n, n, nlev)
        p_s = state.p_s.data      # (6, n, n)
        lat = grid.lat             # (6, n, n)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure at full and half levels
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)  # (6,n,n,nlev)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)  # (6,n,n,nlev+1)

        # Surface temperature = lowest-level temperature
        T_sfc = T[..., -1]  # (6, n, n)

        # Insolation for both gray and RRTMGP backends.
        insol = _compute_insolation(lat, radiation_config)

        # Reshape cubed sphere to columns: (6,n,n,...) -> (ncol, ...)
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]  # 6*n*n
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        insol_col = insol.reshape(ncol)

        # Hydrostatic state is dry-only; pass zero vapor to moist-aware backends.
        q_v_col = jnp.zeros_like(T_col)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
        )

        # Reshape heating rate back to (6, n, n, nlev)
        dT_dt = rad_out.heating_rate.reshape(shape_3d)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        return HydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d), name="du_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d), name="dv_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dT_dt=Field(
                data=dT_dt, name="dT_dt_rad",
                dims=dims_3d, units="K/s",
            ),
            dp_s_dt=Field(
                data=jnp.zeros(shape_2d), name="dp_s_dt_rad",
                dims=dims_2d, units="Pa/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d), name="dphis_dt_rad",
                dims=dims_2d, units="m^2/s^3",
            ),
        )

    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_radiation(
    radiation_config: RadiationConfig,
) -> Callable:
    """Create radiation physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
    """
    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        lat = grid.lat                     # (6, n, n)

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

        # Insolation for both gray and RRTMGP backends.
        insol = _compute_insolation(lat, radiation_config)

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        insol_col = insol.reshape(ncol)

        # NH state stores water vapor in tracer slot 0 when moist tracers exist.
        if state.tracers.data.shape[-1] > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
        )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        return NonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d), name="du_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d), name="dv_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w), name="dw_dt_rad",
                dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d), name="drho_prime_dt_rad",
                dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d), name="dphis_dt_rad",
                dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad",
                dims=dims_tr, units="1/s",
            ),
        )

    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_radiation(
    radiation_config: RadiationConfig,
) -> Callable:
    """Create radiation physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord) -> SpectralHydrostaticState

    Transforms spectral state to Gaussian grid, computes radiation,
    then transforms temperature tendency back to spectral space.
    """
    def physics_fn(state, grid, sigma_coord, grid_fields=None):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import sh_analysis_3d

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
        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full  # (n_lat, n_lon, nlev)
        p_half = p_s[..., None] * sigma_half  # (n_lat, n_lon, nlev+1)

        # Surface temperature = lowest level
        T_sfc = T[..., -1]  # (n_lat, n_lon)

        # Insolation: broadcast lat to (n_lat, n_lon).
        insol_1d = _compute_insolation(lat, radiation_config)
        insol = jnp.broadcast_to(insol_1d[:, None], (n_lat, n_lon))

        # Reshape to columns: (n_lat, n_lon, ...) -> (ncol, ...)
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = jnp.broadcast_to(lat[:, None], (n_lat, n_lon)).reshape(ncol)
        insol_col = insol.reshape(ncol)

        # Spectral PE state is dry-only in current formulation.
        q_v_col = jnp.zeros_like(T_col)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
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

    return physics_fn
