"""Bulk aerodynamic surface layer fluxes.

Simple bulk formulas for surface momentum, sensible heat, and latent
heat fluxes using neutral drag and transfer coefficients.

    |V| = sqrt(u^2 + v^2 + eps)
    u* = sqrt(Cd) * |V|
    tau_x = -rho * Cd * |V| * u
    tau_y = -rho * Cd * |V| * v
    SH = rho * c_pd * Ch * |V| * (T_sfc - T)
    LH = rho * L_v * Ch * |V| * (q_sfc - q_v)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.coupler.bulk_flux import compute_most_fluxes, validate_bulk_scheme


def compute_surface_fluxes(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    config: SurfaceLayerConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute bulk aerodynamic surface fluxes.

    Supports constant neutral coefficients or stability-dependent
    MOST algorithms (COARE 3.0, Large & Yeager 2004) selected via
    ``config.bulk_scheme``.

    Parameters
    ----------
    u : jax.Array
        Lowest-level zonal wind [m/s], shape (ncol,).
    v : jax.Array
        Lowest-level meridional wind [m/s], shape (ncol,).
    T : jax.Array
        Lowest-level temperature [K], shape (ncol,).
    q_v : jax.Array
        Lowest-level water vapor mixing ratio [kg/kg], shape (ncol,).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Lowest-level air density [kg/m^3], shape (ncol,).
    config : SurfaceLayerConfig
        Surface layer parameters.

    Returns
    -------
    tau_x : jax.Array
        Surface zonal stress [Pa], shape (ncol,).
    tau_y : jax.Array
        Surface meridional stress [Pa], shape (ncol,).
    shflx : jax.Array
        Surface sensible heat flux [W/m^2], shape (ncol,). Positive upward.
    lhflx : jax.Array
        Surface latent heat flux [W/m^2], shape (ncol,). Positive upward.
    ustar : jax.Array
        Friction velocity [m/s], shape (ncol,).
    """
    validate_bulk_scheme(config.bulk_scheme)
    if config.bulk_scheme in ("coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
        )
        return tau_x, tau_y, shflx, lhflx, ustar

    # Constant neutral coefficients (default)
    Cd = config.Cd_neutral
    Ch = config.Ch_neutral

    # Wind speed with minimum to avoid division by zero
    wind_speed = jnp.sqrt(u ** 2 + v ** 2 + 1e-4)

    # Friction velocity
    ustar = jnp.sqrt(Cd) * wind_speed

    # Momentum fluxes (stress)
    tau_x = -rho * Cd * wind_speed * u
    tau_y = -rho * Cd * wind_speed * v

    # Sensible heat flux (positive upward = surface warmer than air)
    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T)

    # Latent heat flux (positive upward = surface moister than air)
    lhflx = rho * constants.L_v * Ch * wind_speed * (q_sfc - q_v)

    return tau_x, tau_y, shflx, lhflx, ustar
