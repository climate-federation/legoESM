"""Hines (1997) Doppler-spread gravity wave drag parameterization.

Non-orographic GWD scheme based on Doppler shifting and spectral
saturation of gravity waves. Uses bottom-up propagation with smooth
sigmoid activation for full differentiability.

References
----------
- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
  momentum deposition in the middle atmosphere. 1. Basic formulation.
  J. Atmos. Solar-Terr. Phys., 59, 371-386.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput


def hines_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: HinesConfig,
) -> GWDOutput:
    """Compute Hines Doppler-spread GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. All column arrays (ncol, nlev).

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Brunt-Väisälä frequency
    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    dz_full = jnp.clip(dz_full, 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    N2_half = jnp.clip(N2_half, 1e-8, None)
    N_half = jnp.sqrt(N2_half)

    N_full = jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)

    # Wind magnitude at each level
    U_mag = jnp.sqrt(u ** 2 + v ** 2 + 1e-10)

    # Layer thickness
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)

    # Saturation amplitude per level:
    # As gravity waves propagate upward, their amplitude grows with
    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
    # Saturation occurs when the wave-induced velocity perturbation
    # reaches a fraction of the local wind: sigma_sat ~ U / m_star_norm
    rho_sfc = rho[:, -1:]  # (ncol, 1)
    rho_ratio = jnp.sqrt(jnp.clip(rho_sfc / jnp.clip(rho, 0.01, None), 1.0, None))
    # sigma_sat = wind_fraction * N / (m_star) at each level
    sigma_sat = N_full / jnp.clip(config.m_star * rho_ratio, 1e-6, None)

    # Bottom-up scan: propagate sigma_gw upward from surface
    # As wave propagates up, amplitude grows with sqrt(rho_sfc/rho)
    def scan_fn(carry, k_rev):
        sigma_gw = carry
        k = nlev - 1 - k_rev

        # Amplitude growth from density decrease
        sigma_grown = sigma_gw * rho_ratio[:, k]

        # Dissipation where grown amplitude exceeds saturation
        f_diss = jax.nn.sigmoid(
            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
        )
        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss

        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient
        drag = (sigma_grown - sigma_new) * rho[:, k]
        drag = jnp.clip(drag, -config.Fmax, config.Fmax)

        return sigma_new, drag

    # Pin the carry dtype so the scan body stays at the input precision
    # (defaulting allows x64 to silently promote the launch wind to f64).
    sigma_gw_init = jnp.full((ncol,), config.total_rms_wind, dtype=u.dtype)
    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first

    # Convert to acceleration
    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)

    # Project isotropically along wind direction
    cos_a = u / jnp.clip(U_mag, 0.1, None)
    sin_a = v / jnp.clip(U_mag, 0.1, None)
    du_dt = accel * cos_a
    dv_dt = accel * sin_a

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
