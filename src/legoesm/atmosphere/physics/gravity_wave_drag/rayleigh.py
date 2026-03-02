"""Rayleigh friction gravity wave drag.

Simplest GWD parameterization: applies linear drag proportional to wind
speed in the boundary layer and an upper-atmosphere sponge layer.

Follows the same pattern as held_suarez.py for sigma-based drag profiles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput


def rayleigh_gwd(
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
    config: RayleighConfig,
) -> GWDOutput:
    """Compute Rayleigh friction GWD tendencies.

    Parameters
    ----------
    u, v : jax.Array
        Wind components [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full, z_half : jax.Array
        Heights [m], shapes (ncol, nlev) and (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    lat : jax.Array
        Latitude [rad], shape (ncol,).
    dt : float
        Time step [s].
    config : RayleighConfig

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Sigma coordinate
    p_sfc = p_half[:, -1:]  # (ncol, 1)
    sigma = p_full / jnp.clip(p_sfc, 1.0, None)

    # Boundary layer drag: ramps from 0 at sigma_b to k_max at surface
    k_bl = config.k_max * jnp.clip(
        (sigma - config.sigma_b) / (1.0 - config.sigma_b), 0.0, 1.0
    )

    # Upper sponge: sin^2 taper near model top
    sponge_arg = jnp.clip(
        (config.sponge_top - sigma) / config.sponge_top, 0.0, 1.0
    )
    k_sponge = config.sponge_k * jnp.sin(0.5 * jnp.pi * sponge_arg) ** 2

    # Combined drag coefficient
    k_drag = k_bl + k_sponge

    # Tendencies
    du_dt = -k_drag * u
    dv_dt = -k_drag * v

    # Frictional heating: dT/dt = -(u*du/dt + v*dv/dt) / c_pd
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
