"""Smoothed McFarlane (1987) orographic gravity wave drag.

Extends the Lindzen approach with explicit launch flux control,
minimum wind activation, and directional spreading. Uses smooth
(sigmoid / softmin) approximations for full differentiability.

References
----------
- McFarlane, N. A. (1987). The effect of orographically excited gravity
  wave drag on the general circulation of the lower stratosphere and
  troposphere. J. Atmos. Sci., 44, 1775-1800.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput


def mcfarlane_gwd(
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
    config: McFarlaneConfig,
) -> GWDOutput:
    """Compute McFarlane orographic GWD tendencies.

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

    # Extrapolate N to full levels
    N_full = jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)

    # Low-level wind
    u_sfc = u[:, -1]
    v_sfc = v[:, -1]
    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    cos_a = u_sfc / U_ll
    sin_a = v_sfc / U_ll

    # Smooth minimum wind activation
    U_activated = jax.nn.sigmoid(config.min_wind_sharpness * (U_ll - config.min_wind)) * U_ll

    # Wind projection along wave direction
    U_proj = u * cos_a[:, None] + v * sin_a[:, None]
    U_proj_abs = jnp.clip(jnp.abs(U_proj), 1e-2, None)

    # Launch flux: orographic gravity-wave stress
    #     tau_0 = G_0 * rho * N * k * h^2 * U
    # (after McFarlane 1987 / Palmer 1986).  ``G_0`` is *dimensionless*;
    # the dimensional factors that turn the formula into a stress
    # [Pa = kg/(m·s²)] are ``rho * N * k * h^2 * U``.  The earlier
    # implementation omitted ``k_wave`` and clipped the result to
    # ``[0, 10] Pa`` — which combined with the missing wavenumber gave
    # the formula units of ``kg²/(m²·s⁴)`` and a magnitude of order
    # ``10⁴`` (numerically) → clip truncated to 10 → drag ~1e-21 m/s²
    # (audit's "McFarlane stress dimensionally suspect").
    rho_sfc = rho[:, -1]
    N_sfc = N_full[:, -1]
    tau_0 = (
        config.G_0
        * rho_sfc
        * N_sfc
        * config.k_wave
        * config.h_topo ** 2
        * U_activated
    )
    tau_0 = tau_0 * config.directional_spread
    tau_0 = jnp.clip(tau_0, 0.0, config.tau_max)

    # Saturation stress per level (Lindzen 1981 / McFarlane 1987):
    #     tau_sat = efficiency * rho * U^3 * k_wave / (N * envelope)   [Pa]
    # The earlier formulation omitted ``k_wave`` and had units
    # ``kg/s^2`` rather than ``Pa = kg/(m·s^2)`` — together with the
    # missing ``k_wave`` in the launch stress (fixed earlier in this
    # file) the scheme produced dimensionally inconsistent stresses
    # whose numerical magnitudes were off by a factor of ~k_wave that
    # the ``tau_0`` clip then masked operationally.  Lindzen
    # (``lindzen.py:85``) implements the correct form; McFarlane is
    # now aligned with it.
    envelope = config.envelope_scale
    tau_sat = (
        config.efficiency
        * rho
        * U_proj_abs ** 3
        * config.k_wave
        / (jnp.clip(N_full, 1e-6, None) * envelope)
    )
    tau_sat = jnp.clip(tau_sat, 1e-10, None)

    # Top-down scan with smooth min (softmin via logsumexp)
    alpha = config.softmin_sharpness
    def scan_fn(carry, k_rev):
        tau_carry = carry
        k = nlev - 1 - k_rev
        # Smooth min: softmin(a, b) = -logsumexp(-alpha*[a,b])/alpha
        tau_k = -jax.nn.logsumexp(
            jnp.stack([-alpha * tau_carry, -alpha * tau_sat[:, k]], axis=0),
            axis=0,
        ) / alpha
        drag = tau_carry - tau_k
        return tau_k, drag

    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first

    # Convert to tendency
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)
    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)

    du_dt = accel * cos_a[:, None]
    dv_dt = accel * sin_a[:, None]

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
