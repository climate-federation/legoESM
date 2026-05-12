"""Smoothed Lindzen (1981) orographic gravity wave drag.

Orographic GWD with smooth sigmoid activation for wave breaking,
fully differentiable via jax.lax.scan for the vertical stress profile.

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
from legoesm.atmosphere.physics._shared import safe_divide


def lindzen_gwd(
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
    config: LindzenConfig,
) -> GWDOutput:
    """Compute Lindzen orographic GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. All column arrays (ncol, nlev).

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Brunt-Väisälä frequency at full levels
    # theta_v = T * (p_ref/p)^kappa
    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_full = jnp.clip(dz_full, 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    N2_half = jnp.clip(N2_half, 1e-8, None)
    N_half = jnp.sqrt(N2_half)  # (ncol, nlev-1)

    # Extrapolate N to full levels by padding
    N_full = jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)  # (ncol, nlev)

    # Low-level wind at surface level
    u_sfc = u[:, -1]
    v_sfc = v[:, -1]
    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    cos_a = u_sfc / U_ll
    sin_a = v_sfc / U_ll

    # Wind projection along wave direction at each level
    U_proj = u * cos_a[:, None] + v * sin_a[:, None]  # (ncol, nlev)

    # Source stress at surface: tau_0 = rho * N * k * h^2 * U
    rho_sfc = rho[:, -1]
    N_sfc = N_full[:, -1]
    tau_0 = rho_sfc * N_sfc * config.k_wave * config.h_topo ** 2 * U_ll
    tau_0 = jnp.clip(tau_0, 0.0, None)

    # Saturation stress per level: tau_sat = rho * U^3 * k / N
    # Wave breaks where carried stress exceeds local saturation
    U_proj_abs = jnp.clip(jnp.abs(U_proj), 0.1, None)
    # AD-safe divide near N_full = 0 (uniform-theta column); the forward
    # is bit-identical to the prior ``/ clip(N_full, 1e-6, None)`` for
    # any sample with N_full > 1e-6.
    tau_sat = rho * U_proj_abs ** 3 * config.k_wave * safe_divide(
        jnp.ones_like(N_full), N_full, eps=1e-6,
    )
    tau_sat = jnp.clip(tau_sat, 1e-10, None)

    # Top-down scan: propagate stress from surface upward
    # Levels: 0=top, -1=surface. Scan from surface to top (reversed).
    # Breaking occurs smoothly where tau_carry exceeds tau_sat
    def scan_fn(carry, k_rev):
        tau_carry = carry
        k = nlev - 1 - k_rev
        # Smooth breaking: sigmoid activation where stress exceeds saturation
        excess = safe_divide(tau_carry, tau_sat[:, k], eps=1e-10) - config.critical_Fr
        f_break = jax.nn.sigmoid(config.Fr_sharpness * excess)
        tau_new = tau_carry * (1.0 - f_break) + tau_sat[:, k] * f_break
        tau_new = jnp.minimum(tau_new, tau_carry)
        drag = tau_carry - tau_new
        return tau_new, drag

    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
    # drag_stack: (nlev, ncol) — reverse to get (ncol, nlev) top-first
    drag_all = drag_stack.T  # (ncol, nlev)
    drag_all = drag_all[:, ::-1]  # back to top-first ordering

    # Convert stress deposit to tendency: drag / (rho * dz)
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)
    accel = -drag_all / (jnp.clip(rho * dz, 1e-10, None))

    # Project back to (du_dt, dv_dt)
    du_dt = accel * cos_a[:, None]
    dv_dt = accel * sin_a[:, None]

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
