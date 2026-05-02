"""Holtslag-Boville nonlocal K-profile turbulence scheme.

Nonlocal first-order closure with counter-gradient correction for heat
transport. Uses a smooth bulk-Richardson PBL height diagnostic and a
K-profile that transitions via sigmoid blending to local Ri-based
diffusion above the boundary layer.

References
----------
- Holtslag, A. A. M., & Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. J. Climate, 6,
  1825-1842.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import HoltslagBovilleConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)


def holtslag_boville_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: HoltslagBovilleConfig,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using Holtslag-Boville nonlocal K-profile.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : HoltslagBovilleConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # ----- Half-level gradients (same pattern as louis.py) -----
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Virtual potential temperature
    theta_v = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa * (
        1.0 + 0.61 * q_v
    )

    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri = N2 / S2  # (ncol, nlev-1)

    # ----- Surface fluxes -----
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )
    ustar = jnp.clip(ustar, 1e-4, None)

    # ----- PBL height via smooth bulk-Ri -----
    # Bulk Ri from surface at each level
    theta_v_sfc = theta_v[:, -1]  # (ncol,)
    z_sfc = z_full[:, -1:]  # (ncol, 1)
    u_sfc = u[:, -1:]  # (ncol, 1)
    v_sfc = v[:, -1:]  # (ncol, 1)
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0  # (ncol, nlev)
    dtheta_v_bulk = theta_v - theta_v_sfc[:, None]
    # Wind shear from surface (not absolute wind)
    dV2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2 + 1e-4
    Ri_bulk = (constants.g / jnp.clip(theta_v_sfc[:, None], 1.0, None)) * (
        dtheta_v_bulk * dz_from_sfc / dV2
    )  # (ncol, nlev)

    # Transition-zone weighting: peaks at Ri_crit crossing, not centroid
    sharpness = 20.0
    sigma_pbl = jax.nn.sigmoid(sharpness * (config.Ri_crit - Ri_bulk))  # (ncol, nlev)
    w_pbl = sigma_pbl * (1.0 - sigma_pbl) + 1e-20
    h_pbl = jnp.sum(z_full * w_pbl, axis=1) / jnp.sum(w_pbl, axis=1)  # (ncol,)
    h_pbl = jnp.clip(h_pbl, 100.0, None)

    # ----- K-profile inside PBL -----
    # Km(z) = kappa * u* * z * (1 - z/h)^2
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)
    z_norm = z_half_inner / h_pbl[:, None]  # z/h
    z_norm_clip = jnp.clip(z_norm, 0.0, 1.0)
    Km_profile = (
        constants.kappa_vk * ustar[:, None] * z_half_inner * (1.0 - z_norm_clip) ** 2
    )  # (ncol, nlev-1)

    # Local Ri-based Km above PBL (Louis-style)
    z_abs = jnp.clip(jnp.abs(z_half_inner), 1.0, None)
    l_mix = constants.kappa_vk * z_abs / (
        1.0 + constants.kappa_vk * z_abs / config.l_mix_max
    )
    b_louis = config.b_louis
    Ri_pos = jnp.maximum(Ri, 0.0)
    f_stable = 1.0 / (
        1.0 + 2.0 * b_louis * Ri_pos / jnp.sqrt(1.0 + b_louis * Ri_pos)
    )
    Ri_neg = jnp.minimum(Ri, 0.0)
    f_unstable = 1.0 - 2.0 * b_louis * Ri_neg / (
        1.0 + 3.0 * b_louis * b_louis * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz_half ** 2 + 1e-10)
    )
    blend_ri = jax.nn.sigmoid(100.0 * Ri)
    f_m = (1.0 - blend_ri) * f_unstable + blend_ri * f_stable
    Km_local = l_mix ** 2 * S * f_m  # (ncol, nlev-1)

    # Smooth transition from profile (inside PBL) to local (above)
    blend_pbl = jax.nn.sigmoid(10.0 * (z_norm - 1.0))  # 0 inside PBL, 1 above
    Km_half = (1.0 - blend_pbl) * Km_profile + blend_pbl * Km_local
    Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics — single concat per
    # field instead of the previous ``zeros + 3 .at[].set`` triple
    # scatter (XLA lowers the concat to one HLO op).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Layer thicknesses for diffusion
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ----- Counter-gradient correction for heat -----
    # Modify effective heat surface flux to account for nonlocal transport:
    # gamma_h * (w'theta')_sfc / (Km_max * h_pbl)
    # Applied as an additional correction to the T diffusion RHS
    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic heat flux (ncol,)
    Km_max = jnp.max(Km_half, axis=1)  # (ncol,)
    counter_grad = config.gamma_h * wtheta_sfc / (
        jnp.clip(Km_max, 1e-6, None) * h_pbl
    )  # (ncol,) [K/m]

    # Add counter-gradient to the effective T gradient inside PBL
    # This is equivalent to adding Kh * gamma to the RHS of diffusion
    # We apply it as an enhanced surface flux
    sflx_T_enhanced = sflx_T + rho[:, -1] * jnp.mean(Kh_half, axis=1) * counter_grad

    # ----- Implicit vertical diffusion -----
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion(T, Kh_half, rho, dz_layer, dz_half, dt, sflx_T_enhanced)
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )
