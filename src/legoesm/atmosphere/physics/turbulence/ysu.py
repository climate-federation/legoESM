"""YSU (Yonsei University) PBL turbulence scheme.

Nonlocal K-profile with entrainment flux at PBL top. The K-profile
follows the same structure as Holtslag-Boville but adds an explicit
entrainment term modeled as a Gaussian envelope centered at the PBL top.

References
----------
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon.
  Wea. Rev., 134, 2318-2341.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import YSUConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)


def ysu_turbulence(
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
    config: YSUConfig,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using YSU PBL scheme.

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
    config : YSUConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # ----- Half-level gradients (same as Louis/HB) -----
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Virtual potential temperature
    exner = (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    theta_v = virtual_temperature(T, q_v) * exner

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
    theta_v_sfc = theta_v[:, -1]
    z_sfc = z_full[:, -1:]
    u_sfc = u[:, -1:]
    v_sfc = v[:, -1:]
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0
    dtheta_v_bulk = theta_v - theta_v_sfc[:, None]
    # Wind shear from surface (not absolute wind)
    dV2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2 + 1e-4
    Ri_bulk = (constants.g / jnp.clip(theta_v_sfc[:, None], 1.0, None)) * (
        dtheta_v_bulk * dz_from_sfc / dV2
    )

    # Transition-zone weighting: peaks at Ri_crit crossing, not centroid
    sigma_pbl = jax.nn.sigmoid(config.pbl_smooth_sharpness * (config.Ri_crit - Ri_bulk))
    w_pbl = sigma_pbl * (1.0 - sigma_pbl) + 1e-20
    h_pbl = jnp.sum(z_full * w_pbl, axis=1) / jnp.sum(w_pbl, axis=1)
    h_pbl = jnp.clip(h_pbl, 100.0, None)

    # ----- K-profile -----
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])
    z_norm = z_half_inner / h_pbl[:, None]
    z_norm_clip = jnp.clip(z_norm, 0.0, 1.0)
    Km_profile = (
        constants.kappa_vk * ustar[:, None] * z_half_inner * (1.0 - z_norm_clip) ** 2
    )

    # Local Ri-based Km above PBL
    z_abs = jnp.clip(jnp.abs(z_half_inner), 1.0, None)
    l_mix = constants.kappa_vk * z_abs / (
        1.0 + constants.kappa_vk * z_abs / config.l_mix_max
    )
    b_louis = 5.0
    Ri_pos = jnp.maximum(Ri, 0.0)
    f_stable = 1.0 / (1.0 + 2.0 * b_louis * Ri_pos / jnp.sqrt(1.0 + 5.0 * Ri_pos))
    Ri_neg = jnp.minimum(Ri, 0.0)
    f_unstable = 1.0 - 2.0 * b_louis * Ri_neg / (
        1.0 + 3.0 * b_louis * 5.0 * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz_half ** 2 + 1e-10)
    )
    blend_ri = jax.nn.sigmoid(100.0 * Ri)
    f_m = (1.0 - blend_ri) * f_unstable + blend_ri * f_stable
    Km_local = l_mix ** 2 * S * f_m

    # Smooth blend from K-profile to local
    blend_pbl = jax.nn.sigmoid(10.0 * (z_norm - 1.0))

    # ----- Entrainment flux at PBL top -----
    # Convective velocity scale: w* = (g * h * (w'theta')_sfc / theta_bar)^(1/3)
    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic (ncol,)
    theta_bar = jnp.mean(theta_v, axis=1)  # (ncol,)
    # w_star only meaningful for unstable (positive wtheta)
    buoyancy_flux = constants.g * jnp.maximum(wtheta_sfc, 0.0) * h_pbl / jnp.clip(
        theta_bar, 1.0, None
    )
    w_star = jnp.cbrt(jnp.maximum(buoyancy_flux, 1e-20))  # (ncol,)

    # Gaussian envelope for entrainment: K_ent = c_ent * w* * h * exp(-((z-h)/(0.3*h))^2)
    # Width of 0.3*h is used for robustness at GCM-typical vertical resolution
    width = 0.3 * h_pbl[:, None]  # (ncol, 1)
    K_ent = (
        config.entrainment_coeff * w_star[:, None] * h_pbl[:, None]
        * jnp.exp(-((z_half_inner - h_pbl[:, None]) / jnp.clip(width, 1.0, None)) ** 2)
    )

    # Combined: profile inside PBL, local above, entrainment added everywhere
    # (entrainment Gaussian is self-localizing around PBL top)
    Km_half = (1.0 - blend_pbl) * Km_profile + blend_pbl * Km_local + K_ent
    Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics — single concat per
    # field instead of the previous ``zeros + 3 .at[].set`` triple
    # scatter (lowers to one HLO op).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Layer thicknesses
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # Implicit vertical diffusion
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion(T, Kh_half, rho, dz_layer, dz_half, dt, sflx_T)
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
