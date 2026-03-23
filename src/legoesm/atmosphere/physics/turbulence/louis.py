"""Louis (1979) stability-dependent turbulence scheme.

Standard GCM-type boundary layer parameterization with stability
functions applied to the eddy diffusivity. The mixing length follows
an asymptotic formula, and the Richardson number controls the
stability functions.

References
----------
- Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in the
  atmosphere. Boundary-Layer Meteorol., 17, 187-202.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)


def louis_turbulence(
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
    config: LouisConfig,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using Louis (1979) stability functions.

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
    config : LouisConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # Heights at half-levels between full levels
    # z_half_inner[k] = 0.5 * (z_full[k] + z_full[k+1])  (nlev-1 interfaces)
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)

    # Mixing length at half-levels: l = kappa * z / (1 + kappa * z / l_max)
    z_abs = jnp.clip(jnp.abs(z_half_inner), 1.0, None)
    l_mix = constants.kappa_vk * z_abs / (
        1.0 + constants.kappa_vk * z_abs / config.l_mix_max
    )  # (ncol, nlev-1)

    # Layer thickness for gradient computation
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Wind shear at half-levels
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10  # shear squared, with floor
    S = jnp.sqrt(S2)

    # Virtual potential temperature for buoyancy
    # theta_v = T * (p_ref / p)^kappa * (1 + 0.61 * q_v)
    theta_v = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa * (
        1.0 + 0.61 * q_v
    )

    # Gradient Richardson number at half-levels
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz  # Brunt-Väisälä
    Ri = N2 / S2  # (ncol, nlev-1)

    # Louis (1979) stability functions
    # Smooth blending using sigmoid to avoid if/else branching
    # Unstable (Ri < 0): f(Ri) = 1 - 2b*Ri / (1 + 3b*c * l^2 * |Ri|^0.5 / dz^2)
    # Stable (Ri >= 0): f(Ri) = 1 / (1 + 2b*Ri / sqrt(1 + d*Ri))
    b_louis = config.b_louis
    c_louis = config.c_louis
    d_louis = config.d_louis

    # Unstable branch
    Ri_neg = jnp.minimum(Ri, 0.0)
    f_unstable = 1.0 - 2.0 * b_louis * Ri_neg / (
        1.0 + 3.0 * b_louis * c_louis * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz_half ** 2 + 1e-10)
    )

    # Stable branch
    Ri_pos = jnp.maximum(Ri, 0.0)
    f_stable = 1.0 / (
        1.0 + 2.0 * b_louis * Ri_pos / jnp.sqrt(1.0 + d_louis * Ri_pos)
    )

    # Smooth blending: sigmoid(100 * Ri) transitions from unstable to stable
    blend = jax.nn.sigmoid(100.0 * Ri)
    f_m = (1.0 - blend) * f_unstable + blend * f_stable
    f_h = f_m  # Same stability function for heat (Louis 1979 simplification)

    # Eddy diffusivities at half-levels
    Km_half = l_mix ** 2 * S * f_m  # (ncol, nlev-1)
    Kh_half = l_mix ** 2 * S * f_h

    # Interpolate to full levels for diagnostics
    Km_full = jnp.zeros((ncol, nlev))
    Km_full = Km_full.at[:, 1:-1].set(0.5 * (Km_half[:, :-1] + Km_half[:, 1:]))
    Km_full = Km_full.at[:, 0].set(Km_half[:, 0])
    Km_full = Km_full.at[:, -1].set(Km_half[:, -1])

    Kh_full = jnp.zeros((ncol, nlev))
    Kh_full = Kh_full.at[:, 1:-1].set(0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:]))
    Kh_full = Kh_full.at[:, 0].set(Kh_half[:, 0])
    Kh_full = Kh_full.at[:, -1].set(Kh_half[:, -1])

    # Layer thicknesses for diffusion
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    # Surface fluxes
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # Implicit vertical diffusion
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion(T, Kh_half, rho, dz_layer, dz_half, dt, sflx_T)
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

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
