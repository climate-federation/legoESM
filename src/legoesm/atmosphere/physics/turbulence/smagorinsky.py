"""Smagorinsky (constant Km) turbulence scheme.

The simplest boundary layer parameterization: uses a constant eddy
diffusivity for momentum (Km) and derives heat diffusivity from
the turbulent Prandtl number (Kh = Km / Pr_t).

Vertical mixing is applied implicitly using the Thomas algorithm
to ensure numerical stability at any time step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)


def smagorinsky_turbulence(
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
    config: SmagorinskyConfig,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using constant eddy diffusivity.

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
    config : SmagorinskyConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # Constant diffusivities
    Km_val = config.Km
    Kh_val = Km_val / config.Pr_t

    # Km, Kh at full levels for diagnostics.  Pin the broadcast dtype to
    # the input field dtype: ``jnp.full(shape, scalar)`` defaults to
    # ``float64`` under ``jax_enable_x64=True`` even when the model state
    # is float32, which silently promotes the whole vertical-diffusion
    # solve to f64 (twice the GPU memory bandwidth and a forced cast at
    # the ``rhs.at[].add`` line in implicit_vertical_diffusion).
    _dtype = T.dtype
    Km_full = jnp.full((ncol, nlev), Km_val, dtype=_dtype)
    Kh_full = jnp.full((ncol, nlev), Kh_val, dtype=_dtype)

    # K at half-levels (interfaces): average of adjacent full levels
    K_half_m = jnp.full((ncol, nlev - 1), Km_val, dtype=_dtype)
    K_half_h = jnp.full((ncol, nlev - 1), Kh_val, dtype=_dtype)

    # Layer thicknesses
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz = jnp.clip(dz, 1.0, None)
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Surface fluxes
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )

    # Convert surface fluxes to boundary conditions for diffusion
    # Momentum: flux = tau / rho ~ Cd * |V| * u  (already has rho in it)
    # For the Thomas solver, surface_flux units = [phi_units * kg/m^2/s]
    # tau_x = -rho * Cd * |V| * u  [Pa = kg/(m*s^2)]
    # For u diffusion: surface_flux = tau_x (positive = upward flux of u)
    sflx_u = tau_x   # [Pa]
    sflx_v = tau_y
    # Heat: shflx = rho * c_pd * Ch * |V| * (T_sfc - T) [W/m^2]
    # For T diffusion: surface_flux = shflx / c_pd [kg/(m^2*s) * K]
    sflx_T = shflx / constants.c_pd
    # Moisture: lhflx = rho * L_v * Ch * |V| * (q_sfc - q_v) [W/m^2]
    sflx_q = lhflx / constants.L_v

    # Apply implicit vertical diffusion
    u_new = implicit_vertical_diffusion(u, K_half_m, rho, dz, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, K_half_m, rho, dz, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion(T, K_half_h, rho, dz, dz_half, dt, sflx_T)
    q_new = implicit_vertical_diffusion(q_v, K_half_h, rho, dz, dz_half, dt, sflx_q)

    # Tendencies
    du_dt = (u_new - u) / dt
    dv_dt = (v_new - v) / dt
    dT_dt = (T_new - T) / dt
    dq_v_dt = (q_new - q_v) / dt

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    return TurbulenceOutput(
        du_dt=du_dt,
        dv_dt=dv_dt,
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )
