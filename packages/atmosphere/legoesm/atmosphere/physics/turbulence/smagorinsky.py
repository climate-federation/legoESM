"""Smagorinsky–Lilly turbulence scheme.

Strain-dependent (deformation-based) eddy viscosity

    K_m = (C_s · l)^2 · |S| · √(max(0, 1 − Ri/Pr_t)),   K_h = K_m / Pr_t

following Smagorinsky (1963) with the Lilly (1962) buoyancy correction
that shuts mixing off in strongly stable layers (Ri ≥ Pr_t).  The
deformation is the resolved vertical shear of the horizontal wind,
|S| = √((∂u/∂z)² + (∂v/∂z)²), the only strain component available in a
single-column model; the mixing length ``l`` is the Blackadar (1962)
asymptotic form shared with the other turbulence closures.

Vertical mixing is applied implicitly using the Thomas algorithm
to ensure numerical stability at any time step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)


__physics_contract__ = {
    "summary": (
        "Smagorinsky-Lilly strain-dependent eddy-viscosity turbulence: "
        "K_m = (C_s l)^2 |S| sqrt(max(0, 1 - Ri/Pr_t)), K_h = K_m/Pr_t, "
        "applied by implicit vertical diffusion with surface-flux boundary "
        "conditions."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Down-gradient mixing: du_dt ~ (1/rho) d/dz(rho Km du/dz); Km, Kh >= 0 "
        "and vanish where Ri >= Pr_t (Lilly stable cutoff). shflx, lhflx are "
        "positive UPWARD from the surface and are injected as the lower "
        "boundary condition (a source/sink), so the resolved column budget is "
        "NOT closed. z increases upward; level index -1 is the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Smagorinsky (1963), Mon. Wea. Rev. 91, 99-164; "
        "Lilly (1962), Tellus 14, 148-172"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_turbulence.py: a strongly "
        "stable column (Ri >= Pr_t) shuts mixing off (Km -> 0); with zero "
        "surface flux a neutral column gives ~zero tendencies."
    ),
}


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

    # --- Strain-dependent Smagorinsky–Lilly eddy viscosity ---------------
    # Heights and shear at the (nlev-1) interior half-level interfaces.
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)
    l_mix = mixing_length(z_half_inner, config.l_mix_max)   # Blackadar (1962)

    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])       # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Resolved deformation = vertical shear of the horizontal wind, the
    # only strain component a single-column model carries.  Floor S2 so
    # both ``S = √S2`` and ``Ri = N²/S2`` stay finite and differentiable.
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Gradient Richardson number at the interfaces (virtual θ buoyancy),
    # via the canonical inverse-Exner + shared buoyancy-coefficient helpers
    # (clips kept at the call sites).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri = N2 / S2

    # Lilly (1962) buoyancy factor √(max(0, 1 − Ri/Pr_t)): enhances mixing
    # when unstable (Ri<0), shuts it off at Ri ≥ Pr_t.  Double-``where``
    # keeps the cutoff exact AND the gradient finite at Ri = Pr_t (a bare
    # √(max(·,0)) leaks a 0·∞ NaN cotangent through the dead branch).
    buoy_arg = 1.0 - Ri / config.Pr_t
    buoy_safe = jnp.where(buoy_arg > 0.0, buoy_arg, 1.0)
    f_buoy = jnp.where(buoy_arg > 0.0, jnp.sqrt(buoy_safe), 0.0)

    # K_m = (C_s · l)^2 · |S| · f_buoy ;  K_h = K_m / Pr_t.
    Km_half = (config.C_s * l_mix) ** 2 * S * f_buoy        # (ncol, nlev-1)
    Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics (single concat; same
    # pattern as Louis/TKE/Holtslag-Boville/YSU).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Aliases for the implicit-diffusion solve below.
    K_half_m = Km_half
    K_half_h = Kh_half

    # Layer thicknesses
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz = jnp.clip(dz, 1.0, None)

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

    # Apply implicit vertical diffusion.  Heat is mixed in θ-space so a
    # dry adiabat stays neutral; momentum and moisture are conserved on
    # adiabatic motion and use raw T-style diffusion.
    u_new = implicit_vertical_diffusion(u, K_half_m, rho, dz, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, K_half_m, rho, dz, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, K_half_h, rho, dz, dz_half, p_full, dt, sflx_T,
    )
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
