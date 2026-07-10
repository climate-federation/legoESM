"""Louis (1979) stability-dependent turbulence scheme.

Standard GCM-type boundary layer parameterization with stability
functions applied to the eddy diffusivity. The mixing length follows
an asymptotic formula, and the Richardson number controls the
stability functions.

References
----------
- Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in the
  atmosphere. Boundary-Layer Meteorol., 17, 187-202.
- Louis, J.-F., Tiedtke, M., & Geleyn, J.-F. (1982). A short history of
  the operational PBL parameterization at ECMWF. ECMWF Workshop on
  Planetary Boundary Layer Parameterization, 59-79.  (Separate momentum
  vs heat stability functions, ``b_h/b_m = 3/2``.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    louis_stability_functions,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    diagnose_pbl_height,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Louis (1979) stability-dependent PBL turbulence: mixing-length eddy "
        "diffusivities Km, Kh scaled by Richardson-number stability functions "
        "f_m, f_h; mixes momentum, heat (theta-space) and moisture implicitly."
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
        "Down-gradient eddy diffusion, Km = l^2*|S|*f_m >= 0, Kh = l^2*|S|*f_h "
        ">= 0; stability functions enhance mixing for Ri<0 (unstable) and "
        "suppress it for Ri>0 (stable). The column budget is OPEN: the surface "
        "flux (shflx > 0 upward, lhflx > 0 upward/moistening) is injected as "
        "the bottom boundary condition, top is zero-flux; z increases upward."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Louis (1979), Boundary-Layer Meteorol. 17, 187-202; "
        "Louis, Tiedtke & Geleyn (1982) separate heat stability function"
    ),
    "idealized_test": (
        "rest state with zero surface flux and a neutral column (Ri=0) -> "
        "f_m = f_h = 1 and near-zero interior tendency; Km, Kh >= 0; "
        "stable stratification reduces Km relative to neutral."
    ),
}


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
    surface_flux: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | None = None,
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
    surface_flux : tuple of jax.Array, optional
        Pre-computed surface fluxes ``(tau_x, tau_y, shflx, lhflx, ustar)``,
        each shape ``(ncol,)``, used as the BL bottom boundary condition in
        place of the single-surface ``compute_surface_fluxes`` call.  This
        is how the driver injects the AREA-WEIGHTED tiled (mosaic) surface
        flux — COARE3 on the ocean tile, the land Monin-Obukhov scheme on
        the land tile — instead of running one bulk scheme on the blended
        surface temperature (which runs the ocean scheme over land).  When
        ``None`` (default) the legacy single-surface flux is computed from
        ``T_sfc``/``q_sfc``/``config.surface`` (byte-identical behaviour).

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # Heights at half-levels between full levels
    # z_half_inner[k] = 0.5 * (z_full[k] + z_full[k+1])  (nlev-1 interfaces)
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)

    # Mixing length at half-levels (Blackadar 1962): shared helper, also used
    # by TKE/CLUBB-lite/EDMF/Smagorinsky (no re-inlined l_mix expression).
    l_mix = mixing_length(z_half_inner, config.l_mix_max)  # (ncol, nlev-1)

    # Layer thickness for gradient computation
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Wind shear at half-levels
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10  # shear squared, with floor
    S = jnp.sqrt(S2)

    # Virtual potential temperature for buoyancy:
    # theta_v = T_v(T, q_v) · (p_ref / p)^kappa, with the canonical
    # T_v factor 1 + (R_v/R_d - 1) q_v ≈ 1 + 0.6078 q_v, and the canonical
    # inverse-Exner helper (same 1 Pa pressure floor).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref

    # Gradient Richardson number at half-levels; N² = (g/θ_v)·∂θ_v/∂z via the
    # shared buoyancy coefficient (clip kept at the call site).
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz  # Brunt-Väisälä
    Ri = N2 / S2  # (ncol, nlev-1)

    # Louis stability functions (Louis 1979; separate heat function per
    # Louis, Tiedtke & Geleyn 1982) via the SHARED helper
    # ``physics._shared.louis_stability_functions`` (also used by YSU's
    # free-atmosphere branch — one home, no verbatim copy).
    # Momentum uses b_m = b_louis; heat uses b_h = b_heat_ratio·b_louis
    # (LTG82: 3b heat vs 2b momentum ⇒ ratio 1.5).  The branch denominators
    # are SHARED between momentum and heat inside the helper so only the
    # numerator coefficient differs: K_m is then exactly independent of
    # b_heat_ratio, and b_heat_ratio = 1 recovers the Louis (1979)
    # f_h = f_m form.  The heat function is *more* enhanced when unstable
    # (Pr_t = K_m/K_h < 1) and *more* suppressed when stable (Pr_t > 1).
    f_m, f_h = louis_stability_functions(
        Ri, l_mix, dz_half,
        config.b_louis, config.c_louis, config.d_louis,
        config.blend_ri_sharpness,
        b_heat=config.b_heat_ratio * config.b_louis,
    )

    # Eddy diffusivities at half-levels
    Km_half = l_mix ** 2 * S * f_m  # (ncol, nlev-1)
    Kh_half = l_mix ** 2 * S * f_h

    # Interpolate to full levels for diagnostics — single concat per
    # field instead of the previous ``zeros + 3 .at[].set`` triple
    # scatter (lowers to one HLO op).  Same pattern as TKE,
    # Holtslag-Boville, and YSU.
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

    # Surface fluxes — either the driver-supplied tiled (mosaic) flux or
    # the legacy single-surface bulk flux from the blended T_sfc.
    if surface_flux is not None:
        tau_x, tau_y, shflx, lhflx, ustar = surface_flux
    else:
        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
            T_sfc, q_sfc, rho[:, -1], config.surface,
        )

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # Implicit vertical diffusion.  Heat in θ-space so a dry adiabat
    # stays neutral; moisture and momentum stay in physical space.
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # Thread the scheme's critical Richardson number into the PBL-height
    # diagnosis (LouisConfig.Ri_crit was previously declared but never
    # read — the call silently used PBLHeightConfig's default).
    h_pbl = diagnose_pbl_height(
        T, q_v, u, v, p_full, z_full,
        PBLHeightConfig(Ri_crit=config.Ri_crit),
    )

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
