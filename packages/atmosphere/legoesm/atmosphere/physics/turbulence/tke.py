"""Prognostic TKE, Mellor-Yamada-2.5-INSPIRED k-l turbulence closure.

Carries a prognostic turbulent kinetic energy (TKE) equation and
derives eddy diffusivities from TKE and a mixing length:

    Km = Ck * l * sqrt(max(TKE, tke_min))
    Kh = Km / Pr_t

TKE budget:
    de/dt = Km * S^2 - Kh * N^2 - Ce * e^{3/2} / l + rho^-1 d/dz[rho Km de/dz]

Dissipation is treated semi-implicitly for stability.

References
----------
- Mellor, G. L., & Yamada, T. (1982). Development of a turbulence closure
  model for geophysical fluid problems. Rev. Geophys., 20, 851-875.

Faithfulness to Mellor-Yamada (1982) level 2.5
----------------------------------------------
This is a MY2.5-INSPIRED k-l closure, NOT the literal level-2.5 scheme: it carries the
prognostic TKE budget but REPLACES MY2.5's defining algebraic stability functions with
constant coefficients.
MY2.5-STRUCTURED (faithful forms):
  * the prognostic TKE (``e = q²/2``) budget SKELETON — shear production ``Km·S²``,
    buoyancy ``−Kh·N²``, and dissipation ``∝ e^{3/2}/l`` — matches the Mellor-Yamada
    level-2.5 ``q²/2`` budget (the transport term uses ``Km`` not MY's ``Kq``; see below);
  * the k-l eddy viscosity ``Km ∝ l·√e`` has MY's ``Km = l·q·Sm`` form (``q = √(2e)``, so
    ``Ck = √2·Sm_eff`` for a constant effective ``Sm``);
  * the dissipation ``ε = Ce·e^{3/2}/l`` is MY's ``ε = q³/(B1·l)`` (``Ce = 2^{3/2}/B1
    ≈ 0.17``; default 0.19).
DEPARTURES from MY2.5 (the defining level-2.5 physics, simplified):
  * **CONSTANT Ck replaces the algebraic momentum stability function Sm(GM, GH)** — the
    single defining feature of level 2.5 (``Sm`` depends on BOTH the shear ``GM`` and the
    buoyancy ``GH = −l²N²/q²``).  ``Km = Ck·l·√e`` therefore scales the SAME way at every
    Richardson number; at FIXED ``e`` and ``l`` it is INDEPENDENT of the stratification
    ``N²`` (MY2.5's ``Sm`` decreases with stability).  (Over a prognostic run ``N²`` still
    changes later ``Km`` INDIRECTLY, through its buoyancy effect on ``e``.)
  * **CONSTANT Pr_t = 0.33 sets a fixed ``Kh/Km = 1/Pr_t ≈ 3.03``** — MY2.5's ratio
    ``Sh(GM, GH)/Sm(GM, GH)`` is stability-dependent (and its NEUTRAL value is only
    ``Sh/Sm = A2 / [A1(1−3C1)] ≈ 0.74/0.70 ≈ 1.06``, not ≈3), so ``Pr_t = 0.33`` is this
    scheme's own fixed choice, NOT the MY value.
  * **TKE is diffused with ``Km``, not MY's separate TKE diffusivity ``Kq = l·q·Sq``**
    (``Sq ≈ 0.20``) — the transport term ``ρ⁻¹∂z(ρ Km ∂z e)`` uses the momentum ``Km``.
  * **fixed asymptotic mixing length** ``l = κz/(1 + κz/l_max)`` with a CONSTANT
    ``l_max`` — MY's diagnostic length is likewise Blackadar-like near the wall but with
    an integral, turbulence-dependent asymptote ``l0 = 0.1·∫q z dz / ∫q dz``.
  * **numerics** (not MY): the semi-implicit dissipation linearization
    ``ε ≈ Ce·√e_n·e_{n+1}/l`` (stability + AD-safe) and the ``tke_min`` floor.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_tke_faithful.py``.
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
from legoesm.atmosphere.physics.turbulence.config import TKEConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)


__physics_contract__ = {
    "summary": (
        "Prognostic-TKE (Mellor-Yamada-2.5-INSPIRED k-l closure) turbulence: "
        "advances a turbulent-kinetic-energy budget (shear production, "
        "buoyancy, dissipation, diffusion), sets K_m = Ck l sqrt(max(TKE, "
        "tke_min)) with a CONSTANT Ck (in place of MY2.5's algebraic "
        "Sm(GM, GH) stability function) and K_h = K_m/Pr_t with a constant "
        "Pr_t, then applies implicit vertical diffusion with surface-flux BCs."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "tke": "m^2/s^2", "p_full": "Pa", "p_half": "Pa",
        "z_full": "m", "z_half": "m", "T_sfc": "K", "q_sfc": "kg/kg",
        "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "tke_new": "m^2/s^2",
    },
    "sign_convention": (
        "Down-gradient mixing: du_dt ~ (1/rho) d/dz(rho Km du/dz); Km, Kh >= 0; "
        "TKE >= tke_min with shear production Km S^2 >= 0, buoyancy -Kh N^2 "
        "(a sink in stable stratification), and dissipation Ce TKE^{3/2}/l "
        ">= 0. shflx, lhflx positive UPWARD and injected as the lower boundary "
        "condition (a source), so the resolved column budget is NOT closed. "
        "z increases upward; level index -1 is the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Mellor & Yamada (1982), Rev. Geophys. 20, 851-875",
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_turbulence.py: TKE stays "
        ">= tke_min; a neutral no-shear column has zero production and "
        "buoyancy so TKE decays by dissipation; Km = Ck l sqrt(max(TKE, "
        "tke_min)) >= 0."
    ),
}


def tke_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    tke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: TKEConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute turbulence tendencies using prognostic TKE.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    tke : jax.Array
        Turbulent kinetic energy [m^2/s^2], shape (ncol, nlev).
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
    config : TKEConfig

    Returns
    -------
    TurbulenceOutput
        Turbulence tendencies and diagnostics.
    tke_new : jax.Array
        Updated TKE [m^2/s^2], shape (ncol, nlev).
    """
    ncol, nlev = T.shape
    tke = jnp.maximum(tke, config.tke_min)

    # Half-level quantities
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Mixing length at full levels: l = kappa * z / (1 + kappa * z / l_max)
    l_mix = mixing_length(z_full, config.l_mix_max)  # (ncol, nlev)

    # Eddy diffusivities at full levels
    sqrt_tke = jnp.sqrt(tke)
    Km_full = config.Ck * l_mix * sqrt_tke
    Kh_full = Km_full / config.Pr_t

    # Eddy diffusivities at half-levels (average adjacent full levels)
    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])  # (ncol, nlev-1)
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    # --- TKE budget ---
    # Shear production at half-levels
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2  # (ncol, nlev-1)

    # Brunt-Väisälä at half-levels (canonical inverse-Exner + shared
    # buoyancy-coefficient helpers; clips kept at the call sites).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2_half = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz

    # Interpolate S2 and N2 to full levels.  Single ``concatenate`` of
    # the centered interior with the two endpoint half-values lowers
    # to one HLO op vs the previous ``zeros + 3 .at[].set`` triple
    # scatter (3 wasted scatter ops per field, fired every TKE step).
    S2_interior = 0.5 * (S2_half[:, :-1] + S2_half[:, 1:])
    S2 = jnp.concatenate(
        [S2_half[:, :1], S2_interior, S2_half[:, -1:]], axis=1,
    )
    N2_interior = 0.5 * (N2_half[:, :-1] + N2_half[:, 1:])
    N2 = jnp.concatenate(
        [N2_half[:, :1], N2_interior, N2_half[:, -1:]], axis=1,
    )

    # Production and buoyancy at full levels
    shear_prod = Km_full * S2        # P = Km * S^2
    buoyancy = -Kh_full * N2         # B = -Kh * N^2

    # Dissipation: semi-implicit linearization
    # epsilon = Ce * e^{3/2} / l ≈ Ce * e^{1/2}_n * e_{n+1} / l
    # This makes dissipation linear in the new TKE for stability
    l_mix_safe = jnp.clip(l_mix, 1.0, None)
    diss_coeff = config.Ce * sqrt_tke / l_mix_safe  # [1/s]

    # TKE diffusion using implicit solver
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    # Diffuse TKE (no surface flux for TKE).  Pin the surface_flux dtype
    # to the column dtype so ``rhs.at[:, -1].add(...)`` inside the
    # tridiagonal solve does not see an x64-default zero clashing with
    # the f32 state.
    tke_diffused = implicit_vertical_diffusion(
        tke, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=tke.dtype),
    )

    # Semi-implicit TKE update:
    # tke_new = (tke_diffused + dt * (P + B)) / (1 + dt * diss_coeff)
    tke_new = (tke_diffused + dt * (shear_prod + buoyancy)) / (1.0 + dt * diss_coeff)
    tke_new = jnp.maximum(tke_new, config.tke_min)

    # --- Apply diffusion to u, v, T, q_v ---
    # Surface fluxes
    tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd

    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=h_pbl,
    )

    return output, tke_new
