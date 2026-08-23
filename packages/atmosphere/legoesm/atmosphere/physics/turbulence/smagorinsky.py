"""Smagorinsky–Lilly turbulence scheme.

Strain-dependent (deformation-based) eddy viscosity

    K_m = (C_s · l)^2 · |S| · √(max(0, 1 − Ri/Pr_t)),   K_h = K_m / Pr_t

following Smagorinsky (1963), with a Lilly-type buoyancy cutoff that shuts
mixing off in strongly stable layers (Ri ≥ Pr_t).  Lilly (1962) left the Ri
dependence and ``K_h/K_m`` as undetermined functions in his GENERAL theory,
but his equilibrium EXPERIMENT already gives THIS ``√(1 − Ri/Pr_t)`` stability
factor with ``K_h/K_m = 1`` (hence ``Pr_t = 1``) and ``K_m = 0`` for ``Ri > 1``
— all reproduced here.  Only the JAX AD-safe NUMERICS are modern: the
``max(0, ·)`` clamp and the double-``where`` guard keep the ``√`` and its
reverse-mode cotangent finite at the ``Ri = Pr_t`` kink (plus a ``S²+1e-10``
floor); the ``√`` ramp itself is Lilly's, not a modern replacement of a hard
on/off.  The deformation is a 1-D PROXY: the resolved vertical shear |S| =
√((∂u/∂z)² + (∂v/∂z)²) is SUBSTITUTED for Smagorinsky's horizontal deformation
(a single column carries no horizontal strain); the mixing length ``l`` is the
Blackadar (1962) asymptotic form shared with the other turbulence closures.

Vertical mixing is applied implicitly using the Thomas algorithm
to ensure numerical stability at any time step.

Faithfulness to Smagorinsky (1963) / Lilly (1962)
-------------------------------------------------
This is a MODERN 1-D specialization of the Smagorinsky-Lilly idea, NOT the literal
published scheme: Smagorinsky (1963) used the HORIZONTAL deformation and a GRID
length scale (reporting ``k_s ≈ 0.28``).  Lilly (1962) left ``K_h/K_m`` and the
Richardson dependence as undetermined functions in his GENERAL theory, but his
EXPERIMENTS fixed ``K_h/K_m = 1`` and ``K_m → 0`` for ``Ri > 1``.
FAITHFUL (Smagorinsky deformation structure — the only truly-faithful piece):
  * **Deformation eddy viscosity** ``K_m = (C_s·l)²·|S|`` — the Smagorinsky
    length²·strain form (with the length/strain modernized; see below).
LILLY-EXPERIMENT-CONSISTENT (matches Lilly's specific 1962 equilibrium experiment, not
his undetermined general theory):
  * **Buoyancy stability factor + cutoff + Prandtl closure**: the ``√(1 − Ri/Pr_t)``
    stability factor, the ``Ri ≥ Pr_t`` shut-off, and the default ``Pr_t = 1`` (so
    ``K_h = K_m/Pr_t = K_m``) all match Lilly's equilibrium experiment — his ``√``
    stability factor with ``K_h/K_m = 1`` and ``K_m → 0`` for ``Ri > 1``.  The ``√``
    ramp is Lilly's OWN form (not a modern replacement); only the AD-safe numerics
    (below) are modern.  ``Pr_t`` is exposed as a tunable that generalizes Lilly's
    fixed unity.
MODERN additions (from other authors / not in either 1962-63 paper):
  * **Blackadar (1962) master mixing length** ``l = κz/(1 + κz/l_∞)`` — a separate
    author's length scale, not from Smagorinsky/Lilly.
DEPARTURES / RE-TUNED (vs the published papers):
  * **Modern deformation constant**: ``C_s = 0.2`` (Smagorinsky reported
    ``k_s ≈ 0.28``) and ``l_mix_max = 100 m`` are modern atmospheric single-column
    choices, NOT the 1962/1963 constants.
  * **1-D vertical-shear PROXY for the horizontal deformation**: ``|S| = √((∂u/∂z)² +
    (∂v/∂z)²)`` is the resolved VERTICAL shear.  Smagorinsky (1963) used the
    HORIZONTAL strain + a horizontal grid length; the full strain-rate tensor
    ``√(2·S_ij·S_ij)`` is the modern 3-D LES extension.  Vertical shear is not a
    surviving *component* of the horizontal deformation — it is a CHOSEN 1-D
    substitution for a single-column model (no horizontal strain is available).
  * **AD-safety numerics** (not Smagorinsky): the ``S² + 1e-10`` floor slightly
    perturbs ``K_m`` and ``Ri`` even where mixing is active (negligibly at resolved
    shear), and the double-``where`` at the Lilly-type cutoff yields a finite SELECTED
    reverse-mode cotangent at ``Ri = Pr_t``.  This is the ``differentiable: True``
    AD-safety sense — the forward is continuous but NON-C¹ at the cutoff (the on-side
    √ slope diverges); a bare ``√(max(·,0))`` would instead leak a 0·∞ NaN.
Behavior/assembly pins: ``tests/atmosphere/hydrostatic/unit/test_smagorinsky_faithful.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    deardorff_stable_eddy_viscosity,
    lilly_buoyancy_factor,
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
        "and vanish where Ri >= Pr_t (Lilly-type stable cutoff). shflx, lhflx are "
        "positive UPWARD from the surface and are injected as the lower "
        "boundary condition (a source/sink), so the resolved column budget is "
        "NOT closed. z increases upward; level index -1 is the surface."
    ),
    "conserves": ["none"],
    # AD-safe (finite selected reverse-mode VJP via the double-where guard); NOT a
    # claim of a C1-smooth forward -- the Lilly-type cutoff is a non-C1 point at
    # Ri = Pr_t (the on-side sqrt ramp's slope diverges).
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
    """Compute turbulence tendencies from the deformation/stability-dependent eddy diffusivity.

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

    # Resolved deformation: the vertical shear of the horizontal wind is
    # CHOSEN as a 1-D proxy for Smagorinsky's horizontal deformation (a
    # single column carries no horizontal strain).  Floor S2 so both
    # ``S = √S2`` and ``Ri = N²/S2`` stay finite and differentiable.
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

    # Lilly (1962) buoyancy stability factor √(max(0, 1 − Ri/Pr_t)) — the √
    # form is Lilly's equilibrium result (K_h/K_m = 1, K_m→0 for Ri>1); the
    # AD-safe double-``where`` cutoff (finite cotangent at Ri = Pr_t) lives in the
    # shared helper (with the S²+1e-10 floor above keeping Ri finite). Enhances
    # mixing when unstable (Ri<0), shuts it off at Ri ≥ Pr_t.
    # Stable-stratification treatment.  Dispatch on the STATIC config value at
    # function entry (never a silent ``else``): an unknown value must not run
    # different physics under a typo.
    if config.stability_form == "lilly":
        f_buoy = lilly_buoyancy_factor(Ri, config.Pr_t)
        # K_m = (C_s · l)^2 · |S| · f_buoy ;  K_h = K_m / Pr_t.
        Km_half = (config.C_s * l_mix) ** 2 * S * f_buoy    # (ncol, nlev-1)
    elif config.stability_form == "deardorff":
        # Deardorff (1980) stable-length limit — stratification shrinks the
        # mixing length instead of driving a stability factor to zero.  Takes
        # the RAW S² (the N²/Pr_t subtraction happens inside the helper);
        # passing the already-corrected strain would apply it twice.
        Km_half = deardorff_stable_eddy_viscosity(
            S2, N2, l_mix, config.C_s, config.Pr_t)
    else:
        raise ValueError(
            f"Unknown SmagorinskyConfig.stability_form "
            f"{config.stability_form!r}; expected 'lilly' or 'deardorff'."
        )
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
