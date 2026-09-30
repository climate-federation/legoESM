"""CLUBB-lite — simplified higher-order turbulence closure.

Carries five prognostic second-order moments and diagnoses cloud fraction
from a Gaussian PDF of the saturation deficit, following the approach of
Golaz et al. (2002) and Larson & Golaz (2005).

Prognostic moment (single)
--------------------------
Only ``wp2`` is carried as a prognostic moment.  An earlier version also
carried thlp2, rtp2, wpthlp, wprtp and an assumed-PDF cloud fraction, but
those higher moments and the cloud-fraction block were dead (never returned
through ``TurbulenceOutput``) and were removed (iter-172); cloud fraction is
handled downstream by the Sundqvist / Xu-Randall cloud scheme.

  wp2 = w'^2   vertical-velocity variance, integrated here as a TKE-like
               magnitude (see the faithfulness note on the budget below).

Budget (full levels, semi-implicit dissipation) — AS ACTUALLY INTEGRATED:

  d(wp2)/dt = Km*S^2 - Kh*N^2 - C_eps*wp2^(3/2)/l + diff(wp2)

The sink is written ``C_eps*wp2^(3/2)/l`` because the semi-implicit update
``wp2_new = (wp2 + dt*P)/(1 + dt*diss_wp2)`` with ``diss_wp2 = C_eps*sqrt(wp2)/l``
is the discretisation of ``d(wp2)/dt = P - diss_wp2*wp2 = P - C_eps*wp2^(3/2)/l``;
the linearised rate ``diss_wp2`` multiplies ``wp2``, so the budget term carries
the full ``wp2^(3/2)`` power (see ``c_eps_from_budget`` in
``dynamics/les_closure_diagnosis.py``, the exact inverse of this balance).

NOTE on faithfulness: the canonical CLUBB w'^2 *variance* budget carries a
factor of 2 on the production terms, a buoyancy production
``2*(g/theta_v)*w'theta_v'`` and a ``C1*wp2/tau`` dissipation.  This lite
version instead integrates a TKE-scaled magnitude: shear production
``Km*S^2`` (no factor 2), a down-gradient buoyancy surrogate ``-Kh*N^2``
(in place of ``2*(g/theta_v)*w'theta_v'``), and a ``C_eps*wp2^(3/2)/l``
dissipation (linearised rate ``C_eps*sqrt(wp2)/l`` times ``wp2``).  ``wp2`` is
therefore a TKE-like scale used only to set the
mixing time scale and the down-gradient diffusivities, NOT a strict second
moment.

where tau = l / sqrt(wp2) is the turbulence time scale.

Eddy diffusivities:
  Km = C_K * l * sqrt(wp2)
  Kh = Km / Pr_t

Fidelity vs the full CLUBB (clubb_intr.F90 + CLUBB core)
--------------------------------------------------------
This module is a DELIBERATELY REDUCED surrogate, NOT a port of the
operational CLUBB used in E3SM/CAM.  It is intentionally ~300 lines versus
CLUBB's ~10k lines of higher-order closure.  What clubb_lite approximates,
and where it diverges from the real CLUBB, is stated precisely so callers do
not mistake it for a faithful CLUBB:

WHAT IT KEEPS (qualitatively CLUBB-like):
  * One prognostic second-order moment, ``wp2`` (w'^2), with a
    production - dissipation - diffusion budget (integrated TKE-like; see the
    budget note above), used to set the turbulence mixing time scale.
  * Down-gradient eddy diffusivities ``Km = C_K*l*sqrt(wp2)``, ``Kh =
    Km/Pr_t``.
  (The earlier thlp2/rtp2/wpthlp/wprtp moments and the single-Gaussian PDF
  cloud-fraction diagnosis were dead code and were removed; cloud fraction
  comes from the downstream Sundqvist / Xu-Randall scheme.)

WHAT IT OMITS / SIMPLIFIES (the fidelity gap vs full CLUBB):
  1. PDF shape: CLUBB uses an Analytic Double Gaussian (ADG1) joint PDF of
     (w, theta_l, r_t) closing higher moments (skewness via wp3).  Here the
     PDF is a SINGLE Gaussian in s only -> no skewness, no third moments,
     so no proper updraft/downdraft asymmetry or cumulus-shaped clouds.
  2. wp3 (third moment) is NOT carried -> no skewness-driven nonlocal /
     counter-gradient transport that distinguishes CLUBB from a 2nd-order
     down-gradient scheme.
  3. Pressure terms / return-to-isotropy use the single ``C_eps``
     dissipation instead of CLUBB's full pressure-correlation closure
     with the C-coefficient hierarchy (C2, C6, C7, C8, C11, C14, ...).
  4. No subgrid cloud-water (rcm) feedback into buoyancy production, no
     SILHS sub-columns, no cloud-top radiative/evaporative entrainment
     enhancement, no monotonic flux limiters, no implicit moment matrix
     solve coupling all five moments (each moment is integrated with a
     semi-implicit local dissipation here).
  5. Length scale is the Blackadar master length, not CLUBB's
     Lscale computed from up/down parcel buoyant-sorting integrals.
  6. Diffusivities are diagnostic from wp2; CLUBB transports the moments
     themselves with the closure flux ``w'x' = -K dx/dz + (PDF terms)``.

NET: clubb_lite reproduces the gross structure of a moist 2nd-order TKE-like
closure with a PDF cloud diagnosis, useful as a differentiable, cheap
surrogate, but it does NOT reproduce CLUBB's skewness-based nonlocal
transport, double-Gaussian cloud structure, or operational coefficient set.
Porting full CLUBB is explicitly out of scope; use this only where a reduced
higher-order surrogate is acceptable.

References
----------
- Golaz, J.-C., Larson, V. E., & Cotton, W. R. (2002). A PDF-based
  model for boundary layer clouds. Part I: Method and model description.
  J. Atmos. Sci., 59, 3540-3551.
- Larson, V. E., & Golaz, J.-C. (2005). Using assumed probability
  distribution functions for HoC. J. Atmos. Sci., 62, 3620-3649.
- Larson, V. E. (2022). CLUBB-SILHS: A parameterization of subgrid
  variability in the atmosphere. arXiv:1711.03675 (full-CLUBB reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import (
    broadcast_column_param,
    buoyancy_coefficient,
    exner_function,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
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

from legoesm import constants

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "CLUBB-lite reduced higher-order turbulence closure: one prognostic "
        "moment wp2 (w'^2) sets a TKE-like mixing scale for down-gradient eddy "
        "diffusivities Km, Kh that mix u, v, T (in theta-space), q_v implicitly."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "tke": "m^2/s^2 (carries wp2 = w'^2)",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "tke_new": "m^2/s^2 (updated wp2)",
    },
    "sign_convention": (
        "Down-gradient eddy diffusion, Km >= 0, Kh = Km/Pr_t >= 0; the "
        "tendencies relax the mean state toward a well-mixed profile. The "
        "column budget is OPEN: the surface flux (shflx > 0 upward, lhflx > 0 "
        "upward/moistening) is injected as the bottom boundary condition and "
        "the top is zero-flux; z increases upward."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Golaz, Larson & Cotton (2002), J. Atmos. Sci. 59, 3540-3551; "
        "Larson & Golaz (2005), J. Atmos. Sci. 62, 3620-3649"
    ),
    "idealized_test": (
        "rest state with zero surface flux and a well-mixed neutral column -> "
        "near-zero interior tendency; Km, Kh >= 0; wp2 stays >= tke_min."
    ),
}

def clubb_eddy_diffusivity(
    C_K: jax.Array,
    l_mix: jax.Array,
    sqrt_wp2: jax.Array,
) -> jax.Array:
    """The CLUBB-lite down-gradient momentum diffusivity ``K_m = C_K · ℓ · √wp2``.

    This is the SINGLE definition of the forward eddy-diffusivity closure: the integrator
    (:func:`clubb_lite_turbulence`) builds ``K_m`` from it, and it is the EXACT inverse of the
    LES diagnosis ``C_K = K_m/(ℓ·√wp2)``
    (:func:`legoesm.atmosphere.dynamics.les.les_closure_diagnosis.clubb_coefficient_from_diffusivity`).
    Factored out so the forward/inverse round-trip is pinned against the closure the model
    ACTUALLY integrates (a change to this form is then caught by the round-trip test, not
    silently de-synced from the diagnosis — which would break OSSE parameter recovery).

    ``C_K`` may be a scalar (production) OR a per-column ``(ncol,)`` field (the LES-informed
    eddy-diffusivity correction); ``broadcast_column_param`` keeps the scalar path
    byte-identical and reshapes a per-column field to broadcast over the vertical.  All inputs
    are pure arrays — JAX-traced, differentiable, no side effects.
    """
    return broadcast_column_param(C_K, l_mix) * l_mix * sqrt_wp2


def clubb_heat_diffusivity(
    K_m: jax.Array,
    Pr_t: jax.Array,
    l_mix: jax.Array,
) -> jax.Array:
    """The CLUBB-lite heat/scalar diffusivity ``K_h = K_m / Pr_t`` (turbulent Prandtl number).

    The SINGLE forward definition the integrator builds ``K_h`` from; the EXACT inverse of the
    LES diagnosis ``Pr_t = K_m/K_h``
    (:func:`legoesm.atmosphere.dynamics.les.les_closure_diagnosis.prandtl_number_from_diffusivities`).
    ``Pr_t`` may be a scalar (production) OR a per-column ``(ncol,)`` field (the LES-informed
    correction); ``l_mix`` supplies the column-broadcast shape only.  Pure / differentiable.
    """
    return K_m / broadcast_column_param(Pr_t, l_mix)


def clubb_wp2_production(
    K_m: jax.Array,
    K_h: jax.Array,
    S2: jax.Array,
    N2: jax.Array,
) -> jax.Array:
    """Net ``w'²`` production ``P = K_m·S² − K_h·N²`` (down-gradient shear minus buoyancy
    destruction) — the SINGLE forward definition the integrator's ``wp2`` budget balances, and
    the production the LES diagnosis inverts in
    :func:`legoesm.atmosphere.dynamics.les.les_closure_diagnosis.c_eps_from_budget`
    (``C_eps = P·ℓ/wp2^{3/2}``).  Co-located inputs; pure / differentiable.
    """
    return K_m * S2 - K_h * N2


def clubb_wp2_dissipation_rate(
    C_eps: jax.Array,
    sqrt_wp2: jax.Array,
    l_mix_safe: jax.Array,
) -> jax.Array:
    """The CLUBB-lite ``w'²`` dissipation RATE ``C_eps·√wp2/ℓ`` (so the dissipation TERM is
    ``rate·wp2 = C_eps·wp2^{3/2}/ℓ``).  The SINGLE forward definition the integrator's
    semi-implicit ``wp2`` update uses; at steady state (neglecting transport) it balances the
    production, the relation the LES diagnosis inverts
    (:func:`legoesm.atmosphere.dynamics.les.les_closure_diagnosis.c_eps_from_budget`).  ``C_eps``
    may be a scalar OR a per-column ``(ncol,)`` field; ``l_mix_safe`` is the floored mixing
    length (the integrator clips ℓ ≥ 1 m before the division).  Pure / differentiable.
    """
    return broadcast_column_param(C_eps, l_mix_safe) * sqrt_wp2 / l_mix_safe


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clubb_lite_turbulence(
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
    config: CLUBBLiteConfig,
    surface_flux: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | None = None,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute turbulence tendencies using CLUBB-lite higher-order closure.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    tke : jax.Array
        Vertical velocity variance w'^2 [m^2/s^2], shape (ncol, nlev).
        (Re-uses the TKE array slot for wp2.)
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
    config : CLUBBLiteConfig
    surface_flux : tuple of jax.Array, optional
        Pre-computed surface fluxes ``(tau_x, tau_y, shflx, lhflx, ustar)``,
        each shape ``(ncol,)``, used as the BL bottom boundary condition in
        place of the single-surface ``compute_surface_fluxes`` call — the same
        contract Louis honours, so the driver can inject the AREA-WEIGHTED tiled
        (mosaic) surface flux (COARE3 on the ocean tile, land Monin-Obukhov on
        the land tile) instead of running one bulk scheme on the blended surface
        temperature.  When ``None`` (default) the legacy single-surface flux is
        computed from ``T_sfc``/``q_sfc``/``config.surface`` (identical
        behaviour).  Same units/sign convention as ``compute_surface_fluxes``
        (``tau`` [Pa], ``shflx``/``lhflx`` [W/m^2], ``ustar`` [m/s]).

    Returns
    -------
    TurbulenceOutput
        Turbulence tendencies and diagnostics.
    tke_new : jax.Array
        Updated w'^2 [m^2/s^2], shape (ncol, nlev).
    """
    ncol, nlev = T.shape

    # ===== Geometry =====
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    # ===== Initialize prognostic moments =====
    # wp2 is carried via the tke slot
    wp2 = jnp.maximum(tke, config.tke_min)

    # Higher moments initialized from wp2 (cold start: proportional to wp2)
    # In a full implementation these would be carried as separate state arrays.
    # For "lite" version: diagnose thlp2, rtp2, wpthlp, wprtp from gradients.
    sqrt_wp2 = jnp.sqrt(wp2)

    # ===== Mixing length =====
    l_mix = mixing_length(z_full, config.l_mix_max)  # (ncol, nlev)
    l_mix_safe = jnp.clip(l_mix, 1.0, None)

    # iter-172 F841: removed unused ``tau_turb`` (consumed
    # only by the dead higher-moment / cloud-fraction block
    # — see comment block below).

    # ===== Eddy diffusivities =====
    # ``C_K`` may be a scalar (production) OR a per-column ``(ncol,)`` field
    # (the LES-informed eddy-diffusivity correction); broadcast_column_param
    # keeps the scalar path byte-identical and reshapes a per-column field to
    # broadcast over the vertical axis. See docs/COMPARE_REANALYSIS.md.
    Km_full = clubb_eddy_diffusivity(config.C_K, l_mix, sqrt_wp2)  # (ncol, nlev)
    # ``Pr_t`` likewise may be a scalar (production, byte-identical) OR a
    # per-column LES-informed correction (the turbulent Prandtl number
    # Pr_t = K_m/K_h diagnosed from the LES); broadcast it over the vertical too.
    Kh_full = clubb_heat_diffusivity(Km_full, config.Pr_t, l_mix)

    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])  # (ncol, nlev-1)
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    # ===== Vertical gradients at half-levels =====
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2  # (ncol, nlev-1)

    # Virtual potential temperature for buoyancy. theta_v = T_v / exner, with the
    # canonical exner helper (exner_pref = 1/Π = (p_ref/p)^κ).
    exner_pref = 1.0 / exner_function(jnp.clip(p_full, 1.0, None))
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    # N² = (g/θ_v)·∂θ_v/∂z via the shared buoyancy coefficient (clip at call site).
    N2_half = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz

    # iter-172 F841: removed ``exner`` / ``theta`` /
    # ``dtheta_dz`` / ``drt_dz`` — only consumed by the dead
    # higher-moment + cloud-fraction block (see comment after
    # the eddy-diffusivity section).

    # Interpolate to full levels
    def _half_to_full(field_half):
        """Interpolate (ncol, nlev-1) half-level field to (ncol, nlev)."""
        mid = 0.5 * (field_half[:, :-1] + field_half[:, 1:])
        return jnp.concatenate([
            field_half[:, :1], mid, field_half[:, -1:]
        ], axis=1)

    S2 = _half_to_full(S2_half)
    N2 = _half_to_full(N2_half)
    # iter-172 F841: removed unused ``dtheta_dz_full`` /
    # ``drt_dz_full`` (consumed only by the removed dead
    # higher-moment + cloud-fraction block).

    # ===== Moment budgets (semi-implicit) =====

    # --- w'^2 budget ---
    # Net production P = K_m·S² − K_h·N² (shear minus buoyancy destruction); the
    # diagnosis inverts EXACTLY this form (c_eps_from_budget).
    net_prod = clubb_wp2_production(Km_full, Kh_full, S2, N2)

    # Dissipation coefficient: ``C_eps``-based (semi-implicit). ``C_eps`` may be a scalar
    # (production, byte-identical) OR a per-column LES-informed correction (it sets
    # the GCM's equilibrium wp2 so it tracks the LES w'² — closing the C_K
    # wp2-identification gap); broadcast it over the vertical like C_K / Pr_t.
    diss_wp2 = clubb_wp2_dissipation_rate(config.C_eps, sqrt_wp2, l_mix_safe)

    # Diffuse wp2.  Pin the surface_flux dtype to the input dtype so the
    # tridiagonal solve does not silently promote the column path to f64.
    wp2_diffused = implicit_vertical_diffusion(
        wp2, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=wp2.dtype),
    )

    # Semi-implicit update
    wp2_new = (wp2_diffused + dt * net_prod) / (
        1.0 + dt * diss_wp2
    )
    wp2_new = jnp.maximum(wp2_new, config.tke_min)

    # iter-172 (F841 audit): removed dead higher-moment +
    # cloud-fraction diagnostics that were computed every
    # timestep but never returned through ``TurbulenceOutput``.
    # The dead blocks were:
    #   * ``wpthlp = -Kh_full * dtheta_dz_full`` (line ~224)
    #   * ``wprtp  = -Kh_full * drt_dz_full``    (line ~225)
    #   * ``thlp2`` / ``rtp2`` flux-gradient variances (used
    #     only inside the discarded cloud-fraction block)
    #   * ``sigma_s = sqrt(rtp2)`` (overridden by sigma_s_eff)
    #   * Full Gaussian-PDF cloud fraction with ``erfc`` /
    #     ``q_sat_val`` / ``s_mean`` / ``dqs_dT`` /
    #     ``sigma_s_eff`` (lines ~249-268).
    #
    # These are real physics that someone intended to wire up
    # to a unified CLUBB-style cloud scheme, but the
    # ``TurbulenceOutput`` NamedTuple has never carried
    # ``cloud_fraction`` and downstream cloud-scheme dispatch
    # uses Sundqvist / Xu-Randall instead.  Removing the dead
    # computations saves ~6 jnp ops + 1 ``erfc`` per timestep
    # per AMIP-active column with no behavioural change.
    #
    # If a future CLUBB unified scheme is wired in, restore
    # by:
    #   1. Adding ``cloud_fraction: jax.Array`` to
    #      ``TurbulenceOutput`` (output.py).
    #   2. Re-introducing the PDF block here.
    #   3. Wiring the cloud scheme dispatch to consume
    #      ``turbulence_output.cloud_fraction`` when
    #      ``cloud_scheme == "clubb"``.
    # The block was: ``s_mean = q_v - _q_sat(T, p_full);
    # sigma_s_eff = sqrt(rtp2 + (dqs_dT * exner)**2 * thlp2);
    # cloud_fraction = 0.5 * erfc(-s_mean / (sqrt(2) *
    # sigma_s_eff))``.

    # ===== Surface fluxes =====
    # Either the driver-supplied tiled (mosaic) flux or the legacy single-surface
    # bulk flux from the blended T_sfc (mirrors louis_turbulence).
    if surface_flux is not None:
        tau_x, tau_y, shflx, lhflx, ustar = surface_flux
    else:
        tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
            u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
            T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd

    # ===== Apply implicit vertical diffusion =====
    # Heat in θ-space (dry-adiabat neutral); momentum and moisture raw.
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # ===== PBL height =====
    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
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

    return output, wp2_new
