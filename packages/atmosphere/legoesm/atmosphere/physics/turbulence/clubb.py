"""CLUBB higher-order turbulence closure (fuller port; ``scheme="clubb"``).

This is the operational entry point for the fuller CLUBB port tracked in
``PORT_CLUBB.md`` — substantially richer than :mod:`clubb_lite`. The supporting
machinery (all golden-locked or parity-tested against CLUBB-JAX) is built up in
the ``clubb_*.py`` modules: the staggered grid (:mod:`clubb_grid`), CAM-default
config (:mod:`clubb_config`), Flatau saturation (:mod:`clubb_saturation`),
``sigma_sqd_w``/Brunt-Vaisala (:mod:`clubb_helpers`), skewness diagnostics
(:mod:`clubb_skewness`), the parcel buoyant-sorting mixing length
(:mod:`clubb_mixing_length`), the ADG1 assumed-PDF cloud/buoyancy closure
(:mod:`clubb_pdf`, :mod:`clubb_pdf_moments`), the implicit solvers
(:mod:`clubb_solve`), and the moment advances (:mod:`clubb_moments`).

Phasing (the scheme is wired in and runnable now; fidelity deepens per phase):

  * **Phase 1 (this entry):** uses CLUBB's exact **parcel buoyant-sorting
    length scale** ``Lscale`` (``compute_mixing_length``, golden-locked vs
    CLUBB-JAX) to set the eddy diffusivity ``Km = c_K · Lscale · sqrt(em)`` —
    the distinctive CLUBB feature, replacing clubb_lite's Blackadar length — and
    diagnoses the cloud fraction from the ADG1 PDF. Mean fields (u, v, T, q_v)
    are advanced by the implicit eddy diffusion shared with the other legoESM
    schemes; ``wp2`` is carried (via the ``tke`` slot) with a production /
    dissipation / diffusion budget whose dissipation time scale is
    ``tau = Lscale / sqrt(em)``.
  * **Phase 2+ (in progress):** replace the diagnostic moment budget with the
    full prognostic higher-order moment transport — ``advance_wp2_wp3``,
    ``advance_xp2_xpyp``, ``advance_xm_wpxp`` — coupled through the ADG1 PDF
    buoyancy flux ``wpthvp`` and the implicit tridiag/penta solves.

The ``TurbulenceOutput`` contract carries no ``cloud_fraction`` field (see the
clubb_lite docstring), so the PDF cloud fraction is computed and exposed only
through diagnostics for now; the eddy-diffusion tendencies are the live output.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig, derive_lmin
from legoesm.atmosphere.physics.turbulence.clubb_core import (
    CLUBBForcing,
    CLUBBMomentState,
    advance_clubb_core,
    init_clubb_moments,
    pack_clubb_moments,
    unpack_clubb_moments,
)
from legoesm.atmosphere.physics.turbulence.clubb_diagnostic import (
    diagnose_cloud_and_buoyancy,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import (
    flip_vertical,
    make_clubb_grid_from_levels,
    zt2zm,
)
from legoesm.atmosphere.physics.turbulence.clubb_helpers import (
    calc_brunt_vaisala_freq_sqd,
)
from legoesm.atmosphere.physics.turbulence.clubb_mixing_length import (
    compute_mixing_length,
    set_Lscale_max,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

from legoesm import constants

# Eddy-diffusivity and dissipation coefficients are read from CLUBBParams
# (c_K, beta, ...) — no hardcoded tunables here.
_PR_T = 1.0   # phase-1 turbulent Prandtl number (Kh = Km/_PR_T); refined in P2


def clubb_turbulence(
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
    config: CLUBBConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """CLUBB turbulence tendencies (phase 1: CLUBB ``Lscale`` eddy diffusion).

    Parameters mirror :func:`clubb_lite.clubb_lite_turbulence` (the ``tke`` slot
    carries ``wp2`` [m^2/s^2]); all column fields are TOP-DOWN ``(ncol, nlev)``,
    half-level fields ``(ncol, nlev+1)``.

    Returns
    -------
    TurbulenceOutput
        Tendencies + diffusivities + surface diagnostics.
    wp2_new : jax.Array
        Updated ``w'^2`` [m^2/s^2], ``(ncol, nlev)``, carried to the next step.
    """
    ncol, nlev = T.shape
    params = config.params

    wp2 = jnp.maximum(tke, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    # ---- Thermodynamics (top-down) ----
    exner = exner_function(p_full)                            # (ncol, nlev)
    theta = T / exner
    theta_v = virtual_temperature(T, q_v) / exner             # = theta * (1 + 0.61 q_v)
    # Dry phase-1 mapping to CLUBB variables (rcm = 0): thl ~ theta, rt ~ q_v.
    thlm_td = theta
    rtm_td = q_v
    thvm_td = theta_v
    thv_ds_td = theta_v

    # ---- CLUBB parcel buoyant-sorting mixing length (ascending grid) ----
    gr = make_clubb_grid_from_levels(z_full, z_half)
    thvm = flip_vertical(thvm_td)
    thlm = flip_vertical(thlm_td)
    rtm = flip_vertical(rtm_td)
    exner_a = flip_vertical(exner)
    p_a = flip_vertical(p_full)
    thv_ds = flip_vertical(thv_ds_td)
    # TKE on momentum (zm) levels from the carried wp2 (ascending zt -> zm).
    em_zm = jnp.maximum(zt2zm(flip_vertical(wp2), gr), config.tke_min)

    mu = jnp.full((ncol,), params.mu)
    lmin = derive_lmin(params.lmin_coef)
    Lscale_max = set_Lscale_max(False, None, None, ncol)
    Lscale_a, _, _ = compute_mixing_length(
        thvm, thlm, rtm, em_zm, Lscale_max, p_a, exner_a, thv_ds,
        mu, lmin, False, gr,
    )
    Lscale = flip_vertical(Lscale_a)                          # back to top-down (ncol, nlev)
    Lscale = jnp.clip(Lscale, 1.0, None)

    # ---- Eddy diffusivities from the CLUBB length scale ----
    Km_full = params.c_K * Lscale * sqrt_wp2                  # (ncol, nlev)
    Kh_full = Km_full / _PR_T

    # ---- ADG1 double-Gaussian PDF: cloud fraction + moist buoyancy flux ----
    # (the distinctive CLUBB closure; ascending grid). The buoyancy production of
    # wp2 uses the PDF flux ``w'thv'`` (with its cloud-water latent-heat term),
    # not a plain down-gradient ``-Kh N2``.
    wp2_a = jnp.maximum(flip_vertical(wp2), config.tke_min)   # ascending zt
    Kh_a = (params.c_K / _PR_T) * Lscale_a * jnp.sqrt(wp2_a)
    cloud_frac_a, rcm_a, wpthvp_a = diagnose_cloud_and_buoyancy(
        thlm, rtm, wp2_a, exner_a, p_a, thv_ds, Kh_a, Lscale_a, gr, config)
    buoy_prod = flip_vertical(buoyancy_coefficient(jnp.clip(thvm, 1.0, None)) * wpthvp_a)

    # ---- Geometry + shear (top-down) ----
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2

    def _half_to_full(field_half):
        mid = 0.5 * (field_half[:, :-1] + field_half[:, 1:])
        return jnp.concatenate([field_half[:, :1], mid, field_half[:, -1:]], axis=1)

    S2 = _half_to_full(S2_half)

    # ---- wp2 budget (production - dissipation + diffusion); tau = Lscale/sqrt(wp2) ----
    shear_prod = Km_full * S2
    diss_wp2 = sqrt_wp2 / Lscale                              # 1/tau
    wp2_diffused = implicit_vertical_diffusion(
        wp2, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=wp2.dtype),
    )
    wp2_new = (wp2_diffused + dt * (shear_prod + buoy_prod)) / (1.0 + dt * diss_wp2)
    wp2_new = jnp.clip(wp2_new, config.tke_min, config.wp2_max)

    # ---- Surface fluxes ----
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )
    sflx_u, sflx_v = tau_x, tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ---- Implicit vertical diffusion of the mean state ----
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T)
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
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )
    return output, wp2_new


def clubb_step(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    moments: CLUBBMomentState,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBConfig,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, CLUBBMomentState, dict]:
    """Bridge one prognostic CLUBB step from legoESM top-down column inputs.

    This is the *full prognostic* path (phase 2): it builds the ascending-grid
    host environment CLUBB needs, sets the surface turbulent-flux lower boundary
    conditions, advances the complete higher-order moment set with
    :func:`clubb_core.advance_clubb_core`, and maps the advanced means back to
    top-down ``(ncol, nlev)`` tendencies. The 15-field :class:`CLUBBMomentState`
    (means on zt, moments/fluxes on zm, ``wp3`` zt) is carried in and out.

    Modelling choices (documented; refined as the scheme matures):

      * Dry phase mapping ``thl ~ theta``, ``rt ~ q_v`` (``rcm`` enters only via
        the PDF closure inside ``advance_clubb_core``).
      * Mean vertical velocity ``wm = 0`` (grid-scale subsidence is the dycore's
        job, not the column closure).
      * Geostrophic wind ``ug = um``, ``vg = vm`` and ``fcor = 0`` → the
        Coriolis/geostrophic term in ``advance_windm_edsclrm`` vanishes (rotation
        is handled by the dycore; the turbulence scheme only diffuses).
      * Dry-static reference profiles ``thv_ds``/``rho_ds`` taken as the current
        ``thv``/``rho`` (a per-column Boussinesq-style reference).
      * Large-scale ``CLUBBForcing`` = 0 (other physics supply those tendencies).

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each shape ``(ncol,)``): CLUBB's standard LES/SCM-intercomparison
    interface. When a component is given it OVERRIDES the bulk lower-BC for that
    moment with the prescribed **kinematic** surface flux (``w'thl'`` [K m/s],
    ``w'rt'`` [kg/kg m/s], ``u'w'``/``v'w'`` [m^2/s^2] — same units/sign as the
    internally-computed BCs); components left ``None`` fall back to CLUBB's own
    bulk formula from the air-surface contrast (``T_sfc``/``q_sfc``). The ``None``
    test is a compile-time static choice (CLAUDE.md feature-gating exception), not
    a traced selection. Cases that prescribe all four (BOMEX/DYCOMS/ARM) never
    touch the bulk formula, so the result is independent of ``T_sfc``/``q_sfc``.
    The reported ``shflx``/``lhflx``/``ustar`` diagnostics are made consistent
    with whichever BC was actually used.

    **Heat/moisture vs momentum semantics differ (important):** ``sfc_wpthlp``/
    ``sfc_wprtp`` enter ``advance_xm_wpxp`` directly as the scalar surface-flux
    lower-BC — applied EXACTLY and directionally (a prescribed ``w'thl'_sfc``
    closes the column θl budget to round-off). The momentum components are NOT
    applied as an independent ``(u'w', v'w')`` vector: CAM's
    ``l_imp_sfc_momentum_flux = .true.`` path (``advance_windm_edsclrm``) consumes
    ONLY the surface-stress-vector MAGNITUDE
    ``u_*^2 = sqrt(u'w'_sfc^2 + v'w'_sfc^2)`` (so ``u_* = (u'w'_sfc^2 +
    v'w'_sfc^2)^{1/4}``) and re-applies it implicitly as a drag ANTIPARALLEL to the
    near-surface wind (``-rho u_*^2 u/|V|``). So ``sfc_upwp``/``sfc_vpwp`` set only
    the stress magnitude (equivalently a prescribed ``u_*``); their azimuth is discarded —
    prescribing ``(u'w', 0)`` and ``(0, u'w')`` give identical wind tendencies.
    This is the correct contract for prescribed-``u_*`` LES forcing and is exact
    for the bulk drag (which is already wind-antiparallel by construction), but a
    cross-wind momentum-flux vector cannot be imposed through this interface.

    Returns ``(du_dt, dv_dt, dT_dt, dq_v_dt, new_moments, diagnostics)`` — the
    four mean tendencies (top-down ``(ncol, nlev)``), the advanced moment state,
    and the ``cloud_frac``/``rcm``/``wpthvp``/``Kh_*`` diagnostics dict.
    """
    ncol, nlev = T.shape
    params = config.params

    # ---- Thermodynamics (top-down) ----
    exner = exner_function(p_full)
    theta = T / exner
    thv = virtual_temperature(T, q_v) / exner

    # ---- Ascending CLUBB grid + means on zt ----
    gr = make_clubb_grid_from_levels(z_full, z_half)
    thlm = flip_vertical(theta)          # thl ~ theta (zt)
    rtm = flip_vertical(q_v)             # rt ~ q_v   (zt)
    um = flip_vertical(u)
    vm = flip_vertical(v)
    exner_zt = flip_vertical(exner)
    p_zt = flip_vertical(p_full)
    thv_zt = flip_vertical(thv)

    # ---- Dry-static reference + density profiles (zt and zm) ----
    thv_ds_zt = thv_zt
    thv_ds_zm = zt2zm(thv_zt, gr)
    rho_ds_zt = flip_vertical(rho)
    rho_ds_zm = zt2zm(rho_ds_zt, gr)
    invrs_rho_ds_zt = 1.0 / rho_ds_zt
    invrs_rho_ds_zm = 1.0 / rho_ds_zm

    # ---- Brunt-Vaisala N^2 (dry CAM-default form; rcm/ice unused there) ----
    zeros_zt = jnp.zeros((ncol, nlev), dtype=T.dtype)
    brunt = calc_brunt_vaisala_freq_sqd(
        thlm, exner_zt, rtm, zeros_zt, p_zt, zeros_zt,
        params.bv_efold, config.T0, gr)[0]

    # ---- Parcel buoyant-sorting mixing length ----
    em_zm = jnp.maximum(
        0.5 * (moments.wp2 + moments.up2 + moments.vp2), config.tke_min)  # l_tke_aniso
    mu = jnp.full((ncol,), params.mu)
    lmin = derive_lmin(params.lmin_coef)
    Lscale_max = set_Lscale_max(False, None, None, ncol)
    Lscale, _, _ = compute_mixing_length(
        thv_zt, thlm, rtm, em_zm, Lscale_max, p_zt, exner_zt, thv_ds_zt,
        mu, lmin, False, gr)
    # Enforce the physical minimum mixing length lmin (compute_mixing_length can
    # return < lmin; CLUBB floors it). A strictly positive Lscale keeps the
    # dissipation time tau = Lscale/sqrt(em) finite (invrs_tau = 1/tau).
    Lscale = jnp.maximum(Lscale, lmin)

    # ---- Surface turbulent-flux lower boundary conditions (kinematic) ----
    rho_sfc = rho[:, -1]
    exner_sfc = exner[:, -1]
    # Each BC component is either prescribed (LES/SCM intercomparison cases) or
    # computed by CLUBB's own bulk formula from the air-surface contrast. The
    # bulk formula is evaluated only if at least one component still needs it
    # (static Python branch on None-ness — never a traced selection).
    need_bulk = (sfc_wpthlp is None or sfc_wprtp is None
                 or sfc_upwp is None or sfc_vpwp is None)
    if need_bulk:
        tau_x, tau_y, shflx_b, lhflx_b, ustar_b = compute_surface_fluxes(
            u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
            T_sfc, q_sfc, rho_sfc, config.surface)
        wpthlp_b = shflx_b / (rho_sfc * constants.c_pd * exner_sfc)  # w'thl' [K m/s]
        wprtp_b = lhflx_b / (rho_sfc * constants.L_v)                # w'rt'  [kg/kg m/s]
        # Surface stress convention is tau = -rho*Cd*|V|*u (compute_surface_fluxes),
        # so the kinematic momentum flux is u'w'_sfc = tau_x/rho (NEGATIVE for u>0 —
        # momentum transported downward / drag), NOT -tau_x/rho.
        upwp_b = tau_x / rho_sfc                                     # u'w'   [m^2/s^2]
        vpwp_b = tau_y / rho_sfc
    # Resolve each BC: prescribed kinematic flux when given, else the bulk value.
    wpthlp_sfc = wpthlp_b if sfc_wpthlp is None else sfc_wpthlp
    wprtp_sfc = wprtp_b if sfc_wprtp is None else sfc_wprtp
    upwp_sfc = upwp_b if sfc_upwp is None else sfc_upwp
    vpwp_sfc = vpwp_b if sfc_vpwp is None else sfc_vpwp
    # W/m^2 + ustar diagnostics consistent with the BC actually used: the bulk
    # values pass through unchanged (exact back-compat); prescribed kinematic
    # fluxes are converted back to W/m^2, and ustar from the prescribed stress
    # (|tau|/rho = sqrt(u'w'^2 + v'w'^2), so ustar = (u'w'^2 + v'w'^2)^(1/4)).
    # The 1e-30 inside the fourth root is a pure AD safety floor: at the valid
    # zero-stress prescribed BC (sfc_upwp=sfc_vpwp=0) the bare (.)**0.25 has an
    # +inf slope, so jax.grad of an objective through ustar would be non-finite;
    # the floor pins the gradient to 0 there while leaving any physical stress
    # (|u'w'| >> 1e-8) bit-unchanged.
    shflx = shflx_b if sfc_wpthlp is None else (
        sfc_wpthlp * (rho_sfc * constants.c_pd * exner_sfc))
    lhflx = lhflx_b if sfc_wprtp is None else sfc_wprtp * (rho_sfc * constants.L_v)
    ustar = ustar_b if (sfc_upwp is None and sfc_vpwp is None) else (
        jnp.maximum(upwp_sfc ** 2 + vpwp_sfc ** 2, 1e-30) ** 0.25)
    # The mean state (rtm/thlm/um/vm) is owned by the model and re-read from the
    # live column each step; only the higher-order moments/fluxes persist in
    # ``moments``. Reset the means here, then set the surface flux BCs.
    state = moments._replace(
        rtm=rtm, thlm=thlm, um=um, vm=vm,
        wprtp=moments.wprtp.at[:, 0].set(wprtp_sfc),
        wpthlp=moments.wpthlp.at[:, 0].set(wpthlp_sfc),
        upwp=moments.upwp.at[:, 0].set(upwp_sfc),
        vpwp=moments.vpwp.at[:, 0].set(vpwp_sfc))

    # ---- Advance the full prognostic moment set ----
    zeros_zm = jnp.zeros((ncol, nlev + 1), dtype=T.dtype)
    forcing = CLUBBForcing(
        rtm=zeros_zt, thlm=zeros_zt, um=zeros_zt, vm=zeros_zt, wprtp=zeros_zm,
        wpthlp=zeros_zm, rtp2=zeros_zm, thlp2=zeros_zm, rtpthlp=zeros_zm)
    sfc_elevation = flip_vertical(z_half)[:, 0]
    new_state, diags = advance_clubb_core(
        state, forcing, Lscale=Lscale, brunt_vaisala_freq_sqd=brunt,
        exner_zt=exner_zt, p_in_Pa_zt=p_zt, thv_ds_zt=thv_ds_zt,
        thv_ds_zm=thv_ds_zm, rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
        invrs_rho_ds_zm=invrs_rho_ds_zm, invrs_rho_ds_zt=invrs_rho_ds_zt,
        wm_zt=zeros_zt, wm_zm=zeros_zm, sfc_elevation=sfc_elevation,
        fcor=jnp.zeros((ncol,), dtype=T.dtype), ug=um, vg=vm, dt=dt, gr=gr,
        config=config)

    # ---- Map advanced means back to top-down tendencies ----
    u_new = flip_vertical(new_state.um)
    v_new = flip_vertical(new_state.vm)
    T_new = flip_vertical(new_state.thlm) * exner    # thl ~ theta -> T = theta*exner
    q_new = flip_vertical(new_state.rtm)
    du_dt = (u_new - u) / dt
    dv_dt = (v_new - v) / dt
    dT_dt = (T_new - T) / dt
    dq_v_dt = (q_new - q_v) / dt
    diags = dict(diags, ustar=ustar, shflx=shflx, lhflx=lhflx)
    return du_dt, dv_dt, dT_dt, dq_v_dt, new_state, diags


def clubb_turbulence_prognostic(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    clubb_moments: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBConfig,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Prognostic CLUBB scheme entry (``scheme="clubb"``, ``prognostic=True``).

    Drop-in for :func:`clubb_turbulence` with the SAME carry-slot interface — the
    carried state is the packed :class:`CLUBBMomentState` ``(ncol, 15, nlev+1)``
    (``PhysicsState.clubb_moments``) instead of the single ``wp2`` slot. Unpacks
    it, advances the full higher-order moment closure, and repacks the new moments
    as the carry. No host numerical diffusion is added here: in a coupled run the
    dynamical core supplies it (the bare-SCM stand-in lives in
    :func:`integrate_clubb_column`).

    **CLUBB sub-cycling (CAM fidelity):** CAM runs CLUBB at its own
    ``clubb_timestep`` (``config.clubb_dt``, ~300 s) and sub-cycles it within the
    larger host physics ``dt``. This advances a LOCAL copy of the mean state +
    moments for ``n_sub = ceil(dt / clubb_dt)`` sub-steps of ``dt_sub = dt/n_sub``
    and returns the NET (RAW, unclipped) mean tendency over ``dt`` plus the
    sub-cycled final moments — the same coupling contract as the single-step path
    (positivity limiting stays the host/moisture-fixer's job, not folded into the
    physics tendency). For ``dt <= clubb_dt`` (``n_sub = 1``) it is the single
    step, bit-identical to the un-sub-cycled path. Surface-flux diagnostics are
    the sub-cycle mean; ``Kh`` the final sub-step's.

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each ``(ncol,)`` or ``None``) are forwarded to :func:`clubb_step`
    unchanged — see its docstring. They are held constant across the sub-cycle
    (steady surface forcing, as in BOMEX/DYCOMS/ARM), the natural contract for a
    prescribed-flux case run within one host ``dt``.

    Returns ``(TurbulenceOutput, clubb_moments_new)``; the second element flows
    back into ``PhysicsState.clubb_moments`` via the carry machinery.
    """
    moments = unpack_clubb_moments(clubb_moments)
    n_sub = max(1, math.ceil(dt / config.clubb_dt))

    if n_sub == 1:
        du_dt, dv_dt, dT_dt, dq_v_dt, new_moments, diags = clubb_step(
            u, v, T, q_v, moments, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt, config,
            sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)
        shflx, lhflx, ustar = diags["shflx"], diags["lhflx"], diags["ustar"]
        Kh_full = flip_vertical(diags["Kh_zt"])
    else:
        dt_sub = dt / n_sub
        tv_floor = config.T0 * 0.5

        def _sub(carry, _):
            u_c, v_c, T_c, q_c, m_c = carry
            # Density floor (only) guards a strictly-positive rho if q_c dips
            # slightly negative mid-cycle; q itself is NOT clipped — the host
            # applies the RAW integrated CLUBB tendency, identical to the n_sub=1
            # contract (positivity limiting is the host/moisture-fixer's job, not
            # folded into the physics tendency). clubb_step floors rt internally
            # (rt_tol), so a slightly-negative mean rtm is robust (cloud → 0).
            tv = jnp.maximum(virtual_temperature(T_c, q_c), tv_floor)
            rho_c = p_full / (constants.R_d * tv)
            du, dv, dT, dq, m_new, diag = clubb_step(
                u_c, v_c, T_c, q_c, m_c, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_c, dt_sub, config,
                sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)
            carry = (u_c + dt_sub * du, v_c + dt_sub * dv, T_c + dt_sub * dT,
                     q_c + dt_sub * dq, m_new)
            return carry, diag

        (u_f, v_f, T_f, q_f, new_moments), diag_stk = jax.lax.scan(
            _sub, (u, v, T, q_v, moments), xs=None, length=n_sub)
        du_dt = (u_f - u) / dt
        dv_dt = (v_f - v) / dt
        dT_dt = (T_f - T) / dt
        dq_v_dt = (q_f - q_v) / dt
        # Net surface exchange = sub-cycle-mean flux; Kh from the final sub-step.
        shflx = jnp.mean(diag_stk["shflx"], axis=0)
        lhflx = jnp.mean(diag_stk["lhflx"], axis=0)
        ustar = jnp.mean(diag_stk["ustar"], axis=0)
        Kh_full = flip_vertical(diag_stk["Kh_zt"][-1])

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
    output = TurbulenceOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dq_v_dt=dq_v_dt,
        Km=Kh_full, Kh=Kh_full, shflx=shflx, lhflx=lhflx, ustar=ustar, h_pbl=h_pbl)
    return output, pack_clubb_moments(new_moments)


def integrate_clubb_column(
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
    dt: float,
    nsteps: int,
    config: CLUBBConfig,
    moments: CLUBBMomentState | None = None,
    host_numerical_diffusion: float = 0.05,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, CLUBBMomentState, dict]:
    """Integrate a single-column prognostic CLUBB run for ``nsteps`` steps.

    A self-contained SCM-style driver: ``lax.scan`` over :func:`clubb_step`,
    carrying the full :class:`CLUBBMomentState` *and* the mean state
    ``(u, v, T, q_v)`` (forward-Euler updated by the CLUBB tendencies each step;
    ``rho`` is recomputed hydrostatically from the evolving ``T``/``q_v`` on the
    fixed pressure grid). This exercises the genuinely-prognostic higher-order
    moment closure end-to-end — the moments persist and evolve across steps,
    unlike the diagnostic phase-1 entry — and is the multi-step stability/AD
    test bed for the scheme. All column fields are top-down ``(ncol, nlev)``.

    **Host numerical diffusion.** In a coupled model CLUBB returns *tendencies*
    and the dynamical core advances + numerically diffuses the means; that
    diffusion damps grid-scale (2Δz) vertical noise. A *bare* single-column
    driver advances the means with CLUBB alone, so it must supply that stand-in
    itself — without it, a long near-dry weakly-stratified column grows
    grid-scale ``T`` noise and ``wp2`` (the iter-48 instability; root-caused iter
    49-51 to absent host diffusion, NOT a closure/conservation/port error — a
    tiny ``host_numerical_diffusion`` removes it entirely, ``wp2max`` 10.7→0.06).
    ``host_numerical_diffusion`` is a dimensionless 2nd-order vertical-diffusion
    coefficient (0 ⇒ bare CLUBB, exposes the noise; default 0.05 ⇒ a coupled-
    model-like stand-in). Applied in **flux form** so it conserves the
    column-summed means exactly (zero-flux top/bottom).

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each ``(ncol,)`` or ``None``) are forwarded to :func:`clubb_step`
    unchanged (held constant across the run) — the standard way to drive a
    prescribed-flux LES/SCM case (BOMEX/DYCOMS/ARM). ``None`` ⇒ CLUBB's bulk
    surface formula from ``T_sfc``/``q_sfc`` (the default, back-compatible).

    ``moments`` defaults to a rest state (:func:`clubb_core.init_clubb_moments`).
    ``nsteps`` is a **static** Python int (the ``lax.scan`` length, fixed at trace
    time); jit callers close over it (it is not a traced argument).
    Returns the final ``(u, v, T, q_v, moments)`` and a dict of per-step stacked
    diagnostics (``cloud_frac``/``rcm``/``wpthvp``/``ustar``/...), shape
    ``(nsteps, ...)``. The returned ``moments`` means (rtm/thlm/um/vm) are kept
    consistent with the returned ``(u, v, T, q_v)``.
    """
    ncol, nlev = T.shape
    if moments is None:
        moments = init_clubb_moments(ncol, nlev, config, dtype=T.dtype)
    exner_td = exner_function(p_full)
    nu = host_numerical_diffusion
    # Safety floor for the recomputed virtual temperature so the prognostic
    # density stays strictly positive even if a long/dry SCM run drifts T low
    # (mirrors the shared compute_rho floor; finite-gradient via max).
    tv_floor = config.T0 * 0.5

    def _diffuse(f):
        """Flux-form 2nd-order vertical diffusion (zero-flux BCs → conserves
        sum(f) exactly): f[k] += flux[k] - flux[k-1], flux[k+1/2]=nu*(f[k+1]-f[k]).
        For ``nu <= 0.5`` the update is a convex combination of {f[k-1],f[k],f[k+1]}
        → monotone (stays within neighbour min/max), so it preserves positivity of
        a non-negative input AND the column sum."""
        flux = nu * (f[:, 1:] - f[:, :-1])           # (ncol, nlev-1) interfaces
        return f.at[:, :-1].add(flux).at[:, 1:].add(-flux)

    def _step(carry, _):
        u_c, v_c, T_c, q_c, m_c = carry
        tv = jnp.maximum(virtual_temperature(T_c, q_c), tv_floor)
        rho_c = p_full / (constants.R_d * tv)
        du, dv, dT, dq, m_new, diag = clubb_step(
            u_c, v_c, T_c, q_c, m_c, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho_c, dt, config,
            sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)
        # Moisture positivity is enforced on the CLUBB-tendency update FIRST (the
        # physical moisture-fixer; a non-conservative source, as in any model),
        # THEN the conservative host-stand-in diffusion is applied LAST. Because
        # _diffuse is monotone for nu<=0.5, diffusing a non-negative field keeps it
        # non-negative — so q stays >= 0 AND its (clipped) column sum is conserved
        # by the diffusion (no spurious water creation by the diffusion itself).
        u_n = _diffuse(u_c + dt * du)
        v_n = _diffuse(v_c + dt * dv)
        T_n = _diffuse(T_c + dt * dT)
        q_n = _diffuse(jnp.maximum(q_c + dt * dq, 0.0))
        # Keep the carried CLUBBMomentState means consistent with the updated mean
        # state (they are reset from the column inside clubb_step each step, but a
        # consistent returned state matters for callers/restart inspection).
        m_new = m_new._replace(
            um=flip_vertical(u_n), vm=flip_vertical(v_n),
            thlm=flip_vertical(T_n / exner_td), rtm=flip_vertical(q_n))
        carry = (u_n, v_n, T_n, q_n, m_new)
        return carry, diag

    (u_f, v_f, T_f, q_f, m_f), diags = jax.lax.scan(
        _step, (u, v, T, q_v, moments), xs=None, length=nsteps)
    return u_f, v_f, T_f, q_f, m_f, diags
