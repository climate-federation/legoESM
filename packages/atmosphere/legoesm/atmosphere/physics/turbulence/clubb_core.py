"""CLUBB core orchestration (``advance_clubb_core``) for the CAM-default tree.

This assembles the parity-tested building blocks into the per-step closure: the
**diagnostics** (:func:`compute_clubb_diagnostics` — skewness, ``sigma_sqd_w``,
TKE, the dissipation-time-scale family, and the C6/C7 coefficients) that feed the
ADG1 PDF closure (:mod:`clubb_pdf`/:mod:`clubb_pdf_moments`) and the four
prognostic moment advances, which run in the CAM order
``xm_wpxp(1) -> xp2_xpyp(2) -> wp2_wp3(3) -> windm(4)`` with ``clip_covars_denom``
between. Each constituent is independently bit-exact/round-off parity-tested vs
CLUBB-JAX; this module is the thin (pure / JIT-safe / differentiable) wiring.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_coefficients import compute_C6_C7_Skw_fnc
from legoesm.atmosphere.physics.turbulence.clubb_config import derive_mixt_frac_max_mag
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt, zt2zm
from legoesm.atmosphere.physics.turbulence.clubb_helpers import compute_sigma_sqd_w
from legoesm.atmosphere.physics.turbulence.clubb_moments import (
    advance_windm_edsclrm,
    advance_xp2_xpyp,
    clip_covars_denom,
)
from legoesm.atmosphere.physics.turbulence.clubb_pdf import (
    ADG1_pdf_driver,
    calc_comp_corrs_binormal,
    calc_pdf_liquid_cloud_frac_components,
)
from legoesm.atmosphere.physics.turbulence.clubb_pdf_moments import (
    calc_pdf_higher_order_moments,
    calc_pdf_xprcp_fluxes,
    calc_xpthvp_terms,
)
from legoesm.atmosphere.physics.turbulence.clubb_skewness import (
    compute_gamma_Skw,
    compute_skewness_diagnostics,
)
from legoesm.atmosphere.physics.turbulence.clubb_tau import (
    compute_tau_family,
    compute_tke,
)
from legoesm.atmosphere.physics.turbulence.clubb_wp23 import advance_wp2_wp3
from legoesm.atmosphere.physics.turbulence.clubb_xm_wpxp import advance_xm_wpxp


def compute_clubb_diagnostics(wp2, wp3, up2, vp2, thlp2, rtp2, wpthlp, wprtp,
                              Lscale, brunt_vaisala_freq_sqd, gr: CLUBBGrid, config):
    """Per-step CLUBB closure diagnostics (CAM-default tree).

    Composes the parity-tested helpers into one bundle of the inputs the PDF
    closure and the moment advances need:

      * skewness (``Skw_zm``/``Skw_zt``) and the smoothed ``wp3_on_wp2`` ratio
        (:func:`clubb_skewness.compute_skewness_diagnostics`);
      * the skewness-dependent ``gamma_Skw`` (CAM ``l_gamma_Skw = .true.``) and
        ``sigma_sqd_w`` (:func:`clubb_helpers.compute_sigma_sqd_w`);
      * TKE ``em``/``sqrt_em_zt`` (:func:`clubb_tau.compute_tke`);
      * the ``invrs_tau_*`` family (:func:`clubb_tau.compute_tau_family`);
      * the xm/wpxp coefficients ``C6rt/C6thl/C7_Skw_fnc``
        (:func:`clubb_coefficients.compute_C6_C7_Skw_fnc`).

    All moments/fluxes are zm-level (``wp3`` zt); ``Lscale`` is the parcel
    buoyant-sorting length (zt); ``brunt_vaisala_freq_sqd`` is zm. Returns a dict
    with the above (the tau-family keys + ``Skw_*``/``wp3_on_wp2*``/``wp2_zt``/
    ``wp3_zm``/``em``/``sqrt_em_zt``/``sigma_sqd_w``/``gamma_Skw``/``Lscale_zm``/
    ``C6rt_Skw_fnc``/``C6thl_Skw_fnc``/``C7_Skw_fnc``).
    """
    p = config.params
    skw = compute_skewness_diagnostics(wp2, wp3, config.w_tol, p.Skw_denom_coef, gr)
    # CAM l_gamma_Skw = .true. (CLUBB model_flags default).
    gamma_Skw = compute_gamma_Skw(skw["Skw_zm"], p.gamma_coef, p.gamma_coefb,
                                  p.gamma_coefc, True)
    sigma_sqd_w = compute_sigma_sqd_w(
        gamma_Skw, wp2, thlp2, rtp2, wpthlp, wprtp, gr,
        w_tol=config.w_tol, thl_tol=config.thl_tol, rt_tol=config.rt_tol)
    em, sqrt_em_zt = compute_tke(wp2, up2, vp2, gr, config)
    tau = compute_tau_family(Lscale, em, sqrt_em_zt, brunt_vaisala_freq_sqd, gr, config)
    Lscale_zm = jnp.maximum(zt2zm(Lscale, gr), 0.0)
    C6rt, C6thl, C7 = compute_C6_C7_Skw_fnc(skw["Skw_zm"], Lscale_zm, config, gr)

    # Eddy diffusivities Kh = c_K·Lscale·sqrt(em) (zt and zm levels).
    em_min = 1.5 * config.w_tol ** 2
    Kh_zt = config.params.c_K * Lscale * sqrt_em_zt
    Kh_zm = config.params.c_K * Lscale_zm * jnp.sqrt(jnp.maximum(em, em_min))
    return dict(
        **skw, gamma_Skw=gamma_Skw, sigma_sqd_w=sigma_sqd_w, em=em,
        sqrt_em_zt=sqrt_em_zt, Lscale_zm=Lscale_zm, Kh_zt=Kh_zt, Kh_zm=Kh_zm,
        C6rt_Skw_fnc=C6rt, C6thl_Skw_fnc=C6thl, C7_Skw_fnc=C7, **tau,
    )


def compute_pdf_closure(diag, wp2, wp3, rtp2, thlp2, rtpthlp, up2, vp2,
                        wprtp, wpthlp, upwp, vpwp,
                        wm_zt, rtm, thlm, um, vm, exner_zt, p_in_Pa_zt, thv_ds_zt,
                        gr: CLUBBGrid, config):
    """ADG1 assumed-PDF closure (CAM-default tree, post-advance placement).

    The CAM-default subset of ``pdf_closure_module.F90:pdf_closure_driver`` (and
    its ``adg1_pdf_driver_zt_jax`` helper): given the post-advance moment state it
    invokes the ADG1 double-Gaussian PDF and returns the **buoyancy fluxes**
    (``wpthvp``/``wp2thvp``/``rtpthvp``/``thlpthvp``), the **higher-order velocity
    moments** (``wp4``/``wp2up2``/``wp2vp2``/``wpup2``/``wpvp2``/``wp2rtp``/
    ``wp2thlp``/``wp2up``/``wprtp2``/``wpthlp2``/``wprtpthlp``), the **cloud-water
    turbulent fluxes** (``wprcp``/``rtprcp``/``thlprcp``/``uprcp``/``vprcp``), and
    the **cloud diagnostics** (``cloud_frac``/``rcm``/``rc_coef_zm``) that the four
    moment advances and the host model consume. The CAM-irrelevant pieces (the
    stats writer, the non-ADG1 PDF branches, ice-supersat) are omitted; the
    stats-only intermediates are not returned.

    Inputs: ``diag`` is the :func:`compute_clubb_diagnostics` bundle for the SAME
    (post-advance) state — its ``Skw_zt``/``wp2_zt``/``sigma_sqd_w`` are reused
    rather than re-derived. Prognostic moments ``wp2``/``rtp2``/``thlp2``/
    ``rtpthlp``/``up2``/``vp2`` and fluxes ``wprtp``/``wpthlp``/``upwp``/``vpwp``
    are on zm; ``wp3`` is on zt. Means ``wm_zt``/``rtm``/``thlm``/``um``/``vm`` and
    the thermo fields ``exner_zt``/``p_in_Pa_zt``/``thv_ds_zt`` are on zt.

    Returns a dict (buoyancy/HOM on the levels the advances want: ``wpthvp_zm``
    etc. on zm, ``wp2thvp_zt`` on zt).
    """
    p = config.params
    w_tol_sqd = config.w_tol ** 2

    # --- zt-level fields for the ADG1 driver (adg1_pdf_driver_zt_jax) ---
    Skw_zt = diag["Skw_zt"]
    wp2_zt = diag["wp2_zt"]
    sigma_sqd_w_zt = jnp.maximum(zm2zt(diag["sigma_sqd_w"], gr), 0.0)
    # Raw zt regrids feed the buoyancy assembly (calc_xpthvp_terms); the
    # tolerance-floored ``*_adg`` versions feed ONLY the ADG1 driver. CLUBB-JAX
    # pdf_closure_driver keeps these paths separate (the floor must not leak a
    # tolerance-level variance into rtpthvp/thlpthvp in low-variance columns).
    rtp2_zt = zm2zt(rtp2, gr)
    thlp2_zt = zm2zt(thlp2, gr)
    rtp2_zt_adg = jnp.maximum(rtp2_zt, config.rt_tol ** 2)
    thlp2_zt_adg = jnp.maximum(thlp2_zt, config.thl_tol ** 2)
    up2_zt = jnp.maximum(zm2zt(up2, gr), w_tol_sqd)
    vp2_zt = jnp.maximum(zm2zt(vp2, gr), w_tol_sqd)
    wprtp_zt = zm2zt(wprtp, gr)
    wpthlp_zt = zm2zt(wpthlp, gr)
    upwp_zt = zm2zt(upwp, gr)
    vpwp_zt = zm2zt(vpwp, gr)
    rtpthlp_zt = zm2zt(rtpthlp, gr)

    mixt_frac_max_mag = derive_mixt_frac_max_mag(p.Skw_max_mag)
    adg1 = ADG1_pdf_driver(
        wm_zt, rtm, thlm, um, vm, wp2_zt, rtp2_zt_adg, thlp2_zt_adg, up2_zt, vp2_zt,
        Skw_zt, wprtp_zt, wpthlp_zt, upwp_zt, vpwp_zt, jnp.sqrt(wp2_zt),
        sigma_sqd_w_zt, p.beta, mixt_frac_max_mag)

    # --- per-component rt-thl correlation + liquid cloud-fraction closure ---
    corr_rt_thl_1, corr_rt_thl_2 = calc_comp_corrs_binormal(
        rtpthlp_zt, rtm, thlm, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"],
        adg1["thl_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"],
        adg1["varnce_thl_1"], adg1["varnce_thl_2"], adg1["mixt_frac"])
    comp = calc_pdf_liquid_cloud_frac_components(
        adg1, rtpthlp_zt, rtm, thlm, exner_zt, p_in_Pa_zt)
    rcm_zt = comp["rcm"]

    # --- cloud-water fluxes, higher-order velocity moments, buoyancy fluxes ---
    xprcp = calc_pdf_xprcp_fluxes(adg1, comp, wm_zt, rtm, thlm, um, vm, rcm_zt, gr)
    hom = calc_pdf_higher_order_moments(
        adg1, wm_zt, rtm, thlm, um, vm, corr_rt_thl_1, corr_rt_thl_2, gr)
    (wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm,
     _rc_coef_zt, rc_coef_zm) = calc_xpthvp_terms(
        exner_zt, thv_ds_zt, xprcp["wprcp_zt"], xprcp["wp2rcp_zt"],
        xprcp["rtprcp_zt"], xprcp["thlprcp_zt"], wpthlp_zt, wprtp_zt,
        hom["wp2thlp"], hom["wp2rtp"], rtpthlp_zt, rtp2_zt, thlp2_zt, gr)

    # ADG1 w-component PDF params regridded zt->zm (the MFL turbulent-advection
    # range in advance_xm_wpxp reads these on zm; advance_clubb_core_module.F90).
    w_1_zm = zt2zm(adg1["w_1"], gr)
    w_2_zm = zt2zm(adg1["w_2"], gr)
    varnce_w_1_zm = zt2zm(adg1["varnce_w_1"], gr)
    varnce_w_2_zm = zt2zm(adg1["varnce_w_2"], gr)
    mixt_frac_zm = zt2zm(adg1["mixt_frac"], gr)

    return dict(
        wpthvp=wpthvp_zm, wp2thvp=wp2thvp_zt, rtpthvp=rtpthvp_zm,
        thlpthvp=thlpthvp_zm, rc_coef_zm=rc_coef_zm,
        cloud_frac=comp["cloud_frac"], rcm=comp["rcm"],
        wprcp=xprcp["wprcp_zm"], rtprcp=xprcp["rtprcp_zm"],
        thlprcp=xprcp["thlprcp_zm"], uprcp=xprcp["uprcp_zm"],
        vprcp=xprcp["vprcp_zm"], w_1_zm=w_1_zm, w_2_zm=w_2_zm,
        varnce_w_1_zm=varnce_w_1_zm, varnce_w_2_zm=varnce_w_2_zm,
        mixt_frac_zm=mixt_frac_zm, **hom,
    )


class CLUBBMomentState(NamedTuple):
    """The carried CLUBB prognostic higher-order moment state (CAM-default tree).

    The 15 fields advanced each step by :func:`advance_clubb_core`. Means are on
    thermodynamic (zt) levels; second/third moments and fluxes on momentum (zm)
    levels except ``wp3`` (zt). All ``(ngrdcol, nz*)``. CAM
    ``l_predict_upwp_vpwp = .false.`` so ``upwp``/``vpwp`` are diagnosed inside
    ``advance_windm_edsclrm`` rather than prognosed by ``advance_xm_wpxp``.
    """
    # Means (zt)
    rtm: jax.Array
    thlm: jax.Array
    um: jax.Array
    vm: jax.Array
    # Velocity moments
    wp2: jax.Array      # zm
    wp3: jax.Array      # zt
    up2: jax.Array      # zm
    vp2: jax.Array      # zm
    # Fluxes (zm)
    wprtp: jax.Array
    wpthlp: jax.Array
    upwp: jax.Array
    vpwp: jax.Array
    # Scalar second moments (zm)
    rtp2: jax.Array
    thlp2: jax.Array
    rtpthlp: jax.Array


class CLUBBForcing(NamedTuple):
    """Large-scale forcings (tendency sources) for the moment advances [units/s].

    All on the same level as the advanced field: ``rtm``/``thlm``/``um``/``vm``
    on zt; ``wprtp``/``wpthlp``/``rtp2``/``thlp2``/``rtpthlp`` on zm. Zero by
    default for an isolated SCM driver (the host supplies them when coupled).
    """
    rtm: jax.Array
    thlm: jax.Array
    um: jax.Array
    vm: jax.Array
    wprtp: jax.Array
    wpthlp: jax.Array
    rtp2: jax.Array
    thlp2: jax.Array
    rtpthlp: jax.Array


def advance_clubb_core(state: CLUBBMomentState, forcing: CLUBBForcing, *,
                       Lscale, brunt_vaisala_freq_sqd, exner_zt, p_in_Pa_zt,
                       thv_ds_zt, thv_ds_zm, rho_ds_zm, rho_ds_zt,
                       invrs_rho_ds_zm, invrs_rho_ds_zt, wm_zt, wm_zm,
                       sfc_elevation, fcor, ug, vg, dt, gr: CLUBBGrid, config):
    """One prognostic CLUBB step (the CAM-default ``advance_clubb_core`` core).

    Runs, in the CAM order, ``compute_clubb_diagnostics`` -> the pre-advance ADG1
    PDF closure (``l_call_pdf_closure_twice = .true.``) -> the four moment
    advances ``advance_xm_wpxp -> advance_xp2_xpyp -> advance_wp2_wp3 ->
    advance_windm_edsclrm`` with ``clip_covars_denom`` between the variance/flux
    solves -> the post-advance PDF closure for the cloud/buoyancy diagnostics. The
    advances mutate the shared moment set in sequence (each sees the previous
    advance's update), exactly as the Fortran driver does.

    Parameters
    ----------
    state : CLUBBMomentState
        Start-of-step prognostic moments.
    forcing : CLUBBForcing
        Large-scale tendency sources.
    Lscale : jax.Array
        Parcel buoyant-sorting mixing length (zt) from ``compute_mixing_length``.
    brunt_vaisala_freq_sqd : jax.Array
        ``N^2`` (zm).
    exner_zt, p_in_Pa_zt, thv_ds_zt, thv_ds_zm, rho_ds_zm, rho_ds_zt,
    invrs_rho_ds_zm, invrs_rho_ds_zt, wm_zt, wm_zm, sfc_elevation, fcor, ug, vg :
        Host thermodynamic / dry-static-density / geometry / Coriolis-geostrophic
        fields on their noted grids (``fcor``/``sfc_elevation`` are ``(ngrdcol,)``).
    dt : float
        Time step [s].

    Returns
    -------
    CLUBBMomentState
        End-of-step prognostic moments.
    dict
        Diagnostics — ``cloud_frac``/``rcm`` (post-advance PDF) plus the
        diffusivities ``Kh_zt``/``Kh_zm`` and ``wpthvp`` for the host.
    """
    p = config.params
    ng = state.wp2.shape[0]
    # Background eddy diffusivities are per-column arrays (vertical-resolution
    # scaling is identity on a fixed grid); advance_xp2_xpyp indexes nu[:, None].
    # (The per-moment C2 dissipation coefficients C2rt/C2thl/C2rtthl are owned by
    # advance_xp2_xpyp itself, read from config — CAM uses 3 distinct values.)
    nu2 = jnp.full((ng,), p.nu2)
    nu9 = jnp.full((ng,), p.nu9)

    # ---- (1) closure diagnostics on the start-of-step state ----
    diag = compute_clubb_diagnostics(
        state.wp2, state.wp3, state.up2, state.vp2, state.thlp2, state.rtp2,
        state.wpthlp, state.wprtp, Lscale, brunt_vaisala_freq_sqd, gr, config)

    # ---- (2) pre-advance ADG1 PDF closure -> buoyancy + higher-order moments ----
    pdf = compute_pdf_closure(
        diag, state.wp2, state.wp3, state.rtp2, state.thlp2, state.rtpthlp,
        state.up2, state.vp2, state.wprtp, state.wpthlp, state.upwp, state.vpwp,
        wm_zt, state.rtm, state.thlm, state.um, state.vm, exner_zt, p_in_Pa_zt,
        thv_ds_zt, gr, config)

    # ---- (3) advance_xm_wpxp: rtm/wprtp + thlm/wpthlp ----
    wprtp, rtm, wpthlp, thlm = advance_xm_wpxp(
        state.rtm, state.thlm, state.wprtp, state.wpthlp, forcing.rtm,
        forcing.thlm, forcing.wprtp, forcing.wpthlp, diag["C6rt_Skw_fnc"],
        diag["C6thl_Skw_fnc"], diag["C7_Skw_fnc"], diag["invrs_tau_C6_zm"],
        diag["sigma_sqd_w"], diag["wp3_on_wp2_zt"], state.wp2, diag["Kh_zt"],
        state.rtp2, state.thlp2, pdf["rtpthvp"], pdf["thlpthvp"], thv_ds_zm,
        wm_zm, wm_zt, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
        pdf["w_1_zm"], pdf["w_2_zm"], pdf["varnce_w_1_zm"], pdf["varnce_w_2_zm"],
        pdf["mixt_frac_zm"], dt, gr, config)

    # ---- (4) advance_xp2_xpyp: rtp2/thlp2/rtpthlp/up2/vp2 (uses pre-wp2_wp3 wp2) ----
    rtp2, thlp2, rtpthlp, up2, vp2 = advance_xp2_xpyp(
        rtm, thlm, state.um, state.vm, state.rtp2, state.thlp2, state.rtpthlp,
        state.up2, state.vp2, wprtp, wpthlp, pdf["wpthvp"], state.upwp,
        state.vpwp, state.wp2, diag["wp2_zt"], diag["wp3_on_wp2"],
        diag["wp3_on_wp2_zt"], diag["sigma_sqd_w"], thv_ds_zm, diag["Kh_zt"],
        diag["invrs_tau_xp2_zm"], diag["invrs_tau_C4_zm"],
        diag["invrs_tau_C14_zm"], rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, wm_zm,
        forcing.rtp2, forcing.thlp2, forcing.rtpthlp, nu2, nu9, dt, gr, config)

    # ---- (5) Cauchy-Schwarz clip of the fluxes (pre-wp2_wp3 wp2) ----
    wprtp, wpthlp, upwp, vpwp = clip_covars_denom(
        wprtp, wpthlp, state.upwp, state.vpwp, state.wp2, rtp2, thlp2, up2, vp2,
        l_tke_aniso=config.flags.l_tke_aniso)

    # ---- (6) advance_wp2_wp3: wp2/wp3 (uses post-xp2 up2/vp2 + post-clip fluxes) ----
    wp2, wp3, _wp2_zt = advance_wp2_wp3(
        state.wp2, state.wp3, up2, vp2, diag["sigma_sqd_w"], diag["wp3_on_wp2"],
        pdf["wpup2"], pdf["wpvp2"], pdf["wp2up2_zm"], pdf["wp2vp2_zm"],
        pdf["wp4_zm"], pdf["wpthvp"], pdf["wp2thvp"], state.um, state.vm, upwp,
        vpwp, wm_zm, wm_zt, diag["Kh_zm"], diag["Kh_zt"], diag["invrs_tau_C4_zm"],
        diag["invrs_tau_wp3_zt"], diag["invrs_tau_C1_zm"], diag["Skw_zm"],
        diag["Skw_zt"], rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
        thv_ds_zm, thv_ds_zt, sfc_elevation, dt, gr, config)

    # ---- (7) Cauchy-Schwarz clip with the new wp2 ----
    wprtp, wpthlp, upwp, vpwp = clip_covars_denom(
        wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
        l_tke_aniso=config.flags.l_tke_aniso)

    # ---- (8) advance_windm_edsclrm: um/vm + diagnostic upwp/vpwp ----
    # Kh_zm is the START-OF-STEP eddy diffusivity (advance_clubb_core_module.F90
    # "Block M", computed once before the advance loop) and is intentionally NOT
    # recomputed after wp2/wp3: the reference passes this same Kh_zm to BOTH
    # advance_wp2_wp3 and advance_windm_edsclrm. The new (post-advance) wp2/up2/vp2
    # are used only for the Cauchy-Schwarz flux clip inside windm, matching Fortran.
    um, vm, upwp, vpwp = advance_windm_edsclrm(
        state.um, state.vm, upwp, vpwp, wp2, up2, vp2, wm_zt, diag["Kh_zm"],
        ug, vg, forcing.um, forcing.vm, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zt,
        fcor, p.c_K10, p.nu10, dt, gr, l_tke_aniso=config.flags.l_tke_aniso)

    new_state = CLUBBMomentState(
        rtm=rtm, thlm=thlm, um=um, vm=vm, wp2=wp2, wp3=wp3, up2=up2, vp2=vp2,
        wprtp=wprtp, wpthlp=wpthlp, upwp=upwp, vpwp=vpwp, rtp2=rtp2,
        thlp2=thlp2, rtpthlp=rtpthlp)

    # ---- (9) post-advance PDF closure for the cloud/buoyancy diagnostics ----
    diag_post = compute_clubb_diagnostics(
        wp2, wp3, up2, vp2, thlp2, rtp2, wpthlp, wprtp, Lscale,
        brunt_vaisala_freq_sqd, gr, config)
    pdf_post = compute_pdf_closure(
        diag_post, wp2, wp3, rtp2, thlp2, rtpthlp, up2, vp2, wprtp, wpthlp,
        upwp, vpwp, wm_zt, rtm, thlm, um, vm, exner_zt, p_in_Pa_zt, thv_ds_zt,
        gr, config)

    diagnostics = dict(
        cloud_frac=pdf_post["cloud_frac"], rcm=pdf_post["rcm"],
        wpthvp=pdf_post["wpthvp"], Kh_zt=diag["Kh_zt"], Kh_zm=diag["Kh_zm"])
    return new_state, diagnostics


__all__ = [
    "CLUBBForcing",
    "CLUBBMomentState",
    "advance_clubb_core",
    "compute_clubb_diagnostics",
    "compute_pdf_closure",
]
