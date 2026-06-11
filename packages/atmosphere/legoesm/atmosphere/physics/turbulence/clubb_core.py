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

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_coefficients import compute_C6_C7_Skw_fnc
from legoesm.atmosphere.physics.turbulence.clubb_config import derive_mixt_frac_max_mag
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt, zt2zm
from legoesm.atmosphere.physics.turbulence.clubb_helpers import compute_sigma_sqd_w
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

    return dict(
        wpthvp=wpthvp_zm, wp2thvp=wp2thvp_zt, rtpthvp=rtpthvp_zm,
        thlpthvp=thlpthvp_zm, rc_coef_zm=rc_coef_zm,
        cloud_frac=comp["cloud_frac"], rcm=comp["rcm"],
        wprcp=xprcp["wprcp_zm"], rtprcp=xprcp["rtprcp_zm"],
        thlprcp=xprcp["thlprcp_zm"], uprcp=xprcp["uprcp_zm"],
        vprcp=xprcp["vprcp_zm"], **hom,
    )


__all__ = ["compute_clubb_diagnostics", "compute_pdf_closure"]
