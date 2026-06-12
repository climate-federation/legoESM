"""CLUBB higher-order turbulence closure (fuller port; ``scheme="clubb"``).

This is the single-file home for the fuller CLUBB port tracked in
``PORT_CLUBB.md`` — substantially richer than :mod:`clubb_lite` — restricted to
the call tree exercised by the **CAM-default CLUBB flags** (every piece
golden-locked or parity-tested against CLUBB-JAX). Per the legoESM
one-file-per-scheme convention, the remaining ``clubb_*.py`` helper modules are
being absorbed here section by section (see the table of contents below); the
CAM-default model-flag values are recorded as comments at the end of the file.

Table of contents (sections, in order)
--------------------------------------
  1. Diagnostic ADG1-PDF closure (``diagnose_cloud_and_buoyancy``)
  2. Core orchestration (``compute_clubb_diagnostics`` /
     ``compute_pdf_closure`` / ``advance_clubb_core`` + the
     ``CLUBBMomentState``/``CLUBBForcing`` carry types and pack/unpack)
  3. Scheme entries (``clubb_turbulence`` diagnostic default /
     ``clubb_turbulence_prognostic`` opt-in / ``clubb_step`` bridge /
     ``integrate_clubb_column`` SCM driver)

Phasing (the scheme is wired in and runnable now; fidelity deepens per phase):

  * **Phase 1 (diagnostic default):** uses CLUBB's exact **parcel
    buoyant-sorting length scale** ``Lscale`` (``compute_mixing_length``,
    golden-locked vs CLUBB-JAX) to set the eddy diffusivity
    ``Km = c_K · Lscale · sqrt(em)`` — the distinctive CLUBB feature, replacing
    clubb_lite's Blackadar length — and diagnoses the cloud fraction from the
    ADG1 PDF. Mean fields (u, v, T, q_v) are advanced by the implicit eddy
    diffusion shared with the other legoESM schemes; ``wp2`` is carried (via
    the ``tke`` slot) with a production / dissipation / diffusion budget whose
    dissipation time scale is ``tau = Lscale / sqrt(em)``.
  * **Phase 2 (prognostic, opt-in via ``CLUBBConfig.prognostic=True``):** the
    full prognostic higher-order moment transport — ``advance_wp2_wp3``,
    ``advance_xp2_xpyp``, ``advance_xm_wpxp`` — coupled through the ADG1 PDF
    buoyancy flux ``wpthvp`` and the implicit tridiag/penta solves, with the
    15-field :class:`CLUBBMomentState` carried in
    ``PhysicsState.clubb_moments``.

The ``TurbulenceOutput`` contract carries no ``cloud_fraction`` field (see the
clubb_lite docstring), so the PDF cloud fraction is computed and exposed only
through diagnostics for now; the eddy-diffusion tendencies are the live output.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.clubb_coefficients import (
    compute_C6_C7_Skw_fnc,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import (
    CLUBBConfig,
    derive_lmin,
    derive_mixt_frac_max_mag,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import (
    CLUBBGrid,
    ddzt,
    flip_vertical,
    make_clubb_grid_from_levels,
    zm2zt,
    zt2zm,
)
from legoesm.atmosphere.physics.turbulence.clubb_helpers import (
    calc_brunt_vaisala_freq_sqd,
    compute_sigma_sqd_w,
)
from legoesm.atmosphere.physics.turbulence.clubb_mixing_length import (
    compute_mixing_length,
    set_Lscale_max,
)
from legoesm.atmosphere.physics.turbulence.clubb_moments import (
    advance_windm_edsclrm,
    advance_xp2_xpyp,
    clip_covars_denom,
)
from legoesm.atmosphere.physics.turbulence.clubb_pdf import (
    ADG1_pdf_driver,
    calc_comp_corrs_binormal,
    calc_pdf_liquid_cloud_frac,
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

_EP1 = (1.0 - constants.epsilon) / constants.epsilon
_EP2 = 1.0 / constants.epsilon
_HUNDRED = 100.0


# ===========================================================================
# 1. Diagnostic ADG1-PDF closure
# ===========================================================================
# Given the column mean state and the carried ``wp2`` on the CLUBB grid, this
# diagnoses the second moments with standard mixing-length / down-gradient
# closures, runs the ADG1 double-Gaussian assumed-PDF closure — the
# distinctive CLUBB feature absent from clubb_lite (single Gaussian) — and
# returns the liquid cloud fraction, cloud water ``rcm``, and the moist
# buoyancy flux ``wpthvp`` (including the cloud-water latent-heat term that
# makes a cloudy layer more buoyant). This is the *diagnostic* coupling used
# by the phase-1 runnable ``clubb_turbulence`` entry: skewness is taken
# symmetric (``Skw = 0`` → ``mixt_frac = 1/2``) and the variances are
# mixing-length closures. All fields are on thermodynamic (zt) levels of the
# ascending CLUBB grid.


def _grad_zt(field_zt, gr: CLUBBGrid):
    """d/dz of a zt-level field, returned on zt (``zm2zt(ddzt(.))``)."""
    return zm2zt(ddzt(field_zt, gr), gr)


def diagnose_cloud_and_buoyancy(thlm, rtm, wp2, exner, p_in_Pa, thv_ds, Kh, Lscale,
                                gr: CLUBBGrid, config: CLUBBConfig):
    """ADG1-PDF cloud fraction, cloud water, and moist buoyancy flux (zt levels).

    Parameters (all ``(ncol, nzt)`` on the ascending CLUBB grid)
    ----------
    thlm, rtm : jax.Array
        Liquid-water potential temperature [K] and total water [kg/kg].
    wp2 : jax.Array
        Carried ``w'^2`` [m^2/s^2].
    exner, p_in_Pa, thv_ds : jax.Array
        Exner, pressure [Pa], dry-static virtual potential temperature [K].
    Kh : jax.Array
        Eddy diffusivity for scalars [m^2/s].
    Lscale : jax.Array
        CLUBB parcel mixing length [m].
    gr : CLUBBGrid
    config : CLUBBConfig

    Returns
    -------
    tuple of jax.Array
        ``(cloud_frac, rcm, wpthvp)`` on zt levels — liquid cloud fraction [-],
        cloud water [kg/kg], and the buoyancy flux ``w'thv'`` [K m/s].
    """
    params = config.params
    wp2 = jnp.maximum(wp2, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    ddz_thl = _grad_zt(thlm, gr)
    ddz_rt = _grad_zt(rtm, gr)

    # Down-gradient second-order fluxes and mixing-length variances.
    wpthlp = -Kh * ddz_thl
    wprtp = -Kh * ddz_rt
    thlp2 = jnp.maximum((Lscale * ddz_thl) ** 2, config.thl_tol ** 2)
    rtp2 = jnp.maximum((Lscale * ddz_rt) ** 2, config.rt_tol ** 2)
    rtpthlp = Lscale ** 2 * ddz_thl * ddz_rt
    up2 = vp2 = jnp.maximum(wp2, config.w_tol ** 2)

    # sigma_sqd_w (Skw = 0 -> gamma = gamma_coef), computed directly on zt.
    denom_thl = jnp.sqrt(wp2 * thlp2) + _HUNDRED * config.w_tol * config.thl_tol
    denom_rt = jnp.sqrt(wp2 * rtp2) + _HUNDRED * config.w_tol * config.rt_tol
    max_corr = jnp.maximum((wpthlp / denom_thl) ** 2, (wprtp / denom_rt) ** 2)
    sigma_sqd_w = jnp.clip(params.gamma_coef * (1.0 - jnp.minimum(max_corr, 1.0)), 0.0, 0.99)

    z = jnp.zeros_like(wp2)
    mfmm = derive_mixt_frac_max_mag(params.Skw_max_mag)
    adg1 = ADG1_pdf_driver(
        z, rtm, thlm, z, z, wp2, rtp2, thlp2, up2, vp2, z,
        wprtp, wpthlp, z, z, sqrt_wp2, sigma_sqd_w, params.beta, mfmm)

    rcm, cloud_frac = calc_pdf_liquid_cloud_frac(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa)

    # Moist buoyancy flux: wpthvp = wpthlp + ep1*thv_ds*wprtp + rc_coef*wprcp,
    # with a down-gradient cloud-water flux wprcp (rc_coef = Lv/(exner*Cp) - ep2*thv).
    wprcp = -Kh * _grad_zt(rcm, gr)
    rc_coef = constants.L_v / (exner * constants.c_pd) - _EP2 * thv_ds
    wpthvp = wpthlp + _EP1 * thv_ds * wprtp + rc_coef * wprcp
    return cloud_frac, rcm, wpthvp


# ===========================================================================
# 2. Core orchestration
# ===========================================================================
# Assembles the parity-tested building blocks into the per-step closure: the
# diagnostics (``compute_clubb_diagnostics`` — skewness, ``sigma_sqd_w``, TKE,
# the dissipation-time-scale family, and the C6/C7 coefficients) that feed the
# ADG1 PDF closure and the four prognostic moment advances, which run in the
# CAM order ``xm_wpxp(1) -> xp2_xpyp(2) -> wp2_wp3(3) -> windm(4)`` with
# ``clip_covars_denom`` between. Each constituent is independently
# bit-exact/round-off parity-tested vs CLUBB-JAX; this section is the thin
# (pure / JIT-safe / differentiable) wiring.


def compute_clubb_diagnostics(wp2, wp3, up2, vp2, thlp2, rtp2, wpthlp, wprtp,
                              Lscale, brunt_vaisala_freq_sqd, gr: CLUBBGrid, config):
    """Per-step CLUBB closure diagnostics (CAM-default tree).

    Composes the parity-tested helpers into one bundle of the inputs the PDF
    closure and the moment advances need:

      * skewness (``Skw_zm``/``Skw_zt``) and the smoothed ``wp3_on_wp2`` ratio
        (:func:`compute_skewness_diagnostics`);
      * the skewness-dependent ``gamma_Skw`` (CAM ``l_gamma_Skw = .true.``) and
        ``sigma_sqd_w`` (:func:`compute_sigma_sqd_w`);
      * TKE ``em``/``sqrt_em_zt`` (:func:`compute_tke`);
      * the ``invrs_tau_*`` family (:func:`compute_tau_family`);
      * the xm/wpxp coefficients ``C6rt/C6thl/C7_Skw_fnc``
        (:func:`compute_C6_C7_Skw_fnc`).

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


def init_clubb_moments(ncol: int, nlev: int, config, dtype=jnp.float64) -> CLUBBMomentState:
    """Seed a fresh :class:`CLUBBMomentState` at rest (CAM-default floors).

    Means are zero (the bridge resets them from the live column each step);
    velocity variances start at the floor ``tke_min`` (``w_tol^2`` scale), scalar
    variances at their tolerance-squared floors, all fluxes and ``wp3`` zero.
    ``nlev`` thermo (zt) levels → ``nzm = nlev + 1`` momentum levels.
    """
    nzm = nlev + 1
    zt = jnp.zeros((ncol, nlev), dtype=dtype)
    zm0 = jnp.zeros((ncol, nzm), dtype=dtype)
    wtol2 = jnp.full((ncol, nzm), config.tke_min, dtype=dtype)
    return CLUBBMomentState(
        rtm=zt, thlm=zt, um=zt, vm=zt,
        wp2=wtol2, wp3=zt, up2=wtol2, vp2=wtol2,
        wprtp=zm0, wpthlp=zm0, upwp=zm0, vpwp=zm0,
        rtp2=jnp.full((ncol, nzm), config.rt_tol ** 2, dtype=dtype),
        thlp2=jnp.full((ncol, nzm), config.thl_tol ** 2, dtype=dtype),
        rtpthlp=zm0)


# Field layout for packing CLUBBMomentState into a single (ncol, NFIELDS, nzm)
# array carried in PhysicsState (like gwd_spectrum). zt-level fields (nlev) use
# the first nlev slots of the nzm axis with the trailing slot zero-padded.
N_MOMENT_FIELDS = len(CLUBBMomentState._fields)   # 15
_ZT_FIELD_NAMES = frozenset(("rtm", "thlm", "um", "vm", "wp3"))  # rest are zm


def pack_clubb_moments(state: CLUBBMomentState) -> jax.Array:
    """Pack a :class:`CLUBBMomentState` into one ``(ncol, 15, nzm)`` array.

    zm-level fields (``wp2``/variances/fluxes) fill the full ``nzm`` axis; zt-level
    fields (``rtm``/``thlm``/``um``/``vm``/``wp3``, length ``nlev = nzm-1``) fill
    ``[:, :nlev]`` with a zero in the trailing slot. Inverse of
    :func:`unpack_clubb_moments`. Used to carry the moment state in
    ``PhysicsState`` as a single regular array.
    """
    cols = []
    for name in CLUBBMomentState._fields:
        f = getattr(state, name)
        if name in _ZT_FIELD_NAMES:                       # (ncol, nlev) -> (ncol, nzm)
            f = jnp.concatenate([f, jnp.zeros_like(f[:, :1])], axis=1)
        cols.append(f[:, None, :])                        # (ncol, 1, nzm)
    return jnp.concatenate(cols, axis=1)                  # (ncol, 15, nzm)


def unpack_clubb_moments(arr: jax.Array) -> CLUBBMomentState:
    """Unpack a ``(ncol, 15, nzm)`` array into a :class:`CLUBBMomentState`.

    Inverse of :func:`pack_clubb_moments`: zt-level fields are sliced back to
    ``nlev = nzm-1`` (dropping the zero pad).
    """
    nlev = arr.shape[2] - 1
    fields = {}
    for i, name in enumerate(CLUBBMomentState._fields):
        col = arr[:, i, :]
        fields[name] = col[:, :nlev] if name in _ZT_FIELD_NAMES else col
    return CLUBBMomentState(**fields)


# ===========================================================================
# 3. Scheme entries
# ===========================================================================


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
    :func:`advance_clubb_core`, and maps the advanced means back to
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

    ``moments`` defaults to a rest state (:func:`init_clubb_moments`).
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
