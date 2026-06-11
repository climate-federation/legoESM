"""CLUBB ADG1 PDF moment integrals + buoyancy-flux assembly (pdf_closure_module).

Faithful port of the higher-order-moment and ``x'thv'`` sections of
``CLUBB-JAX/.../pdf_closure_module.py`` for the CAM-default ADG1 path. Given the
ADG1 double-Gaussian component parameters (:mod:`clubb_pdf`) plus the cloud
diagnosis, these integrate the assumed PDF for:

  * the velocity-scalar higher moments ``<w'^2 x'>``, ``<w'x'^2>``,
    ``<w'^2 x'^2>``, ``<w'^4>``, ``<w'x'y'>`` (the down-gradient + skewness
    transport that feeds the wp2/wp3/xp2/xp3 advance), and
  * the cloud-water fluxes ``x'rc'`` and the buoyancy fluxes ``x'thv'``
    (``wpthvp`` is the buoyancy production of ``wp2`` — the term that makes
    CLUBB a moist, nonlocal closure).

For ADG1 the velocity-scalar component correlations
``corr_w_rt = corr_w_thl = corr_u_w = corr_v_w = 0`` and ``corr_w_chi = 0``;
only the per-component ``corr_rt_thl`` is nonzero (feeds ``wprtpthlp``). The
moment integrals use no physical constants (bit-exact to the reference); the
buoyancy assembly uses ``L_v``/``c_pd``/``epsilon`` from ``legoesm.constants``.

Grid: thermodynamic (zt) inputs; the zm-output fluxes are regridded zt->zm and
the top momentum level (``k_ub_zm = nzm-1``, ascending) is zeroed, matching the
``pdf_closure_driver`` regrid. ``wp4`` additionally zeroes the surface level.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zt2zm
from legoesm.atmosphere.physics.turbulence.clubb_helpers import safe_sqrt

from legoesm import constants

_EP1 = (1.0 - constants.epsilon) / constants.epsilon
_EP2 = 1.0 / constants.epsilon




# ---------------------------------------------------------------------------
# Two-/tri-normal PDF moment integrals (pure; no physical constants)
# ---------------------------------------------------------------------------

def calc_wp2xp_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'^2 x'>`` over the binormal (w, x) PDF (``calc_wp2xp_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    return (a * ((dw1 ** 2 + varnce_w_1) * dx1
                 + 2.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dw1)
            + (1.0 - a) * ((dw2 ** 2 + varnce_w_2) * dx2
                           + 2.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dw2))


def calc_wpxp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'x'^2>`` over the binormal (w, x) PDF (``calc_wpxp2_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    return (a * (dw1 * (dx1 ** 2 + varnce_x_1)
                 + 2.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dx1)
            + (1.0 - a) * (dw2 * (dx2 ** 2 + varnce_x_2)
                           + 2.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dx2))


def calc_wp2xp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                    varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'^2 x'^2>`` over the binormal (w, x) PDF (``calc_wp2xp2_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    term1 = (dw1 ** 2 * (dx1 ** 2 + varnce_x_1)
             + 4.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dx1 * dw1
             + (dx1 ** 2 + (1.0 + 2.0 * corr_w_x_1 ** 2) * varnce_x_1) * varnce_w_1)
    term2 = (dw2 ** 2 * (dx2 ** 2 + varnce_x_2)
             + 4.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dx2 * dw2
             + (dx2 ** 2 + (1.0 + 2.0 * corr_w_x_2 ** 2) * varnce_x_2) * varnce_w_2)
    return a * term1 + (1.0 - a) * term2


def calc_wp4_pdf(wm, w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac):
    """``<w'^4>`` over the two-component normal w-PDF (``calc_wp4_pdf``)."""
    a = mixt_frac
    d1, d2 = w_1 - wm, w_2 - wm
    return (a * (3.0 * varnce_w_1 ** 2 + 6.0 * d1 ** 2 * varnce_w_1 + d1 ** 4)
            + (1.0 - a) * (3.0 * varnce_w_2 ** 2 + 6.0 * d2 ** 2 * varnce_w_2 + d2 ** 4))


def calc_wpxpyp_pdf(wm, xm, ym, w_1, w_2, x_1, x_2, y_1, y_2,
                    varnce_w_1, varnce_w_2, varnce_x_1, varnce_x_2, varnce_y_1, varnce_y_2,
                    corr_w_x_1, corr_w_x_2, corr_w_y_1, corr_w_y_2,
                    corr_x_y_1, corr_x_y_2, mixt_frac):
    """``<w'x'y'>`` over the trinormal (w, x, y) PDF (``calc_wpxpyp_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    dy1, dy2 = y_1 - ym, y_2 - ym
    comp1 = (dw1 * dx1 * dy1 + corr_x_y_1 * safe_sqrt(varnce_x_1 * varnce_y_1) * dw1
             + corr_w_y_1 * safe_sqrt(varnce_w_1 * varnce_y_1) * dx1
             + corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dy1)
    comp2 = (dw2 * dx2 * dy2 + corr_x_y_2 * safe_sqrt(varnce_x_2 * varnce_y_2) * dw2
             + corr_w_y_2 * safe_sqrt(varnce_w_2 * varnce_y_2) * dx2
             + corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dy2)
    return a * comp1 + (1.0 - a) * comp2


# ---------------------------------------------------------------------------
# Higher-order-moment orchestration (ADG1: velocity-scalar corrs = 0)
# ---------------------------------------------------------------------------

def calc_pdf_higher_order_moments(adg1, wm_zt, rtm, thlm, um, vm,
                                  corr_rt_thl_1, corr_rt_thl_2, gr: CLUBBGrid):
    """Integrate the ADG1 PDF for the velocity-scalar higher-order moments.

    Returns a dict with the Fortran moment names (``wp2rtp``, ``wp2thlp``,
    ``wp2up``, ``wpup2``, ``wpvp2``, ``wp2up2_zm``, ``wp2vp2_zm``, ``wp4_zm``,
    ``wprtp2``, ``wpthlp2``, ``wprtpthlp``). For ADG1 the w-scalar correlations
    are zero; only ``corr_rt_thl`` (per component) is nonzero.
    """
    mf = adg1["mixt_frac"]
    z = jnp.zeros_like(mf)
    w1, w2 = adg1["w_1"], adg1["w_2"]
    vw1, vw2 = adg1["varnce_w_1"], adg1["varnce_w_2"]
    nzm = gr.zm.shape[1]
    k_ub = nzm - 1
    k_lb = 0

    def _wp2xp(xm, x1, x2, vx1, vx2):
        return calc_wp2xp_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    def _wpxp2(xm, x1, x2, vx1, vx2):
        return calc_wpxp2_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    def _wp2xp2(xm, x1, x2, vx1, vx2):
        return calc_wp2xp2_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    wp2rtp = _wp2xp(rtm, adg1["rt_1"], adg1["rt_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"])
    wp2thlp = _wp2xp(thlm, adg1["thl_1"], adg1["thl_2"], adg1["varnce_thl_1"], adg1["varnce_thl_2"])
    wp2up = _wp2xp(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wpup2 = _wpxp2(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wpvp2 = _wpxp2(vm, adg1["v_1"], adg1["v_2"], adg1["varnce_v_1"], adg1["varnce_v_2"])

    wp2up2_zt = _wp2xp2(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wp2vp2_zt = _wp2xp2(vm, adg1["v_1"], adg1["v_2"], adg1["varnce_v_1"], adg1["varnce_v_2"])
    wp2up2_zm = zt2zm(wp2up2_zt, gr).at[:, k_ub].set(0.0)
    wp2vp2_zm = zt2zm(wp2vp2_zt, gr).at[:, k_ub].set(0.0)

    wp4_zt = calc_wp4_pdf(wm_zt, w1, w2, vw1, vw2, mf)
    wp4_zm = zt2zm(wp4_zt, gr, zm_min=0.0).at[:, k_lb].set(0.0).at[:, k_ub].set(0.0)

    wprtp2 = _wpxp2(rtm, adg1["rt_1"], adg1["rt_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"])
    wpthlp2 = _wpxp2(thlm, adg1["thl_1"], adg1["thl_2"], adg1["varnce_thl_1"], adg1["varnce_thl_2"])

    wprtpthlp = calc_wpxpyp_pdf(
        wm_zt, rtm, thlm, w1, w2, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"], adg1["thl_2"],
        vw1, vw2, adg1["varnce_rt_1"], adg1["varnce_rt_2"],
        adg1["varnce_thl_1"], adg1["varnce_thl_2"],
        z, z, z, z, corr_rt_thl_1, corr_rt_thl_2, mf)

    return {
        "wp2rtp": wp2rtp, "wp2thlp": wp2thlp, "wp2up": wp2up,
        "wpup2": wpup2, "wpvp2": wpvp2,
        "wp2up2_zm": wp2up2_zm, "wp2vp2_zm": wp2vp2_zm, "wp4_zm": wp4_zm,
        "wprtp2": wprtp2, "wpthlp2": wpthlp2, "wprtpthlp": wprtpthlp,
    }


# ---------------------------------------------------------------------------
# Cloud-water fluxes x'rc' (ADG1: corr_w_chi = 0)
# ---------------------------------------------------------------------------

def calc_xprcp_component(wm, rtm, thlm, um, vm, rcm,
                         w_i, rt_i, thl_i, u_i, v_i, varnce_w_i,
                         stdev_chi_i, stdev_eta_i, corr_chi_eta_i, crt_i, cthl_i,
                         rc_i, cloud_frac_i):
    """Per-component cloud-water covariances (``calc_xprcp_component``, ADG1).

    Returns ``(wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp)``. ``crt_i``/``cthl_i``
    are the chi sensitivities from the chi/eta transform; the ``cthl=0`` (rsatl=0)
    limit is guarded (the result is masked by ``cloud_frac=0`` there).
    """
    drc = rc_i - rcm
    wprcp = (w_i - wm) * drc
    wp2rcp = ((w_i - wm) ** 2 + varnce_w_i) * drc
    crt_safe = jnp.where(crt_i == 0.0, 1.0, crt_i)
    rtprcp = ((rt_i - rtm) * drc
              + (corr_chi_eta_i * stdev_eta_i + stdev_chi_i) / (2.0 * crt_safe)
              * stdev_chi_i * cloud_frac_i)
    cthl_safe = jnp.where(cthl_i == 0.0, 1.0, cthl_i)
    thlprcp = ((thl_i - thlm) * drc
               + (corr_chi_eta_i * stdev_eta_i - stdev_chi_i) / (2.0 * cthl_safe)
               * stdev_chi_i * cloud_frac_i)
    uprcp = (u_i - um) * drc
    vprcp = (v_i - vm) * drc
    return wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp


def calc_pdf_xprcp_fluxes(adg1, comp, wm_zt, rtm, thlm, um, vm, rcm_zt, gr: CLUBBGrid):
    """Mixed cloud-water turbulent fluxes from the ADG1 PDF (``calc_pdf_xprcp_fluxes``).

    Calls :func:`calc_xprcp_component` for each PDF component, mixes the six
    fluxes (``w'rc'``, ``w'^2 rc'``, ``rt'rc'``, ``thl'rc'``, ``u'rc'``,
    ``v'rc'``) by ``mixt_frac`` on zt, then regrids the five zm-output fluxes
    zt->zm with the top momentum level (``k_ub_zm``) zeroed (corr_w_chi = 0 for
    ADG1). ``comp`` is the per-component dict from
    :func:`clubb_pdf.calc_pdf_liquid_cloud_frac_components`.

    Returns a dict with the zt-grid fluxes (consumed by the buoyancy-flux
    assembly, which wants the native pdf-grid values) and the regridded zm fluxes.
    """
    mf = comp["mixt_frac"]
    k_ub = gr.zm.shape[1] - 1

    c1 = calc_xprcp_component(
        wm_zt, rtm, thlm, um, vm, rcm_zt,
        adg1["w_1"], adg1["rt_1"], adg1["thl_1"], adg1["u_1"], adg1["v_1"],
        adg1["varnce_w_1"], comp["stdev_chi_1"], comp["stdev_eta_1"],
        comp["corr_ce_1"], comp["crt_1"], comp["cthl_1"], comp["rc_1"], comp["cf_1"])
    c2 = calc_xprcp_component(
        wm_zt, rtm, thlm, um, vm, rcm_zt,
        adg1["w_2"], adg1["rt_2"], adg1["thl_2"], adg1["u_2"], adg1["v_2"],
        adg1["varnce_w_2"], comp["stdev_chi_2"], comp["stdev_eta_2"],
        comp["corr_ce_2"], comp["crt_2"], comp["cthl_2"], comp["rc_2"], comp["cf_2"])

    wprcp_zt, wp2rcp_zt, rtprcp_zt, thlprcp_zt, uprcp_zt, vprcp_zt = (
        mf * a + (1.0 - mf) * b for a, b in zip(c1, c2))

    def _to_zm(field_zt):
        return zt2zm(field_zt, gr).at[:, k_ub].set(0.0)

    return {
        "wprcp_zt": wprcp_zt, "wp2rcp_zt": wp2rcp_zt, "rtprcp_zt": rtprcp_zt,
        "thlprcp_zt": thlprcp_zt, "uprcp_zt": uprcp_zt, "vprcp_zt": vprcp_zt,
        "wprcp_zm": _to_zm(wprcp_zt), "rtprcp_zm": _to_zm(rtprcp_zt),
        "thlprcp_zm": _to_zm(thlprcp_zt), "uprcp_zm": _to_zm(uprcp_zt),
        "vprcp_zm": _to_zm(vprcp_zt),
    }


# ---------------------------------------------------------------------------
# Buoyancy fluxes x'thv' (uses legoesm.constants)
# ---------------------------------------------------------------------------

def calc_xpthvp_terms(exner, thv_ds_zt, wprcp_zt, wp2rcp_zt, rtprcp_zt, thlprcp_zt,
                      wpthlp_zt, wprtp_zt, wp2thlp_zt, wp2rtp_zt,
                      rtpthlp_zt, rtp2_zt, thlp2_zt, gr: CLUBBGrid):
    """Virtual-potential-temperature (buoyancy) fluxes (``calc_xpthvp_terms``).

    ``rc_coef = L_v/(exner*c_pd) - ep2*thv_ds`` and
    ``x'thv' = x'thl' + ep1*thv_ds*x'rt' + rc_coef*x'rc'`` for ``x in {w, w^2,
    rt, thl}``. The three zm-output fluxes (and ``rc_coef``) are regridded zt->zm
    with the top momentum level zeroed; ``wp2thvp`` stays on zt.

    Returns ``(wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt,
    rc_coef_zm)``.
    """
    lv, cp = constants.L_v, constants.c_pd
    rc_coef_zt = lv / (exner * cp) - _EP2 * thv_ds_zt
    wpthvp_zt = wpthlp_zt + _EP1 * thv_ds_zt * wprtp_zt + rc_coef_zt * wprcp_zt
    wp2thvp_zt = wp2thlp_zt + _EP1 * thv_ds_zt * wp2rtp_zt + rc_coef_zt * wp2rcp_zt
    rtpthvp_zt = rtpthlp_zt + _EP1 * thv_ds_zt * rtp2_zt + rc_coef_zt * rtprcp_zt
    thlpthvp_zt = thlp2_zt + _EP1 * thv_ds_zt * rtpthlp_zt + rc_coef_zt * thlprcp_zt
    k_ub = gr.zm.shape[1] - 1
    wpthvp_zm = zt2zm(wpthvp_zt, gr).at[:, k_ub].set(0.0)
    rtpthvp_zm = zt2zm(rtpthvp_zt, gr).at[:, k_ub].set(0.0)
    thlpthvp_zm = zt2zm(thlpthvp_zt, gr).at[:, k_ub].set(0.0)
    rc_coef_zm = zt2zm(rc_coef_zt, gr).at[:, k_ub].set(0.0)
    return wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt, rc_coef_zm


__all__ = [
    "calc_wp2xp_pdf",
    "calc_wpxp2_pdf",
    "calc_wp2xp2_pdf",
    "calc_wp4_pdf",
    "calc_wpxpyp_pdf",
    "calc_pdf_higher_order_moments",
    "calc_xprcp_component",
    "calc_pdf_xprcp_fluxes",
    "calc_xpthvp_terms",
]
