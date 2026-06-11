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
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zt2zm
from legoesm.atmosphere.physics.turbulence.clubb_helpers import compute_sigma_sqd_w
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


__all__ = ["compute_clubb_diagnostics"]
