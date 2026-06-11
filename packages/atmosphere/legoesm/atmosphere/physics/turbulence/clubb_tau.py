"""CLUBB dissipation time-scale (tau) family for the CAM-default tree.

The prognostic moment advances (:mod:`clubb_wp23`, :mod:`clubb_moments`,
:mod:`clubb_xm_wpxp`) each need a dissipation inverse-time-scale
(``invrs_tau_C1/C4/C6/C14/xp2_zm`` and ``invrs_tau_wp3_zt``). For the CAM-default
``l_diag_Lscale_from_tau = .false.`` the tau model is simple: ``tau =
min(Lscale/sqrt(em), taumax)``, so all the base inverse time-scales equal
``invrs_tau_zm`` (zm) / ``invrs_tau_zt`` (zt), and ``l_stability_correct_tau_zm =
.true.`` multiplies the C1/C6 branch by a Brunt-Vaisala stability correction
(``calc_stability_correction``). ``l_use_invrs_tau_N2_iso = .false.`` so C4 uses
the plain wp2 tau.

This isolates the tau computation from the larger ``advance_clubb_core``
orchestration; the parcel buoyant-sorting ``Lscale`` itself comes from
:mod:`clubb_mixing_length`. Pure / JIT-safe / differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt, zt2zm

_MAX_STABILITY_CORR = 3.0   # cap on the N2 stability enhancement (advance_helper)
_EM_MIN_COEF = 1.5          # em_min = 1.5 * w_tol^2 (constants_clubb)


def compute_tke(wp2, up2, vp2, gr: CLUBBGrid, config):
    """Turbulent kinetic energy ``em`` (zm) and ``sqrt_em_zt`` (zt).

    CAM ``l_tke_aniso = .true.`` → ``em = 0.5·(wp2 + vp2 + up2)`` (the anisotropic
    TKE); the ``.false.`` branch uses ``em = 1.5·wp2``. ``sqrt_em_zt =
    sqrt(max(zm2zt(em), em_min))`` with ``em_min = 1.5·w_tol^2``. All moment
    inputs are zm-level. Returns ``(em, sqrt_em_zt)`` — the TKE the tau model and
    MFL consume.
    """
    if config.flags.l_tke_aniso:
        em = 0.5 * (wp2 + vp2 + up2)
    else:
        em = 1.5 * wp2
    em_min = _EM_MIN_COEF * config.w_tol ** 2
    sqrt_em_zt = jnp.sqrt(jnp.maximum(zm2zt(em, gr), em_min))
    return em, sqrt_em_zt


def calc_stability_correction(brunt_vaisala_freq_sqd, Lscale_zm, em,
                              lambda0_stability_coef):
    """Brunt-Vaisala stability correction factor (``calc_stability_correction``).

    ``1 + min(lambda0·N^2·Lscale_zm^2/em, 3)`` where ``lambda0`` is zeroed in
    unstable layers (``N^2 <= 0``). All zm-level ``(ncol, nzm)``;
    ``lambda0_stability_coef`` is the tunable coefficient (scalar or per-column).
    """
    lambda0_eff = jnp.where(brunt_vaisala_freq_sqd > 0.0, lambda0_stability_coef, 0.0)
    return 1.0 + jnp.minimum(
        lambda0_eff * brunt_vaisala_freq_sqd * Lscale_zm ** 2 / em, _MAX_STABILITY_CORR)


def compute_tau_family(Lscale, em, sqrt_em_zt, brunt_vaisala_freq_sqd, gr: CLUBBGrid,
                       config):
    """CAM-default ``invrs_tau_*`` family from the parcel ``Lscale`` and TKE.

    ``tau_zt = min(Lscale/sqrt_em_zt, taumax)``, ``tau_zm =
    min(Lscale_zm/sqrt(max(em_min, em)), taumax)`` with ``Lscale_zm = max(zt2zm
    (Lscale), 0)``. The stability correction (``l_stability_correct_tau_zm =
    True``) scales the C1/C6 branch; C4/C14/xp2 use the plain wp2 tau
    (``l_use_invrs_tau_N2_iso = False``); wp3 uses the zt tau. ``Lscale``/
    ``sqrt_em_zt`` are zt-level; ``em`` (TKE) / ``brunt_vaisala_freq_sqd`` are
    zm-level. Returns a dict of the inverse time-scales the advances consume.
    """
    params = config.params
    taumax = params.taumax
    em_min = _EM_MIN_COEF * config.w_tol ** 2

    tau_zt = jnp.minimum(Lscale / sqrt_em_zt, taumax)
    Lscale_zm = jnp.maximum(zt2zm(Lscale, gr), 0.0)
    tau_zm = jnp.minimum(Lscale_zm / jnp.sqrt(jnp.maximum(em_min, em)), taumax)
    invrs_tau_zm = 1.0 / tau_zm
    invrs_tau_zt = 1.0 / tau_zt

    # em is floored to em_min here for the stability-correction division (em is
    # physically TKE >= em_min, so this is forward-identical to the reference's
    # raw-em form, but keeps the 1/em gradient finite at the floor — AD safety).
    stability_correction = calc_stability_correction(
        brunt_vaisala_freq_sqd, Lscale_zm, jnp.maximum(em, em_min),
        params.lambda0_stability_coef)
    invrs_tau_N2_zm = invrs_tau_zm * stability_correction

    return dict(
        invrs_tau_zm=invrs_tau_zm, invrs_tau_zt=invrs_tau_zt,
        invrs_tau_C1_zm=invrs_tau_N2_zm, invrs_tau_C6_zm=invrs_tau_N2_zm,
        invrs_tau_C4_zm=invrs_tau_zm, invrs_tau_C14_zm=invrs_tau_zm,
        invrs_tau_xp2_zm=invrs_tau_zm, invrs_tau_wp3_zt=invrs_tau_zt,
        tau_zm=tau_zm, tau_zt=tau_zt, stability_correction=stability_correction,
    )


__all__ = ["compute_tke", "calc_stability_correction", "compute_tau_family"]
