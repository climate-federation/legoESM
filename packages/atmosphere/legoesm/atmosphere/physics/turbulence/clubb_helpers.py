"""CLUBB advance-helper kernels (CAM-default tree): sigma_sqd_w + Brunt-Vaisala.

Faithful ports of the parts of ``CLUBB-JAX/.../sigma_sqd_w_module.py`` and
``advance_helper_module.py`` that are live under the CAM-default ``clubb_*``
flags (see ``PORT_CLUBB.md`` for the authoritative flag table). Both kernels
feed the mixing-length / time-scale and PDF closures.

CAM-default branch pruning (documented, not laziness — these branches are
*off* in the CAM-default tree the task targets):
  * ``compute_sigma_sqd_w``: ``l_predict_upwp_vpwp = .false.`` -> the
    ``upwp``/``vpwp`` correlation terms are NOT included (only the rt/thl
    w-correlations enter ``max_corr``).
  * ``calc_brunt_vaisala_freq_sqd``: ``l_use_thvm_in_bv_freq = .false.``,
    ``l_brunt_vaisala_freq_moist = .false.``,
    ``l_modify_limiters_for_cnvg_test = .false.`` -> the returned
    ``brunt_vaisala_freq_sqd`` is the *dry* form ``(g/T0) d(thlm)/dz``; the
    moist/mixed forms (``bv_moist``, ``bv_mixed``, ``bv_smth``) are still
    computed and returned because downstream mixing-length / Ri code consumes
    them regardless of the returned-form flag.

Constants come from ``legoesm.constants`` (CLAUDE.md), NOT CLUBB's own module
globals. CLUBB tunes against slightly different base constants (its grav, Cp,
Lv, Rd); the legoESM values differ by ~0.1 %, below CLUBB's tuning uncertainty
(documented in ``PORT_CLUBB.md``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import (
    CLUBBGrid,
    ddzt,
    zm2zt2zm,
    zt2zm,
)
from legoesm.atmosphere.physics.turbulence.clubb_saturation import sat_mixrat_liq

from legoesm import constants

_ONE_HUNDRED = 100.0
_ZERO_THRESHOLD = 0.0
# bv_mixed clip: Fortran min(bv, 1e8*|bv|^3) (advance_helper_module.F90).
_BV_CLIP_COEF = 1.0e8


def compute_sigma_sqd_w(
    gamma_Skw_fnc: jax.Array,
    wp2: jax.Array,
    thlp2: jax.Array,
    rtp2: jax.Array,
    wpthlp: jax.Array,
    wprtp: jax.Array,
    gr: CLUBBGrid,
    *,
    w_tol: float,
    thl_tol: float,
    rt_tol: float,
) -> jax.Array:
    """PDF width parameter ``sigma_sqd_w`` (CAM default, l_predict_upwp_vpwp=F).

    ``sigma_sqd_w = gamma_Skw_fnc * (1 - min(max_x corr_wx^2, 1))`` smoothed
    zm->zt->zm with a zero floor (``sigma_sqd_w_module.F90``). All fields are on
    momentum (zm) levels, shape ``(ngrdcol, nzm)``.

    Parameters
    ----------
    gamma_Skw_fnc : jax.Array
        Skewness-dependent gamma coefficient on zm levels.
    wp2, thlp2, rtp2 : jax.Array
        w, thl, rt variances on zm levels.
    wpthlp, wprtp : jax.Array
        w'thl', w'rt' fluxes on zm levels.
    gr : CLUBBGrid
        CLUBB staggered grid (for the zm->zt->zm smoother).
    w_tol, thl_tol, rt_tol : float
        Tolerances (from ``CLUBBConfig``) regularizing the correlation
        denominators (``100 * w_tol * x_tol``).

    Returns
    -------
    jax.Array
        ``sigma_sqd_w`` on zm levels, shape ``(ngrdcol, nzm)``.
    """
    denom_thl = jnp.sqrt(wp2 * thlp2) + _ONE_HUNDRED * w_tol * thl_tol
    denom_rtp = jnp.sqrt(wp2 * rtp2) + _ONE_HUNDRED * w_tol * rt_tol

    corr_thl_sqd = (wpthlp / denom_thl) ** 2
    corr_rtp_sqd = (wprtp / denom_rtp) ** 2
    max_corr = jnp.maximum(corr_thl_sqd, corr_rtp_sqd)

    sigma_sqd_w_tmp = gamma_Skw_fnc * (1.0 - jnp.minimum(max_corr, 1.0))
    return zm2zt2zm(sigma_sqd_w_tmp, gr, zm_min=_ZERO_THRESHOLD)


def calc_brunt_vaisala_freq_sqd(
    thlm: jax.Array,
    exner: jax.Array,
    rtm: jax.Array,
    rcm: jax.Array,
    p_in_Pa: jax.Array,
    ice_supersat_frac: jax.Array,
    bv_efold: jax.Array | float,
    T0: float,
    gr: CLUBBGrid,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Brunt-Vaisala frequency squared (CAM-default branch).

    CAM defaults assumed: ``l_use_thvm_in_bv_freq = .false.``,
    ``l_brunt_vaisala_freq_moist = .false.``,
    ``l_modify_limiters_for_cnvg_test = .false.``. The returned
    ``brunt_vaisala_freq_sqd`` is therefore the dry form ``(g/T0) d(thlm)/dz``;
    ``bv_moist``/``bv_mixed``/``bv_smth`` are still computed (downstream
    mixing-length/Ri consume them).

    All thermodynamic inputs are on thermodynamic (zt) levels,
    shape ``(ngrdcol, nzt)``; outputs are on momentum (zm) levels,
    shape ``(ngrdcol, nzm)`` (the ``ddzt``/``zt2zm`` operators move zt->zm).

    Parameters
    ----------
    thlm : jax.Array
        Liquid-water potential temperature [K] (zt).
    exner : jax.Array
        Exner function [-] (zt).
    rtm : jax.Array
        Total water mixing ratio [kg/kg] (zt).
    rcm : jax.Array
        Cloud water mixing ratio [kg/kg] (zt) (from the PDF closure).
    p_in_Pa : jax.Array
        Pressure [Pa] (zt).
    ice_supersat_frac : jax.Array
        Ice supersaturation fraction [-] (zt).
    bv_efold : jax.Array or float
        e-folding coefficient for the dry<->moist blend [-]; per-column
        ``(ngrdcol,)`` or scalar (``CLUBBParams.bv_efold``).
    T0 : float
        Reference absolute temperature [K] for the dry BV frequency.
    gr : CLUBBGrid
        CLUBB staggered grid.

    Returns
    -------
    tuple of jax.Array
        ``(brunt_vaisala_freq_sqd, bv_mixed, bv_smth, bv_dry, bv_moist)``,
        each on zm levels [1/s^2].
    """
    g = constants.g
    cp = constants.c_pd
    lv = constants.L_v
    rd = constants.R_d
    ep = constants.epsilon

    ddzt_thlm = ddzt(thlm, gr)
    bv_dry_main = (g / T0) * ddzt_thlm   # l_use_thvm_in_bv_freq = False

    T_in_K = thlm * exner + (lv / cp) * rcm
    T_in_K_zm = zt2zm(T_in_K, gr, zm_min=_ZERO_THRESHOLD)

    rsat = sat_mixrat_liq(p_in_Pa, T_in_K)
    rsat_zm = zt2zm(rsat, gr, zm_min=_ZERO_THRESHOLD)
    ddzt_rsat = ddzt(rsat, gr)

    thm = thlm + (lv / (cp * exner)) * rcm
    thm_zm = zt2zm(thm, gr, zm_min=_ZERO_THRESHOLD)
    ddzt_thm = ddzt(thm, gr)
    ddzt_rtm = ddzt(rtm, gr)

    bv_dry = (g / thm_zm) * ddzt_thm

    num_fac = 1.0 + lv * rsat_zm / (rd * T_in_K_zm)
    den_fac = 1.0 + ep * lv ** 2 * rsat_zm / (cp * rd * T_in_K_zm ** 2)
    bv_moist = g * (
        (num_fac / den_fac) * (ddzt_thm / thm_zm + (lv / (cp * T_in_K_zm)) * ddzt_rsat)
        - ddzt_rtm
    )

    bv_efold_arr = jnp.asarray(bv_efold)
    if bv_efold_arr.ndim == 1:
        bv_efold_arr = bv_efold_arr[:, None]
    ice_supersat_frac_zm = zt2zm(ice_supersat_frac, gr, zm_min=_ZERO_THRESHOLD)
    bv_mixed = bv_moist + jnp.exp(-bv_efold_arr * ice_supersat_frac_zm) * (bv_dry - bv_moist)

    # l_modify_limiters_for_cnvg_test = False -> clip then smooth (no min clamp).
    bv_clipped = jnp.minimum(bv_mixed, _BV_CLIP_COEF * jnp.abs(bv_mixed) ** 3)
    bv_smth = zm2zt2zm(bv_clipped, gr)

    # l_brunt_vaisala_freq_moist = False -> return the dry form.
    brunt_vaisala_freq_sqd = bv_dry_main
    return brunt_vaisala_freq_sqd, bv_mixed, bv_smth, bv_dry, bv_moist


__all__ = ["compute_sigma_sqd_w", "calc_brunt_vaisala_freq_sqd"]
