"""CLUBB coupled xm/wpxp advance — term builders (CAM-default tree).

Faithful port of the LHS/RHS sub-functions of
``advance_xm_wpxp_module.F90:advance_xm_wpxp`` for the CAM-default flags. The
mean scalars ``xm`` (rtm/thlm, thermodynamic/zt levels) and their vertical
fluxes ``wpxp`` (wprtp/wpthlp, momentum/zm levels) are solved as one coupled
**tridiagonal-per-pair / pentadiagonal interleaved** system (xm[k] at global
index 2k, wpxp[k] at 2k+1); these builders assemble the per-equation band
contributions that the later assembly (:func:`xm_wpxp_lhs`/`xm_wpxp_rhs`)
interleaves.

CAM-default context: ``l_predict_upwp_vpwp = .false.`` (the momentum fluxes
upwp/vpwp are diagnosed, not advanced here — this module advances only the
scalar fluxes wprtp/wpthlp), ``l_upwind_xm_ma = .true.`` (upwind xm mean
advection — the shared ``term_ma_zt_lhs_upwind``). The ``C6_Skw_fnc`` /
``C7_Skw_fnc`` skewness-damping coefficients and ``invrs_tau_C6_zm`` come from
the orchestration layer (a later chunk). All arrays on the ascending CLUBB grid;
pure / JIT-safe / differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_fill_holes import fill_holes_vertical
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, ddzt, zm2zt
from legoesm.atmosphere.physics.turbulence.clubb_mfl import (
    MFL_UM,
    MFL_VM,
    monotonic_turbulent_flux_limit,
)
from legoesm.atmosphere.physics.turbulence.clubb_moments import (
    clip_covar,
    diffusion_zm_lhs,
    term_ma_zm_lhs,
    term_ma_zt_lhs_upwind,
)
from legoesm.atmosphere.physics.turbulence.clubb_solve import penta_solve

from legoesm import constants

_GAMMA = 1.5   # gamma_over_implicit_ts (constants_clubb)


def _weights_zm2zt(gr: CLUBBGrid):
    """zm->zt interpolation weights ``(ncol, nzt, 2)`` = ``[m_above, m_below]``
    (``calc_zm2zt_weights``, ascending grid), computed inline from the grid."""
    total = (gr.zm[:, 1:] - gr.zm[:, :-1]) + 1.0e-30
    w_above = (gr.zt - gr.zm[:, :-1]) / total
    w_below = (gr.zm[:, 1:] - gr.zt) / total
    return jnp.stack([w_above, w_below], axis=-1)


def xpyp_term_ta_pdf_lhs_centered(coef_zt, rho_ds_zt, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Centered ADG1 turbulent-advection LHS for w'x' (``xpyp_term_ta_pdf_lhs``).

    The CAM-default wpxp turbulent advection is implicit and centered
    (``l_explicit_turbulent_adv_wpxp = .false.``, ``l_godunov_upwind_wpxp_ta =
    .false.``) — distinct from the *upwind* operator used by xp2/xpyp
    (``l_upwind_xpyp_ta = .true.``). Discretizes
    ``(1/rho_ds_zm)·d(rho_ds_zt·coef·var_zm)/dz`` at interior zm levels using the
    inline zm->zt weights. ``coef_zt`` is ``(ncol, nzt)``. ``(3, ncol, nzm)`` =
    ``[super, main, sub]``; boundaries zero.
    """
    w2zt = _weights_zm2zt(gr)
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    rho_coef_k = rho_ds_zt[:, 1:] * coef_zt[:, 1:]
    rho_coef_km1 = rho_ds_zt[:, :-1] * coef_zt[:, :-1]
    super_int = fac * rho_coef_k * w2zt[:, 1:, 0]
    main_int = fac * (rho_coef_k * w2zt[:, 1:, 1] - rho_coef_km1 * w2zt[:, :-1, 0])
    sub_int = -fac * rho_coef_km1 * w2zt[:, :-1, 1]
    zb = jnp.zeros((coef_zt.shape[0], 1), dtype=coef_zt.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([zb, main_int, zb], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def calc_xm_wpxp_ta_terms(sigma_sqd_w, wp3_on_wp2_zt, rho_ds_zt, invrs_rho_ds_zm,
                          gr: CLUBBGrid):
    """ADG1 turbulent-advection LHS for w'x' (``calc_xm_wpxp_ta_terms``), ``(3, ncol, nzm)``.

    ``coef = a1_coef_zt·wp3_on_wp2_zt`` with ``a1_coef = 1/(1 - sigma_sqd_w)``
    regridded zm->zt, fed to :func:`xpyp_term_ta_pdf_lhs_centered`. The same
    operator serves wprtp and wpthlp (shared ADG1 TA LHS).
    """
    a1_coef_zt = zm2zt(1.0 / (1.0 - sigma_sqd_w), gr)
    coef_zt = a1_coef_zt * wp3_on_wp2_zt
    return xpyp_term_ta_pdf_lhs_centered(coef_zt, rho_ds_zt, invrs_rho_ds_zm, gr)


def calc_xm_wpxp_lhs_terms(wm_zm, wm_zt, wp2, Kw6, nu6, C7_Skw_fnc,
                           invrs_rho_ds_zm, rho_ds_zt, rho_ds_zm, invrs_rho_ds_zt,
                           gr: CLUBBGrid):
    """Shared LHS terms for the xm/w'x' system (``calc_xm_wpxp_lhs_terms``).

    Computes once and shares: the w'x' diffusion (``Kw6 = c_K6·Kh_zt``) + mean
    advection (zm centered, zt upwind — CAM ``l_upwind_xm_ma = True``) + the
    xm<->wpxp turbulent-advection / production / accumulation operators. The ADG1
    TA operator (:func:`calc_xm_wpxp_ta_terms`) is computed separately. ``nu6`` is
    a scalar background diffusivity. Returns a dict with keys ``lhs_diff_zm,
    lhs_ma_zm, lhs_ma_zt, lhs_ta_xm, lhs_tp, lhs_ac_pr2``.
    """
    nu6_arr = jnp.full((invrs_rho_ds_zm.shape[0],), nu6, dtype=wp2.dtype)
    return dict(
        lhs_diff_zm=diffusion_zm_lhs(Kw6, nu6_arr, invrs_rho_ds_zm, rho_ds_zt, gr),
        lhs_ma_zm=term_ma_zm_lhs(wm_zm, gr),
        lhs_ma_zt=term_ma_zt_lhs_upwind(wm_zt, gr),
        lhs_ta_xm=xm_term_ta_lhs(invrs_rho_ds_zt, rho_ds_zm, gr),
        lhs_tp=wpxp_term_tp_lhs(wp2, gr),
        lhs_ac_pr2=wpxp_terms_ac_pr2_lhs(C7_Skw_fnc, wm_zt, gr),
    )


def solve_xm_wpxp_with_single_lhs(wpxp, xm, wpxp_forcing, xm_forcing, C6_Skw_fnc,
                                  C7_Skw_fnc, invrs_tau_C6_zm, lhs_ta_wpxp,
                                  lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_xm,
                                  lhs_tp, lhs_ac_pr2, thv_ds_zm, xpthvp, wm_zt,
                                  dt, gr: CLUBBGrid, wp2=None, xp2_relaxed=None):
    """Solve one xm/w'x' variable pair (``solve_xm_wpxp_with_single_lhs``).

    Builds the field-specific pressure-1 LHS and buoyancy/pr3 RHS, assembles the
    coupled penta system with the shared LHS terms, solves + de-interleaves, and
    (if ``wp2``/``xp2_relaxed`` given) applies the Cauchy-Schwarz flux clip.
    ``rhs_ta = 0`` for ADG1. Returns ``(wpxp_new, xm_new)``.
    """
    lhs_pr1 = wpxp_term_pr1_lhs(C6_Skw_fnc, invrs_tau_C6_zm)
    rhs_bp_pr3 = wpxp_terms_bp_pr3_rhs(C7_Skw_fnc, thv_ds_zm, xpthvp)
    rhs_ta = jnp.zeros_like(wpxp)
    lhs = xm_wpxp_lhs(lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_wpxp, lhs_ta_xm,
                      lhs_tp, lhs_ac_pr2, lhs_pr1, dt)
    rhs = xm_wpxp_rhs(wpxp, xm, wpxp_forcing, xm_forcing, rhs_bp_pr3, rhs_ta,
                      lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm=0)
    wpxp_new, xm_new = xm_wpxp_solve(lhs, rhs)
    if wp2 is not None and xp2_relaxed is not None:
        wpxp_new = clip_covar(wpxp_new, wp2, xp2_relaxed)
    return wpxp_new, xm_new


def xm_wpxp_clipping_and_stats(solve_type, xm, wpxp_preclip, xm_old, xp2, xp2_clip,
                               wp2, wm_zt, xm_forcing, rho_ds_zm, rho_ds_zt,
                               invrs_rho_ds_zm, invrs_rho_ds_zt, xp2_threshold,
                               xm_tol, low_lev_effect, high_lev_effect, field_tol,
                               fill_holes_type, l_mono_flux_lim, dt, gr: CLUBBGrid):
    """Per-field post-solve clipping for advance_xm_wpxp (``xm_wpxp_clipping_and_stats``).

    Applied once per scalar after its solve: (1) the monotonic turbulent-flux
    limiter (no-op unless ``l_mono_flux_lim``; adjusts both ``xm`` and the flux),
    (2) ``fill_holes_vertical`` on the mean field (gated ``fill_holes_type != 0``
    and not a wind component), (3) the Cauchy-Schwarz flux clip (``clip_covar``,
    bounded by ``wp2``/``xp2_clip``). ``solve_type``/``fill_holes_type``/
    ``l_mono_flux_lim`` are static. Returns ``(xm, wpxp)``.

    CAM gating: ``l_pos_def = .false.`` (the CLUBB default — absent from the CAM
    namelist), so the Fortran's RTM ``pos_definite_adj`` branch (which would
    adjust both ``xm`` and ``wpxp`` when the new mean goes negative) is OUT of the
    CAM-default tree and not ported; CLUBB-JAX omits it identically (verified by
    the bit-exact ``xm_wpxp_clipping_and_stats`` parity test).
    """
    if l_mono_flux_lim:
        xm, wpxp_preclip = monotonic_turbulent_flux_limit(
            solve_type, xm, wpxp_preclip, xm_old, xp2, wm_zt, xm_forcing,
            rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
            xp2_threshold, xm_tol, low_lev_effect, high_lev_effect, gr, dt)
    if fill_holes_type != 0 and solve_type not in (MFL_UM, MFL_VM):
        nzt = xm.shape[1]
        xm = fill_holes_vertical(xm, rho_ds_zt, gr.dzt, field_tol, 0, nzt - 1,
                                 fill_holes_type)
    wpxp = clip_covar(wpxp_preclip, wp2, xp2_clip)
    return xm, wpxp


def diagnose_upxp(ypwp, xm, wpxp, ym, C6x_Skw_fnc, tau_C6_zm, C7_Skw_fnc, gr: CLUBBGrid):
    """Diagnose a horizontal turbulent scalar flux ``y'x'`` (``diagnose_upxp``).

    Andre et al. (1978) eqn. 7 / Bougeault et al. (1981) eqn. 4 (CAM
    ``l_predict_upwp_vpwp = .false.``): ``y'x' = (tau_C6/C6x)·(-y'w'·d(xm)/dz
    - (1-C7)·w'x'·d(ym)/dz)`` at interior zm levels; boundaries zero. ``ym`` is
    the smoothed velocity. ``(ncol, nzm)``.
    """
    ddzt_xm = ddzt(xm, gr)
    ddzt_ym = ddzt(ym, gr)
    interior = (tau_C6_zm[:, 1:-1] / C6x_Skw_fnc[:, 1:-1]) * (
        -ypwp[:, 1:-1] * ddzt_xm[:, 1:-1]
        - (1.0 - C7_Skw_fnc[:, 1:-1]) * wpxp[:, 1:-1] * ddzt_ym[:, 1:-1])
    return jnp.zeros_like(ypwp).at[:, 1:-1].set(interior)


def xm_term_ta_lhs(invrs_rho_ds_zt, rho_ds_zm, gr: CLUBBGrid):
    """Turbulent-advection LHS for xm (``xm_term_ta_lhs``), ``(2, ncol, nzt)``.

    The implicit ``(1/rho_ds_zt)·d(rho_ds_zm·w'x')/dz`` couples xm (zt) to the
    bracketing wpxp (zm) levels: band 0 is the coefficient of ``wpxp[k+1]``, band
    1 of ``wpxp[k]``.
    """
    invrs_dzt = gr.invrs_dzt
    sup = invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, 1:]
    sub = -invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, :-1]
    return jnp.stack([sup, sub], axis=0)


def wpxp_term_tp_lhs(wp2, gr: CLUBBGrid):
    """Turbulent-production LHS for w'x' (``wpxp_term_tp_lhs``), ``(2, ncol, nzm)``.

    Couples wpxp (zm) to the bracketing xm (zt) levels: band 0 is the coefficient
    of ``xm[k]`` (``+wp2·invrs_dzm``), band 1 of ``xm[k-1]`` (``-wp2·invrs_dzm``).
    Boundaries zero.
    """
    interior = wp2[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    zeros_col = jnp.zeros((wp2.shape[0], 1), dtype=wp2.dtype)
    sup = jnp.concatenate([zeros_col, interior, zeros_col], axis=1)
    sub = jnp.concatenate([zeros_col, -interior, zeros_col], axis=1)
    return jnp.stack([sup, sub], axis=0)


def wpxp_terms_ac_pr2_lhs(C7_Skw_fnc, wm_zt, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for w'x' (``wpxp_terms_ac_pr2_lhs``), ``(ncol, nzm)``.

    ``(1 - C7_Skw_fnc)·d(wm_zt)/dz`` at interior zm levels; boundaries zero.
    ``C7_Skw_fnc`` is ``(ncol, nzm)``; ``wm_zt`` is ``(ncol, nzt)``.
    """
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]
    interior = (1.0 - C7_Skw_fnc[:, 1:-1]) * gr.invrs_dzm[:, 1:-1] * d_wm
    zeros_col = jnp.zeros((C7_Skw_fnc.shape[0], 1), dtype=C7_Skw_fnc.dtype)
    return jnp.concatenate([zeros_col, interior, zeros_col], axis=1)


def wpxp_term_pr1_lhs(C6_Skw_fnc, invrs_tau_C6_zm):
    """Pressure-1 LHS for w'x' (``wpxp_term_pr1_lhs``), ``(ncol, nzm)``.

    ``C6_Skw_fnc·invrs_tau_C6_zm·w'x'`` at each zm level; boundaries zeroed.
    """
    result = C6_Skw_fnc * invrs_tau_C6_zm
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


def wpxp_terms_bp_pr3_rhs(C7_Skw_fnc, thv_ds_zm, xpthvp):
    """Buoyancy-production + pressure-3 RHS for w'x' (``wpxp_terms_bp_pr3_rhs``), ``(ncol, nzm)``.

    ``(1 - C7_Skw_fnc)·(g/thv_ds)·x'thv'`` at each zm level (``xpthvp`` = r'thv'
    or thl'thv'); boundaries zeroed. Uses ``constants.g`` (CLUBB ``grav``).
    """
    result = (constants.g / thv_ds_zm) * (1.0 - C7_Skw_fnc) * xpthvp
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


# ---------------------------------------------------------------------------
# Pentadiagonal assembly + solve (interleaved wpxp[2k] / xm[2k+1])
# ---------------------------------------------------------------------------

def xm_wpxp_lhs(lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_wpxp, lhs_ta_xm,
                lhs_tp, lhs_ac_pr2, lhs_pr1, dt):
    """Assemble the coupled xm/wpxp pentadiagonal LHS (``xm_wpxp_lhs``).

    Faithful port of ``advance_xm_wpxp_module.F90:xm_wpxp_lhs`` for the CAM tree
    (``l_implemented = False`` standalone, ``l_diffuse_rtm_and_thlm = False`` so
    the xm rows carry NO diffusion, ``l_iter = True``). Interleaving: wpxp[k] at
    global index 2k, xm[k] at 2k+1. Bands ``[super2, super1, main, sub1, sub2]``;
    the wpxp turbulent advection and pressure-1 are over-implicit (scaled by
    gamma). wpxp lower/upper rows are identity BCs. Returns ``(5, ncol, 2*nzm-1)``.
    """
    ngrdcol = lhs_diff_zm.shape[1]
    nzm = lhs_diff_zm.shape[2]
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    g = _GAMMA
    lhs = jnp.zeros((5, ngrdcol, ndim), dtype=lhs_diff_zm.dtype)

    # xm rows (odd global indices 1,3,..): mean advection (upwind) + xm<->wpxp TA.
    lhs = lhs.at[0, :, 1::2].set(lhs_ma_zt[0])          # super2: xm[k+1]
    lhs = lhs.at[1, :, 1::2].set(lhs_ta_xm[0])          # super1: wpxp[k+1]
    lhs = lhs.at[2, :, 1::2].set(invrs_dt + lhs_ma_zt[1])  # diag
    lhs = lhs.at[3, :, 1::2].set(lhs_ta_xm[1])          # sub1: wpxp[k]
    lhs = lhs.at[4, :, 1::2].set(lhs_ma_zt[2])          # sub2: xm[k-1]

    # wpxp interior rows (even global indices 2..2*(nzm-2)).
    sl = slice(2, 2 * nzm - 2, 2)
    lhs = lhs.at[0, :, sl].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1]
                              + g * lhs_ta_wpxp[0, :, 1:-1])
    lhs = lhs.at[1, :, sl].set(lhs_tp[0, :, 1:-1])      # super1: xm[k]
    lhs = lhs.at[2, :, sl].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1] + lhs_ac_pr2[:, 1:-1]
        + g * (lhs_ta_wpxp[1, :, 1:-1] + lhs_pr1[:, 1:-1]) + invrs_dt)
    lhs = lhs.at[3, :, sl].set(lhs_tp[1, :, 1:-1])      # sub1: xm[k-1]
    lhs = lhs.at[4, :, sl].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1]
                              + g * lhs_ta_wpxp[2, :, 1:-1])

    # wpxp lower (j=0) and upper (j=ndim-1) identity BC rows.
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[2, :, -1].set(1.0)
    return lhs


def xm_wpxp_rhs(wpxp, xm, wpxp_forcing, xm_forcing, rhs_bp_pr3, rhs_ta,
                lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm=0):
    """Assemble the coupled xm/wpxp explicit RHS (``xm_wpxp_rhs``).

    Faithful port of ``advance_xm_wpxp_module.F90:xm_wpxp_rhs`` (``l_iter = True``
    so ``wpxp/dt`` is added). xm rows: ``xm/dt + xm_forcing``; wpxp interior:
    buoyancy/pr3 + forcing + ``rhs_ta`` (0 for ADG1) + the over-implicit TA/pr1
    contributions + ``wpxp/dt``. wpxp lower BC carries the current value, upper BC
    is 0. Returns ``(ncol, 2*nzm-1)``.
    """
    ngrdcol, nzm = wpxp.shape
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    g = _GAMMA
    rhs = jnp.zeros((ngrdcol, ndim), dtype=wpxp.dtype)

    rhs = rhs.at[:, 0].set(wpxp[:, k_lb_zm])
    rhs = rhs.at[:, 1::2].set(xm * invrs_dt + xm_forcing)

    ta = lhs_ta_wpxp[:, :, 1:-1]
    pr1 = lhs_pr1[:, 1:-1]
    rhs_int = (
        rhs_bp_pr3[:, 1:-1] + wpxp_forcing[:, 1:-1] + rhs_ta[:, 1:-1]
        + (1.0 - g) * (-ta[0] * wpxp[:, 2:] - ta[1] * wpxp[:, 1:-1]
                       - ta[2] * wpxp[:, :-2] - pr1 * wpxp[:, 1:-1])
        + wpxp[:, 1:-1] * invrs_dt)
    rhs = rhs.at[:, 2:-1:2].set(rhs_int)
    rhs = rhs.at[:, -1].set(0.0)
    return rhs


def xm_wpxp_solve(lhs, rhs):
    """Pentadiagonal solve + de-interleave (``xm_wpxp_solve``).

    Solves with the CLUBB-band penta LU (:func:`clubb_solve.penta_solve`) and
    splits: wpxp on even slots, xm on odd. Returns ``(wpxp_new, xm_new)``.
    """
    soln = penta_solve(lhs, rhs)
    return soln[:, 0::2], soln[:, 1::2]


__all__ = [
    "xm_term_ta_lhs",
    "wpxp_term_tp_lhs",
    "wpxp_terms_ac_pr2_lhs",
    "wpxp_term_pr1_lhs",
    "wpxp_terms_bp_pr3_rhs",
    "xm_wpxp_lhs",
    "xm_wpxp_rhs",
    "xm_wpxp_solve",
    "xpyp_term_ta_pdf_lhs_centered",
    "calc_xm_wpxp_ta_terms",
    "calc_xm_wpxp_lhs_terms",
    "solve_xm_wpxp_with_single_lhs",
    "xm_wpxp_clipping_and_stats",
    "diagnose_upxp",
]
