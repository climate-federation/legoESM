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
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid
from legoesm.atmosphere.physics.turbulence.clubb_solve import penta_solve

from legoesm import constants

_GAMMA = 1.5   # gamma_over_implicit_ts (constants_clubb)


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
]
