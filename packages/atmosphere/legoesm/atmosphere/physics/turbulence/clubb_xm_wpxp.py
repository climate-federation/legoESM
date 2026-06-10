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

from legoesm import constants


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


__all__ = [
    "xm_term_ta_lhs",
    "wpxp_term_tp_lhs",
    "wpxp_terms_ac_pr2_lhs",
    "wpxp_term_pr1_lhs",
    "wpxp_terms_bp_pr3_rhs",
]
