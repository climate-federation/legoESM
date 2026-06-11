"""CLUBB monotonic turbulent-flux limiter — JAX-compatible helpers (CAM ON).

The CAM-default tree runs the monotonic turbulent-flux limiter
(``l_mono_flux_lim_{rtm,thlm,um,vm,spikefix} = .true.``): after the xm/wpxp
solve it adjusts the mean field ``xm`` and its flux ``wpxp`` so the turbulent
flux cannot create spurious vertical extrema. This module ports the pieces of
``mono_flux_limiter.F90`` that are pure / JIT-safe / differentiable:

  * the mean up/down vertical velocity of the assumed PDF
    (:func:`calc_mean_w_up_down_component`, :func:`mean_vert_vel_up_down`) — the
    erf-based component means used to bound the reachable level range, and
  * the implicit xm re-solve (:func:`mfl_xm_lhs`/:func:`mfl_xm_rhs`/
    :func:`mfl_xm_solve`) that re-advances ``xm`` once the limited flux is known.

The reference's ``calc_turb_adv_range`` (the data-dependent integer level-range
search) runs as host numpy with Python early-stops; a pure-JAX (masked
``lax`` loop) reformulation and the limiter core are a later chunk so the whole
limiter stays inside the jitted, differentiable step.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid
from legoesm.atmosphere.physics.turbulence.clubb_moments import term_ma_zt_lhs_upwind
from legoesm.atmosphere.physics.turbulence.clubb_solve import tridiag_solve

# Monotonic-flux-limiter field ids + their max-variance caps (constants_clubb).
MFL_RTM, MFL_THLM, MFL_UM, MFL_VM = "rtm", "thlm", "um", "vm"
_MAX_XP2 = {MFL_RTM: 5.0e-6, MFL_THLM: 5.0, MFL_UM: 10.0, MFL_VM: 10.0}

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def calc_mean_w_up_down_component(w_i, varnce_w_i, wm):
    """Mean down/up vertical velocity of one PDF component (``calc_mean_w_up_down_component``).

    Returns ``(mean_w_down, mean_w_up)`` on the zm grid for the assumed-Gaussian
    component ``(w_i, varnce_w_i)``: the truncated-Gaussian means split at
    ``w = 0``, with three saturating branches (the component is too weak vs the
    grid velocity ``wm``, all-down, or all-up). Domain boundaries zeroed.
    Pure-jnp (erf/exp) → differentiable.
    """
    wi = w_i
    # AD-safe sqrt: sqrt is never evaluated at <=0 in the primal, so the VJP
    # stays finite (sqrt(0) has an infinite derivative). Forward-identical to
    # sqrt(max(varnce,0)) since both give 0 where varnce<=0.
    var_pos = varnce_w_i > 0.0
    sig_raw = jnp.sqrt(jnp.where(var_pos, varnce_w_i, 1.0))
    sig = jnp.where(var_pos, sig_raw, 0.0)
    sig_s = jnp.where(var_pos, sig_raw, 1.0)
    z = (0.0 - wi) / (_SQRT_2 * sig_s)
    ev = jnp.exp(-z ** 2)
    ef = jax.scipy.special.erf(z)
    too_weak = jnp.abs(wi) + 3.0 * sig <= wm
    all_dn = (~too_weak) & (wi + 3.0 * sig <= 0.0)
    all_up = (~too_weak) & (~all_dn) & (wi - 3.0 * sig >= 0.0)
    mwd_m = -sig / _SQRT_2PI * ev + wi * 0.5 * (1.0 + ef)
    mwu_m = sig / _SQRT_2PI * ev + wi * 0.5 * (1.0 - ef)
    mwd = jnp.where(too_weak, 0.0, jnp.where(all_dn, wi, jnp.where(all_up, 0.0, mwd_m)))
    mwu = jnp.where(too_weak, 0.0, jnp.where(all_dn, 0.0, jnp.where(all_up, wi, mwu_m)))
    mwd = mwd.at[:, 0].set(0.0).at[:, -1].set(0.0)
    mwu = mwu.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return mwd, mwu


def mean_vert_vel_up_down(w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac, wm):
    """Mixt-frac-weighted mean down/up vertical velocity (``mean_vert_vel_up_down``).

    Combines the two PDF components' :func:`calc_mean_w_up_down_component` results.
    Returns ``(mean_w_down, mean_w_up)`` on the zm grid.
    """
    mwd1, mwu1 = calc_mean_w_up_down_component(w_1, varnce_w_1, wm)
    mwd2, mwu2 = calc_mean_w_up_down_component(w_2, varnce_w_2, wm)
    mean_w_down = mixt_frac * mwd1 + (1.0 - mixt_frac) * mwd2
    mean_w_up = mixt_frac * mwu1 + (1.0 - mixt_frac) * mwu2
    return mean_w_down, mean_w_up


def mfl_xm_lhs(wm_zt, invrs_dt, gr: CLUBBGrid):
    """LHS of the MFL xm re-solve tridiagonal system (``mfl_xm_lhs``).

    The re-solve advances ``xm`` alone (the flux ``w'x'`` is now known), so it is
    a plain tridiagonal: the upwind mean-advection operator (CAM
    ``l_upwind_xm_ma = True``) plus ``1/dt`` on the main diagonal.
    ``(3, ncol, nzt)``.
    """
    return term_ma_zt_lhs_upwind(wm_zt, gr).at[1, :, :].add(invrs_dt)


def mfl_xm_rhs(xm_old, wpxp, xm_forcing, invrs_dt, invrs_rho_ds_zt, invrs_dzt, rho_ds_zm):
    """RHS of the MFL xm re-solve (``mfl_xm_rhs``), ``(ncol, nzt)``.

    ``xm_old/dt + xm_forcing - (1/rho_ds_zt)·d(rho_ds_zm·w'x')/dz`` with the
    limited flux ``wpxp``.
    """
    return (xm_old * invrs_dt + xm_forcing
            - invrs_rho_ds_zt * invrs_dzt
            * (rho_ds_zm[:, 1:] * wpxp[:, 1:] - rho_ds_zm[:, :-1] * wpxp[:, :-1]))


def mfl_xm_solve(lhs, rhs):
    """Solve the MFL xm re-solve tridiagonal system (``mfl_xm_solve``).

    Implicit tridiagonal re-solve (``l_mfl_xm_imp_adj = True``) via the CLUBB-band
    Thomas solver (:func:`clubb_solve.tridiag_solve`).
    """
    return tridiag_solve(lhs, rhs)


__all__ = [
    "MFL_RTM", "MFL_THLM", "MFL_UM", "MFL_VM",
    "calc_mean_w_up_down_component",
    "mean_vert_vel_up_down",
    "mfl_xm_lhs",
    "mfl_xm_rhs",
    "mfl_xm_solve",
]
