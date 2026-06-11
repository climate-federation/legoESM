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
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt, zt2zm
from legoesm.atmosphere.physics.turbulence.clubb_moments import term_ma_zt_lhs_upwind
from legoesm.atmosphere.physics.turbulence.clubb_solve import tridiag_solve

# Monotonic-flux-limiter field ids + their max-variance caps (constants_clubb).
MFL_RTM, MFL_THLM, MFL_UM, MFL_VM = "rtm", "thlm", "um", "vm"
_MAX_XP2 = {MFL_RTM: 5.0e-6, MFL_THLM: 5.0, MFL_UM: 10.0, MFL_VM: 10.0}

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_F64_EPS = float(jnp.finfo(jnp.float64).eps)


def _safe_sqrt(x):
    """AD-safe ``sqrt`` with a finite gradient at ``x<=0``, propagating NaN.

    ``sqrt`` is never evaluated at ``<=0`` in the primal (finite VJP), giving 0
    where ``x==0``; a NaN input is passed through (rather than masked to 0) so a
    corrupted variance field surfaces instead of being silently clipped — matches
    the reference ``sqrt(NaN)=NaN`` propagation.
    """
    is_pos = x > 0.0
    safe = jnp.where(is_pos, x, 1.0)
    root = jnp.where(is_pos, jnp.sqrt(safe), 0.0)
    return jnp.where(jnp.isnan(x), x, root)


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


def calc_turb_adv_range(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                        mixt_frac_zm, gr: CLUBBGrid, dt):
    """Range of zt levels reachable by turbulent advection in one step (``calc_turb_adv_range``).

    Pure-JAX (masked ``lax.fori_loop``) reformulation of the host-numpy
    level-range search (``l_constant_thickness=False``, ascending grid): from
    each zt level ``k`` it walks down (using the mean **up** velocity ``vvu``) and
    up (using the mean **down** velocity ``vvd``) accumulating travel time until
    it exceeds ``dt`` or hits a velocity-sign barrier. Returns integer index
    arrays ``(low_lev_effect, high_lev_effect)``, each ``(ncol, nzt)``, that widen
    the limiter's allowable min/max window. The integer outputs are used only as
    masks downstream (no gradient flows through them). JIT-safe; the early-stops
    are replaced by a ``done`` mask over the fixed ``nzt`` loop bound.
    """
    ng, nzm = w_1_zm.shape
    nzt = nzm - 1
    gd = 1.0   # grid_dir, ascending
    dzm = gr.dzm
    wm = gd * dzm / dt
    vvd, vvu = mean_vert_vel_up_down(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                                     mixt_frac_zm, wm)

    def low_col(vvu_c, dzm_c):
        def low_at_k(k):
            def body(i, carry):
                da, done, low = carry
                j = k - 1 - i
                active = (j >= 0) & jnp.logical_not(done)
                ja = jnp.clip(j + 1, 1, nzm - 1)
                vu = vvu_c[ja]
                barrier = vu <= 0.0
                contrib = gd * dzm_c[ja] / jnp.where(vu > 0.0, vu, 1.0)
                new_da = jnp.where(active & jnp.logical_not(barrier), da + contrib, da)
                time_stop = active & jnp.logical_not(barrier) & (new_da >= dt)
                low_j = jnp.where(active, j, low)
                new_low = jnp.where(active & barrier, jnp.minimum(j + 1, nzt - 1), low_j)
                return (new_da, done | (active & (barrier | time_stop)), new_low)
            _, _, low = jax.lax.fori_loop(0, nzt, body, (0.0, False, 0))
            return low
        return jax.vmap(low_at_k)(jnp.arange(nzt))

    def high_col(vvd_c, dzm_c):
        def high_at_k(k):
            def body(i, carry):
                da, done, high = carry
                j = k + 1 + i
                active = (j <= nzt - 1) & jnp.logical_not(done)
                ja = jnp.clip(j, 0, nzm - 1)
                vd = vvd_c[ja]
                barrier = vd >= 0.0
                contrib = -gd * dzm_c[ja] / jnp.where(vd < 0.0, vd, -1.0)
                new_da = jnp.where(active & jnp.logical_not(barrier), da + contrib, da)
                time_stop = active & jnp.logical_not(barrier) & (new_da >= dt)
                high_j = jnp.where(active, j, high)
                new_high = jnp.where(active & barrier, jnp.maximum(j - 1, 0), high_j)
                return (new_da, done | (active & (barrier | time_stop)), new_high)
            _, _, high = jax.lax.fori_loop(0, nzt, body, (0.0, False, k))
            return high
        return jax.vmap(high_at_k)(jnp.arange(nzt))

    low = jax.vmap(low_col)(vvu, dzm)
    high = jax.vmap(high_col)(vvd, dzm)

    # Boundary levels are set explicitly (the reference does not search them).
    k = jnp.arange(nzt)[None, :]
    low = jnp.where(k == 0, 0, low)
    low = jnp.where(k == nzt - 2, nzt - 2, low)
    low = jnp.where(k == nzt - 1, nzt - 1, low)
    high = jnp.where(k == 0, 0, high)
    high = jnp.where(k == nzt - 2, nzt - 1, high)
    high = jnp.where(k == nzt - 1, nzt - 1, high)
    return low, high


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


def monotonic_turbulent_flux_limit(
    solve_type, xm, wpxp, xm_old, xp2, wm_zt, xm_forcing,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
    xp2_threshold, xm_tol, low_lev_effect, high_lev_effect,
    gr: CLUBBGrid, dt, l_mono_flux_lim_spikefix=True,
):
    """Monotonic turbulent-flux limiter core (``monotonic_turbulent_flux_limit``).

    Pure-JAX reformulation of the host-numpy ``_monotonic_turbulent_flux_limit``:
    bounds the turbulent flux ``wpxp`` so the mean field ``xm`` stays within
    ``±max(2·sqrt(x'^2), xm_tol)`` of its no-flux value over the
    turbulent-advection-reachable level window ``[low_lev_effect,
    high_lev_effect]`` (the masked min/max), then re-advances ``xm`` implicitly
    with the limited flux. The sequential per-level flux clip (each level's bound
    uses the already-clipped level below) is a :func:`lax.scan`; the level window
    is a mask; the top spike-fix conserves the column. ``solve_type`` (static MFL
    id) selects the variance cap, the ``rtm`` spike-fix, and the wind (uv)
    non-negativity skip. Returns ``(xm, wpxp)``. Differentiable in the field
    inputs (the integer level bounds are stop-gradient masks).
    """
    ng, nzt = xm.shape
    nzm = nzt + 1
    is_uv = solve_type in (MFL_UM, MFL_VM)
    max_xp2 = _MAX_XP2[solve_type]
    spikefix_rtm = bool(l_mono_flux_lim_spikefix) and solve_type == MFL_RTM
    invrs_dt = 1.0 / dt
    gd = 1.0   # grid_dir, ascending
    dzt = gr.dzt
    invrs_dzt = gr.invrs_dzt

    xm_enter = xm

    xp2_zt = jnp.clip(zm2zt(xp2, gr), xp2_threshold, max_xp2)
    max_dev = jnp.maximum(2.0 * _safe_sqrt(xp2_zt), xm_tol)
    xm_without_ta = xm_old + dt * xm_forcing
    min_x_lev = xm_without_ta - max_dev
    if not is_uv:
        min_x_lev = jnp.maximum(min_x_lev, 0.0)
    max_x_lev = xm_without_ta + max_dev

    # Windowed min/max over the reachable level window [low, high] (masked).
    j = jnp.arange(nzt)[None, None, :]
    in_win = (j >= low_lev_effect[:, :, None]) & (j <= high_lev_effect[:, :, None])
    min_x_allowable = jnp.min(jnp.where(in_win, min_x_lev[:, None, :], jnp.inf), axis=2)
    max_x_allowable = jnp.max(jnp.where(in_win, max_x_lev[:, None, :], -jnp.inf), axis=2)

    thr_term_zt = invrs_dt * gd * dzt * (xm_without_ta - min_x_allowable)
    mfl_max_term_zt = rho_ds_zt * thr_term_zt
    mfl_min_term_zt = rho_ds_zt * invrs_dt * gd * dzt * (xm_without_ta - max_x_allowable)
    thr_term_zm = zt2zm(thr_term_zt, gr)   # (ng, nzm)

    # Sequential clip of wpxp over interior zm levels k=1..nzm-2 (scan over levels).
    # step s -> k=s+1, k_zt=s, k-1=s. Each step's bound uses the clipped k-1 flux.
    def step(wp_prev, xs):
        max_term, min_term, thr_km1, irho_k, rho_km1, wp_k = xs
        spikefix_cond = (spikefix_rtm & (jnp.abs(wp_prev) > thr_km1) & (wp_prev < 0.0))
        mfl_max = jnp.where(spikefix_cond, 0.0,
                            irho_k * (max_term + rho_km1 * wp_prev))
        mfl_min = irho_k * (min_term + rho_km1 * wp_prev)
        clipped = jnp.where(wp_k > mfl_max, mfl_max,
                            jnp.where(wp_k < mfl_min, mfl_min, wp_k))
        needed = jnp.abs(clipped - wp_k) > _F64_EPS
        return clipped, (clipped, needed)

    xs = (mfl_max_term_zt[:, :nzm - 2].T, mfl_min_term_zt[:, :nzm - 2].T,
          thr_term_zm[:, :nzm - 2].T, invrs_rho_ds_zm[:, 1:nzm - 1].T,
          rho_ds_zm[:, :nzm - 2].T, wpxp[:, 1:nzm - 1].T)
    _, (clipped_T, needed_T) = jax.lax.scan(step, wpxp[:, 0], xs)
    clipped_interior = clipped_T.T          # (ng, nzm-2)
    wpxp_new = jnp.concatenate([wpxp[:, :1], clipped_interior, wpxp[:, -1:]], axis=1)
    adj_needed = jnp.any(needed_T.T, axis=1)   # (ng,)

    # Re-solve xm implicitly with the limited flux (applied where adjustment fired).
    lhs = mfl_xm_lhs(wm_zt, invrs_dt, gr)
    rhs = mfl_xm_rhs(xm_old, wpxp_new, xm_forcing, invrs_dt, invrs_rho_ds_zt,
                     invrs_dzt, rho_ds_zm)
    xm_mfl = mfl_xm_solve(lhs, rhs)
    xm = jnp.where(adj_needed[:, None], xm_mfl, xm)

    # Top spike-fix: conserve column xm if the top level moved a lot.
    dz_top = gr.zm[:, -1] - gr.zm[:, -2]
    moved = jnp.abs(xm[:, -1] - xm_enter[:, -1]) > 10.0 * xm_tol
    xm_dw = rho_ds_zt[:, -1] * (xm[:, -1] - xm_enter[:, -1]) * dz_top
    k_idx = jnp.arange(nzt)[None, :]
    below_top = k_idx < (nzt - 1)
    xm_vint = jnp.sum(jnp.where(below_top, rho_ds_zt * xm * gd * dzt, 0.0), axis=1)
    small_vint = jnp.abs(xm_vint) < _F64_EPS
    coef = jnp.maximum(xm_dw / jnp.where(small_vint, 1.0, xm_vint), -0.99)
    xm_scaled = (xm * (1.0 + coef[:, None])).at[:, -1].set(xm_enter[:, -1])
    xm_smallv = xm.at[:, -1].set(xm_enter[:, -1])
    xm_fixed = jnp.where(small_vint[:, None], xm_smallv, xm_scaled)
    xm = jnp.where(moved[:, None], xm_fixed, xm)
    return xm, wpxp_new


__all__ = [
    "MFL_RTM", "MFL_THLM", "MFL_UM", "MFL_VM",
    "calc_mean_w_up_down_component",
    "mean_vert_vel_up_down",
    "calc_turb_adv_range",
    "mfl_xm_lhs",
    "mfl_xm_rhs",
    "mfl_xm_solve",
    "monotonic_turbulent_flux_limit",
]
