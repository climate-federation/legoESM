"""CLUBB nonlocal mixing length ``Lscale`` — buoyant-sorting parcel integrals.

Faithful port of the CAM-default-tree parcel method in
``CLUBB-JAX/.../mixing_length.py`` (``compute_mixing_length`` /
``calc_Lscale_directly``). CAM sets ``l_diag_Lscale_from_tau = .false.`` so this
parcel buoyant-sorting path (Golaz et al. 2002) is the live one — NOT the
tau-diagnostic path.

Method: from each level a parcel is launched up and down; it is entrained
toward the environment (rate ``mu``) and rises/sinks while it has positive
remaining TKE (the running CAPE integral). ``Lscale_up``/``Lscale_down`` are the
distances reached; ``Lscale = sqrt(Lscale_up * Lscale_down)``, floored by a
surface-layer minimum ``lminh`` and capped by ``Lscale_max``.

legoESM adaptations (vs the reference):
  * constants from ``legoesm.constants`` (g/c_pd/L_v/R_d/epsilon), not CLUBB
    module globals (~0.1 % tuning-scale difference, documented in PORT_CLUBB.md);
  * saturation from :func:`clubb_saturation.sat_mixrat_liq` (Flatau — the CAM
    default ``saturation_formula``), so the ``saturation_formula`` argument is
    dropped throughout;
  * grid operators / spacings from :mod:`clubb_grid` (the ``CLUBBGrid`` pytree);
  * the per-column loop is a ``jax.vmap`` over the column axis (CLAUDE.md:
    vmap over Python loops on array dims) rather than the reference's
    ``for i in range(ngrdcol)``, and each column uses its OWN grid spacings
    (the reference assumes a single shared grid ``gr.zt[0]``);
  * the dynamic-trip ``while`` parcel ascents/descents use a fixed-length
    ``lax.scan`` (``_bounded_while``) that freezes once the parcel's TKE is
    exhausted — bit-exact to the ``while_loop`` final state and reverse-mode
    differentiable (the reference's REFACTOR B3).

Ascending grid only (CLUBB ``grid_dir_indx = +1``): index 0 = surface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import buoyancy_coefficient
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt
from legoesm.atmosphere.physics.turbulence.clubb_saturation import sat_mixrat_liq

from legoesm import constants

# Derived thermodynamic ratios (legoESM constants).
_EP = constants.epsilon
_EP1 = (1.0 - _EP) / _EP                       # (1 - eps)/eps
_EP2 = 1.0 / _EP                               # 1/eps
_LV2_COEF = _EP * constants.L_v ** 2 / (constants.R_d * constants.c_pd)  # K^2

_ZERO = 0.0
_EPS = 1.0e-10                 # comparison tolerance (CLUBB ``eps``)
_ZLMIN = 0.1                   # minimum Lscale [m]
_LSCALE_SFCLYR_DEPTH = 500.0   # surface-layer depth for lminh [m]


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x,0))`` with a finite (0) gradient at ``x<=0`` (double-where)."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


def _bounded_while(cond_fn, body_fn, init_state, max_iters):
    """Reverse-mode-differentiable fixed-length replacement for ``lax.while_loop``.

    Runs ``max_iters`` ``lax.scan`` steps; each applies ``body_fn`` only where
    ``cond_fn`` is still true (pytree select), freezing the state otherwise. For
    a body that no-ops once its ``done`` flag is set this reproduces the
    ``while_loop`` final state bit-exactly when ``max_iters >= true trip count``;
    unlike ``while_loop`` it supports ``jax.grad``.
    """
    def step(state, _):
        run = cond_fn(state)
        new = body_fn(state)
        state2 = jax.tree_util.tree_map(lambda o, n: jnp.where(run, n, o), state, new)
        return state2, None

    final, _ = jax.lax.scan(step, init_state, None, length=max_iters)
    return final


def set_Lscale_max(l_implemented: bool, host_dx, host_dy, ngrdcol: int) -> jax.Array:
    """Maximum allowable ``Lscale`` [m] (``mixing_length.F90:set_Lscale_max``).

    In a host model the cap is ``0.25 * min(host_dx, host_dy)``; standalone it is
    ``1e5``. Returns a ``(ngrdcol,)`` array.
    """
    if l_implemented:
        return 0.25 * jnp.minimum(jnp.asarray(host_dx), jnp.asarray(host_dy))
    return jnp.full((ngrdcol,), 1.0e5)


def _parcel_thv(thl_par, rt_par, exner_j, p_j, thv_ds_j, Lv_coef_j):
    """Virtual potential temperature of the parcel at one level (Lewellen-Yoh 1993)."""
    tl_j = thl_par * exner_j
    rsat_j = sat_mixrat_liq(p_j, tl_j)
    tl_sqd = tl_j ** 2
    s_j = (rt_par - rsat_j) * tl_sqd / (tl_sqd + _LV2_COEF * rsat_j)
    rc_j = jnp.maximum(s_j, _ZERO)
    return thl_par + _EP1 * thv_ds_j * rt_par + Lv_coef_j * rc_j


def _upward_inner_while(
    k_py, tke_0, thl_init, rt_init, dCAPE_init,
    thl_precalc_up, rt_precalc_up, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py,
):
    """Upward parcel trajectory from ``k_py+2`` (ascending). See module docstring."""
    init_state = (
        k_py + 2, tke_0, thl_init, rt_init, dCAPE_init,
        jnp.bool_(False), k_py + 1, tke_0, dCAPE_init, jnp.float64(0.0),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcp, done, _jl, _tex, _dep, _dej = state
        return ~done & (j < k_ub_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_prev, done, j_last, tke_exit, dep, dej = state
        thl_new = thl_precalc_up[j] + thl * exp_mu_dzm[j]
        rt_new = rt_precalc_up[j] + rt * exp_mu_dzm[j]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j])
        dCAPE_j = grav_on_thvm[j] * (thv_new - thvm[j])
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_prev) * dzm[j]

        new_tke = tke + CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex = exhausted & ~done

        tke_exit_out = jnp.where(newly_ex, tke, tke_exit)
        dep_out = jnp.where(newly_ex, dCAPE_prev, dep)
        dej_out = jnp.where(newly_ex, dCAPE_j, dej)

        j_out = jnp.where(exhausted, j, j + 1)
        tke_out = jnp.where(exhausted, tke, new_tke)
        thl_out = jnp.where(exhausted, thl, thl_new)
        rt_out = jnp.where(exhausted, rt, rt_new)
        dCAPE_out = jnp.where(exhausted, dCAPE_prev, dCAPE_j)
        j_last_out = jnp.where(exhausted, j_last, j)
        done_out = done | exhausted
        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tke_exit_out, dep_out, dej_out)

    final = _bounded_while(cond_fn, body_fn, init_state, k_ub_zt_py)
    j_final, _tke, _thl, _rt, _dcp, done_final, j_last, tke_exit, dep, dej = final
    return j_last, done_final, j_final, tke_exit, dep, dej


def _compute_lscale_up_col(
    tke_i_col, thl_par_1_up, rt_par_1_up, dCAPE_dz_1_up, CAPE_incr_1_up,
    thl_precalc_up, rt_precalc_up, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py, nzt,
):
    """``Lscale_up`` for a single column (outer scan over launch levels)."""
    def outer_step(max_alt, k_py):
        tke_i_k = tke_i_col[k_py]
        tke_0 = tke_i_k + CAPE_incr_1_up[k_py + 1]

        # Case A: TKE exhausted before reaching k+1.
        dCAPE_1_kp1 = dCAPE_dz_1_up[k_py + 1]
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_kp1) > 0.0, dCAPE_1_kp1, 1.0)
        frac_a = -_safe_sqrt(-2.0 * tke_i_k * dzm[k_py + 1] * dCAPE_1_kp1) / safe_dCAPE_a

        # Case B/C: parcel survives the initial step -> inner ascent.
        j_last, exited_early, j_final, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = (
            _upward_inner_while(
                k_py, tke_0, thl_par_1_up[k_py + 1], rt_par_1_up[k_py + 1],
                dCAPE_dz_1_up[k_py + 1], thl_precalc_up, rt_precalc_up, exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm, zt, k_ub_zt_py,
            )
        )

        base_dist = zt[j_last] - zt[k_py]
        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_prev
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_prev) * _EPS)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = -tke_exit / safe_dCAPE_j
        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = dCAPE_exit_prev ** 2 - 2.0 * tke_exit * invrs_dzm[j_final] * dCAPE_diff
        frac_quad = (-dCAPE_exit_prev * invrs_diff * dzm[j_final]
                     - _safe_sqrt(disc) * invrs_diff * dzm[j_final])
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_up_k = jnp.where(tke_0 > 0.0, _ZLMIN + base_dist + frac_bc, _ZLMIN + frac_a)

        k_alt = zt[k_py] + Lscale_up_k
        Lscale_up_k_smooth = jnp.where(k_alt < max_alt, max_alt - zt[k_py], Lscale_up_k)
        new_max_alt = jnp.where(k_alt < max_alt, max_alt, k_alt)
        return new_max_alt, Lscale_up_k_smooth

    _, vals = jax.lax.scan(outer_step, jnp.float64(0.0), jnp.arange(nzt - 2))
    return jnp.concatenate([vals, jnp.full(2, _ZLMIN)])


def _downward_inner_while(
    k_py, tke_0, thl_init, rt_init, dCAPE_init,
    thl_precalc_down, rt_precalc_down, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_lb_zt_py, max_iters,
):
    """Downward parcel trajectory from ``k_py-2`` (ascending). See module docstring."""
    init_state = (
        k_py - 2, tke_0, thl_init, rt_init, dCAPE_init,
        jnp.bool_(False), k_py - 1, tke_0, dCAPE_init, jnp.float64(0.0),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcp, done, _jl, _tex, _dep1, _dej = state
        return ~done & (j >= k_lb_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_plus1, done, j_last, tex, dep1, dej = state
        thl_new = thl_precalc_down[j] + thl * exp_mu_dzm[j + 1]
        rt_new = rt_precalc_down[j] + rt * exp_mu_dzm[j + 1]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j])
        dCAPE_j = grav_on_thvm[j] * (thv_new - thvm[j])
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_plus1) * dzm[j + 1]

        new_tke = tke - CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex = exhausted & ~done

        tex_out = jnp.where(newly_ex, tke, tex)
        dep1_out = jnp.where(newly_ex, dCAPE_plus1, dep1)
        dej_out = jnp.where(newly_ex, dCAPE_j, dej)

        j_out = jnp.where(exhausted, j, j - 1)
        tke_out = jnp.where(exhausted, tke, new_tke)
        thl_out = jnp.where(exhausted, thl, thl_new)
        rt_out = jnp.where(exhausted, rt, rt_new)
        dCAPE_out = jnp.where(exhausted, dCAPE_plus1, dCAPE_j)
        j_last_out = jnp.where(exhausted, j_last, j)
        done_out = done | exhausted
        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tex_out, dep1_out, dej_out)

    final = _bounded_while(cond_fn, body_fn, init_state, max_iters)
    j_final, _tke, _thl, _rt, _dcp, done_final, j_last, tke_exit, dep1, dej = final
    return j_last, done_final, j_final, tke_exit, dep1, dej


def _compute_lscale_down_col(
    tke_i_col, thl_par_1_down, rt_par_1_down, dCAPE_dz_1_down, CAPE_incr_1_down,
    thl_precalc_down, rt_precalc_down, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py, k_lb_zt_py, nzt,
):
    """``Lscale_down`` for a single column (outer scan descending from the top)."""
    def outer_step(min_alt, i):
        k_py = nzt - 1 - i

        tke_i_k = tke_i_col[k_py]
        tke_0 = tke_i_k - CAPE_incr_1_down[k_py - 1]

        dCAPE_1_km1 = dCAPE_dz_1_down[k_py - 1]
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_km1) > 0.0, dCAPE_1_km1, 1.0)
        frac_a = _safe_sqrt(2.0 * tke_i_k * dzm[k_py] * dCAPE_1_km1) / safe_dCAPE_a

        j_last, exited_early, j_final, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = (
            _downward_inner_while(
                k_py, tke_0, thl_par_1_down[k_py - 1], rt_par_1_down[k_py - 1],
                dCAPE_dz_1_down[k_py - 1], thl_precalc_down, rt_precalc_down, exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm, zt, k_lb_zt_py, k_ub_zt_py,
            )
        )

        base_dist = zt[k_py] - zt[j_last]
        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_plus1
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_plus1) * _EPS)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = tke_exit / safe_dCAPE_j
        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = dCAPE_exit_plus1 ** 2 + 2.0 * tke_exit * invrs_dzm[j_final + 1] * dCAPE_diff
        frac_quad = (-dCAPE_exit_plus1 * invrs_diff * dzm[j_final + 1]
                     + _safe_sqrt(disc) * invrs_diff * dzm[j_final + 1])
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_down_k = jnp.where(tke_0 > 0.0, _ZLMIN + base_dist + frac_bc, _ZLMIN + frac_a)

        k_alt = zt[k_py] - Lscale_down_k
        Lscale_down_k_smooth = jnp.where(k_alt > min_alt, zt[k_py] - min_alt, Lscale_down_k)
        new_min_alt = jnp.where(k_alt > min_alt, min_alt, k_alt)
        return new_min_alt, (k_py, Lscale_down_k_smooth)

    init_min_alt = zt[k_ub_zt_py]
    _, (k_indices, vals) = jax.lax.scan(outer_step, init_min_alt, jnp.arange(nzt - 1))
    col = jnp.full(nzt, _ZLMIN)
    return col.at[k_indices].set(vals)


def compute_mixing_length(
    thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
    mu, lmin, l_implemented, gr: CLUBBGrid,
):
    """Nonlocal parcel mixing length (``mixing_length.F90:compute_mixing_length``).

    Parameters
    ----------
    thvm, thlm, rtm, p_in_Pa, exner, thv_ds : jax.Array
        Virtual potential temp, liquid-water potential temp, total water,
        pressure [Pa], Exner, dry-static virtual potential temp; all on
        thermodynamic (zt) levels, shape ``(ngrdcol, nzt)``.
    em : jax.Array
        TKE on momentum (zm) levels, shape ``(ngrdcol, nzm)``.
    Lscale_max : jax.Array
        Per-column cap [m], shape ``(ngrdcol,)`` (see :func:`set_Lscale_max`).
    mu : jax.Array
        Entrainment rate [1/m], shape ``(ngrdcol,)`` (``CLUBBParams.mu``).
    lmin : float
        Surface-layer minimum length [m] (``CLUBBParams.lmin_coef``-scaled).
    l_implemented : bool
        True when CLUBB runs inside a host model (surface layer above ground).
    gr : CLUBBGrid
        Ascending CLUBB staggered grid (provides ``zt``/``zm``/``dzm``/
        ``invrs_dzm``).

    Returns
    -------
    tuple of jax.Array
        ``(Lscale, Lscale_up, Lscale_down)`` on zt levels, ``(ngrdcol, nzt)``.
    """
    ngrdcol, nzt = thvm.shape
    k_ub_zt_py = nzt - 1
    k_lb_zt_py = 0

    # ---- Shared precomputations (vectorized over columns) ----
    tke_i = zm2zt(em, gr)                                  # (ngrdcol, nzt)
    grav_on_thvm = buoyancy_coefficient(thvm)
    Lv_coef = constants.L_v / (exner * constants.c_pd) - _EP2 * thv_ds

    exp_mu_dzm = jnp.exp(-mu[:, None] * gr.dzm)            # (ngrdcol, nzm)
    entrain_coef = (1.0 - exp_mu_dzm) * gr.invrs_dzm / mu[:, None]

    _pad0 = jnp.zeros((ngrdcol, 1))
    _pad2 = jnp.zeros((ngrdcol, 2))

    # Upward precalcs (parcel-from-below recurrence coefficients).
    thl_mid, thl_blw = thlm[:, 1:nzt - 1], thlm[:, 0:nzt - 2]
    rt_mid, rt_blw = rtm[:, 1:nzt - 1], rtm[:, 0:nzt - 2]
    emu_up, ec_up = exp_mu_dzm[:, 1:nzt - 1], entrain_coef[:, 1:nzt - 1]
    thl_precalc_up = jnp.concatenate(
        [_pad0, thl_mid - thl_blw * emu_up - (thl_mid - thl_blw) * ec_up, _pad2], axis=1)
    rt_precalc_up = jnp.concatenate(
        [_pad0, rt_mid - rt_blw * emu_up - (rt_mid - rt_blw) * ec_up, _pad2], axis=1)

    ec_init_up = entrain_coef[:, 1:nzt]
    thl_par_1_up_int = thlm[:, 1:] - (thlm[:, 1:] - thlm[:, :-1]) * ec_init_up
    rt_par_1_up_int = rtm[:, 1:] - (rtm[:, 1:] - rtm[:, :-1]) * ec_init_up
    tl_par_1_up_int = thl_par_1_up_int * exner[:, 1:]
    rsat_1_up_int = sat_mixrat_liq(p_in_Pa[:, 1:], tl_par_1_up_int)
    tl_sqd_up = tl_par_1_up_int ** 2
    s_1_up = (rt_par_1_up_int - rsat_1_up_int) * tl_sqd_up / (tl_sqd_up + _LV2_COEF * rsat_1_up_int)
    rc_1_up = jnp.maximum(s_1_up, _ZERO)
    thv_1_up = (thl_par_1_up_int + _EP1 * thv_ds[:, 1:] * rt_par_1_up_int
                + Lv_coef[:, 1:] * rc_1_up)
    dCAPE_dz_1_up_int = grav_on_thvm[:, 1:] * (thv_1_up - thvm[:, 1:])
    CAPE_incr_1_up_int = 0.5 * dCAPE_dz_1_up_int * gr.dzm[:, 1:nzt]

    thl_par_1_up = jnp.concatenate([_pad0, thl_par_1_up_int], axis=1)
    rt_par_1_up = jnp.concatenate([_pad0, rt_par_1_up_int], axis=1)
    dCAPE_dz_1_up = jnp.concatenate([_pad0, dCAPE_dz_1_up_int], axis=1)
    CAPE_incr_1_up = jnp.concatenate([_pad0, CAPE_incr_1_up_int], axis=1)

    # Downward precalcs (parcel-from-above recurrence coefficients).
    thl_abv, thl_at_j = thlm[:, 1:], thlm[:, :-1]
    rt_abv, rt_at_j = rtm[:, 1:], rtm[:, :-1]
    emu_dn, ec_dn = exp_mu_dzm[:, 1:nzt], entrain_coef[:, 1:nzt]
    thl_precalc_down = jnp.concatenate(
        [thl_at_j - thl_abv * emu_dn - (thl_at_j - thl_abv) * ec_dn, _pad2], axis=1)
    rt_precalc_down = jnp.concatenate(
        [rt_at_j - rt_abv * emu_dn - (rt_at_j - rt_abv) * ec_dn, _pad2], axis=1)

    ec_init_dn = entrain_coef[:, 1:nzt]
    thl_par_1_dn_int = thlm[:, :-1] - (thlm[:, :-1] - thlm[:, 1:]) * ec_init_dn
    rt_par_1_dn_int = rtm[:, :-1] - (rtm[:, :-1] - rtm[:, 1:]) * ec_init_dn
    tl_par_1_dn_int = thl_par_1_dn_int * exner[:, :-1]
    rsat_1_dn_int = sat_mixrat_liq(p_in_Pa[:, :-1], tl_par_1_dn_int)
    tl_sqd_dn = tl_par_1_dn_int ** 2
    s_1_dn = (rt_par_1_dn_int - rsat_1_dn_int) * tl_sqd_dn / (tl_sqd_dn + _LV2_COEF * rsat_1_dn_int)
    rc_1_dn = jnp.maximum(s_1_dn, _ZERO)
    thv_1_dn = (thl_par_1_dn_int + _EP1 * thv_ds[:, :-1] * rt_par_1_dn_int
                + Lv_coef[:, :-1] * rc_1_dn)
    dCAPE_dz_1_dn_int = grav_on_thvm[:, :-1] * (thv_1_dn - thvm[:, :-1])
    CAPE_incr_1_dn_int = 0.5 * dCAPE_dz_1_dn_int * gr.dzm[:, 1:nzt]

    thl_par_1_down = jnp.concatenate([thl_par_1_dn_int, _pad0], axis=1)
    rt_par_1_down = jnp.concatenate([rt_par_1_dn_int, _pad0], axis=1)
    dCAPE_dz_1_down = jnp.concatenate([dCAPE_dz_1_dn_int, _pad0], axis=1)
    CAPE_incr_1_down = jnp.concatenate([CAPE_incr_1_dn_int, _pad0], axis=1)

    # ---- Per-column up/down via vmap over the column axis ----
    def up_col(args):
        (tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
         got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c) = args
        return _compute_lscale_up_col(
            tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
            got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c,
            k_ub_zt_py, nzt)

    def down_col(args):
        (tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
         got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c) = args
        return _compute_lscale_down_col(
            tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
            got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c,
            k_ub_zt_py, k_lb_zt_py, nzt)

    up_args = (tke_i, thl_par_1_up, rt_par_1_up, dCAPE_dz_1_up, CAPE_incr_1_up,
               thl_precalc_up, rt_precalc_up, exp_mu_dzm,
               grav_on_thvm, Lv_coef, thv_ds, exner, p_in_Pa, thvm,
               gr.dzm, gr.invrs_dzm, gr.zt)
    down_args = (tke_i, thl_par_1_down, rt_par_1_down, dCAPE_dz_1_down, CAPE_incr_1_down,
                 thl_precalc_down, rt_precalc_down, exp_mu_dzm,
                 grav_on_thvm, Lv_coef, thv_ds, exner, p_in_Pa, thvm,
                 gr.dzm, gr.invrs_dzm, gr.zt)

    Lscale_up_all = jax.vmap(up_col)(up_args)
    Lscale_down_all = jax.vmap(down_col)(down_args)

    # ---- Surface-layer floor lminh + Lscale_max cap ----
    invrs_sfclyr = 1.0 / _LSCALE_SFCLYR_DEPTH
    if l_implemented:
        zm_sfc = gr.zm[:, 0]   # ascending: bottom zm level = ground
        lminh = (jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - (gr.zt - zm_sfc[:, None]))
                 * lmin * invrs_sfclyr)
    else:
        lminh = jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - gr.zt) * lmin * invrs_sfclyr

    Lscale_up = jnp.maximum(lminh, Lscale_up_all)
    Lscale_down = jnp.maximum(lminh, Lscale_down_all)
    Lscale = _safe_sqrt(Lscale_up * Lscale_down)

    # Upper boundary: Lscale[k_ub] = Lscale[k_ub - 1].
    Lscale = Lscale.at[:, k_ub_zt_py].set(Lscale[:, k_ub_zt_py - 1])
    Lscale = jnp.minimum(Lscale, Lscale_max[:, None])
    return Lscale, Lscale_up, Lscale_down


__all__ = ["compute_mixing_length", "set_Lscale_max"]
