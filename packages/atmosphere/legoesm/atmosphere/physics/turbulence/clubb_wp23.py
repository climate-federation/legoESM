"""CLUBB coupled wp2/wp3 advance — LHS term builders (CAM-default tree).

Faithful port of the LHS sub-functions of
``advance_wp2_wp3_module.F90:advance_wp2_wp3`` for the CAM-default flags. The
wp2 (second moment, momentum/zm levels) and wp3 (third moment, thermodynamic/zt
levels) equations are solved as one **coupled pentadiagonal system** on the
interleaved grid (wp2[k] at global index 2k, wp3[k] at 2k+1); these builders
assemble the per-equation band contributions that
:func:`wp23_lhs` (a later chunk) interleaves into the penta bands.

CAM-default branches taken here (verified against the namelist):
``l_standard_term_ta = .false.`` (the ADG1 non-standard turbulent advection —
:func:`wp3_term_ta_ADG1_lhs`), ``l_tke_aniso = .true.`` (the wp2 pressure-1 term
:func:`wp2_term_pr1_lhs` is active), ``l_damp_wp3_Skw_squared = .false.`` and
``l_damp_wp2_using_em = .false.`` (the orchestration passes ``C8b = 0`` so the
Skw^2 factor in :func:`wp3_term_pr1_lhs` reduces to ``C8·invrs_tau``).

The zt->zm interpolation weights (``weights_zt2zm``) are computed inline from the
grid geometry (matching ``grid_class.F90:calc_zt2zm_weights``, ascending grid),
mirroring the inline ``weights_zm2zt`` in
:func:`clubb_moments.term_ma_zm_lhs` — no new ``CLUBBGrid`` field is required.

All arrays are on the ascending CLUBB grid; the ``C*_Skw_fnc`` skewness-damping
coefficients, ``a1/a3_coef_zt``, ``invrs_tau_*`` and ``C4``/``C8``/``C8b``/
``C11`` come from the orchestration layer (a later chunk). Pure / JIT-safe /
differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_fill_holes import (
    fill_holes_vertical,
    fill_holes_wp2_from_horz_tke,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, ddzt, zm2zt
from legoesm.atmosphere.physics.turbulence.clubb_moments import (
    clip_variance,
    diffusion_zm_lhs,
    diffusion_zt_lhs,
    term_ma_zm_lhs,
    term_ma_zt_lhs_upwind,
)
from legoesm.atmosphere.physics.turbulence.clubb_solve import penta_solve

from legoesm import constants

_TWO_THIRDS = 2.0 / 3.0
_GAMMA = 1.5   # gamma_over_implicit_ts (constants_clubb)
_EPS = 1.0e-10  # constants_clubb eps = max(1e-10, machine eps): branch threshold floor

# clip_skewness (clip_explicit) algorithm constants (CLUBB; not tunable):
_WP3_MAX = 100.0           # absolute |wp3| limit [m^3/s^3] ("known magic number")
_SFC_AGL_THRESH_M = 100.0  # surface-layer threshold for the tighter skewness limit [m AGL]
_SFC_SKW_FACTOR = 0.0021   # surface-layer wp3_lim_sqd factor (clip_explicit.F90)


def _safe_sqrt(x):
    """``sqrt(max(x,0))`` with a finite (0) gradient at ``x<=0`` (double-where)."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


def clip_skewness(wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag):
    """Limit ``|Sk_w| = |wp3|/wp2_zt^(3/2)`` (``clip_skewness``, CAM branch).

    Faithful port of the ``l_use_wp3_lim_with_smth_Heaviside = .false.`` branch
    of ``clip_explicit.F90:clip_skewness_core`` (the CAM default — the smooth
    Heaviside path is the conv-test variant): a sharp 100 m-AGL threshold caps
    ``wp3^2`` to ``Skw_max_mag^2·wp2_zt^3`` aloft and to
    ``0.0021·Skw_max_mag^2·wp2_zt^3`` in the surface layer, then clips
    ``|wp3| <= 100``. ``wp2_zt`` (``>= 0``) and ``wp3`` are zt-level
    ``(ncol, nzt)``; ``sfc_elevation``/``Skw_max_mag`` are ``(ncol,)``. Pure /
    JIT-safe / differentiable.
    """
    wp2_zt_cubed = wp2_zt ** 3
    zagl = zt - sfc_elevation[:, None]
    skw_sq = Skw_max_mag[:, None] ** 2
    wp3_lim_sqd = jnp.where(zagl <= _SFC_AGL_THRESH_M,
                            _SFC_SKW_FACTOR * skw_sq * wp2_zt_cubed,
                            skw_sq * wp2_zt_cubed)
    exceed = wp3 ** 2 > wp3_lim_sqd
    wp3 = jnp.where(exceed, jnp.sign(wp3) * _safe_sqrt(wp3_lim_sqd), wp3)
    return jnp.clip(wp3, -_WP3_MAX, _WP3_MAX)


def weights_zt2zm(gr: CLUBBGrid):
    """zt->zm interpolation weights ``(ncol, nzm, 2)`` (``calc_zt2zm_weights``).

    Ascending grid (grid_dir = +1): for interior momentum level ``k`` the two
    columns are ``[T_ABOVE, T_BELOW]`` weighting ``zt[k-1]`` and ``zt[k]`` onto
    ``zm[k]``; the boundary levels (``k=0``, ``k=nzm-1``) use the reference's
    linear extension. On a uniform grid both interior weights are ``1/2``.
    """
    zm, zt = gr.zm, gr.zt
    denom = (zt[:, 1:] - zt[:, :-1]) + 1.0e-30        # (ncol, nzm-2), interior k=1..nzm-2
    t_above_int = (zm[:, 1:-1] - zt[:, :-1]) / denom
    t_below_int = (zt[:, 1:] - zm[:, 1:-1]) / denom

    denom0 = (zt[:, 1] - zt[:, 0]) + 1.0e-30
    t_above_0 = (zm[:, 0] - zt[:, 0]) / denom0
    t_below_0 = (zt[:, 1] - zm[:, 0]) / denom0

    denomn = (zt[:, -1] - zt[:, -2]) + 1.0e-30
    t_above_n = (zm[:, -1] - zt[:, -2]) / denomn
    t_below_n = (zt[:, -1] - zm[:, -1]) / denomn

    t_above = jnp.concatenate([t_above_0[:, None], t_above_int, t_above_n[:, None]], axis=1)
    t_below = jnp.concatenate([t_below_0[:, None], t_below_int, t_below_n[:, None]], axis=1)
    return jnp.stack([t_above, t_below], axis=-1)


def wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr: CLUBBGrid):
    """Turbulent-advection LHS for wp2 (``wp2_term_ta_lhs``), ``(2, ncol, nzm)``.

    The wp2 (zm) turbulent advection couples to wp3 (zt): band 0 is the
    coefficient of ``wp3[k]`` (super1 in the penta system), band 1 of
    ``wp3[k-1]`` (sub1). Boundaries zero.
    """
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    lhs = jnp.zeros((2,) + invrs_rho_ds_zm.shape, dtype=invrs_rho_ds_zm.dtype)
    lhs = lhs.at[0, :, 1:-1].set(fac * rho_ds_zt[:, 1:])
    lhs = lhs.at[1, :, 1:-1].set(-fac * rho_ds_zt[:, :-1])
    return lhs


def wp3_term_ta_ADG1_lhs(wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
                         rho_ds_zm, invrs_rho_ds_zt, gr: CLUBBGrid):
    """5-band ADG1 turbulent-advection LHS for wp3 (``wp3_term_ta_ADG1_lhs``).

    The non-standard TA branch (CAM ``l_standard_term_ta = .false.``): wp3 (zt)
    couples to neighbouring wp3 (a1 coefficient) and wp2 (a3 coefficient) levels.
    Bands ``[super2(wp3[k+1]), super1(wp2[k+1]), main(wp3[k]), sub1(wp2[k]),
    sub2(wp3[k-1])]`` → ``(5, ncol, nzt)``; boundaries zero. Uses the inline
    zt->zm weights.
    """
    w_zt2zm = weights_zt2zm(gr)
    lhs = jnp.zeros((5,) + invrs_rho_ds_zt.shape, dtype=invrs_rho_ds_zt.dtype)

    inv = invrs_rho_ds_zt[:, 1:-1]
    a1 = a1_coef_zt[:, 1:-1]
    a3 = a3_coef_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]

    rho_up, rho_lo = rho_ds_zm[:, 2:-1], rho_ds_zm[:, 1:-2]
    wp2_up, wp2_lo = wp2[:, 2:-1], wp2[:, 1:-2]
    w3w2_up, w3w2_lo = wp3_on_wp2[:, 2:-1], wp3_on_wp2[:, 1:-2]
    wt_up_tab = w_zt2zm[:, 2:-1, 0]
    wt_up_tbe = w_zt2zm[:, 2:-1, 1]
    wt_lo_tab = w_zt2zm[:, 1:-2, 0]
    wt_lo_tbe = w_zt2zm[:, 1:-2, 1]

    lhs = lhs.at[0, :, 1:-1].set(inv * a1 * idzt * rho_up * w3w2_up * wt_up_tab)
    lhs = lhs.at[1, :, 1:-1].set(inv * a3 * idzt * rho_up * wp2_up)
    lhs = lhs.at[2, :, 1:-1].set(
        inv * a1 * idzt * (rho_up * w3w2_up * wt_up_tbe - rho_lo * w3w2_lo * wt_lo_tab))
    lhs = lhs.at[3, :, 1:-1].set(-inv * a3 * idzt * rho_lo * wp2_lo)
    lhs = lhs.at[4, :, 1:-1].set(-inv * a1 * idzt * rho_lo * w3w2_lo * wt_lo_tbe)
    return lhs


def wp3_term_tp_lhs(coef, wp2, rho_ds_zm, invrs_rho_ds_zt, gr: CLUBBGrid):
    """Turbulent-production LHS for wp3 (``wp3_term_tp_lhs``), ``(2, ncol, nzt)``.

    ``coef`` is ``(ncol,)`` per-column. Band 0 is the coefficient of
    ``wp2[k+1]``, band 1 of ``wp2[k]``. The main calls this twice (advection
    ``coef = 1`` and pressure ``coef = -C_wp3_pr_tp``). Boundaries zero.
    """
    lhs = jnp.zeros((2,) + invrs_rho_ds_zt.shape, dtype=invrs_rho_ds_zt.dtype)
    c = coef[:, None]
    inv = invrs_rho_ds_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]
    rho_up, wp2_up = rho_ds_zm[:, 2:-1], wp2[:, 2:-1]
    rho_lo, wp2_lo = rho_ds_zm[:, 1:-2], wp2[:, 1:-2]
    lhs = lhs.at[0, :, 1:-1].set(c * (-3.0 * inv * idzt * rho_up * wp2_up + 1.5 * idzt * wp2_up))
    lhs = lhs.at[1, :, 1:-1].set(c * (3.0 * inv * idzt * rho_lo * wp2_lo - 1.5 * idzt * wp2_lo))
    return lhs


def wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for wp3 (``wp3_terms_ac_pr2_lhs``), ``(ncol, nzt)``.

    ``(1 - C11_Skw_fnc)·3·d(wm_zm)/dz`` at interior zt levels; boundaries zero.
    """
    lhs = jnp.zeros(C11_Skw_fnc.shape, dtype=C11_Skw_fnc.dtype)
    d_wm = wm_zm[:, 2:-1] - wm_zm[:, 1:-2]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0 * gr.invrs_dzt[:, 1:-1] * d_wm)
    return lhs


def wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for wp2 (``wp2_terms_ac_pr2_lhs``), ``(ncol, nzm)``.

    ``(1 - C_uu_shr)·2·d(wm_zt)/dz`` at interior zm levels; ``C_uu_shr`` is
    ``(ncol,)``. Boundaries zero.
    """
    lhs = jnp.zeros((C_uu_shr.shape[0], gr.invrs_dzm.shape[1]), dtype=gr.invrs_dzm.dtype)
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C_uu_shr[:, None]) * 2.0 * gr.invrs_dzm[:, 1:-1] * d_wm)
    return lhs


def wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm):
    """Dissipation-1 LHS for wp2 (``wp2_term_dp1_lhs``), ``(ncol, nzm)``.

    ``C1_Skw_fnc·invrs_tau_C1_zm`` at interior zm levels; boundaries zero.
    """
    lhs = jnp.zeros_like(C1_Skw_fnc)
    lhs = lhs.at[:, 1:-1].set(C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1])
    return lhs


def wp2_term_pr1_lhs(C4, invrs_tau_C4_zm):
    """Pressure-1 LHS for wp2 (``wp2_term_pr1_lhs``, CAM ``l_tke_aniso = True``).

    ``(2·C4·invrs_tau_C4_zm)/3`` at interior zm levels; ``C4`` is ``(ncol,)``.
    Boundaries zero. ``(ncol, nzm)``.
    """
    lhs = jnp.zeros_like(invrs_tau_C4_zm)
    lhs = lhs.at[:, 1:-1].set((2.0 * C4[:, None] * invrs_tau_C4_zm[:, 1:-1]) / 3.0)
    return lhs


def wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt):
    """Pressure-1 LHS for wp3 (``wp3_term_pr1_lhs``), ``(ncol, nzt)``.

    ``C8·invrs_tau_wp3_zt·(3·C8b·Skw_zt^2 + 1)`` at interior zt levels; ``C8``,
    ``C8b`` are ``(ncol,)``. In the CAM-default tree ``l_damp_wp3_Skw_squared =
    .false.`` so the orchestration passes ``C8b = 0`` and the factor reduces to
    ``C8·invrs_tau_wp3_zt``. Boundaries zero.
    """
    lhs = jnp.zeros_like(invrs_tau_wp3_zt)
    lhs = lhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1]
        * (3.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2 + 1.0))
    return lhs


# ---------------------------------------------------------------------------
# RHS term builders
# ---------------------------------------------------------------------------
# Most match CLUBB-JAX (CAM == ARM for these). Two diverge: the CLUBB-JAX port
# hardcodes the ARM defaults l_damp_wp2_using_em=True and
# l_use_tke_in_wp3_pr_turb_term=True, but CAM sets BOTH False — so for those the
# CAM-branch formula is taken from the CLUBB Fortran
# (CESM .../CLUBB_core/advance_wp2_wp3_module.F90), not from CLUBB-JAX (which
# only implements the True branch). Those two carry no JAX bit-exact oracle and
# are validated structurally + golden-pinned + against the Fortran by hand.


def wp2_term_pr_dfsn_rhs(C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm,
                         wpup2, wpvp2, wp3, gr: CLUBBGrid):
    """Pressure-diffusion RHS for wp2 (``wp2_term_pr_dfsn_rhs``), ``(ncol, nzm)``.

    ``C_wp2_pr_dfsn·d/dz[rho_ds·(w'u'^2 + w'v'^2 + w'^3)]/rho_ds`` at interior zm
    levels; the lower boundary copies the first interior value (per the
    reference). ``wpup2``/``wpvp2``/``wp3`` are zt-level; ``C_wp2_pr_dfsn`` is
    ``(ncol,)``.
    """
    wpuip2 = wpup2 + wpvp2 + wp3
    rhs = jnp.zeros_like(invrs_rho_ds_zm)
    fac = C_wp2_pr_dfsn[:, None] * invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    interior = fac * (rho_ds_zt[:, 1:] * wpuip2[:, 1:] - rho_ds_zt[:, :-1] * wpuip2[:, :-1])
    rhs = rhs.at[:, 1:-1].set(interior)
    rhs = rhs.at[:, 0].set(rhs[:, 1])
    return rhs


def wp3_term_pr_dfsn_rhs(C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
                         wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr: CLUBBGrid):
    """Pressure-diffusion RHS for wp3 (``wp3_term_pr_dfsn_rhs``), ``(ncol, nzt)``.

    ``C_wp3_pr_dfsn·d/dz[rho_ds·net]/rho_ds`` where ``net = (w'^2 u'^2 + w'^2 v'^2
    + w'^4) - w'^2(u'^2 + v'^2 + w'^2)`` (all zm-level), at interior zt levels;
    boundaries zero. ``C_wp3_pr_dfsn`` is ``(ncol,)``.
    """
    net = (wp2up2 + wp2vp2 + wp4) - wp2 * (up2 + vp2 + wp2)
    rhs = jnp.zeros_like(invrs_rho_ds_zt)
    fac = C_wp3_pr_dfsn[:, None] * invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
    val = rho_ds_zm[:, 2:-1] * net[:, 2:-1] - rho_ds_zm[:, 1:-2] * net[:, 1:-2]
    rhs = rhs.at[:, 1:-1].set(fac * val)
    return rhs


def wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp):
    """Buoyancy + pressure-2 RHS for wp2 (``wp2_terms_bp_pr2_rhs``), ``(ncol, nzm)``.

    ``(1 - C_uu_buoy)·2·(g/thv_ds)·w'thv'`` at interior zm levels; boundaries
    zero. ``C_uu_buoy`` is ``(ncol,)``; uses ``constants.g`` (CLUBB ``grav``).
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C_uu_buoy[:, None]) * 2.0
        * (constants.g / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1])
    return rhs


def wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, um, vpwp, vm,
                     gr: CLUBBGrid):
    """Pressure-3 RHS for wp2 (``wp2_term_pr3_rhs``), ``(ncol, nzm)``.

    ``(2/3)·[C_uu_buoy·(g/thv_ds)·w'thv' + C_uu_shr·(-u'w'·d(um)/dz
    - v'w'·d(vm)/dz)]`` clamped to ``>= 0`` at interior zm levels; boundaries
    zero. ``um``/``vm`` are zt-level; ``upwp``/``vpwp`` zm-level;
    ``C_uu_shr``/``C_uu_buoy`` are ``(ncol,)``; uses ``constants.g``.
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    buoy = C_uu_buoy[:, None] * (constants.g / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
    shear = C_uu_shr[:, None] * (
        -upwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
        - vpwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1]))
    val = jnp.maximum(_TWO_THIRDS * (buoy + shear), 0.0)
    rhs = rhs.at[:, 1:-1].set(val)
    return rhs


def wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm):
    """Pressure-1 RHS for wp2 (``wp2_term_pr1_rhs``, CAM ``l_tke_aniso = True``).

    ``C4·(u'^2 + v'^2)·invrs_tau_C4_zm / 3`` at interior zm levels; boundaries
    zero. ``C4`` is ``(ncol,)``. ``(ncol, nzm)``.
    """
    rhs = jnp.zeros_like(invrs_tau_C4_zm)
    rhs = rhs.at[:, 1:-1].set(
        (C4[:, None] * (up2[:, 1:-1] + vp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]) / 3.0)
    return rhs


def wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp):
    """Buoyancy + pressure-2 RHS for wp3 (``wp3_terms_bp1_pr2_rhs``), ``(ncol, nzt)``.

    ``(1 - C11_Skw_fnc)·3·(g/thv_ds)·w'^2 thv'`` at interior zt levels;
    boundaries zero. Uses ``constants.g``.
    """
    rhs = jnp.zeros_like(thv_ds_zt)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0
        * (constants.g / thv_ds_zt[:, 1:-1]) * wp2thvp[:, 1:-1])
    return rhs


def wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3):
    """Pressure-1 RHS for wp3 (``wp3_term_pr1_rhs``), ``(ncol, nzt)``.

    ``C8·invrs_tau_wp3_zt·(2·C8b·Skw_zt^2)·wp3`` at interior zt levels;
    boundaries zero. In the CAM-default tree ``l_damp_wp3_Skw_squared = .false.``
    so the orchestration passes ``C8b = 0`` and this term vanishes. ``C8``,
    ``C8b`` are ``(ncol,)``.
    """
    rhs = jnp.zeros_like(wp3)
    rhs = rhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1]
        * (2.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2) * wp3[:, 1:-1])
    return rhs


# --- CAM-branch builders taken from the CLUBB Fortran (CLUBB-JAX has only the
#     ARM/True branch of each) ------------------------------------------------

def wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, threshold):
    """Dissipation-1 RHS for wp2 — CAM ``l_damp_wp2_using_em = .false.`` branch.

    From ``advance_wp2_wp3_module.F90:wp2_term_dp1_rhs`` (the ``.false.`` path):
    the wp2 dissipation damps ``w'^2`` only toward its floor ``threshold``
    (``w_tol^2``), so the explicit RHS is ``+(C1_Skw_fnc·invrs_tau)·threshold``
    at interior zm levels (boundaries zero). Note: in this branch
    ``C1_Skw_fnc`` carries NO ``1/3`` factor (the ``1/3`` is the ARM/True path).
    CLUBB-JAX implements only the True path (``-(C1_Skw_fnc·invrs_tau)·(u'^2 +
    v'^2)``), so this has no JAX bit-exact oracle.
    """
    rhs = jnp.zeros_like(C1_Skw_fnc)
    rhs = rhs.at[:, 1:-1].set(
        (C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1]) * threshold)
    return rhs


def wp3_term_pr_turb_rhs(C_wp3_pr_turb, Kh_zt, wpthvp, dum_dz, dvm_dz,
                         upwp, vpwp, thv_ds_zt, gr: CLUBBGrid):
    """Pressure-turbulence RHS for wp3 — CAM ``l_use_tke_in_wp3_pr_turb_term =
    .false.`` branch (the experimental shear term, CLUBB TRAC #411).

    From ``advance_wp2_wp3_module.F90:wp3_term_pr_turb_rhs`` (the ``.false.``
    path):

      ``-C_wp3_pr_turb·Kh_zt·d/dz{ (g/thv_ds)·Δw'thv'
          - Δ(u'w'·du/dz) - Δ(v'w'·dv/dz) }``

    at interior zt levels, where ``Δ`` is the zm-level difference bracketing the
    zt level. ``wpthvp``/``upwp``/``vpwp``/``dum_dz``/``dvm_dz`` are zm-level
    (``dum_dz = ddzt(um)``); ``Kh_zt``/``thv_ds_zt`` are zt-level;
    ``C_wp3_pr_turb`` is ``(ncol,)``; uses ``constants.g``. Boundaries zero.
    CLUBB-JAX implements only the TKE (True) path, so this has no JAX bit-exact
    oracle — verified against the Fortran formula + golden-pinned.
    """
    rhs = jnp.zeros_like(Kh_zt)
    C = C_wp3_pr_turb[:, None]
    buoy = (constants.g / thv_ds_zt[:, 1:-1]) * (wpthvp[:, 2:-1] - wpthvp[:, 1:-2])
    shr_u = upwp[:, 2:-1] * dum_dz[:, 2:-1] - upwp[:, 1:-2] * dum_dz[:, 1:-2]
    shr_v = vpwp[:, 2:-1] * dvm_dz[:, 2:-1] - vpwp[:, 1:-2] * dvm_dz[:, 1:-2]
    rhs = rhs.at[:, 1:-1].set(
        -C * Kh_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1] * (buoy - shr_u - shr_v))
    return rhs


def compute_a1_a3_coef(sigma_sqd_w, a3_coef_min, gr: CLUBBGrid):
    """ADG1 ``a1``/``a3`` coefficients on zm and zt levels (``advance_wp2_wp3`` pre-compute).

    ``a1 = 1/(1 - sigma_sqd_w)``; ``a3 = max(-2·(1 - sigma_sqd_w)^2 + 3,
    a3_coef_min)`` (zm-level), then interpolated to zt. ``sigma_sqd_w`` is
    ``(ncol, nzm)`` (``< 1``); ``a3_coef_min`` is ``(ncol,)``. Returns
    ``(a1_coef, a3_coef, a1_coef_zt, a3_coef_zt)``.
    """
    one_minus = 1.0 - sigma_sqd_w
    a1_coef = 1.0 / one_minus
    a3_coef = jnp.maximum(-2.0 * one_minus ** 2 + 3.0, a3_coef_min[:, None])
    return a1_coef, a3_coef, zm2zt(a1_coef, gr), zm2zt(a3_coef, gr)


def compute_skw_fnc(C, Cb, Cc, Skw):
    """Skewness-dependent CLUBB coefficient (``C1_Skw_fnc`` / ``C11_Skw_fnc``).

    ``Cb + (C - Cb)·exp(-½·(Skw/Cc)^2)`` where ``|C - Cb|`` exceeds the
    floor ``|C + Cb|·eps/2`` (else just ``Cb``). ``C``/``Cb``/``Cc`` are
    ``(ncol,)`` params; ``Skw`` is ``(ncol, nz)`` on the matching grid (zm for
    C1, zt for C11). NOTE: the CAM-default ``l_damp_wp2_using_em = .false.`` path
    does NOT apply the extra ``1/3`` factor to ``C1_Skw_fnc`` (that is the True
    path); the caller decides.
    """
    diff = jnp.abs(C - Cb)[:, None]
    thresh = (jnp.abs(C + Cb) * _EPS / 2.0)[:, None]
    smooth = Cb[:, None] + (C[:, None] - Cb[:, None]) * jnp.exp(-0.5 * (Skw / Cc[:, None]) ** 2)
    return jnp.where(diff > thresh, smooth, Cb[:, None] * jnp.ones_like(Skw))


# ---------------------------------------------------------------------------
# Pentadiagonal assembly + solve (interleaved wp2[2k] / wp3[2k+1])
# ---------------------------------------------------------------------------

def wp23_rhs(*, nzm, invrs_dt, rhs_pr_turb_wp3, rhs_pr_dfsn_wp3, rhs_pr_dfsn_wp2,
             rhs_pr1_wp2, rhs_bp1_pr2_wp3, rhs_pr1_wp3, rhs_bp_pr2_wp2,
             rhs_pr3_wp2, rhs_dp1_wp2, lhs_pr1_wp2, lhs_tp_wp3, lhs_pr1_wp3,
             lhs_dp1_wp2, lhs_ta_wp3, wp2, wp3, w_tol_sqd,
             l_ho_nontrad_coriolis=False, fcor_y=None, wp2up=None, upwp=None):
    """Assemble the coupled wp2/wp3 explicit RHS vector (``wp23_rhs``).

    Faithful port of ``advance_wp2_wp3_module.F90:wp23_rhs``. Interleaving: wp2[k]
    at global index 2k, wp3[k] at 2k+1 (ascending grid). Over-implicit terms use
    the ``(1 - gamma)·(-lhs·field)`` contributions; the four corner rows are the
    BCs (lower wp2 = current value, lower/upper wp3 = 0, upper wp2 = ``w_tol_sqd``).
    ``l_ho_nontrad_coriolis`` (CAM ``.false.``) is a static feature gate. Returns
    ``(ncol, 2*nzm-1)``.
    """
    ngrdcol = wp2.shape[0]
    ndim = 2 * nzm - 1
    rhs = jnp.zeros((ngrdcol, ndim), dtype=wp2.dtype)

    # wp3 interior (global 3,5,..): pressure-turb + pressure-dfsn
    rhs = rhs.at[:, 3:-2:2].set(rhs_pr_turb_wp3[:, 1:-1] + rhs_pr_dfsn_wp3[:, 1:-1])
    # wp2 interior (global 2,4,..): pressure-dfsn
    rhs = rhs.at[:, 2:-1:2].set(rhs_pr_dfsn_wp2[:, 1:-1])

    # l_tke_aniso=True: wp2 pr1 + over-implicit pr1
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _GAMMA) * (-lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # wp3 time tendency + turbulent production (over-implicit) + buoyancy/pr2/pr1
    rhs = rhs.at[:, 3:-2:2].add(invrs_dt * wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _GAMMA) * (-lhs_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                          - lhs_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]))
    rhs = rhs.at[:, 3:-2:2].add(rhs_bp1_pr2_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(rhs_pr1_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add((1.0 - _GAMMA) * (-lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1]))

    # wp2 time tendency + buoyancy/pr2 + pr3 + dp1 (over-implicit)
    rhs = rhs.at[:, 2:-1:2].add(invrs_dt * wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_bp_pr2_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr3_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_dp1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _GAMMA) * (-lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # ADG1 TA for wp3 (over-implicit, 5 bands)
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _GAMMA) * (
            -lhs_ta_wp3[0, :, 1:-1] * wp3[:, 2:]
            - lhs_ta_wp3[1, :, 1:-1] * wp2[:, 2:-1]
            - lhs_ta_wp3[2, :, 1:-1] * wp3[:, 1:-1]
            - lhs_ta_wp3[3, :, 1:-1] * wp2[:, 1:-2]
            - lhs_ta_wp3[4, :, 1:-1] * wp3[:, :-2]))

    if l_ho_nontrad_coriolis:   # CAM default False (static gate)
        fy = jnp.asarray(fcor_y)
        fy = fy[:, None] if fy.ndim == 1 else fy
        rhs = rhs.at[:, 2:-1:2].add(2.0 * fy * upwp[:, 1:-1])
        rhs = rhs.at[:, 3:-2:2].add(3.0 * fy * jnp.asarray(wp2up)[:, 1:-1])

    # Boundary conditions (k_lb_zm = 0 on the ascending grid)
    rhs = rhs.at[:, 0].set(wp2[:, 0])
    rhs = rhs.at[:, 1].set(0.0)
    rhs = rhs.at[:, 2 * nzm - 3].set(0.0)
    rhs = rhs.at[:, 2 * nzm - 2].set(w_tol_sqd)
    return rhs


def wp23_lhs(*, nzm, ndim, invrs_dt, lhs_ma_zm, lhs_diff_zm, lhs_ta_wp2,
             lhs_ac_pr2_wp2, lhs_dp1_wp2, lhs_pr1_wp2, lhs_splat_wp2,
             lhs_ma_zt, lhs_diff_zt, lhs_tp_wp3, lhs_ac_pr2_wp3, lhs_pr1_wp3,
             lhs_splat_wp3, lhs_ta_wp3):
    """Assemble the coupled wp2/wp3 pentadiagonal LHS matrix (``wp23_lhs``).

    Faithful port of ``advance_wp2_wp3_module.F90:wp23_lhs``. Bands
    ``[super2, super1, main, sub1, sub2]``; wp2[k] at 2k, wp3[k] at 2k+1.
    Over-implicit implicit terms are scaled by gamma; identity rows pin the four
    BC corners. ``l_tke_aniso=True`` (wp2 pr1 on the main diagonal) and the splat
    terms (CAM ``C_wp2_splat=0`` → zero) are included. Returns ``(5, ncol, 2*nzm-1)``.
    """
    ngrdcol = lhs_dp1_wp2.shape[0]
    lhs = jnp.zeros((5, ngrdcol, ndim), dtype=lhs_dp1_wp2.dtype)

    # Lower BC identity (wp2 global 0, wp3 global 1)
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[2, :, 1].set(1.0)

    # wp2 interior rows
    lhs = lhs.at[0, :, 2:-1:2].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1])
    lhs = lhs.at[1, :, 2:-1:2].set(lhs_ta_wp2[0, :, 1:-1])
    lhs = lhs.at[2, :, 2:-1:2].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1] + lhs_ac_pr2_wp2[:, 1:-1]
        + _GAMMA * lhs_dp1_wp2[:, 1:-1] + invrs_dt)
    lhs = lhs.at[3, :, 2:-1:2].set(lhs_ta_wp2[1, :, 1:-1])
    lhs = lhs.at[4, :, 2:-1:2].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1])
    lhs = lhs.at[2, :, 2:-1:2].add(_GAMMA * lhs_pr1_wp2[:, 1:-1])   # l_tke_aniso=True
    lhs = lhs.at[2, :, 2:-1:2].add(lhs_splat_wp2[:, 1:-1])

    # wp3 interior rows
    lhs = lhs.at[0, :, 3:-2:2].set(lhs_ma_zt[0, :, 1:-1] + lhs_diff_zt[0, :, 1:-1])
    lhs = lhs.at[1, :, 3:-2:2].set(_GAMMA * lhs_tp_wp3[0, :, 1:-1])
    lhs = lhs.at[2, :, 3:-2:2].set(
        lhs_ma_zt[1, :, 1:-1] + lhs_diff_zt[1, :, 1:-1] + lhs_ac_pr2_wp3[:, 1:-1]
        + _GAMMA * lhs_pr1_wp3[:, 1:-1] + lhs_splat_wp3[:, 1:-1] + invrs_dt)
    lhs = lhs.at[3, :, 3:-2:2].set(_GAMMA * lhs_tp_wp3[1, :, 1:-1])
    lhs = lhs.at[4, :, 3:-2:2].set(lhs_ma_zt[2, :, 1:-1] + lhs_diff_zt[2, :, 1:-1])

    # ADG1 TA for wp3: add gamma*lhs_ta_wp3 to all 5 bands of the wp3 rows
    for b in range(5):
        lhs = lhs.at[b, :, 3:-2:2].add(_GAMMA * lhs_ta_wp3[b, :, 1:-1])

    # Upper BC identity (wp3 global 2*nzm-3, wp2 global 2*nzm-2)
    lhs = lhs.at[2, :, 2 * nzm - 3].set(1.0)
    lhs = lhs.at[2, :, 2 * nzm - 2].set(1.0)
    return lhs


def wp23_solve(lhs, rhs):
    """Pentadiagonal solve + de-interleave (``wp23_solve``).

    Solves the coupled system with the CLUBB-band penta LU
    (:func:`clubb_solve.penta_solve`) and splits the solution: wp2 on even
    slots, wp3 on odd. Returns ``(wp2_new, wp3_new)``.
    """
    solution = penta_solve(lhs, rhs)
    return solution[:, 0::2], solution[:, 1::2]


def advance_wp2_wp3(wp2, wp3, up2, vp2, sigma_sqd_w, wp3_on_wp2,
                    wpup2, wpvp2, wp2up2, wp2vp2, wp4, wpthvp, wp2thvp,
                    um, vm, upwp, vpwp, wm_zm, wm_zt, Kh_zm, Kh_zt,
                    invrs_tau_C4_zm, invrs_tau_wp3_zt, invrs_tau_C1_zm,
                    Skw_zm, Skw_zt, rho_ds_zm, rho_ds_zt,
                    invrs_rho_ds_zm, invrs_rho_ds_zt, thv_ds_zm, thv_ds_zt,
                    sfc_elevation, dt, gr: CLUBBGrid, config):
    """Advance the coupled wp2 (zm) / wp3 (zt) second/third moments one step.

    Faithful port of the core (non-budget) path of
    ``advance_wp2_wp3_module.F90:advance_wp2_wp3`` for the CAM-default tree.
    Assembles the RHS/LHS term builders (:mod:`clubb_wp23`), the diffusion LHS
    (:mod:`clubb_moments`) and the centered mean-advection operators, solves the
    interleaved pentadiagonal system (:func:`wp23_solve`), then applies the
    post-solve chain: ``fill_holes_vertical`` → ``fill_holes_wp2_from_horz_tke``
    (``l_wp2_fill_holes_tke``) → ``clip_variance`` (``l_min_wp2_from_corr_wx =
    False`` → the simple ``w_tol^2`` floor) → ``clip_skewness``.

    CAM gating (static / param): ``l_damp_wp2_using_em = False`` (the
    ``wp2_term_dp1_rhs`` floor form; ``C1_Skw_fnc`` carries no ``1/3``),
    ``l_damp_wp3_Skw_squared = False`` (``C8b = 0``),
    ``l_use_tke_in_wp3_pr_turb_term = False`` (the shear ``wp3_term_pr_turb_rhs``
    using ``dum/dvm_dz = ddzt(um/vm)``), ``l_tke_aniso = True``,
    ``C_wp2/wp3_splat = 0``, ``l_ho_nontrad_coriolis = False``,
    ``l_use_wp3_lim_with_smth_Heaviside = False``. Means on zt; moments/fluxes on
    the noted grids. Budget/stats (``l_sample``) are not part of the live path.

    Returns ``(wp2, wp3, wp2_zt)`` — the clipped wp2 (zm) and wp3 (zt) and the
    positive-definite wp2 interpolated to zt.
    """
    params = config.params
    flags = config.flags
    ng, nzm = wp2.shape
    nzt = nzm - 1
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    w_tol_sqd = config.w_tol ** 2

    def col(v):
        return jnp.full((ng,), v, dtype=wp2.dtype)

    C4, C8, C8b = col(params.C4), col(params.C8), col(params.C8b)
    C_uu_shr, C_uu_buoy = col(params.C_uu_shr), col(params.C_uu_buoy)
    C_wp2_pr_dfsn = col(params.C_wp2_pr_dfsn)
    C_wp3_pr_tp = col(params.C_wp3_pr_tp)
    C_wp3_pr_turb = col(params.C_wp3_pr_turb)
    C_wp3_pr_dfsn = col(params.C_wp3_pr_dfsn)
    c_K1, c_K8 = col(params.c_K1), col(params.c_K8)
    nu1, nu8 = col(params.nu1), col(params.nu8)
    C12 = col(params.C12)
    a3_min, skw_max = col(params.a3_coef_min), col(params.Skw_max_mag)

    # Skewness-dependent coefficients (CAM l_damp_wp2_using_em=False → no 1/3 on C1).
    C1_Skw_fnc = compute_skw_fnc(col(params.C1), col(params.C1b), col(params.C1c), Skw_zm)
    C11_Skw_fnc = compute_skw_fnc(col(params.C11), col(params.C11b), col(params.C11c), Skw_zt)

    a1_coef, a3_coef, a1_coef_zt, a3_coef_zt = compute_a1_a3_coef(sigma_sqd_w, a3_min, gr)
    Kw1 = c_K1[:, None] * Kh_zt   # zt-level diffusivity for wp2
    Kw8 = c_K8[:, None] * Kh_zm   # zm-level diffusivity for wp3
    dum_dz = ddzt(um, gr)
    dvm_dz = ddzt(vm, gr)

    # ---- explicit RHS terms ----
    rhs_pr_turb_wp3 = wp3_term_pr_turb_rhs(C_wp3_pr_turb, Kh_zt, wpthvp, dum_dz,
                                           dvm_dz, upwp, vpwp, thv_ds_zt, gr)
    rhs_pr_dfsn_wp3 = wp3_term_pr_dfsn_rhs(C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
                                           wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr)
    rhs_pr_dfsn_wp2 = wp2_term_pr_dfsn_rhs(C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm,
                                           wpup2, wpvp2, wp3, gr)
    rhs_bp_pr2_wp2 = wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp)
    rhs_dp1_wp2 = wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, w_tol_sqd)
    rhs_pr3_wp2 = wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp,
                                   upwp, um, vpwp, vm, gr)
    rhs_pr1_wp2 = wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm)
    rhs_bp1_pr2_wp3 = wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp)
    rhs_pr1_wp3 = wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3)

    # ---- LHS terms needed by wp23_rhs (over-implicit) ----
    lhs_diff_zm = diffusion_zm_lhs(Kw1, nu1, invrs_rho_ds_zm, rho_ds_zt, gr)
    lhs_diff_zt = diffusion_zt_lhs(Kw8, nu8, invrs_rho_ds_zt, rho_ds_zm, gr)
    lhs_tp_wp3 = (wp3_term_tp_lhs(jnp.ones((ng,), dtype=wp2.dtype), wp2, rho_ds_zm,
                                  invrs_rho_ds_zt, gr)
                  + wp3_term_tp_lhs(-C_wp3_pr_tp, wp2, rho_ds_zm, invrs_rho_ds_zt, gr))
    lhs_pr1_wp3 = wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt)
    lhs_dp1_wp2 = wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm)
    lhs_pr1_wp2 = wp2_term_pr1_lhs(C4, invrs_tau_C4_zm)
    lhs_ta_wp3 = wp3_term_ta_ADG1_lhs(wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
                                      rho_ds_zm, invrs_rho_ds_zt, gr)
    lhs_splat_wp2 = jnp.zeros((ng, nzm), dtype=wp2.dtype)   # C_wp2_splat = 0
    lhs_splat_wp3 = jnp.zeros((ng, nzt), dtype=wp2.dtype)   # C_wp3_splat = 0

    rhs = wp23_rhs(
        nzm=nzm, invrs_dt=invrs_dt, rhs_pr_turb_wp3=rhs_pr_turb_wp3,
        rhs_pr_dfsn_wp3=rhs_pr_dfsn_wp3, rhs_pr_dfsn_wp2=rhs_pr_dfsn_wp2,
        rhs_pr1_wp2=rhs_pr1_wp2, rhs_bp1_pr2_wp3=rhs_bp1_pr2_wp3,
        rhs_pr1_wp3=rhs_pr1_wp3, rhs_bp_pr2_wp2=rhs_bp_pr2_wp2,
        rhs_pr3_wp2=rhs_pr3_wp2, rhs_dp1_wp2=rhs_dp1_wp2, lhs_pr1_wp2=lhs_pr1_wp2,
        lhs_tp_wp3=lhs_tp_wp3, lhs_pr1_wp3=lhs_pr1_wp3, lhs_dp1_wp2=lhs_dp1_wp2,
        lhs_ta_wp3=lhs_ta_wp3, wp2=wp2, wp3=wp3, w_tol_sqd=w_tol_sqd)

    # C12 scales the wp3 diffusion LHS (after the RHS, before the LHS assembly).
    lhs_diff_zt = lhs_diff_zt * C12[None, :, None]

    # wp2 mean advection is centered; wp3 uses upwind (CAM l_upwind_xm_ma=True).
    lhs_ma_zm = term_ma_zm_lhs(wm_zm, gr)
    lhs_ma_zt = term_ma_zt_lhs_upwind(wm_zt, gr)
    lhs_ta_wp2 = wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr)
    lhs_ac_pr2_wp2 = wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr)
    lhs_ac_pr2_wp3 = wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr)

    lhs = wp23_lhs(
        nzm=nzm, ndim=ndim, invrs_dt=invrs_dt, lhs_ma_zm=lhs_ma_zm,
        lhs_diff_zm=lhs_diff_zm, lhs_ta_wp2=lhs_ta_wp2, lhs_ac_pr2_wp2=lhs_ac_pr2_wp2,
        lhs_dp1_wp2=lhs_dp1_wp2, lhs_pr1_wp2=lhs_pr1_wp2, lhs_splat_wp2=lhs_splat_wp2,
        lhs_ma_zt=lhs_ma_zt, lhs_diff_zt=lhs_diff_zt, lhs_tp_wp3=lhs_tp_wp3,
        lhs_ac_pr2_wp3=lhs_ac_pr2_wp3, lhs_pr1_wp3=lhs_pr1_wp3,
        lhs_splat_wp3=lhs_splat_wp3, lhs_ta_wp3=lhs_ta_wp3)

    wp2_new, wp3_new = wp23_solve(lhs, rhs)

    # ---- post-solve fill_holes / clip ----
    wp2_c = fill_holes_vertical(wp2_new, rho_ds_zm, gr.dzm, w_tol_sqd,
                                1, nzm - 2, flags.fill_holes_type)
    if flags.l_wp2_fill_holes_tke:
        wp2_c, _, _ = fill_holes_wp2_from_horz_tke(wp2_c, up2, vp2, w_tol_sqd, 0, nzm - 3)
    wp2_c = clip_variance(wp2_c, w_tol_sqd)
    wp2_zt = jnp.maximum(zm2zt(wp2_c, gr), w_tol_sqd)
    wp3_c = clip_skewness(wp3_new, wp2_zt, gr.zt, sfc_elevation, skw_max)
    return wp2_c, wp3_c, wp2_zt


__all__ = [
    "weights_zt2zm",
    "wp2_term_ta_lhs",
    "wp3_term_ta_ADG1_lhs",
    "wp3_term_tp_lhs",
    "wp3_terms_ac_pr2_lhs",
    "wp2_terms_ac_pr2_lhs",
    "wp2_term_dp1_lhs",
    "wp2_term_pr1_lhs",
    "wp3_term_pr1_lhs",
    "wp2_term_pr_dfsn_rhs",
    "wp3_term_pr_dfsn_rhs",
    "wp2_terms_bp_pr2_rhs",
    "wp2_term_pr3_rhs",
    "wp2_term_pr1_rhs",
    "wp3_terms_bp1_pr2_rhs",
    "wp3_term_pr1_rhs",
    "wp2_term_dp1_rhs",
    "wp3_term_pr_turb_rhs",
    "wp23_rhs",
    "wp23_lhs",
    "wp23_solve",
    "compute_a1_a3_coef",
    "compute_skw_fnc",
    "clip_skewness",
    "advance_wp2_wp3",
]
