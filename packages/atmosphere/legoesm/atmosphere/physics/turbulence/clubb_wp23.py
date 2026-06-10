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
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid


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
]
