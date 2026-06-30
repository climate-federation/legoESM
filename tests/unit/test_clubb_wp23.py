"""Tests for the CLUBB wp2/wp3 advance LHS term builders (now in ``clubb.py``).

Bit-exact parity vs CLUBB-JAX ``advance_wp2_wp3_module`` for the CAM-default
tree, plus boundary/shape sanity that runs in CI without the reference. The
zt->zm interpolation weights are checked vs ``grid_class.calc_zt2zm_weights``
(and analytically on a uniform grid).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as W  # noqa: E402, N812

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


def _gr(ng=2, nzt=10, stretched=True):
    nzm = nzt + 1
    if stretched:
        zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.1 ** np.arange(nzm)[:-1]]))
    else:
        zm_1d = np.linspace(0.0, 2000.0, nzm)
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def _refgr(gr, ng, nzm):
    """Build the reference Grid NamedTuple from a CLUBBGrid."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.derived_types.grid_class import (
        Grid,
        calc_zm2zt_weights,
        calc_zt2zm_weights,
    )
    nzt = nzm - 1
    zm_np, zt_np, dzt_np = np.asarray(gr.zm), np.asarray(gr.zt), np.asarray(gr.dzt)
    return Grid(
        nzm=nzm, nzt=nzt, ngrdcol=ng, zm=gr.zm, zt=gr.zt, dzm=gr.dzm, dzt=gr.dzt,
        invrs_dzm=gr.invrs_dzm, invrs_dzt=gr.invrs_dzt,
        weights_zt2zm=jnp.asarray(calc_zt2zm_weights(nzm, nzt, ng, zm_np, zt_np)),
        weights_zm2zt=jnp.asarray(calc_zm2zt_weights(nzm, nzt, ng, zm_np, zt_np, dzt_np)),
        k_lb_zm=0, k_ub_zm=nzm - 1, k_lb_zt=0, k_ub_zt=nzt - 1,
        grid_dir_indx=1, grid_dir=1.0)


# --------------------------------------------------------------------------
# zt2zm weights
# --------------------------------------------------------------------------

def test_weights_zt2zm_uniform_is_half():
    gr, ng, nzm = _gr(stretched=False)
    w = np.asarray(W.weights_zt2zm(gr))
    assert w.shape == (ng, nzm, 2)
    # interior weights are 1/2 on a uniform grid
    np.testing.assert_allclose(w[:, 1:-1, :], 0.5, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_weights_zt2zm_parity_stretched():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.derived_types.grid_class import calc_zt2zm_weights
    gr, ng, nzm = _gr(stretched=True)
    ref = calc_zt2zm_weights(nzm, nzm - 1, ng, np.asarray(gr.zm), np.asarray(gr.zt))
    np.testing.assert_allclose(np.asarray(W.weights_zt2zm(gr)), np.asarray(ref),
                               rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# LHS builders: inputs + shared parity sweep
# --------------------------------------------------------------------------

def _inputs(gr, ng, nzm, seed=5):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def zm(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzm)))

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzt)))

    return dict(
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt)))),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        a1_coef_zt=jnp.asarray(1.0 + 0.3 * rng.random((ng, nzt))),
        a3_coef_zt=jnp.asarray(0.5 + 0.3 * rng.random((ng, nzt))),
        wp3_on_wp2=zm(0.1),
        coef=jnp.asarray(0.4 + rng.random((ng,))),
        C11_Skw_fnc=zt(0.1, 0.5), wm_zm=zm(0.02), wm_zt=zt(0.02),
        C_uu_shr=jnp.asarray(0.3 + 0.1 * rng.random((ng,))),
        C1_Skw_fnc=zm(0.1, 1.0),
        invrs_tau_C1_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        C4=jnp.asarray(5.0 + rng.random((ng,))),
        invrs_tau_C4_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        C8=jnp.asarray(4.0 + rng.random((ng,))),
        C8b=jnp.asarray(0.3 + rng.random((ng,))),
        invrs_tau_wp3_zt=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzt))),
        Skw_zt=zt(0.5),
        # --- RHS inputs ---
        C_wp2_pr_dfsn=jnp.asarray(0.2 + 0.3 * rng.random((ng,))),
        C_wp3_pr_dfsn=jnp.asarray(0.2 + 0.3 * rng.random((ng,))),
        C_wp3_pr_turb=jnp.asarray(0.4 + 0.3 * rng.random((ng,))),
        C_uu_buoy=jnp.asarray(0.3 + 0.1 * rng.random((ng,))),
        wpup2=zt(0.02), wpvp2=zt(0.02), wp3=zt(0.05),
        wp2up2=zm(0.02), wp2vp2=zm(0.02), wp4=zm(0.2, 0.5),
        up2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        vp2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        thv_ds_zm=zm(1.0, 300.0), thv_ds_zt=zt(1.0, 300.0),
        wpthvp=zm(0.02), wp2thvp=zt(0.02),
        upwp=zm(0.05), vpwp=zm(0.05), um=zt(2.0, 5.0), vm=zt(1.0),
        Kh_zt=jnp.asarray(1.0 + 3.0 * rng.random((ng, nzt))),
        dum_dz=zm(1e-2), dvm_dz=zm(1e-2),
        threshold=float((2.0e-2) ** 2),
    )


def test_shapes_and_boundaries():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    p = _inputs(gr, ng, nzm)
    ta2 = W.wp2_term_ta_lhs(p["invrs_rho_ds_zm"], p["rho_ds_zt"], gr)
    assert ta2.shape == (2, ng, nzm)
    assert np.allclose(np.asarray(ta2)[:, :, 0], 0.0) and np.allclose(np.asarray(ta2)[:, :, -1], 0.0)
    ta3 = W.wp3_term_ta_ADG1_lhs(p["wp2"], p["a1_coef_zt"], p["a3_coef_zt"],
                                 p["wp3_on_wp2"], p["rho_ds_zm"], p["invrs_rho_ds_zt"], gr)
    assert ta3.shape == (5, ng, nzt)
    assert np.allclose(np.asarray(ta3)[:, :, 0], 0.0) and np.allclose(np.asarray(ta3)[:, :, -1], 0.0)
    dp1 = W.wp2_term_dp1_lhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"])
    assert dp1.shape == (ng, nzm)
    assert np.allclose(np.asarray(dp1)[:, 0], 0.0) and np.allclose(np.asarray(dp1)[:, -1], 0.0)


def test_pr1_wp3_reduces_when_C8b_zero():
    """CAM l_damp_wp3_Skw_squared=False → C8b=0 → factor reduces to C8*invrs_tau."""
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    C8b0 = jnp.zeros((ng,))
    out = np.asarray(W.wp3_term_pr1_lhs(p["C8"], C8b0, p["invrs_tau_wp3_zt"], p["Skw_zt"]))
    exp = np.asarray(p["C8"])[:, None] * np.asarray(p["invrs_tau_wp3_zt"])
    np.testing.assert_allclose(out[:, 1:-1], exp[:, 1:-1], rtol=1e-12, atol=1e-14)


_FIX = Path(__file__).resolve().parent / "clubb_fixtures"


def _all_lhs_outputs(gr, p):
    return dict(
        weights_zt2zm=W.weights_zt2zm(gr),
        wp2_term_ta=W.wp2_term_ta_lhs(p["invrs_rho_ds_zm"], p["rho_ds_zt"], gr),
        wp3_term_ta=W.wp3_term_ta_ADG1_lhs(p["wp2"], p["a1_coef_zt"], p["a3_coef_zt"],
                                           p["wp3_on_wp2"], p["rho_ds_zm"],
                                           p["invrs_rho_ds_zt"], gr),
        wp3_term_tp=W.wp3_term_tp_lhs(p["coef"], p["wp2"], p["rho_ds_zm"],
                                      p["invrs_rho_ds_zt"], gr),
        wp3_ac_pr2=W.wp3_terms_ac_pr2_lhs(p["C11_Skw_fnc"], p["wm_zm"], gr),
        wp2_ac_pr2=W.wp2_terms_ac_pr2_lhs(p["C_uu_shr"], p["wm_zt"], gr),
        wp2_dp1=W.wp2_term_dp1_lhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"]),
        wp2_pr1=W.wp2_term_pr1_lhs(p["C4"], p["invrs_tau_C4_zm"]),
        wp3_pr1=W.wp3_term_pr1_lhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"]),
    )


def test_all_lhs_builders_match_golden():
    """Non-skipped CI guard: all 8 builders + weights vs the committed golden.

    The golden was generated from this implementation when the bit-exact parity
    vs CLUBB-JAX (``test_all_lhs_builders_parity``) passed, so it pins the
    band-ordering / slice conventions even when the reference tree is absent.
    """
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    g = np.load(_FIX / "clubb_wp23_lhs_golden.npz")
    out = _all_lhs_outputs(gr, p)
    for key, ref in [("weights_zt2zm", "weights_zt2zm"), ("wp2_term_ta", "wp2_term_ta"),
                     ("wp3_term_ta", "wp3_term_ta"), ("wp3_term_tp", "wp3_term_tp"),
                     ("wp3_ac_pr2", "wp3_ac_pr2"), ("wp2_ac_pr2", "wp2_ac_pr2"),
                     ("wp2_dp1", "wp2_dp1"), ("wp2_pr1", "wp2_pr1"), ("wp3_pr1", "wp3_pr1")]:
        # FP-reassociation tolerance: the C1-C8 "condense" refactor fused/re-
        # associated the algebra (~1e-14 rel), so an exact match is no longer the
        # right invariant — pin to round-off, not bit-identity.
        np.testing.assert_allclose(np.asarray(out[key]), g[ref], rtol=1e-13, atol=1e-16)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_all_lhs_builders_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_wp2_wp3_module as R  # noqa: N812

    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    rg = _refgr(gr, ng, nzm)

    def chk(a, b):
        # Pure algebra (no solve) → bit-exact, not round-off.
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    chk(W.wp2_term_ta_lhs(p["invrs_rho_ds_zm"], p["rho_ds_zt"], gr),
        R.wp2_term_ta_lhs(p["invrs_rho_ds_zm"], p["rho_ds_zt"], rg))
    chk(W.wp3_term_ta_ADG1_lhs(p["wp2"], p["a1_coef_zt"], p["a3_coef_zt"], p["wp3_on_wp2"],
                               p["rho_ds_zm"], p["invrs_rho_ds_zt"], gr),
        R.wp3_term_ta_ADG1_lhs(p["wp2"], p["a1_coef_zt"], p["a3_coef_zt"], p["wp3_on_wp2"],
                               p["rho_ds_zm"], p["invrs_rho_ds_zt"], rg))
    chk(W.wp3_term_tp_lhs(p["coef"], p["wp2"], p["rho_ds_zm"], p["invrs_rho_ds_zt"], gr),
        R.wp3_term_tp_lhs(p["coef"], p["wp2"], p["rho_ds_zm"], p["invrs_rho_ds_zt"], rg))
    chk(W.wp3_terms_ac_pr2_lhs(p["C11_Skw_fnc"], p["wm_zm"], gr),
        R.wp3_terms_ac_pr2_lhs(p["C11_Skw_fnc"], p["wm_zm"], rg))
    chk(W.wp2_terms_ac_pr2_lhs(p["C_uu_shr"], p["wm_zt"], gr),
        R.wp2_terms_ac_pr2_lhs(p["C_uu_shr"], p["wm_zt"], rg))
    chk(W.wp2_term_dp1_lhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"]),
        R.wp2_term_dp1_lhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"]))
    chk(W.wp2_term_pr1_lhs(p["C4"], p["invrs_tau_C4_zm"]),
        R.wp2_term_pr1_lhs(p["C4"], p["invrs_tau_C4_zm"]))
    chk(W.wp3_term_pr1_lhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"]),
        R.wp3_term_pr1_lhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"]))


def test_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)

    def loss(wp2):
        ta3 = W.wp3_term_ta_ADG1_lhs(wp2, p["a1_coef_zt"], p["a3_coef_zt"], p["wp3_on_wp2"],
                                     p["rho_ds_zm"], p["invrs_rho_ds_zt"], gr)
        return jnp.sum(ta3 ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wp2"])))


# --------------------------------------------------------------------------
# RHS builders
# --------------------------------------------------------------------------

def _all_rhs_outputs(gr, p):
    return dict(
        wp2_pr_dfsn=W.wp2_term_pr_dfsn_rhs(p["C_wp2_pr_dfsn"], p["rho_ds_zt"],
                                           p["invrs_rho_ds_zm"], p["wpup2"], p["wpvp2"],
                                           p["wp3"], gr),
        wp3_pr_dfsn=W.wp3_term_pr_dfsn_rhs(p["C_wp3_pr_dfsn"], p["rho_ds_zm"],
                                           p["invrs_rho_ds_zt"], p["wp2up2"], p["wp2vp2"],
                                           p["wp4"], p["up2"], p["vp2"], p["wp2"], gr),
        wp2_bp_pr2=W.wp2_terms_bp_pr2_rhs(p["C_uu_buoy"], p["thv_ds_zm"], p["wpthvp"]),
        wp2_pr3=W.wp2_term_pr3_rhs(p["C_uu_shr"], p["C_uu_buoy"], p["thv_ds_zm"],
                                   p["wpthvp"], p["upwp"], p["um"], p["vpwp"], p["vm"], gr),
        wp2_pr1=W.wp2_term_pr1_rhs(p["C4"], p["up2"], p["vp2"], p["invrs_tau_C4_zm"]),
        wp3_bp1_pr2=W.wp3_terms_bp1_pr2_rhs(p["C11_Skw_fnc"], p["thv_ds_zt"], p["wp2thvp"]),
        wp3_pr1=W.wp3_term_pr1_rhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"], p["wp3"]),
        wp2_dp1=W.wp2_term_dp1_rhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"], p["threshold"]),
        wp3_pr_turb=W.wp3_term_pr_turb_rhs(p["C_wp3_pr_turb"], p["Kh_zt"], p["wpthvp"],
                                           p["dum_dz"], p["dvm_dz"], p["upwp"], p["vpwp"],
                                           p["thv_ds_zt"], gr),
    )


def test_rhs_builders_match_golden():
    """Non-skipped CI guard for all 9 RHS builders vs the committed golden."""
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    g = np.load(_FIX / "clubb_wp23_rhs_golden.npz")
    out = _all_rhs_outputs(gr, p)
    for key in out:
        # FP-reassociation tolerance (C1-C8 condense refactor; ~1e-14 rel).
        np.testing.assert_allclose(np.asarray(out[key]), g[key], rtol=1e-13, atol=1e-16)


def test_wp2_term_dp1_rhs_cam_branch_formula():
    """CAM l_damp_wp2_using_em=False: interior == C1_Skw_fnc*invrs_tau*threshold."""
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    out = np.asarray(W.wp2_term_dp1_rhs(p["C1_Skw_fnc"], p["invrs_tau_C1_zm"], p["threshold"]))
    exp = np.asarray(p["C1_Skw_fnc"]) * np.asarray(p["invrs_tau_C1_zm"]) * p["threshold"]
    np.testing.assert_allclose(out[:, 1:-1], exp[:, 1:-1], rtol=1e-12, atol=1e-14)
    assert np.allclose(out[:, 0], 0.0) and np.allclose(out[:, -1], 0.0)


def test_wp3_term_pr_turb_rhs_zero_on_uniform_fields():
    """Experimental shear term #411 vanishes when its bracketed fields are uniform."""
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    C = jnp.full((ng,), 0.5)
    Kh = jnp.full((ng, nzt), 2.0)
    thv = jnp.full((ng, nzt), 300.0)
    const_zm = jnp.full((ng, nzm), 0.3)
    out = np.asarray(W.wp3_term_pr_turb_rhs(C, Kh, const_zm, const_zm, const_zm,
                                            const_zm, const_zm, thv, gr))
    # wpthvp & u'w'*du/dz constant in z -> all bracketed differences vanish -> 0
    np.testing.assert_allclose(out, 0.0, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_rhs_cam_eq_arm_builders_parity():
    """Parity for the 7 RHS builders where CAM == ARM (vs CLUBB-JAX).

    The 3 buoyancy terms use gravity; the reference module `_grav` (CLUBB 9.81)
    is patched to legoESM constants.g so parity isolates the algebra. The 2
    CAM-divergent builders (wp2_term_dp1_rhs/.false., wp3_term_pr_turb_rhs/.false.)
    have no CLUBB-JAX oracle and are covered by golden + analytic tests above.
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_wp2_wp3_module as R  # noqa: N812
    from legoesm import constants
    R._grav = float(constants.g)

    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    rg = _refgr(gr, ng, nzm)

    def chk(a, b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    chk(W.wp2_term_pr_dfsn_rhs(p["C_wp2_pr_dfsn"], p["rho_ds_zt"], p["invrs_rho_ds_zm"],
                               p["wpup2"], p["wpvp2"], p["wp3"], gr),
        R.wp2_term_pr_dfsn_rhs(p["C_wp2_pr_dfsn"], p["rho_ds_zt"], p["invrs_rho_ds_zm"],
                               p["wpup2"], p["wpvp2"], p["wp3"], rg))
    chk(W.wp3_term_pr_dfsn_rhs(p["C_wp3_pr_dfsn"], p["rho_ds_zm"], p["invrs_rho_ds_zt"],
                               p["wp2up2"], p["wp2vp2"], p["wp4"], p["up2"], p["vp2"], p["wp2"], gr),
        R.wp3_term_pr_dfsn_rhs(p["C_wp3_pr_dfsn"], p["rho_ds_zm"], p["invrs_rho_ds_zt"],
                               p["wp2up2"], p["wp2vp2"], p["wp4"], p["up2"], p["vp2"], p["wp2"], rg))
    chk(W.wp2_terms_bp_pr2_rhs(p["C_uu_buoy"], p["thv_ds_zm"], p["wpthvp"]),
        R.wp2_terms_bp_pr2_rhs(p["C_uu_buoy"], p["thv_ds_zm"], p["wpthvp"]))
    chk(W.wp2_term_pr3_rhs(p["C_uu_shr"], p["C_uu_buoy"], p["thv_ds_zm"], p["wpthvp"],
                           p["upwp"], p["um"], p["vpwp"], p["vm"], gr),
        R.wp2_term_pr3_rhs(p["C_uu_shr"], p["C_uu_buoy"], p["thv_ds_zm"], p["wpthvp"],
                           p["upwp"], p["um"], p["vpwp"], p["vm"], rg))
    chk(W.wp2_term_pr1_rhs(p["C4"], p["up2"], p["vp2"], p["invrs_tau_C4_zm"]),
        R.wp2_term_pr1_rhs(p["C4"], p["up2"], p["vp2"], p["invrs_tau_C4_zm"]))
    chk(W.wp3_terms_bp1_pr2_rhs(p["C11_Skw_fnc"], p["thv_ds_zt"], p["wp2thvp"]),
        R.wp3_terms_bp1_pr2_rhs(p["C11_Skw_fnc"], p["thv_ds_zt"], p["wp2thvp"]))
    chk(W.wp3_term_pr1_rhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"], p["wp3"]),
        R.wp3_term_pr1_rhs(p["C8"], p["C8b"], p["invrs_tau_wp3_zt"], p["Skw_zt"], p["wp3"]))


# --------------------------------------------------------------------------
# Pentadiagonal assembly + solve
# --------------------------------------------------------------------------

def _assembly_inputs(ng, nzm, seed=8):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def a(*shape):
        return jnp.asarray(rng.standard_normal(shape))

    return dict(
        invrs_dt=1.0 / 300.0, w_tol_sqd=float((2.0e-2) ** 2),
        # RHS term arrays
        rhs_pr_turb_wp3=a(ng, nzt), rhs_pr_dfsn_wp3=a(ng, nzt),
        rhs_pr_dfsn_wp2=a(ng, nzm), rhs_pr1_wp2=a(ng, nzm),
        rhs_bp1_pr2_wp3=a(ng, nzt), rhs_pr1_wp3=a(ng, nzt),
        rhs_bp_pr2_wp2=a(ng, nzm), rhs_pr3_wp2=a(ng, nzm), rhs_dp1_wp2=a(ng, nzm),
        lhs_pr1_wp2=a(ng, nzm), lhs_tp_wp3=a(2, ng, nzt), lhs_pr1_wp3=a(ng, nzt),
        lhs_dp1_wp2=a(ng, nzm), lhs_ta_wp3=a(5, ng, nzt),
        wp2=jnp.asarray(0.2 + rng.random((ng, nzm))),
        wp3=a(ng, nzt),
        # LHS term arrays
        lhs_ma_zm=a(3, ng, nzm), lhs_diff_zm=a(3, ng, nzm), lhs_ta_wp2=a(2, ng, nzm),
        lhs_ac_pr2_wp2=a(ng, nzm), lhs_splat_wp2=jnp.zeros((ng, nzm)),
        lhs_ma_zt=a(3, ng, nzt), lhs_diff_zt=a(3, ng, nzt),
        lhs_ac_pr2_wp3=a(ng, nzt), lhs_splat_wp3=jnp.zeros((ng, nzt)),
    )


def _call_wp23_rhs(p, nzm):
    return W.wp23_rhs(
        nzm=nzm, invrs_dt=p["invrs_dt"], rhs_pr_turb_wp3=p["rhs_pr_turb_wp3"],
        rhs_pr_dfsn_wp3=p["rhs_pr_dfsn_wp3"], rhs_pr_dfsn_wp2=p["rhs_pr_dfsn_wp2"],
        rhs_pr1_wp2=p["rhs_pr1_wp2"], rhs_bp1_pr2_wp3=p["rhs_bp1_pr2_wp3"],
        rhs_pr1_wp3=p["rhs_pr1_wp3"], rhs_bp_pr2_wp2=p["rhs_bp_pr2_wp2"],
        rhs_pr3_wp2=p["rhs_pr3_wp2"], rhs_dp1_wp2=p["rhs_dp1_wp2"],
        lhs_pr1_wp2=p["lhs_pr1_wp2"], lhs_tp_wp3=p["lhs_tp_wp3"],
        lhs_pr1_wp3=p["lhs_pr1_wp3"], lhs_dp1_wp2=p["lhs_dp1_wp2"],
        lhs_ta_wp3=p["lhs_ta_wp3"], wp2=p["wp2"], wp3=p["wp3"],
        w_tol_sqd=p["w_tol_sqd"])


def _call_wp23_lhs(p, nzm):
    ndim = 2 * nzm - 1
    return W.wp23_lhs(
        nzm=nzm, ndim=ndim, invrs_dt=p["invrs_dt"], lhs_ma_zm=p["lhs_ma_zm"],
        lhs_diff_zm=p["lhs_diff_zm"], lhs_ta_wp2=p["lhs_ta_wp2"],
        lhs_ac_pr2_wp2=p["lhs_ac_pr2_wp2"], lhs_dp1_wp2=p["lhs_dp1_wp2"],
        lhs_pr1_wp2=p["lhs_pr1_wp2"], lhs_splat_wp2=p["lhs_splat_wp2"],
        lhs_ma_zt=p["lhs_ma_zt"], lhs_diff_zt=p["lhs_diff_zt"],
        lhs_tp_wp3=p["lhs_tp_wp3"], lhs_ac_pr2_wp3=p["lhs_ac_pr2_wp3"],
        lhs_pr1_wp3=p["lhs_pr1_wp3"], lhs_splat_wp3=p["lhs_splat_wp3"],
        lhs_ta_wp3=p["lhs_ta_wp3"])


def test_wp23_assembly_shapes_and_bc():
    ng, nzm = 2, 11
    p = _assembly_inputs(ng, nzm)
    rhs = np.asarray(_call_wp23_rhs(p, nzm))
    lhs = np.asarray(_call_wp23_lhs(p, nzm))
    ndim = 2 * nzm - 1
    assert rhs.shape == (ng, ndim) and lhs.shape == (5, ng, ndim)
    # BC corner rows: main band = 1, off-diagonals = 0
    for c in (0, 1, ndim - 2, ndim - 1):
        assert np.allclose(lhs[2, :, c], 1.0)
        for b in (0, 1, 3, 4):
            assert np.allclose(lhs[b, :, c], 0.0)
    # RHS BCs: lower wp2 (0)=value, lower wp3 (1)=0, upper wp3 (ndim-2)=0,
    # upper wp2 (ndim-1)=w_tol_sqd  [globals 2*nzm-3 and 2*nzm-2]
    np.testing.assert_array_equal(rhs[:, 0], np.asarray(p["wp2"])[:, 0])
    assert np.allclose(rhs[:, 1], 0.0) and np.allclose(rhs[:, ndim - 2], 0.0)
    np.testing.assert_array_equal(rhs[:, ndim - 1], p["w_tol_sqd"])


def test_wp23_assembly_matches_golden():
    ng, nzm = 2, 11
    p = _assembly_inputs(ng, nzm)
    g = np.load(_FIX / "clubb_wp23_assembly_golden.npz")
    np.testing.assert_array_equal(np.asarray(_call_wp23_rhs(p, nzm)), g["rhs"])
    np.testing.assert_array_equal(np.asarray(_call_wp23_lhs(p, nzm)), g["lhs"])


def test_wp23_solve_deinterleaves():
    ng, nzm = 2, 9
    ndim = 2 * nzm - 1
    rng = np.random.default_rng(3)
    # Diagonally dominant penta system
    lhs = jnp.asarray(0.1 * rng.standard_normal((5, ng, ndim)))
    lhs = lhs.at[2].set(10.0 + rng.random((ng, ndim)))
    rhs = jnp.asarray(rng.standard_normal((ng, ndim)))
    wp2, wp3 = W.wp23_solve(lhs, rhs)
    assert wp2.shape == (ng, nzm) and wp3.shape == (ng, nzm - 1)
    assert np.all(np.isfinite(np.asarray(wp2))) and np.all(np.isfinite(np.asarray(wp3)))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_wp23_assembly_parity():
    """Bit-exact parity of wp23_rhs / wp23_lhs vs CLUBB-JAX (pure interleaving)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_wp2_wp3_module as R  # noqa: N812

    ng, nzm = 2, 11
    nzt = nzm - 1
    p = _assembly_inputs(ng, nzm)
    gr, _, _ = _gr(ng=ng, nzt=nzt)
    rg = _refgr(gr, ng, nzm)
    ndim = 2 * nzm - 1

    ref_rhs = R.wp23_rhs(
        nzm=nzm, ngrdcol=ng, invrs_dt=p["invrs_dt"],
        rhs_pr_turb_wp3=p["rhs_pr_turb_wp3"], rhs_pr_dfsn_wp3=p["rhs_pr_dfsn_wp3"],
        rhs_pr_dfsn_wp2=p["rhs_pr_dfsn_wp2"], rhs_pr1_wp2=p["rhs_pr1_wp2"],
        rhs_bp1_pr2_wp3=p["rhs_bp1_pr2_wp3"], rhs_pr1_wp3=p["rhs_pr1_wp3"],
        rhs_bp_pr2_wp2=p["rhs_bp_pr2_wp2"], rhs_pr3_wp2=p["rhs_pr3_wp2"],
        rhs_dp1_wp2=p["rhs_dp1_wp2"], lhs_pr1_wp2=p["lhs_pr1_wp2"],
        lhs_tp_wp3=p["lhs_tp_wp3"], lhs_pr1_wp3=p["lhs_pr1_wp3"],
        lhs_dp1_wp2=p["lhs_dp1_wp2"], lhs_ta_wp3=p["lhs_ta_wp3"],
        wp2=p["wp2"], wp3=p["wp3"], wp2up=jnp.zeros((ng, nzt)),
        upwp=jnp.zeros((ng, nzm)), l_ho_nontrad_coriolis=False, fcor_y=None, gr=rg)
    np.testing.assert_array_equal(np.asarray(_call_wp23_rhs(p, nzm)), np.asarray(ref_rhs))

    ref_lhs = R.wp23_lhs(
        nzm=nzm, ngrdcol=ng, ndim=ndim, invrs_dt=p["invrs_dt"],
        lhs_ma_zm=p["lhs_ma_zm"], lhs_diff_zm=p["lhs_diff_zm"], lhs_ta_wp2=p["lhs_ta_wp2"],
        lhs_ac_pr2_wp2=p["lhs_ac_pr2_wp2"], lhs_dp1_wp2=p["lhs_dp1_wp2"],
        lhs_pr1_wp2=p["lhs_pr1_wp2"], lhs_splat_wp2=p["lhs_splat_wp2"],
        lhs_ma_zt=p["lhs_ma_zt"], lhs_diff_zt=p["lhs_diff_zt"], lhs_tp_wp3=p["lhs_tp_wp3"],
        lhs_ac_pr2_wp3=p["lhs_ac_pr2_wp3"], lhs_pr1_wp3=p["lhs_pr1_wp3"],
        lhs_splat_wp3=p["lhs_splat_wp3"], lhs_ta_wp3=p["lhs_ta_wp3"])
    np.testing.assert_array_equal(np.asarray(_call_wp23_lhs(p, nzm)), np.asarray(ref_lhs))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_wp23_solve_parity():
    """Round-off parity of wp23_solve vs CLUBB-JAX (legoESM penta LU vs reference)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_wp2_wp3_module as R  # noqa: N812
    ng, nzm = 2, 9
    ndim = 2 * nzm - 1
    rng = np.random.default_rng(4)
    lhs = jnp.asarray(0.1 * rng.standard_normal((5, ng, ndim)))
    lhs = lhs.at[2].set(10.0 + rng.random((ng, ndim)))
    rhs = jnp.asarray(rng.standard_normal((ng, ndim)))
    m2, m3 = W.wp23_solve(lhs, rhs)
    r2, r3 = R.wp23_solve(lhs, rhs)
    np.testing.assert_allclose(np.asarray(m2), np.asarray(r2), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m3), np.asarray(r3), rtol=1e-9, atol=1e-12)


def test_wp23_assembly_jit_grad():
    ng, nzm = 2, 11
    p = _assembly_inputs(ng, nzm)

    def loss(wp2):
        rhs = _call_wp23_rhs(dict(p, wp2=wp2), nzm)
        lhs = _call_wp23_lhs(dict(p, wp2=wp2), nzm)
        a, b = W.wp23_solve(lhs, rhs)
        return jnp.sum(a ** 2) + jnp.sum(b ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wp2"])))


# --------------------------------------------------------------------------
# Centered term_ma_zt_lhs + a1/a3 + skewness-function pre-computes
# --------------------------------------------------------------------------

def _penta_matvec(lhs, x):
    """Apply a CLUBB-band penta matrix [super2,super1,main,sub1,sub2] to x."""
    ng, ndim = x.shape
    y = lhs[2] * x
    y = y.at[:, :-1].add(lhs[1, :, :-1] * x[:, 1:])     # super1 -> x[k+1]
    y = y.at[:, :-2].add(lhs[0, :, :-2] * x[:, 2:])     # super2 -> x[k+2]
    y = y.at[:, 1:].add(lhs[3, :, 1:] * x[:, :-1])      # sub1   -> x[k-1]
    y = y.at[:, 2:].add(lhs[4, :, 2:] * x[:, :-2])      # sub2   -> x[k-2]
    return y


def test_term_ma_zt_lhs_flows_through_wp23_lhs_correctly():
    """Integration: assembling the wp3 mean-advection (upwind, CAM
    l_upwind_xm_ma=True) into wp23_lhs and applying the penta matrix to an
    interleaved [wp2=0, wp3=field] vector reproduces the direct tridiagonal
    action of term_ma_zt_lhs_upwind on wp3 — proving the band mapping
    (super->band0/super2, sub->band4/sub2) end to end (codex review)."""
    from legoesm.atmosphere.physics.turbulence.clubb import term_ma_zt_lhs_upwind
    gr, ng, nzm = _gr(stretched=True)
    nzt = nzm - 1
    ndim = 2 * nzm - 1
    rng = np.random.default_rng(99)
    wm_zt = jnp.asarray(0.03 * rng.standard_normal((ng, nzt)))
    field = jnp.asarray(rng.standard_normal((ng, nzt)))   # a wp3 field

    lhs_ma_zt = term_ma_zt_lhs_upwind(wm_zt, gr)
    z3 = jnp.zeros((3, ng, nzt))
    z3m = jnp.zeros((3, ng, nzm))
    z2 = jnp.zeros((2, ng, nzt))
    z5 = jnp.zeros((5, ng, nzt))
    lhs = W.wp23_lhs(
        nzm=nzm, ndim=ndim, invrs_dt=0.0,
        lhs_ma_zm=z3m, lhs_diff_zm=z3m, lhs_ta_wp2=jnp.zeros((2, ng, nzm)),
        lhs_ac_pr2_wp2=jnp.zeros((ng, nzm)), lhs_dp1_wp2=jnp.zeros((ng, nzm)),
        lhs_pr1_wp2=jnp.zeros((ng, nzm)), lhs_splat_wp2=jnp.zeros((ng, nzm)),
        lhs_ma_zt=lhs_ma_zt, lhs_diff_zt=z3, lhs_tp_wp3=z2,
        lhs_ac_pr2_wp3=jnp.zeros((ng, nzt)), lhs_pr1_wp3=jnp.zeros((ng, nzt)),
        lhs_splat_wp3=jnp.zeros((ng, nzt)), lhs_ta_wp3=z5)

    x = jnp.zeros((ng, ndim)).at[:, 1::2].set(field)      # wp3 on odd slots
    y = np.asarray(_penta_matvec(lhs, x))

    # direct tridiag action of [super,main,sub] on the wp3 field, interior levels
    ma = np.asarray(lhs_ma_zt)
    direct = (ma[0, :, 1:-1] * np.asarray(field)[:, 2:]
              + ma[1, :, 1:-1] * np.asarray(field)[:, 1:-1]
              + ma[2, :, 1:-1] * np.asarray(field)[:, :-2])
    # wp3 interior global indices 3,5,..,2*nzm-5
    np.testing.assert_allclose(y[:, 3:-2:2], direct, rtol=1e-12, atol=1e-14)


def test_compute_a1_a3_coef_formula():
    gr, ng, nzm = _gr()
    rng = np.random.default_rng(15)
    sigma = jnp.asarray(0.1 + 0.4 * rng.random((ng, nzm)))
    a3_min = jnp.asarray(0.4 + 0.1 * rng.random((ng,)))
    a1, a3, a1_zt, a3_zt = W.compute_a1_a3_coef(sigma, a3_min, gr)
    one_minus = 1.0 - np.asarray(sigma)
    np.testing.assert_allclose(np.asarray(a1), 1.0 / one_minus, rtol=1e-12, atol=1e-14)
    exp_a3 = np.maximum(-2.0 * one_minus ** 2 + 3.0, np.asarray(a3_min)[:, None])
    np.testing.assert_allclose(np.asarray(a3), exp_a3, rtol=1e-12, atol=1e-14)
    # zt interpolation matches the grid operator
    from legoesm.atmosphere.physics.turbulence.clubb import zm2zt
    np.testing.assert_array_equal(np.asarray(a1_zt), np.asarray(zm2zt(a1, gr)))
    np.testing.assert_array_equal(np.asarray(a3_zt), np.asarray(zm2zt(a3, gr)))


def test_compute_skw_fnc_formula_and_degenerate():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(16)
    Skw = jnp.asarray(rng.standard_normal((ng, nzt)))
    C = jnp.asarray(0.7 + rng.random((ng,)))
    Cb = jnp.asarray(0.3 + rng.random((ng,)))
    Cc = jnp.asarray(1.0 + rng.random((ng,)))
    out = np.asarray(W.compute_skw_fnc(C, Cb, Cc, Skw))
    exp = (np.asarray(Cb)[:, None] + (np.asarray(C) - np.asarray(Cb))[:, None]
           * np.exp(-0.5 * (np.asarray(Skw) / np.asarray(Cc)[:, None]) ** 2))
    np.testing.assert_allclose(out, exp, rtol=1e-12, atol=1e-14)
    # degenerate C == Cb -> constant Cb (the |C-Cb| <= floor branch)
    out2 = np.asarray(W.compute_skw_fnc(Cb, Cb, Cc, Skw))
    np.testing.assert_allclose(out2, np.asarray(Cb)[:, None] * np.ones((ng, nzt)),
                               rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# clip_skewness (CAM l_use_wp3_lim_with_smth_Heaviside=False branch)
# --------------------------------------------------------------------------

def _clip_skw_inputs(ng=3, nzt=12, seed=21):
    rng = np.random.default_rng(seed)
    zt = jnp.asarray(np.tile(np.linspace(20.0, 3000.0, nzt), (ng, 1)))
    wp2_zt = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzt)))
    sfc = jnp.zeros((ng,))
    skw_max = jnp.full((ng,), 4.5)
    return zt, wp2_zt, sfc, skw_max


def test_clip_skewness_enforces_limit():
    zt, wp2_zt, sfc, skw_max = _clip_skw_inputs()
    ng, nzt = wp2_zt.shape
    # large wp3 that must be clipped everywhere
    wp3 = jnp.asarray(50.0 * np.ones((ng, nzt)))
    out = np.asarray(W.clip_skewness(wp3, wp2_zt, zt, sfc, skw_max))
    assert np.all(np.abs(out) <= 100.0 + 1e-9)
    # |Sk_w| = |wp3|/wp2_zt^1.5 within Skw_max_mag aloft (>100 m AGL)
    aloft = np.asarray(zt) > 100.0
    skw = np.abs(out) / np.asarray(wp2_zt) ** 1.5
    assert np.all(skw[aloft] <= 4.5 + 1e-9)


def test_clip_skewness_passes_small_wp3():
    zt, wp2_zt, sfc, skw_max = _clip_skw_inputs()
    ng, nzt = wp2_zt.shape
    wp3 = jnp.asarray(1e-3 * np.ones((ng, nzt)))   # well within the limit
    out = np.asarray(W.clip_skewness(wp3, wp2_zt, zt, sfc, skw_max))
    np.testing.assert_allclose(out, np.asarray(wp3), rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_clip_skewness_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.clip_explicit as RC  # noqa: N812
    zt, wp2_zt, sfc, skw_max = _clip_skw_inputs(seed=22)
    ng, nzt = wp2_zt.shape
    rng = np.random.default_rng(23)
    wp3 = jnp.asarray(20.0 * rng.standard_normal((ng, nzt)))
    out = W.clip_skewness(wp3, wp2_zt, zt, sfc, skw_max)
    ref = RC.clip_skewness_core(wp3, wp2_zt, zt, sfc, skw_max,
                                l_use_wp3_lim_with_smth_Heaviside=False)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-12, atol=1e-14)


def test_clip_skewness_jit_grad():
    zt, wp2_zt, sfc, skw_max = _clip_skw_inputs(seed=24)
    ng, nzt = wp2_zt.shape
    wp3 = jnp.asarray(10.0 * np.ones((ng, nzt)))

    def loss(w):
        return jnp.sum(W.clip_skewness(w, wp2_zt, zt, sfc, skw_max) ** 2)

    assert jnp.isfinite(jax.jit(loss)(wp3))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(wp3)))


# --------------------------------------------------------------------------
# advance_wp2_wp3 main
# --------------------------------------------------------------------------

def _wp23_main_inputs(seed=30, ng=2, nzt=10):
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    nzm = nzt + 1
    rng = np.random.default_rng(seed)
    gr, _, _ = _gr(ng=ng, nzt=nzt, stretched=True)

    def zm(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzm)))

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzt)))

    return dict(
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        wp3=zt(0.05), up2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        vp2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        sigma_sqd_w=jnp.asarray(0.1 + 0.3 * rng.random((ng, nzm))),
        wp3_on_wp2=zm(0.1), wpup2=zt(0.02), wpvp2=zt(0.02),
        wp2up2=zm(0.02), wp2vp2=zm(0.02), wp4=jnp.asarray(0.5 + 0.3 * rng.random((ng, nzm))),
        wpthvp=zm(0.02), wp2thvp=zt(0.02), um=zt(2.0, 5.0), vm=zt(1.0),
        upwp=zm(0.05), vpwp=zm(0.05), wm_zm=zm(0.02), wm_zt=zt(0.02),
        Kh_zm=jnp.asarray(1.0 + 2.0 * rng.random((ng, nzm))),
        Kh_zt=jnp.asarray(1.0 + 2.0 * rng.random((ng, nzt))),
        invrs_tau_C4_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        invrs_tau_wp3_zt=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzt))),
        invrs_tau_C1_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        Skw_zm=zm(0.5), Skw_zt=zt(0.5),
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt)))),
        thv_ds_zm=zm(1.0, 300.0), thv_ds_zt=zt(1.0, 300.0),
        sfc_elevation=jnp.zeros((ng,)), dt=300.0, gr=gr, config=CLUBBConfig(),
    )


def test_advance_wp2_wp3_runs_and_bounds():
    kw = _wp23_main_inputs()
    cfg = kw["config"]
    wp2, wp3, wp2_zt = W.advance_wp2_wp3(**kw)
    ng, nzm = kw["wp2"].shape
    assert wp2.shape == (ng, nzm) and wp3.shape == (ng, nzm - 1)
    assert wp2_zt.shape == (ng, nzm - 1)
    for f in (wp2, wp3, wp2_zt):
        assert np.all(np.isfinite(np.asarray(f)))
    w_tol_sqd = cfg.w_tol ** 2
    assert np.all(np.asarray(wp2)[:, :-1] >= w_tol_sqd - 1e-12)     # clip_variance floor
    assert np.all(np.asarray(wp2_zt) >= w_tol_sqd - 1e-12)
    assert np.all(np.abs(np.asarray(wp3)) <= 100.0 + 1e-9)          # clip_skewness


def test_advance_wp2_wp3_jit_grad():
    kw = _wp23_main_inputs()

    def loss(wp2):
        a, b, c = W.advance_wp2_wp3(**dict(kw, wp2=wp2))
        return jnp.sum(a ** 2) + jnp.sum(b ** 2) + jnp.sum(c ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wp2"])))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_advance_wp2_wp3_composition_parity():
    """Round-off parity of the full main vs the reference, in the configuration
    where the two CAM-divergent terms vanish (C1=C1b=0 -> C1_Skw_fnc=0 kills the
    dp1 term in both ARM and CAM branches; C_wp3_pr_turb=0 kills pr_turb in both).
    This verifies the main's COMPOSITION/wiring against the reference (the
    reference is hardwired ARM and cannot compute the CAM dp1/pr_turb branches;
    those branches are golden/analytic-tested separately)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_wp2_wp3_module as R  # noqa: N812
    from clubb_jax.src.CLUBB_core import parameter_indices as PI  # noqa: N812
    from legoesm import constants

    kw = _wp23_main_inputs(seed=31)
    cfg = kw["config"]
    # Zero the divergent-term coefficients so ARM == CAM.
    params2 = cfg.params._replace(C1=0.0, C1b=0.0, C_wp3_pr_turb=0.0)
    cfg2 = cfg._replace(params=params2)
    kw2 = dict(kw, config=cfg2)

    ng, nzm = kw["wp2"].shape
    nzt = nzm - 1
    p = params2
    gr = kw["gr"]
    rg = _refgr(gr, ng, nzm)

    cp = np.zeros((ng, 102))
    for idx, val in [
        (PI.iC4, p.C4), (PI.iC8, p.C8), (PI.iC8b, p.C8b), (PI.iC11, p.C11),
        (PI.iC11b, p.C11b), (PI.iC11c, p.C11c), (PI.iC1, p.C1), (PI.iC1b, p.C1b),
        (PI.iC1c, p.C1c), (PI.iC12, p.C12), (PI.ia3_coef_min, p.a3_coef_min),
        (PI.iC_uu_shr, p.C_uu_shr), (PI.iC_uu_buoy, p.C_uu_buoy),
        (PI.iC_wp2_pr_dfsn, p.C_wp2_pr_dfsn), (PI.iC_wp3_pr_tp, p.C_wp3_pr_tp),
        (PI.iC_wp3_pr_turb, p.C_wp3_pr_turb), (PI.iC_wp3_pr_dfsn, p.C_wp3_pr_dfsn),
        (PI.ic_K1, p.c_K1), (PI.ic_K8, p.c_K8), (PI.iSkw_max_mag, p.Skw_max_mag),
    ]:
        cp[:, idx - 1] = val

    flags = SimpleNamespace(
        fill_holes_type=2, l_wp2_fill_holes_tke=True, l_min_wp2_from_corr_wx=False,
        l_use_wp3_lim_with_smth_Heaviside=False, l_lmm_stepping=False,
        l_standard_term_ta=False)

    R._grav = float(constants.g)
    ref = R.advance_wp2_wp3(
        kw["wp2"], kw["wp3"], kw["up2"], kw["vp2"], kw["sigma_sqd_w"], kw["wp3_on_wp2"],
        jnp.zeros((ng, nzm)), kw["wpup2"], kw["wpvp2"], kw["wp2up2"], kw["wp2vp2"],
        kw["wp4"], kw["wpthvp"], kw["wp2thvp"], kw["um"], kw["vm"], kw["upwp"], kw["vpwp"],
        kw["wm_zm"], kw["wm_zt"], kw["Kh_zm"], kw["Kh_zt"], kw["invrs_tau_C4_zm"],
        kw["invrs_tau_wp3_zt"], kw["invrs_tau_C1_zm"], kw["Skw_zm"], kw["Skw_zt"],
        kw["rho_ds_zm"], kw["rho_ds_zt"], kw["invrs_rho_ds_zm"], kw["invrs_rho_ds_zt"],
        kw["thv_ds_zm"], kw["thv_ds_zt"], jnp.zeros((ng, nzm)), jnp.zeros((ng, nzt)),
        jnp.zeros((ng, nzm)), jnp.zeros((ng, nzm)), jnp.ones((ng, nzm)), jnp.ones((ng, nzm)),
        jnp.asarray(cp), kw["dt"], p.nu1, p.nu8, rg, flags, kw["sfc_elevation"])

    mine = W.advance_wp2_wp3(**kw2)
    for a, b in zip(mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
