"""Tests for the CLUBB wp2/wp3 advance LHS term builders (``clubb_wp23.py``).

Bit-exact parity vs CLUBB-JAX ``advance_wp2_wp3_module`` for the CAM-default
tree, plus boundary/shape sanity that runs in CI without the reference. The
zt->zm interpolation weights are checked vs ``grid_class.calc_zt2zm_weights``
(and analytically on a uniform grid).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb_wp23 as W  # noqa: E402, N812

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
        np.testing.assert_array_equal(np.asarray(out[key]), g[ref])


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
        np.testing.assert_array_equal(np.asarray(out[key]), g[key])


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
