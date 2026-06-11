"""Tests for the CLUBB xm/wpxp advance term builders (``clubb_xm_wpxp.py``).

Bit-exact parity vs CLUBB-JAX ``advance_xm_wpxp_module`` (the grav term patched
to legoESM ``constants.g``), a committed golden for CI coverage without the
reference, and boundary/shape + jit/grad sanity.
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

from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb_xm_wpxp as X  # noqa: E402, N812

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_FIX = Path(__file__).resolve().parent / "clubb_fixtures"


def _gr(ng=2, nzt=10):
    nzm = nzt + 1
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.1 ** np.arange(nzm)[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def _refgr(gr, ng, nzm):
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


def _inputs(gr, ng, nzm, seed=4):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    return dict(
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt)))),
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
        C6_Skw_fnc=jnp.asarray(2.0 + rng.random((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        invrs_tau_C6_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        thv_ds_zm=jnp.asarray(300.0 + rng.standard_normal((ng, nzm))),
        xpthvp=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
    )


def _outputs(gr, p):
    return dict(
        xm_ta=X.xm_term_ta_lhs(p["invrs_rho_ds_zt"], p["rho_ds_zm"], gr),
        wpxp_tp=X.wpxp_term_tp_lhs(p["wp2"], gr),
        wpxp_ac_pr2=X.wpxp_terms_ac_pr2_lhs(p["C7_Skw_fnc"], p["wm_zt"], gr),
        wpxp_pr1=X.wpxp_term_pr1_lhs(p["C6_Skw_fnc"], p["invrs_tau_C6_zm"]),
        wpxp_bp_pr3=X.wpxp_terms_bp_pr3_rhs(p["C7_Skw_fnc"], p["thv_ds_zm"], p["xpthvp"]),
    )


def test_shapes_and_boundaries():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    p = _inputs(gr, ng, nzm)
    out = _outputs(gr, p)
    assert out["xm_ta"].shape == (2, ng, nzt)
    assert out["wpxp_tp"].shape == (2, ng, nzm)
    for key in ("wpxp_ac_pr2", "wpxp_pr1", "wpxp_bp_pr3"):
        a = np.asarray(out[key])
        assert a.shape == (ng, nzm)
        assert np.allclose(a[:, 0], 0.0) and np.allclose(a[:, -1], 0.0)
    # wpxp_tp boundaries zero
    assert np.allclose(np.asarray(out["wpxp_tp"])[:, :, 0], 0.0)
    assert np.allclose(np.asarray(out["wpxp_tp"])[:, :, -1], 0.0)


def test_matches_golden():
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    g = np.load(_FIX / "clubb_xm_wpxp_builders_golden.npz")
    out = _outputs(gr, p)
    for key in out:
        np.testing.assert_array_equal(np.asarray(out[key]), g[key])


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    from legoesm import constants

    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)
    rg = _refgr(gr, ng, nzm)

    def chk(a, b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    chk(X.xm_term_ta_lhs(p["invrs_rho_ds_zt"], p["rho_ds_zm"], gr),
        R.xm_term_ta_lhs(p["invrs_rho_ds_zt"], p["rho_ds_zm"], rg))
    chk(X.wpxp_term_tp_lhs(p["wp2"], gr), R.wpxp_term_tp_lhs(p["wp2"], rg))
    chk(X.wpxp_terms_ac_pr2_lhs(p["C7_Skw_fnc"], p["wm_zt"], gr),
        R.wpxp_terms_ac_pr2_lhs(p["C7_Skw_fnc"], p["wm_zt"], rg))
    chk(X.wpxp_term_pr1_lhs(p["C6_Skw_fnc"], p["invrs_tau_C6_zm"]),
        R.wpxp_term_pr1_lhs(p["C6_Skw_fnc"], p["invrs_tau_C6_zm"]))
    chk(X.wpxp_terms_bp_pr3_rhs(p["C7_Skw_fnc"], p["thv_ds_zm"], p["xpthvp"]),
        R.wpxp_terms_bp_pr3_rhs(p["C7_Skw_fnc"], p["thv_ds_zm"], p["xpthvp"],
                                grav=float(constants.g)))


def test_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _inputs(gr, ng, nzm)

    def loss(wp2):
        return jnp.sum(X.wpxp_term_tp_lhs(wp2, gr) ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wp2"])))


# --------------------------------------------------------------------------
# Pentadiagonal assembly + solve
# --------------------------------------------------------------------------

def _asm_inputs(ng, nzm, seed=7):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def a(*shape):
        return jnp.asarray(rng.standard_normal(shape))

    return dict(
        dt=300.0,
        lhs_diff_zm=a(3, ng, nzm), lhs_ma_zm=a(3, ng, nzm), lhs_ma_zt=a(3, ng, nzt),
        lhs_ta_wpxp=a(3, ng, nzm), lhs_ta_xm=a(2, ng, nzt), lhs_tp=a(2, ng, nzm),
        lhs_ac_pr2=a(ng, nzm), lhs_pr1=a(ng, nzm),
        wpxp=jnp.asarray(0.01 * rng.standard_normal((ng, nzm))), xm=a(ng, nzt),
        wpxp_forcing=a(ng, nzm) * 1e-4, xm_forcing=a(ng, nzt) * 1e-4,
        rhs_bp_pr3=a(ng, nzm), rhs_ta=a(ng, nzm),
    )


def _call_lhs(p):
    return X.xm_wpxp_lhs(p["lhs_diff_zm"], p["lhs_ma_zm"], p["lhs_ma_zt"],
                         p["lhs_ta_wpxp"], p["lhs_ta_xm"], p["lhs_tp"],
                         p["lhs_ac_pr2"], p["lhs_pr1"], p["dt"])


def _call_rhs(p):
    return X.xm_wpxp_rhs(p["wpxp"], p["xm"], p["wpxp_forcing"], p["xm_forcing"],
                         p["rhs_bp_pr3"], p["rhs_ta"], p["lhs_ta_wpxp"], p["lhs_pr1"],
                         p["dt"])


def test_assembly_shapes_and_bc():
    ng, nzm = 2, 11
    p = _asm_inputs(ng, nzm)
    lhs = np.asarray(_call_lhs(p))
    rhs = np.asarray(_call_rhs(p))
    ndim = 2 * nzm - 1
    assert lhs.shape == (5, ng, ndim) and rhs.shape == (ng, ndim)
    # wpxp BC corners (j=0, j=ndim-1): diag=1, off-diagonals 0
    for c in (0, ndim - 1):
        assert np.allclose(lhs[2, :, c], 1.0)
        for b in (0, 1, 3, 4):
            assert np.allclose(lhs[b, :, c], 0.0)
    np.testing.assert_array_equal(rhs[:, 0], np.asarray(p["wpxp"])[:, 0])
    assert np.allclose(rhs[:, ndim - 1], 0.0)


def test_assembly_matches_golden():
    ng, nzm = 2, 11
    p = _asm_inputs(ng, nzm)
    g = np.load(_FIX / "clubb_xm_wpxp_assembly_golden.npz")
    np.testing.assert_array_equal(np.asarray(_call_lhs(p)), g["lhs"])
    np.testing.assert_array_equal(np.asarray(_call_rhs(p)), g["rhs"])


def test_solve_deinterleaves():
    ng, nzm = 2, 9
    ndim = 2 * nzm - 1
    rng = np.random.default_rng(2)
    lhs = jnp.asarray(0.1 * rng.standard_normal((5, ng, ndim)))
    lhs = lhs.at[2].set(10.0 + rng.random((ng, ndim)))
    rhs = jnp.asarray(rng.standard_normal((ng, ndim)))
    wpxp, xm = X.xm_wpxp_solve(lhs, rhs)
    assert wpxp.shape == (ng, nzm) and xm.shape == (ng, nzm - 1)
    assert np.all(np.isfinite(np.asarray(wpxp))) and np.all(np.isfinite(np.asarray(xm)))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_assembly_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    ng, nzm = 2, 11
    p = _asm_inputs(ng, nzm)
    ref_lhs = R.xm_wpxp_lhs(p["lhs_diff_zm"], p["lhs_ma_zm"], p["lhs_ma_zt"],
                            p["lhs_ta_wpxp"], p["lhs_ta_xm"], p["lhs_tp"],
                            p["lhs_ac_pr2"], p["lhs_pr1"], p["dt"])
    np.testing.assert_array_equal(np.asarray(_call_lhs(p)), np.asarray(ref_lhs))
    ref_rhs = R.xm_wpxp_rhs(p["wpxp"], p["xm"], p["wpxp_forcing"], p["xm_forcing"],
                            p["rhs_bp_pr3"], p["rhs_ta"], p["lhs_ta_wpxp"], p["lhs_pr1"],
                            p["dt"], 0)
    np.testing.assert_array_equal(np.asarray(_call_rhs(p)), np.asarray(ref_rhs))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_solve_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    ng, nzm = 2, 9
    ndim = 2 * nzm - 1
    rng = np.random.default_rng(3)
    lhs = jnp.asarray(0.1 * rng.standard_normal((5, ng, ndim)))
    lhs = lhs.at[2].set(10.0 + rng.random((ng, ndim)))
    rhs = jnp.asarray(rng.standard_normal((ng, ndim)))
    m2, m3 = X.xm_wpxp_solve(lhs, rhs)
    r2, r3 = R.xm_wpxp_solve(lhs, rhs)
    np.testing.assert_allclose(np.asarray(m2), np.asarray(r2), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m3), np.asarray(r3), rtol=1e-9, atol=1e-12)


def test_assembly_jit_grad():
    ng, nzm = 2, 11
    p = _asm_inputs(ng, nzm)

    def loss(wpxp):
        lhs = _call_lhs(p)
        rhs = _call_rhs(dict(p, wpxp=wpxp))
        a, b = X.xm_wpxp_solve(lhs, rhs)
        return jnp.sum(a ** 2) + jnp.sum(b ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wpxp"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wpxp"])))


# --------------------------------------------------------------------------
# calc_xm_wpxp_ta_terms / calc_xm_wpxp_lhs_terms / diagnose_upxp
# --------------------------------------------------------------------------

def _ta_inputs(gr, ng, nzm, seed=9):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    return dict(
        sigma_sqd_w=jnp.asarray(0.1 + 0.3 * rng.random((ng, nzm))),
        wp3_on_wp2_zt=jnp.asarray(0.1 * rng.standard_normal((ng, nzt))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        coef_zt=jnp.asarray(0.5 * rng.standard_normal((ng, nzt))),
    )


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_centered_ta_and_calc_ta_terms_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.turbulent_adv_pdf as RT  # noqa: N812
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    gr, ng, nzm = _gr()
    p = _ta_inputs(gr, ng, nzm)
    rg = _refgr(gr, ng, nzm)
    # centered xpyp TA operator
    np.testing.assert_array_equal(
        np.asarray(X.xpyp_term_ta_pdf_lhs_centered(p["coef_zt"], p["rho_ds_zt"],
                                                   p["invrs_rho_ds_zm"], gr)),
        np.asarray(RT.xpyp_term_ta_pdf_lhs_jax(p["coef_zt"], p["rho_ds_zt"],
                                               p["invrs_rho_ds_zm"], rg)))
    # calc_xm_wpxp_ta_terms
    np.testing.assert_allclose(
        np.asarray(X.calc_xm_wpxp_ta_terms(p["sigma_sqd_w"], p["wp3_on_wp2_zt"],
                                           p["rho_ds_zt"], p["invrs_rho_ds_zm"], gr)),
        np.asarray(R.calc_xm_wpxp_ta_terms(p["sigma_sqd_w"], p["wp3_on_wp2_zt"],
                                           p["rho_ds_zt"], p["invrs_rho_ds_zm"], rg)),
        rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_calc_lhs_terms_and_diagnose_upxp_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(10)
    rg = _refgr(gr, ng, nzm)
    args = dict(
        wm_zm=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        Kw6=jnp.asarray(0.5 + rng.random((ng, nzt))), nu6=10.0,
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt)))),
    )
    mine = X.calc_xm_wpxp_lhs_terms(gr=gr, **args)
    ref = R.calc_xm_wpxp_lhs_terms(gr=rg, **args)
    for key in ("lhs_diff_zm", "lhs_ma_zm", "lhs_ma_zt", "lhs_ta_xm", "lhs_tp", "lhs_ac_pr2"):
        np.testing.assert_array_equal(np.asarray(mine[key]), np.asarray(ref[key]))

    # diagnose_upxp
    d = dict(
        ypwp=jnp.asarray(0.05 * rng.standard_normal((ng, nzm))),
        xm=jnp.asarray(290.0 + rng.standard_normal((ng, nzt))),
        wpxp=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
        ym=jnp.asarray(5.0 + rng.standard_normal((ng, nzt))),
        C6x_Skw_fnc=jnp.asarray(2.0 + rng.random((ng, nzm))),
        tau_C6_zm=jnp.asarray(100.0 + rng.random((ng, nzm))),
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
    )
    np.testing.assert_array_equal(
        np.asarray(X.diagnose_upxp(gr=gr, **d)),
        np.asarray(R.diagnose_upxp(gr=rg, **d)))


def _helper_outputs(gr, ng, nzm):
    """Deterministic outputs of the centered-TA + calc helpers for the golden."""
    rng = np.random.default_rng(77)
    nzt = nzm - 1
    sigma = jnp.asarray(0.1 + 0.3 * rng.random((ng, nzm)))
    w3w2_zt = jnp.asarray(0.1 * rng.standard_normal((ng, nzt)))
    rho_zt = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt)))
    irho_zm = jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm))))
    coef = jnp.asarray(0.5 * rng.standard_normal((ng, nzt)))
    lhs = X.calc_xm_wpxp_lhs_terms(
        wm_zm=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        Kw6=jnp.asarray(0.5 + rng.random((ng, nzt))), nu6=10.0,
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
        invrs_rho_ds_zm=irho_zm, rho_ds_zt=rho_zt,
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt)))), gr=gr)
    return dict(
        centered_ta=X.xpyp_term_ta_pdf_lhs_centered(coef, rho_zt, irho_zm, gr),
        ta_terms=X.calc_xm_wpxp_ta_terms(sigma, w3w2_zt, rho_zt, irho_zm, gr),
        lhs_ta_xm=lhs["lhs_ta_xm"], lhs_tp=lhs["lhs_tp"], lhs_ac_pr2=lhs["lhs_ac_pr2"],
        lhs_ma_zt=lhs["lhs_ma_zt"],
    )


def test_helpers_match_golden():
    gr, ng, nzm = _gr()
    g = np.load(_FIX / "clubb_xm_wpxp_helpers_golden.npz")
    out = _helper_outputs(gr, ng, nzm)
    for key in out:
        np.testing.assert_array_equal(np.asarray(out[key]), g[key])


def test_centered_ta_uniform_grid_and_jit():
    gr, ng, nzm = _gr()
    p = _ta_inputs(gr, ng, nzm)
    band = np.asarray(X.xpyp_term_ta_pdf_lhs_centered(p["coef_zt"], p["rho_ds_zt"],
                                                      p["invrs_rho_ds_zm"], gr))
    assert band.shape == (3, ng, nzm)
    assert np.allclose(band[:, :, 0], 0.0) and np.allclose(band[:, :, -1], 0.0)

    def loss(coef):
        return jnp.sum(X.calc_xm_wpxp_ta_terms(p["sigma_sqd_w"], coef, p["rho_ds_zt"],
                                               p["invrs_rho_ds_zm"], gr) ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wp3_on_wp2_zt"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wp3_on_wp2_zt"])))


# --------------------------------------------------------------------------
# solve_xm_wpxp_with_single_lhs + xm_wpxp_clipping_and_stats
# --------------------------------------------------------------------------

def _solve_inputs(gr, ng, nzm, seed=12):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    return dict(
        wpxp=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
        xm=jnp.asarray(290.0 + rng.standard_normal((ng, nzt))),
        wpxp_forcing=jnp.asarray(1e-5 * rng.standard_normal((ng, nzm))),
        xm_forcing=jnp.asarray(1e-4 * rng.standard_normal((ng, nzt))),
        C6_Skw_fnc=jnp.asarray(4.0 + rng.random((ng, nzm))),
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
        invrs_tau_C6_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        lhs_ta_wpxp=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_diff_zm=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_ma_zm=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_ma_zt=jnp.asarray(rng.standard_normal((3, ng, nzt))),
        lhs_ta_xm=jnp.asarray(rng.standard_normal((2, ng, nzt))),
        lhs_tp=jnp.asarray(rng.standard_normal((2, ng, nzm))),
        lhs_ac_pr2=jnp.asarray(rng.standard_normal((ng, nzm))),
        thv_ds_zm=jnp.asarray(300.0 + rng.standard_normal((ng, nzm))),
        xpthvp=jnp.asarray(0.02 * rng.standard_normal((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        dt=300.0,
    )


def test_solve_runs_and_shapes():
    gr, ng, nzm = _gr()
    p = _solve_inputs(gr, ng, nzm)
    wpxp, xm = X.solve_xm_wpxp_with_single_lhs(gr=gr, **p)
    assert wpxp.shape == (ng, nzm) and xm.shape == (ng, nzm - 1)
    assert np.all(np.isfinite(np.asarray(wpxp))) and np.all(np.isfinite(np.asarray(xm)))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_solve_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    from legoesm import constants
    # wpxp_terms_bp_pr3_rhs binds grav as a def-time default kwarg (CLUBB 9.81);
    # rebind it to legoESM g so parity isolates numerics from the constant basis.
    R.wpxp_terms_bp_pr3_rhs.__defaults__ = (float(constants.g),)
    gr, ng, nzm = _gr()
    rg = _refgr(gr, ng, nzm)
    p = _solve_inputs(gr, ng, nzm)
    m_wp, m_xm = X.solve_xm_wpxp_with_single_lhs(gr=gr, **p)
    r_wp, r_xm = R.solve_xm_wpxp_with_single_lhs(gr=rg, **p)
    np.testing.assert_allclose(np.asarray(m_wp), np.asarray(r_wp), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m_xm), np.asarray(r_xm), rtol=1e-9, atol=1e-12)


def _clip_inputs(gr, ng, nzm, seed=13):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    rho_zm = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    rho_zt = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt)))
    w1 = jnp.asarray(0.8 * rng.standard_normal((ng, nzm)))
    w2 = jnp.asarray(0.8 * rng.standard_normal((ng, nzm)))
    v1 = jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm)))
    v2 = jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm)))
    mf = jnp.asarray(0.3 + 0.4 * rng.random((ng, nzm)))
    from legoesm.atmosphere.physics.turbulence.clubb_mfl import calc_turb_adv_range
    lo, hi = calc_turb_adv_range(w1, w2, v1, v2, mf, gr, 300.0)
    return dict(
        solve_type="rtm",
        xm=jnp.asarray(0.01 + 0.01 * rng.standard_normal((ng, nzt))),
        wpxp_preclip=jnp.asarray(0.5 * rng.standard_normal((ng, nzm))),
        xm_old=jnp.asarray(0.01 + 0.01 * rng.standard_normal((ng, nzt))),
        xp2=jnp.asarray(0.01 + 0.5 * rng.random((ng, nzm))),
        xp2_clip=jnp.asarray(0.01 + 0.5 * rng.random((ng, nzm))),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        xm_forcing=jnp.asarray(1e-5 * rng.standard_normal((ng, nzt))),
        rho_ds_zm=rho_zm, rho_ds_zt=rho_zt,
        invrs_rho_ds_zm=1.0 / rho_zm, invrs_rho_ds_zt=1.0 / rho_zt,
        xp2_threshold=1e-9, xm_tol=1e-4, low_lev_effect=lo, high_lev_effect=hi,
        field_tol=1e-8, fill_holes_type=2, l_mono_flux_lim=True, dt=300.0,
    )


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_clipping_and_stats_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    gr, ng, nzm = _gr(ng=3, nzt=14)
    rg = _refgr(gr, ng, nzm)
    p = _clip_inputs(gr, ng, nzm)
    m_xm, m_wp = X.xm_wpxp_clipping_and_stats(gr=gr, **p)
    r_xm, r_wp = R.xm_wpxp_clipping_and_stats(gr=rg, **p)
    np.testing.assert_allclose(np.asarray(m_xm), np.asarray(r_xm), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(m_wp), np.asarray(r_wp), rtol=1e-9, atol=1e-10)


def test_clipping_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _clip_inputs(gr, ng, nzm)

    def loss(wpxp_preclip):
        xm, wp = X.xm_wpxp_clipping_and_stats(gr=gr, **dict(p, wpxp_preclip=wpxp_preclip))
        return jnp.sum(xm ** 2) + jnp.sum(wp ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wpxp_preclip"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wpxp_preclip"])))


# --------------------------------------------------------------------------
# advance_xm_wpxp main
# --------------------------------------------------------------------------

def _main_inputs(gr, ng, nzm, seed=20):
    from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def zm(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzm)))

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzt)))

    rho_zm = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    rho_zt = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt)))
    return dict(
        rtm=zt(1e-3, 8e-3), thlm=zt(0.5, 290.0), wprtp=zm(1e-4), wpthlp=zm(1e-2),
        rtm_forcing=zt(1e-8), thlm_forcing=zt(1e-5),
        wprtp_forcing=zm(1e-8), wpthlp_forcing=zm(1e-5),
        C6rt_Skw_fnc=zm(0.5, 4.0), C6thl_Skw_fnc=zm(0.5, 4.0),
        C7_Skw_fnc=jnp.asarray(0.3 + 0.2 * rng.random((ng, nzm))),
        invrs_tau_C6_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        sigma_sqd_w=jnp.asarray(0.1 + 0.3 * rng.random((ng, nzm))),
        wp3_on_wp2_zt=zt(0.1), wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        Kh_zt=jnp.asarray(1.0 + 3.0 * rng.random((ng, nzt))),
        rtp2=jnp.asarray(1e-7 + 1e-6 * rng.random((ng, nzm))),
        thlp2=jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm))),
        rtpthvp=zm(1e-3), thlpthvp=zm(1e-2), thv_ds_zm=zm(1.0, 300.0),
        wm_zm=zm(0.02), wm_zt=zt(0.02), rho_ds_zm=rho_zm, rho_ds_zt=rho_zt,
        invrs_rho_ds_zm=1.0 / rho_zm, invrs_rho_ds_zt=1.0 / rho_zt,
        w_1_zm=zm(0.8), w_2_zm=zm(0.8),
        varnce_w_1_zm=jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm))),
        varnce_w_2_zm=jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm))),
        mixt_frac_zm=jnp.asarray(0.3 + 0.4 * rng.random((ng, nzm))),
        dt=300.0, gr=gr, config=CLUBBConfig(),
    )


def test_advance_xm_wpxp_runs_and_shapes():
    gr, ng, nzm = _gr()
    kw = _main_inputs(gr, ng, nzm)
    wprtp, rtm, wpthlp, thlm = X.advance_xm_wpxp(**kw)
    nzt = nzm - 1
    assert wprtp.shape == (ng, nzm) and rtm.shape == (ng, nzt)
    assert wpthlp.shape == (ng, nzm) and thlm.shape == (ng, nzt)
    for f in (wprtp, rtm, wpthlp, thlm):
        assert np.all(np.isfinite(np.asarray(f)))


def test_advance_xm_wpxp_wiring():
    """Independent wiring check: each pair's output reproduces a from-scratch
    solve+clip with explicitly-transcribed per-field args (catches swapped
    xpthvp/C6/forcing/xp2/tol between the rt and thl pairs)."""
    from legoesm.atmosphere.physics.turbulence.clubb_mfl import (
        MFL_RTM, MFL_THLM, calc_turb_adv_range)
    gr, ng, nzm = _gr()
    kw = _main_inputs(gr, ng, nzm)
    cfg = kw["config"]
    p = cfg.params
    lhs_ta = X.calc_xm_wpxp_ta_terms(kw["sigma_sqd_w"], kw["wp3_on_wp2_zt"],
                                     kw["rho_ds_zt"], kw["invrs_rho_ds_zm"], gr)
    sh = X.calc_xm_wpxp_lhs_terms(
        kw["wm_zm"], kw["wm_zt"], kw["wp2"], p.c_K6 * kw["Kh_zt"], p.nu6,
        kw["C7_Skw_fnc"], kw["invrs_rho_ds_zm"], kw["rho_ds_zt"], kw["rho_ds_zm"],
        kw["invrs_rho_ds_zt"], gr)
    lo, hi = calc_turb_adv_range(kw["w_1_zm"], kw["w_2_zm"], kw["varnce_w_1_zm"],
                                 kw["varnce_w_2_zm"], kw["mixt_frac_zm"], gr, 300.0)

    def pair(wpxp, xm, wpf, xmf, C6, xpthvp, xp2, mfl, xm_tol, tol_mfl, l_mfl):
        wp_pre, xm_new = X.solve_xm_wpxp_with_single_lhs(
            wpxp, xm, wpf, xmf, C6, kw["C7_Skw_fnc"], kw["invrs_tau_C6_zm"], lhs_ta,
            sh["lhs_diff_zm"], sh["lhs_ma_zm"], sh["lhs_ma_zt"], sh["lhs_ta_xm"],
            sh["lhs_tp"], sh["lhs_ac_pr2"], kw["thv_ds_zm"], xpthvp, kw["wm_zt"], 300.0, gr)
        return X.xm_wpxp_clipping_and_stats(
            mfl, xm_new, wp_pre, xm, xp2, xp2, kw["wp2"], kw["wm_zt"], xmf,
            kw["rho_ds_zm"], kw["rho_ds_zt"], kw["invrs_rho_ds_zm"], kw["invrs_rho_ds_zt"],
            xm_tol ** 2, tol_mfl, lo, hi, xm_tol, cfg.flags.fill_holes_type, l_mfl, 300.0, gr)

    exp_rtm, exp_wprtp = pair(kw["wprtp"], kw["rtm"], kw["wprtp_forcing"], kw["rtm_forcing"],
                              kw["C6rt_Skw_fnc"], kw["rtpthvp"], kw["rtp2"], MFL_RTM,
                              cfg.rt_tol, 1.0e-4, cfg.flags.l_mono_flux_lim_rtm)
    exp_thlm, exp_wpthlp = pair(kw["wpthlp"], kw["thlm"], kw["wpthlp_forcing"], kw["thlm_forcing"],
                                kw["C6thl_Skw_fnc"], kw["thlpthvp"], kw["thlp2"], MFL_THLM,
                                cfg.thl_tol, 0.2, cfg.flags.l_mono_flux_lim_thlm)

    wprtp, rtm, wpthlp, thlm = X.advance_xm_wpxp(**kw)
    np.testing.assert_array_equal(np.asarray(rtm), np.asarray(exp_rtm))
    np.testing.assert_array_equal(np.asarray(wprtp), np.asarray(exp_wprtp))
    np.testing.assert_array_equal(np.asarray(thlm), np.asarray(exp_thlm))
    np.testing.assert_array_equal(np.asarray(wpthlp), np.asarray(exp_wpthlp))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_advance_xm_wpxp_full_main_parity():
    """Gold-standard: the full main's scalar-pair outputs (wprtp/rtm/wpthlp/thlm)
    match the reference main, with the reference configured to ARM-matching
    constants where my main passes per-column constants for C6 and the C7 array
    via Cx_fnc_Richardson. Round-off (the penta solve differs)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xm_wpxp_module as R  # noqa: N812
    from clubb_jax.src.CLUBB_core import parameter_indices as PI  # noqa: N812
    from legoesm import constants
    R.wpxp_terms_bp_pr3_rhs.__defaults__ = (float(constants.g),)

    gr, ng, nzm = _gr(ng=2, nzt=12)
    nzt = nzm - 1
    rg = _refgr(gr, ng, nzm)
    kw = _main_inputs(gr, ng, nzm, seed=44)
    # Use per-column-constant C6 so ARM (clubb_params[iC6rt]) == my C6rt_Skw_fnc.
    C6rt_c, C6thl_c = 4.0, 4.5
    kw = dict(kw, C6rt_Skw_fnc=jnp.full((ng, nzm), C6rt_c),
              C6thl_Skw_fnc=jnp.full((ng, nzm), C6thl_c))
    mine = X.advance_xm_wpxp(**kw)   # (wprtp, rtm, wpthlp, thlm)

    cp = np.zeros((ng, 102))
    cp[:, PI.iC6rt - 1] = C6rt_c
    cp[:, PI.iC6thl - 1] = C6thl_c
    cp[:, PI.ic_K6 - 1] = kw["config"].params.c_K6
    cp[:, PI.ibeta - 1] = kw["config"].params.beta
    cp[:, PI.iC_uu_shr - 1] = kw["config"].params.C_uu_shr
    flags = SimpleNamespace(
        l_enable_relaxed_clipping=False, l_mono_flux_lim_rtm=True,
        l_mono_flux_lim_thlm=True, l_mono_flux_lim_spikefix=True, fill_holes_type=2,
        l_ho_nontrad_coriolis=False, l_uv_nudge=False, l_predict_upwp_vpwp=True,
        ipdf_call_placement=0, iiPDF_type=1, l_call_pdf_closure_twice=True,
        l_standard_term_ta=False)
    z_zm = jnp.zeros((ng, nzm))
    z_zt = jnp.zeros((ng, nzt))
    ref = R.advance_xm_wpxp(
        Cx_fnc_Richardson=kw["C7_Skw_fnc"], Kh_zt=kw["Kh_zt"], clubb_params=jnp.asarray(cp),
        dt_advance=kw["dt"], fcor=jnp.full((ng,), 1e-4), fcor_y=jnp.zeros((ng,)),
        flags=flags, gr=rg, invrs_rho_ds_zm=kw["invrs_rho_ds_zm"],
        invrs_rho_ds_zt=kw["invrs_rho_ds_zt"], invrs_tau_C6_zm=kw["invrs_tau_C6_zm"],
        l_sample=False, mixt_frac_zm=kw["mixt_frac_zm"], ngrdcol=ng,
        nu_vert_res_dep=SimpleNamespace(nu6=kw["config"].params.nu6), nzm=nzm, nzt=nzt,
        rc_coef_zm=z_zm, rho_ds_zm=kw["rho_ds_zm"], rho_ds_zt=kw["rho_ds_zt"],
        rtm_forcing=kw["rtm_forcing"], rtm_ref=kw["rtm"], rtp2=kw["rtp2"],
        rtpthvp=kw["rtpthvp"], sigma_sqd_w=kw["sigma_sqd_w"], sponge_cfg=None,
        stats_writer=None, thlm_forcing=kw["thlm_forcing"], thlm_ref=kw["thlm"],
        thlp2=kw["thlp2"], thlpthvp=kw["thlpthvp"], thv_ds_zm=kw["thv_ds_zm"],
        ts_nudge=0.0, ug=z_zt, um_forcing=z_zt, um_ref=z_zt, up2=jnp.full((ng, nzm), 0.4),
        uprcp=z_zm, varnce_w_1_zm=kw["varnce_w_1_zm"], varnce_w_2_zm=kw["varnce_w_2_zm"],
        vg=z_zt, vm_forcing=z_zt, vm_ref=z_zt, vp2=jnp.full((ng, nzm), 0.4), vprcp=z_zm,
        w_1_zm=kw["w_1_zm"], w_2_zm=kw["w_2_zm"], wm_zm=kw["wm_zm"], wm_zt=kw["wm_zt"],
        wp2=kw["wp2"], wp3_on_wp2_zt=kw["wp3_on_wp2_zt"], wprtp_forcing=kw["wprtp_forcing"],
        wpthlp_forcing=kw["wpthlp_forcing"], rcm=z_zt, rtm=kw["rtm"], thlm=kw["thlm"],
        um=z_zt, upwp=z_zm, vm=z_zt, vpwp=z_zm, wprtp=kw["wprtp"], wpthlp=kw["wpthlp"])

    # ref dict: wprtp/rtm/wpthlp/thlm
    for key, val in zip(("wprtp", "rtm", "wpthlp", "thlm"), mine):
        np.testing.assert_allclose(np.asarray(val), np.asarray(ref[key]),
                                   rtol=1e-9, atol=1e-11)


def test_advance_xm_wpxp_jit_grad():
    gr, ng, nzm = _gr()
    kw = _main_inputs(gr, ng, nzm)

    def loss(wprtp):
        out = X.advance_xm_wpxp(**dict(kw, wprtp=wprtp))
        return sum(jnp.sum(f ** 2) for f in out)

    assert jnp.isfinite(jax.jit(loss)(kw["wprtp"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wprtp"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
