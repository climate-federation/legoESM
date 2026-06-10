"""Tests for the CLUBB xm/wpxp advance term builders (``clubb_xm_wpxp.py``).

Bit-exact parity vs CLUBB-JAX ``advance_xm_wpxp_module`` (the grav term patched
to legoESM ``constants.g``), a committed golden for CI coverage without the
reference, and boundary/shape + jit/grad sanity.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
