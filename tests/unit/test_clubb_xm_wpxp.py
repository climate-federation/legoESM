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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
