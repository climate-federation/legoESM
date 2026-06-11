"""Tests for the CLUBB monotonic-flux-limiter JAX helpers (``clubb_mfl.py``).

Bit-exact/round-off parity vs CLUBB-JAX ``mono_flux_limiter`` for the erf-based
mean up/down velocity and the xm re-solve, a committed golden for CI coverage
without the reference, and jit/grad on the differentiable helpers.
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
from legoesm.atmosphere.physics.turbulence import clubb_mfl as M  # noqa: E402, N812

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


def _vel_inputs(ng, nzm, seed=1):
    rng = np.random.default_rng(seed)
    return dict(
        w_1=jnp.asarray(0.5 * rng.standard_normal((ng, nzm))),
        w_2=jnp.asarray(0.5 * rng.standard_normal((ng, nzm))),
        varnce_w_1=jnp.asarray(0.1 + 0.5 * rng.random((ng, nzm))),
        varnce_w_2=jnp.asarray(0.1 + 0.5 * rng.random((ng, nzm))),
        mixt_frac=jnp.asarray(0.3 + 0.4 * rng.random((ng, nzm))),
        wm=jnp.asarray(0.1 + 0.2 * rng.random((ng, nzm))),
    )


def test_mean_vert_vel_shapes_and_boundaries():
    gr, ng, nzm = _gr()
    p = _vel_inputs(ng, nzm)
    mwd, mwu = M.calc_mean_w_up_down_component(p["w_1"], p["varnce_w_1"], p["wm"])
    assert mwd.shape == (ng, nzm)
    assert np.allclose(np.asarray(mwd)[:, 0], 0.0) and np.allclose(np.asarray(mwd)[:, -1], 0.0)
    assert np.allclose(np.asarray(mwu)[:, 0], 0.0) and np.allclose(np.asarray(mwu)[:, -1], 0.0)


def test_matches_golden():
    gr, ng, nzm = _gr()
    p = _vel_inputs(ng, nzm)
    g = np.load(_FIX / "clubb_mfl_golden.npz")
    mwd, mwu = M.mean_vert_vel_up_down(p["w_1"], p["w_2"], p["varnce_w_1"],
                                       p["varnce_w_2"], p["mixt_frac"], p["wm"])
    np.testing.assert_array_equal(np.asarray(mwd), g["mean_w_down"])
    np.testing.assert_array_equal(np.asarray(mwu), g["mean_w_up"])
    lhs = M.mfl_xm_lhs(jnp.asarray(g["wm_zt"]), 1.0 / 300.0, gr)
    np.testing.assert_array_equal(np.asarray(lhs), g["mfl_xm_lhs"])


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mono_flux_limiter as R  # noqa: N812
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    p = _vel_inputs(ng, nzm)
    rg = _refgr(gr, ng, nzm)

    mwd, mwu = M.mean_vert_vel_up_down(p["w_1"], p["w_2"], p["varnce_w_1"],
                                       p["varnce_w_2"], p["mixt_frac"], p["wm"])
    r_mwd, r_mwu = R.mean_vert_vel_up_down(p["w_1"], p["w_2"], p["varnce_w_1"],
                                           p["varnce_w_2"], p["mixt_frac"], p["wm"])
    np.testing.assert_allclose(np.asarray(mwd), np.asarray(r_mwd), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(mwu), np.asarray(r_mwu), rtol=1e-12, atol=1e-14)

    rng = np.random.default_rng(5)
    wm_zt = jnp.asarray(0.02 * rng.standard_normal((ng, nzt)))
    np.testing.assert_array_equal(
        np.asarray(M.mfl_xm_lhs(wm_zt, 1.0 / 300.0, gr)),
        np.asarray(R.mfl_xm_lhs(wm_zt, 1.0 / 300.0, rg)))

    xm_old = jnp.asarray(290.0 + rng.standard_normal((ng, nzt)))
    wpxp = jnp.asarray(0.02 * rng.standard_normal((ng, nzm)))
    xm_forcing = jnp.asarray(1e-4 * rng.standard_normal((ng, nzt)))
    irho_zt = jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzt))))
    rho_zm = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    np.testing.assert_allclose(
        np.asarray(M.mfl_xm_rhs(xm_old, wpxp, xm_forcing, 1.0 / 300.0, irho_zt,
                                gr.invrs_dzt, rho_zm)),
        np.asarray(R.mfl_xm_rhs(xm_old, wpxp, xm_forcing, 1.0 / 300.0, irho_zt,
                                gr.invrs_dzt, rho_zm)),
        rtol=1e-12, atol=1e-14)


def _range_inputs(gr, ng, nzm, seed=3):
    rng = np.random.default_rng(seed)
    return dict(
        w_1_zm=jnp.asarray(0.8 * rng.standard_normal((ng, nzm))),
        w_2_zm=jnp.asarray(0.8 * rng.standard_normal((ng, nzm))),
        varnce_w_1_zm=jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm))),
        varnce_w_2_zm=jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm))),
        mixt_frac_zm=jnp.asarray(0.3 + 0.4 * rng.random((ng, nzm))),
    )


def test_turb_adv_range_matches_golden():
    """Non-skipped CI guard: the JAX level-range search vs the committed golden."""
    gr, ng, nzm = _gr()
    p = _range_inputs(gr, ng, nzm)
    lo, hi = M.calc_turb_adv_range(gr=gr, dt=300.0, **p)
    g = np.load(_FIX / "clubb_mfl_range_golden.npz")
    np.testing.assert_array_equal(np.asarray(lo), g["low"])
    np.testing.assert_array_equal(np.asarray(hi), g["high"])
    # bounds are valid zt indices, low<=high
    nzt = nzm - 1
    assert np.all((np.asarray(lo) >= 0) & (np.asarray(lo) <= nzt - 1))
    assert np.all((np.asarray(hi) >= 0) & (np.asarray(hi) <= nzt - 1))
    assert np.all(np.asarray(lo) <= np.asarray(hi))


def test_turb_adv_range_jit():
    gr, ng, nzm = _gr()
    p = _range_inputs(gr, ng, nzm)
    jf = jax.jit(lambda **kw: M.calc_turb_adv_range(gr=gr, dt=300.0, **kw))
    lo, hi = jf(**p)
    assert lo.shape == (ng, nzm - 1) and hi.shape == (ng, nzm - 1)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_turb_adv_range_parity():
    """Bit-exact integer-index parity vs the host-numpy reference (multiple sizes)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mono_flux_limiter as R  # noqa: N812
    for seed in range(4):
        for nzt in (8, 13):
            gr, ng, nzm = _gr(ng=3, nzt=nzt)
            rg = _refgr(gr, ng, nzm)
            p = _range_inputs(gr, ng, nzm, seed=seed)
            mlo, mhi = M.calc_turb_adv_range(gr=gr, dt=300.0, **p)
            rlo, rhi = R.calc_turb_adv_range(p["w_1_zm"], p["w_2_zm"],
                                             p["varnce_w_1_zm"], p["varnce_w_2_zm"],
                                             p["mixt_frac_zm"], rg, 300.0)
            np.testing.assert_array_equal(np.asarray(mlo), np.asarray(rlo))
            np.testing.assert_array_equal(np.asarray(mhi), np.asarray(rhi))


def test_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _vel_inputs(ng, nzm)

    def loss(w_1):
        mwd, mwu = M.mean_vert_vel_up_down(w_1, p["w_2"], p["varnce_w_1"],
                                           p["varnce_w_2"], p["mixt_frac"], p["wm"])
        return jnp.sum(mwd ** 2 + mwu ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["w_1"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["w_1"])))

    # Gradient w.r.t. variance must stay finite even at zero/negative variance
    # (the sqrt-at-zero AD hazard) — codex review.
    ng, nzm = p["w_1"].shape
    varnce = jnp.zeros((ng, nzm)).at[:, : nzm // 2].set(-0.5)   # zero and negative

    def vloss(v):
        mwd, mwu = M.calc_mean_w_up_down_component(p["w_1"], v, p["wm"])
        return jnp.sum(mwd ** 2 + mwu ** 2)

    assert jnp.all(jnp.isfinite(jax.grad(vloss)(varnce)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
