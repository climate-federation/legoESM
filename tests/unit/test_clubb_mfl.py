"""Tests for the CLUBB monotonic-flux-limiter JAX helpers (now in ``clubb.py``).

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

from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as M  # noqa: E402, N812

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clubb_ref_api as _ref  # noqa: E402

_CLUBB_JAX_ROOT = _ref.ROOT
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
    # FP-reassociation tolerance: the C1-C8 "condense" refactor fused/re-
    # associated the algebra (~1e-14 rel) -> round-off, not bit-identity.
    np.testing.assert_allclose(np.asarray(mwd), g["mean_w_down"], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.asarray(mwu), g["mean_w_up"], rtol=1e-13, atol=1e-16)
    lhs = M.mfl_xm_lhs(jnp.asarray(g["wm_zt"]), 1.0 / 300.0, gr)
    np.testing.assert_allclose(np.asarray(lhs), g["mfl_xm_lhs"], rtol=1e-13, atol=1e-16)


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


def _limiter_inputs(gr, ng, nzm, solve_type="rtm", seed=8):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    base = 290.0 if solve_type in ("rtm", "thlm") else 0.0
    xm = jnp.asarray(base + 2.0 * rng.standard_normal((ng, nzt)))
    rho_zm = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    rho_zt = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt)))
    w1 = jnp.asarray(0.8 * rng.standard_normal((ng, nzm)))
    w2 = jnp.asarray(0.8 * rng.standard_normal((ng, nzm)))
    v1 = jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm)))
    v2 = jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm)))
    mf = jnp.asarray(0.3 + 0.4 * rng.random((ng, nzm)))
    lo, hi = M.calc_turb_adv_range(w1, w2, v1, v2, mf, gr, 300.0)
    thr, tol = (1e-9, 1e-4) if solve_type in ("rtm",) else (1e-4, 0.2)
    return dict(
        solve_type=solve_type, xm=xm,
        xm_old=jnp.asarray(np.asarray(xm) + 0.1 * rng.standard_normal((ng, nzt))),
        wpxp=jnp.asarray(0.5 * rng.standard_normal((ng, nzm))),
        xp2=jnp.asarray(0.01 + 0.5 * rng.random((ng, nzm))),
        wm_zt=jnp.asarray(0.02 * rng.standard_normal((ng, nzt))),
        xm_forcing=jnp.asarray(1e-4 * rng.standard_normal((ng, nzt))),
        rho_ds_zm=rho_zm, rho_ds_zt=rho_zt,
        invrs_rho_ds_zm=1.0 / rho_zm, invrs_rho_ds_zt=1.0 / rho_zt,
        xp2_threshold=thr, xm_tol=tol, low_lev_effect=lo, high_lev_effect=hi,
        gr=gr, dt=300.0,
    )


def test_limiter_matches_golden():
    gr, ng, nzm = _gr()
    p = _limiter_inputs(gr, ng, nzm)
    xm, wpxp = M.monotonic_turbulent_flux_limit(**p)
    g = np.load(_FIX / "clubb_mfl_limiter_golden.npz")
    np.testing.assert_allclose(np.asarray(xm), g["xm"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(wpxp), g["wpxp"], rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_limiter_parity():
    """Round-off parity vs the host-numpy limiter for all 4 field types."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mono_flux_limiter as R  # noqa: N812
    for st in ("rtm", "thlm", "um", "vm"):
        gr, ng, nzm = _gr(ng=3, nzt=14)
        rg = _refgr(gr, ng, nzm)
        p = _limiter_inputs(gr, ng, nzm, solve_type=st, seed=hash(st) % 100)
        xm, wpxp = M.monotonic_turbulent_flux_limit(**p)
        rxm, rwp = R._monotonic_turbulent_flux_limit_numpy(
            st, np.asarray(p["xm"]), np.asarray(p["wpxp"]), np.asarray(p["xm_old"]),
            np.asarray(p["xp2"]), np.asarray(p["wm_zt"]), np.asarray(p["xm_forcing"]),
            np.asarray(p["rho_ds_zm"]), np.asarray(p["rho_ds_zt"]),
            np.asarray(p["invrs_rho_ds_zm"]), np.asarray(p["invrs_rho_ds_zt"]),
            p["xp2_threshold"], p["xm_tol"], np.asarray(p["low_lev_effect"]),
            np.asarray(p["high_lev_effect"]), rg, 300.0)
        np.testing.assert_allclose(np.asarray(xm), np.asarray(rxm), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(np.asarray(wpxp), np.asarray(rwp), rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_limiter_nan_xp2_matches_reference():
    """With a NaN variance the AD-safe sqrt must match the reference (no
    masking-to-0 that would clip with spurious finite bounds) — codex review.
    A NaN variance makes the level bounds NaN, so the clip comparisons are False
    and the flux is left unchanged in BOTH implementations."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mono_flux_limiter as R  # noqa: N812
    gr, ng, nzm = _gr(ng=3, nzt=14)
    rg = _refgr(gr, ng, nzm)
    p = _limiter_inputs(gr, ng, nzm, solve_type="thlm", seed=1)
    xp2 = np.asarray(p["xp2"]).copy()
    xp2[0, 3] = np.nan
    p = dict(p, xp2=jnp.asarray(xp2))
    xm, wpxp = M.monotonic_turbulent_flux_limit(**p)
    rxm, rwp = R._monotonic_turbulent_flux_limit_numpy(
        "thlm", np.asarray(p["xm"]), np.asarray(p["wpxp"]), np.asarray(p["xm_old"]),
        xp2, np.asarray(p["wm_zt"]), np.asarray(p["xm_forcing"]),
        np.asarray(p["rho_ds_zm"]), np.asarray(p["rho_ds_zt"]),
        np.asarray(p["invrs_rho_ds_zm"]), np.asarray(p["invrs_rho_ds_zt"]),
        p["xp2_threshold"], p["xm_tol"], np.asarray(p["low_lev_effect"]),
        np.asarray(p["high_lev_effect"]), rg, 300.0)
    # finite entries match (round-off); NaN pattern matches
    m_x, r_x = np.asarray(xm), np.asarray(rxm)
    np.testing.assert_array_equal(np.isnan(m_x), np.isnan(r_x))
    fin = ~np.isnan(m_x)
    np.testing.assert_allclose(m_x[fin], r_x[fin], rtol=1e-9, atol=1e-10)


def test_limiter_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _limiter_inputs(gr, ng, nzm)

    def loss(wpxp):
        xm, wp = M.monotonic_turbulent_flux_limit(**dict(p, wpxp=wpxp))
        return jnp.sum(xm ** 2) + jnp.sum(wp ** 2)

    assert jnp.isfinite(jax.jit(loss)(p["wpxp"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["wpxp"])))


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
