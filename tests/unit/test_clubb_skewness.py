"""Unit tests for CLUBB skewness diagnostics (now in ``clubb.py``).

Part of the fuller CLUBB port — see ``docs/md_files/clubb.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid, zm2zt  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    LG_2005_ansatz,
    Skx_func,
    calc_wp3_on_wp2,
    compute_gamma_Skw,
    compute_skewness_diagnostics,
    xp3_LG_2005_ansatz,
)

# CAM-default gamma coefficients (CLUBBParams).
_GC, _GB, _GCF = 0.308, 0.32, 5.0
_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


def _gr(ng=2, nzt=10):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def test_calc_wp3_on_wp2_clip_and_roundtrip():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(0)
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    wp3 = jnp.asarray(2.0 * rng.standard_normal((ng, nzt)))
    r_zm, r_zt = calc_wp3_on_wp2(wp2, wp3, 2.0e-2, gr)
    assert r_zm.shape == (ng, nzm) and r_zt.shape == (ng, nzt)
    assert np.all(np.abs(np.asarray(r_zt)) <= 1000.0 + 1e-9)
    assert np.all(np.isfinite(np.asarray(r_zm)))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_calc_wp3_on_wp2_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_helper_module as R  # noqa: N812
    from clubb_jax.src.derived_types.grid_class import (
        Grid, calc_zm2zt_weights, calc_zt2zm_weights)
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(2)
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    wp3 = jnp.asarray(2.0 * rng.standard_normal((ng, nzt)))
    zm_np, zt_np, dzt_np = np.asarray(gr.zm), np.asarray(gr.zt), np.asarray(gr.dzt)
    rg = Grid(nzm=nzm, nzt=nzt, ngrdcol=ng, zm=gr.zm, zt=gr.zt, dzm=gr.dzm, dzt=gr.dzt,
              invrs_dzm=gr.invrs_dzm, invrs_dzt=gr.invrs_dzt,
              weights_zt2zm=jnp.asarray(calc_zt2zm_weights(nzm, nzt, ng, zm_np, zt_np)),
              weights_zm2zt=jnp.asarray(calc_zm2zt_weights(nzm, nzt, ng, zm_np, zt_np, dzt_np)),
              k_lb_zm=0, k_ub_zm=nzm - 1, k_lb_zt=0, k_ub_zt=nzt - 1,
              grid_dir_indx=1, grid_dir=1.0)
    m_zm, m_zt = calc_wp3_on_wp2(wp2, wp3, 2.0e-2, gr)
    r_zm, r_zt = R.calc_wp3_on_wp2(wp2, wp3, rg)
    np.testing.assert_allclose(np.asarray(m_zm), np.asarray(r_zm), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(m_zt), np.asarray(r_zt), rtol=1e-12, atol=1e-14)


def test_skewness_diagnostics_consistency():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(7)
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    wp3 = jnp.asarray(0.5 * rng.standard_normal((ng, nzt)))
    out = compute_skewness_diagnostics(wp2, wp3, 2.0e-2, 0.0, gr)
    # Skw_zt with Skw_denom_coef=0 is wp3 / wp2_zt^1.5
    exp_skw_zt = np.asarray(wp3) * np.asarray(out["wp2_zt"]) ** -1.5
    np.testing.assert_allclose(np.asarray(out["Skw_zt"]), exp_skw_zt, rtol=1e-12)
    # wp2_zt is floored zm2zt
    np.testing.assert_array_equal(
        np.asarray(out["wp2_zt"]),
        np.maximum(np.asarray(zm2zt(wp2, gr)), (2.0e-2) ** 2))
    for k in ("Skw_zm", "wp3_on_wp2", "wp3_on_wp2_zt"):
        assert np.all(np.isfinite(np.asarray(out[k])))


def test_diagnostics_jit_grad():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    rng = np.random.default_rng(9)
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    wp3 = jnp.asarray(0.5 * rng.standard_normal((ng, nzt)))

    def loss(wp3):
        out = compute_skewness_diagnostics(wp2, wp3, 2.0e-2, 0.0, gr)
        return jnp.sum(out["Skw_zm"] ** 2) + jnp.sum(out["wp3_on_wp2"] ** 2)

    assert jnp.isfinite(jax.jit(loss)(wp3))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(wp3)))


# ---------------------------------------------------------------------------
# Skx_func
# ---------------------------------------------------------------------------

def test_skx_func_zero_denom_coef_is_normalized_third_moment():
    xp2 = jnp.asarray([0.5, 1.0, 2.0])
    xp3 = jnp.asarray([0.1, -0.2, 0.3])
    out = Skx_func(xp2, xp3, x_tol=2e-2, Skw_denom_coef=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(xp3 * xp2 ** -1.5), rtol=1e-12)


def test_skx_func_sign_follows_third_moment():
    xp2 = jnp.asarray([1.0, 1.0])
    xp3 = jnp.asarray([0.5, -0.5])
    out = Skx_func(xp2, xp3, x_tol=2e-2, Skw_denom_coef=4.0)
    assert float(out[0]) > 0 and float(out[1]) < 0
    # Odd symmetry in xp3.
    out_neg = Skx_func(xp2, -xp3, x_tol=2e-2, Skw_denom_coef=4.0)
    np.testing.assert_allclose(np.asarray(out_neg), -np.asarray(out), rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_gamma_Skw
# ---------------------------------------------------------------------------

def test_gamma_skw_constant_when_flag_off():
    Skw = jnp.linspace(-3.0, 3.0, 20)
    out = compute_gamma_Skw(Skw, _GC, _GB, _GCF, l_gamma_Skw=False)
    np.testing.assert_allclose(np.asarray(out), _GC, rtol=1e-12)


def test_gamma_skw_limits():
    """At Skw=0 -> gamma_coef; at large |Skw| -> gamma_coefb; bounded between."""
    out0 = compute_gamma_Skw(jnp.asarray([0.0]), _GC, _GB, _GCF)
    assert float(out0[0]) == pytest.approx(_GC, rel=1e-12)
    out_big = compute_gamma_Skw(jnp.asarray([50.0]), _GC, _GB, _GCF)
    assert float(out_big[0]) == pytest.approx(_GB, rel=1e-6)
    Skw = jnp.linspace(-10.0, 10.0, 100)
    out = compute_gamma_Skw(Skw, _GC, _GB, _GCF)
    lo, hi = min(_GC, _GB), max(_GC, _GB)
    assert jnp.all(out >= lo - 1e-12) and jnp.all(out <= hi + 1e-12)


def test_gamma_skw_degenerate_coefficients_constant():
    Skw = jnp.linspace(-3.0, 3.0, 10)
    out = compute_gamma_Skw(Skw, 0.3, 0.3, _GCF)  # gc == gb -> constant
    np.testing.assert_allclose(np.asarray(out), 0.3, rtol=1e-12)


def test_gamma_skw_golden():
    out = compute_gamma_Skw(jnp.asarray([1.0]), _GC, _GB, _GCF)
    expected = _GB + (_GC - _GB) * np.exp(-0.5 * (1.0 / _GCF) ** 2)
    assert float(out[0]) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# LG_2005_ansatz + xp3 inverse
# ---------------------------------------------------------------------------

def test_lg2005_zero_flux_zero_skewness():
    Skw = jnp.asarray([0.5, -0.5])
    out = LG_2005_ansatz(
        Skw, wpxp=jnp.zeros(2), wp2=jnp.full(2, 0.5), xp2=jnp.full(2, 0.1),
        sigma_sqd_w=jnp.full(2, 0.3), beta=2.4, x_tol=1e-2, w_tol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_xp3_ansatz_inverts_skx_func():
    """Skx_func(xp2, xp3_LG_2005_ansatz(...)) == LG_2005_ansatz(...)."""
    Skw = jnp.asarray([0.6, -0.3, 0.0])
    wpxp = jnp.asarray([0.05, -0.02, 0.0])
    wp2 = jnp.full(3, 0.5)
    xp2 = jnp.full(3, 0.1)
    ssw = jnp.full(3, 0.3)
    beta, x_tol, w_tol, denom = 2.4, 1e-2, 2e-2, 4.0
    skx_direct = LG_2005_ansatz(Skw, wpxp, wp2, xp2, ssw, beta, x_tol, w_tol)
    xp3 = xp3_LG_2005_ansatz(Skw, wpxp, wp2, xp2, ssw, beta, x_tol, w_tol, denom)
    skx_round = Skx_func(xp2, xp3, x_tol, denom)
    np.testing.assert_allclose(np.asarray(skx_round), np.asarray(skx_direct), rtol=1e-10)


# ---------------------------------------------------------------------------
# AD / JIT
# ---------------------------------------------------------------------------

def test_gamma_skw_differentiable_wrt_coef():
    def loss(gc):
        return jnp.sum(compute_gamma_Skw(jnp.linspace(-2.0, 2.0, 10), gc, _GB, _GCF))

    g = jax.grad(loss)(_GC)
    assert jnp.isfinite(g)


def test_skewness_jit_clean():
    Skw = jnp.asarray([0.5, -0.5])
    out = jax.jit(lambda s: compute_gamma_Skw(s, _GC, _GB, _GCF))(Skw)
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
