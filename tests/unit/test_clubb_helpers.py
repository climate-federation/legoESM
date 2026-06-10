"""Unit tests for CLUBB advance-helper kernels (sigma_sqd_w, Brunt-Vaisala).

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_grid import (  # noqa: E402
    ddzt,
    make_clubb_grid,
)
from legoesm.atmosphere.physics.turbulence.clubb_helpers import (  # noqa: E402
    calc_brunt_vaisala_freq_sqd,
    compute_sigma_sqd_w,
)

from legoesm import constants  # noqa: E402


def _grid(ngrdcol=2, nzm=11):
    idx = np.arange(nzm, dtype=np.float64)
    zm_1d = np.cumsum(np.concatenate([[0.0], 50.0 * 1.1 ** idx[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ngrdcol, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ngrdcol, nzm


# ---------------------------------------------------------------------------
# sigma_sqd_w
# ---------------------------------------------------------------------------

def _tols():
    return dict(w_tol=2.0e-2, thl_tol=1.0e-2, rt_tol=1.0e-8)


def test_sigma_sqd_w_shape_and_bounds():
    gr, ng, nzm = _grid()
    gamma = jnp.full((ng, nzm), 0.3)
    wp2 = jnp.full((ng, nzm), 0.5)
    thlp2 = jnp.full((ng, nzm), 0.1)
    rtp2 = jnp.full((ng, nzm), 1e-6)
    wpthlp = jnp.full((ng, nzm), 0.05)
    wprtp = jnp.full((ng, nzm), 1e-4)
    out = compute_sigma_sqd_w(gamma, wp2, thlp2, rtp2, wpthlp, wprtp, gr, **_tols())
    assert out.shape == (ng, nzm)
    # 0 <= sigma_sqd_w <= gamma (factor (1 - min(corr,1)) in [0,1]); smoothing
    # of a field in [0, gamma] stays in [0, gamma] up to interpolation.
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 0.3 + 1e-12)


def test_sigma_sqd_w_zero_flux_gives_gamma():
    """With zero w-correlations, sigma_sqd_w = gamma (interior, after smoothing)."""
    gr, ng, nzm = _grid()
    gamma = jnp.full((ng, nzm), 0.25)
    wp2 = jnp.full((ng, nzm), 0.5)
    thlp2 = jnp.full((ng, nzm), 0.1)
    rtp2 = jnp.full((ng, nzm), 1e-6)
    zero = jnp.zeros((ng, nzm))
    out = compute_sigma_sqd_w(gamma, wp2, thlp2, rtp2, zero, zero, gr, **_tols())
    # constant 0.25 field is preserved by zm->zt->zm in the interior.
    np.testing.assert_allclose(np.asarray(out[:, 1:-1]), 0.25, rtol=1e-9)


def test_sigma_sqd_w_high_correlation_reduces_sigma():
    """Near-perfect w'thl' correlation drives sigma_sqd_w toward zero."""
    gr, ng, nzm = _grid()
    gamma = jnp.full((ng, nzm), 0.3)
    wp2 = jnp.full((ng, nzm), 0.5)
    thlp2 = jnp.full((ng, nzm), 0.1)
    rtp2 = jnp.full((ng, nzm), 1e-8)
    wpthlp = jnp.full((ng, nzm), jnp.sqrt(0.5 * 0.1))  # corr ~ 1
    wprtp = jnp.zeros((ng, nzm))
    out = compute_sigma_sqd_w(gamma, wp2, thlp2, rtp2, wpthlp, wprtp, gr, **_tols())
    assert jnp.all(out < 0.05)  # << gamma


def test_sigma_sqd_w_grad_and_jit():
    gr, ng, nzm = _grid()
    gamma = jnp.full((ng, nzm), 0.3)
    wp2 = jnp.full((ng, nzm), 0.5)
    thlp2 = jnp.full((ng, nzm), 0.1)
    rtp2 = jnp.full((ng, nzm), 1e-6)
    wprtp = jnp.full((ng, nzm), 1e-4)

    def loss(wpthlp):
        return jnp.sum(compute_sigma_sqd_w(gamma, wp2, thlp2, rtp2, wpthlp, wprtp, gr, **_tols()))

    jitted = jax.jit(loss)
    assert jnp.isfinite(jitted(jnp.full((ng, nzm), 0.05)))
    g = jax.grad(loss)(jnp.full((ng, nzm), 0.05))
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Brunt-Vaisala frequency squared
# ---------------------------------------------------------------------------

def _bv_inputs(gr, ng, nzm, dthl_dz=5.0e-3, rcm_val=0.0, ice_val=0.0):
    nzt = nzm - 1
    zt = gr.zt
    thlm = 300.0 + dthl_dz * zt                      # stable if dthl_dz>0
    p = 1.0e5 * jnp.exp(-zt / 8000.0)                # decreasing upward
    exner = (p / constants.p_ref) ** constants.kappa
    rtm = jnp.full((ng, nzt), 5e-3)
    rcm = jnp.full((ng, nzt), rcm_val)
    ice = jnp.full((ng, nzt), ice_val)
    return thlm, exner, rtm, rcm, p, ice


def test_bv_shapes_and_dry_form_identity():
    gr, ng, nzm = _grid()
    thlm, exner, rtm, rcm, p, ice = _bv_inputs(gr, ng, nzm)
    bv, bv_mixed, bv_smth, bv_dry, bv_moist = calc_brunt_vaisala_freq_sqd(
        thlm, exner, rtm, rcm, p, ice, bv_efold=5.0, T0=300.0, gr=gr
    )
    for arr in (bv, bv_mixed, bv_smth, bv_dry, bv_moist):
        assert arr.shape == (ng, nzm)
    # Returned (CAM default l_brunt_vaisala_freq_moist=False) form is the dry one.
    expected = (constants.g / 300.0) * ddzt(thlm, gr)
    np.testing.assert_allclose(np.asarray(bv), np.asarray(expected), rtol=1e-12)


def test_bv_stable_profile_positive():
    gr, ng, nzm = _grid()
    thlm, exner, rtm, rcm, p, ice = _bv_inputs(gr, ng, nzm, dthl_dz=6.0e-3)
    bv, *_ = calc_brunt_vaisala_freq_sqd(
        thlm, exner, rtm, rcm, p, ice, bv_efold=5.0, T0=300.0, gr=gr
    )
    # ddzt boundary replicates interior; interior of a stable column is > 0.
    assert jnp.all(bv[:, 1:-1] > 0.0)


def test_bv_unstable_profile_negative():
    gr, ng, nzm = _grid()
    thlm, exner, rtm, rcm, p, ice = _bv_inputs(gr, ng, nzm, dthl_dz=-6.0e-3)
    bv, *_ = calc_brunt_vaisala_freq_sqd(
        thlm, exner, rtm, rcm, p, ice, bv_efold=5.0, T0=300.0, gr=gr
    )
    assert jnp.all(bv[:, 1:-1] < 0.0)


def test_bv_mixed_equals_dry_when_no_ice():
    """ice_supersat_frac = 0 => exp(0)=1 => bv_mixed == bv_dry exactly."""
    gr, ng, nzm = _grid()
    thlm, exner, rtm, rcm, p, ice = _bv_inputs(gr, ng, nzm, rcm_val=0.0, ice_val=0.0)
    _, bv_mixed, _, bv_dry, _ = calc_brunt_vaisala_freq_sqd(
        thlm, exner, rtm, rcm, p, ice, bv_efold=5.0, T0=300.0, gr=gr
    )
    np.testing.assert_allclose(np.asarray(bv_mixed), np.asarray(bv_dry), rtol=1e-10)


def test_bv_grad_and_jit():
    gr, ng, nzm = _grid()
    thlm, exner, rtm, rcm, p, ice = _bv_inputs(gr, ng, nzm, rcm_val=1e-4, ice_val=0.1)

    def loss(thlm_):
        bv, *_ = calc_brunt_vaisala_freq_sqd(
            thlm_, exner, rtm, rcm, p, ice, bv_efold=5.0, T0=300.0, gr=gr
        )
        return jnp.sum(bv ** 2)

    assert jnp.isfinite(jax.jit(loss)(thlm))
    g = jax.grad(loss)(thlm)
    assert jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
