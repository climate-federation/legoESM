"""Unit tests for the Andreae (2009) AOD → CCN diagnostic.

Reference anchors come straight from the paper (ACP 9, 543–556):
  - Fit: AOT500 = 0.0027 · CCN0.4^0.640 (Fig. 1, r²=0.88)
  - Clean continental average: CCN0.4 = 200±90 cm⁻³ at AOT = 0.075±0.025
  - Polluted continental average: CCN0.4 = 2900±2800 cm⁻³ at AOT = 0.45±0.27
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
    CCNFromAODConfig,
    ccn_from_aod,
)

jax.config.update("jax_enable_x64", True)


def test_forward_inverse_roundtrip():
    cfg = CCNFromAODConfig()
    n_cm3 = jnp.array([50.0, 200.0, 1000.0, 5000.0])
    aot = cfg.aot_coeff * n_cm3 ** cfg.aot_exponent
    n_back = ccn_from_aod(aot) / 1.0e6
    assert jnp.allclose(n_back, n_cm3, rtol=1e-10)


def test_paper_clean_continental_anchor():
    # AOT 0.075 → ~200 cm⁻³ (paper Table 2 average 200±90)
    n = float(ccn_from_aod(jnp.array(0.075))) / 1.0e6
    assert 110.0 < n < 290.0, n


def test_paper_polluted_continental_anchor():
    # AOT 0.45 → within 2900±2800 cm⁻³ (paper Table 2 average)
    n = float(ccn_from_aod(jnp.array(0.45))) / 1.0e6
    assert 100.0 < n < 5700.0, n


def test_monotone_in_aod():
    aod = jnp.linspace(0.0, 1.5, 200)
    n = ccn_from_aod(aod)
    assert jnp.all(jnp.diff(n) >= 0.0)


def test_floor_and_cap():
    cfg = CCNFromAODConfig()
    n_lo = float(ccn_from_aod(jnp.array(0.0))) / 1.0e6
    n_hi = float(ccn_from_aod(jnp.array(50.0))) / 1.0e6
    assert n_lo == pytest.approx(cfg.n_ccn_min_cm3)
    assert n_hi == pytest.approx(cfg.n_ccn_max_cm3)


def test_differentiable_and_finite_grad_at_zero():
    g = jax.grad(lambda a: ccn_from_aod(a) / 1.0e6)(0.0)
    assert jnp.isfinite(g)
    g_mid = jax.grad(lambda a: ccn_from_aod(a) / 1.0e6)(0.2)
    assert jnp.isfinite(g_mid) and g_mid > 0.0


def test_shape_preserved():
    aod = jnp.ones((6, 4, 4)) * 0.1
    n = ccn_from_aod(aod)
    assert n.shape == aod.shape
