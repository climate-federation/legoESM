"""Unit tests for canopy/stability.py boundary-layer + MOST parameters.

Covers the DifferBESS aa6e8b9 boundary-layer corrections:
- ``compute_boundary_layer_resistance`` reads ``cv`` / ``d_leaf`` from the
  caller (no longer hard-coded 0.01 / 0.04).
- The CLM5-aligned config defaults (cv = 0.0135) and the per-PFT leaf-width
  table are present.
- MOST runs finite/positive after the kB^-1 = 0 (z0h = z0m) change.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.land.canopy.config import CanopyConfig, PFT_LEAF_WIDTH
from legoesm.land.canopy.stability import (
    compute_boundary_layer_resistance,
    monin_obukhov_stability,
)


def test_boundary_layer_resistance_matches_forced_convection_formula():
    uav = jnp.array([2.0]); LAI = jnp.array([3.0]); fSun = jnp.array([0.6])
    cv = jnp.array([0.0135]); d_leaf = jnp.array([0.025])
    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, LAI, fSun, cv, d_leaf)
    rb = 1.0 / (cv * jnp.sqrt(uav / d_leaf))
    assert jnp.allclose(Rb_Sun, rb / (LAI * fSun), rtol=1e-6)
    assert jnp.allclose(Rb_Sh, rb / (LAI * (1.0 - fSun)), rtol=1e-6)


def test_boundary_layer_resistance_depends_on_cv_and_dleaf():
    """Proves cv/d_leaf are live parameters, not the old hard-coded literals."""
    uav = jnp.array([2.0]); LAI = jnp.array([3.0]); fSun = jnp.array([0.5])
    Rb_a, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.0135]), jnp.array([0.025]))
    # Old BESS values (cv=0.01, d_leaf=0.04) must give a different Rb.
    Rb_b, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.01]), jnp.array([0.04]))
    assert not jnp.allclose(Rb_a, Rb_b)
    # Larger cv -> smaller resistance.
    Rb_hi, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.02]), jnp.array([0.025]))
    assert float(Rb_hi[0]) < float(Rb_a[0])


def test_canopy_config_clm5_boundary_layer_defaults():
    cfg = CanopyConfig()
    assert cfg.cv == 0.0135
    # PFT leaf-width table: small needles vs broad leaves (Schuepp 1993).
    assert PFT_LEAF_WIDTH["ENF"] == 0.01
    assert PFT_LEAF_WIDTH["EBF"] == 0.04
    assert PFT_LEAF_WIDTH["DBF"] == 0.025


def test_most_finite_after_kb_inv_zero():
    """MOST returns finite, positive resistances after z0h = z0m (kB^-1 = 0)."""
    one = jnp.array([1.0])
    ustar, rah, raw, uav, _zeta = monin_obukhov_stability(
        ur=3.0 * one, Ta=300.0 * one, Tv_atm=300.5 * one, Tc=301.0 * one,
        q_atm=0.010 * one, q_c=0.011 * one, zldis=10.0 * one, z0m=0.5 * one)
    for v in (ustar, rah, raw, uav):
        assert bool(jnp.all(jnp.isfinite(v)))
    assert float(ustar[0]) > 0.0
    assert float(rah[0]) > 0.0
