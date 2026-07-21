"""Root-zone vs top-boundary split of the multilayer-land latent stream.

Regression for the transpiration/soil-evap conflation (codex 2026-07-20): the
water-limited L_v evaporation stream must be split into a top-boundary
(bare-soil) part and a root-zone (transpiration) part using the canopy scheme's
ACTUAL LE_canopy/LE_soil ratio when available, not the ``f_veg`` root-zone-
wetness heuristic.  The split must be conservation-neutral.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.land.multilayer_land import _partition_latent_root_top

_NO = jnp.zeros(1, bool)          # no snow


def _call(soil_evap, f_veg, le_c, le_s, has_snow=_NO):
    b, t = _partition_latent_root_top(
        jnp.asarray([soil_evap]), has_snow, jnp.asarray([f_veg]),
        None if le_c is None else jnp.asarray([le_c]),
        None if le_s is None else jnp.asarray([le_s]))
    return float(b[0]), float(t[0])


def test_conservation_bare_plus_transp_equals_total():
    """evap_bare + evap_transp == soil_evap exactly, across scenarios."""
    for se, fv, lc, ls in [(5.0, 0.3, 80.0, 20.0), (5.0, 0.3, None, None),
                           (2.0, 0.9, 10.0, 90.0), (-1.0, 0.5, 0.0, 5.0)]:
        b, t = _call(se, fv, lc, ls)
        np.testing.assert_allclose(float(b + t), se, atol=1e-12)


def test_uses_canopy_split_not_fveg():
    """With LE_canopy=80, LE_soil=20, the transp fraction is 0.8 — NOT f_veg=0.3."""
    b, t = _call(10.0, f_veg=0.3, le_c=80.0, le_s=20.0)
    np.testing.assert_allclose(float(t), 8.0, atol=1e-10)   # 0.8 * 10
    np.testing.assert_allclose(float(b), 2.0, atol=1e-10)
    # the OLD f_veg heuristic would have given transp = 0.3*10 = 3 (mis-sourced)
    assert abs(float(t) - 3.0) > 1.0


def test_falls_back_to_fveg_when_no_split():
    """SimpleSEB reports no split -> use the f_veg heuristic."""
    b, t = _call(10.0, f_veg=0.3, le_c=None, le_s=None)
    np.testing.assert_allclose(float(t), 3.0, atol=1e-10)   # 0.3 * 10


def test_mixed_sign_components_fall_back_to_fveg():
    """codex: transpiration (LE_canopy>0) with soil dew (LE_soil<0) cannot be
    routed by one fraction of the net stream -> defer to f_veg (no worse than the
    prior net-based behaviour), NOT the (wrong) all-to-root ratio=1."""
    b, t = _call(10.0, f_veg=0.3, le_c=20.0, le_s=-10.0)
    np.testing.assert_allclose(float(t), 3.0, atol=1e-10)   # f_veg fallback, not 10
    assert abs(float(t) - 10.0) > 1.0


def test_zero_total_le_falls_back_and_grad_finite():
    """LE_canopy==LE_soil==0 -> fall back to f_veg, and AD stays finite (the
    double-where guard prevents a 0/0 NaN gradient)."""
    b, t = _call(4.0, f_veg=0.6, le_c=0.0, le_s=0.0)
    np.testing.assert_allclose(float(t), 2.4, atol=1e-10)   # 0.6 * 4 (fallback)

    def loss(le_c):
        _, tt = _partition_latent_root_top(
            jnp.asarray([4.0]), _NO, jnp.asarray([0.6]),
            le_c, jnp.asarray([0.0]))
        return jnp.sum(tt)
    g = jax.grad(loss)(jnp.zeros(1))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_snow_routes_all_to_root_zone():
    """Over snow the stream is pure transpiration -> transp_frac = 1."""
    b, t = _call(3.0, f_veg=0.2, le_c=50.0, le_s=50.0, has_snow=jnp.ones(1, bool))
    np.testing.assert_allclose(float(t), 3.0, atol=1e-10)
    np.testing.assert_allclose(float(b), 0.0, atol=1e-10)


def test_dew_routes_all_to_top_boundary():
    """Dew (soil_evap<0, downward deposition) routes entirely to bare soil."""
    b, t = _call(-2.0, f_veg=0.7, le_c=90.0, le_s=10.0)
    np.testing.assert_allclose(float(b), -2.0, atol=1e-12)
    np.testing.assert_allclose(float(t), 0.0, atol=1e-12)


def test_grad_flows_through_canopy_split():
    """A finite, nonzero gradient flows from LE_canopy into the root-zone sink."""
    def loss(le_c):
        _, tt = _partition_latent_root_top(
            jnp.asarray([10.0]), _NO, jnp.asarray([0.3]),
            le_c, jnp.asarray([20.0]))
        return jnp.sum(tt)
    g = jax.grad(loss)(jnp.asarray([80.0]))
    assert bool(jnp.all(jnp.isfinite(g))) and abs(float(g[0])) > 0.0
