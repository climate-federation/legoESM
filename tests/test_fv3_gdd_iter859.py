"""FV3_3D iter 859: growing_degree_days_fv3.

GDD = Σ max(T_daily − T_base, 0)  [°C·days].

Tests
-----

1. ``test_uniform_above_base``: 20°C daily / 100 days base=10 → GDD=1000.
2. ``test_below_base_zero``: T=5°C < base=10 → GDD=0.
3. ``test_at_base_zero``: T=base → GDD=0.
4. ``test_corn_maturity_band``: 180-day growing season warm summer → 2500-2800.
5. ``test_wheat_t_base_0``: wheat base=0 → larger GDD same conditions.
6. ``test_t_cap_modified``: T_cap=30 caps contribution at extreme heat.
7. ``test_warming_increases_gdd``: +2 K shift → ↑GDD.
8. ``test_batched_columns``: (lat, lon, time) → (lat, lon).
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import growing_degree_days_fv3


def test_uniform_above_base():
    """T=20°C / 100 days, base=10 → GDD = 100·10 = 1000."""
    t = jnp.full((100,), 20.0)
    gdd = growing_degree_days_fv3(t)
    np.testing.assert_allclose(np.asarray(gdd), 1000.0, rtol=1e-12)


def test_below_base_zero():
    """T=5°C / 30 days, base=10 → GDD=0."""
    t = jnp.full((30,), 5.0)
    gdd = growing_degree_days_fv3(t)
    np.testing.assert_allclose(np.asarray(gdd), 0.0, atol=1e-14)


def test_at_base_zero():
    """T = base → GDD = 0."""
    t = jnp.full((30,), 10.0)
    gdd = growing_degree_days_fv3(t)
    np.testing.assert_allclose(np.asarray(gdd), 0.0, atol=1e-14)


def test_corn_maturity_band():
    """Warm 180-day summer (mean 25°C, base 10) → 2700 GDD (corn mature)."""
    n_days = 180
    t = jnp.full((n_days,), 25.0)
    gdd = growing_degree_days_fv3(t)
    # 25 − 10 = 15 × 180 = 2700
    np.testing.assert_allclose(np.asarray(gdd), 2700.0, rtol=1e-12)
    # Mid corn-maturity band 2500-2800
    assert 2500.0 < float(gdd) < 2800.0


def test_wheat_t_base_0():
    """Wheat base=0 → much larger GDD than corn base=10 in same climate."""
    t = jnp.full((100,), 15.0)
    gdd_corn = growing_degree_days_fv3(t, t_base=10.0)
    gdd_wheat = growing_degree_days_fv3(t, t_base=0.0)
    assert float(gdd_wheat) > float(gdd_corn)
    np.testing.assert_allclose(np.asarray(gdd_wheat), 1500.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gdd_corn), 500.0, rtol=1e-12)


def test_t_cap_modified():
    """T_cap=30 caps extreme-heat contribution.  T=35°C uncapped GDD=2500,
    capped at 30 gives GDD=2000."""
    t = jnp.full((100,), 35.0)
    gdd_uncapped = growing_degree_days_fv3(t)
    gdd_capped = growing_degree_days_fv3(t, t_cap=30.0)
    np.testing.assert_allclose(np.asarray(gdd_uncapped), 2500.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gdd_capped), 2000.0, rtol=1e-12)
    assert float(gdd_capped) < float(gdd_uncapped)


def test_warming_increases_gdd():
    """+2 K uniform shift → larger GDD (positive crop impact below T_cap)."""
    n = 180
    t = jnp.full((n,), 20.0)
    t_warm = t + 2.0
    gdd_base = growing_degree_days_fv3(t)
    gdd_warm = growing_degree_days_fv3(t_warm)
    # GDD_warm = (22-10)*180 = 2160; GDD_base = (20-10)*180 = 1800
    assert float(gdd_warm) > float(gdd_base)


def test_batched_columns():
    """(lat, lon, time) → (lat, lon)."""
    rng = np.random.default_rng(seed=859)
    n_lat, n_lon, n_t = 4, 5, 100
    t = jnp.asarray(rng.uniform(5.0, 30.0, size=(n_lat, n_lon, n_t)))
    gdd = growing_degree_days_fv3(t)
    assert gdd.shape == (n_lat, n_lon)


def test_shapes_finite():
    """3-D batched random arrays: finite, non-negative."""
    rng = np.random.default_rng(seed=859)
    n_x, n_y, n_t = 6, 8, 90
    t = jnp.asarray(rng.uniform(-5.0, 35.0, size=(n_x, n_y, n_t)))
    gdd = growing_degree_days_fv3(t)
    assert gdd.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(gdd))
    assert jnp.all(gdd >= 0.0)
