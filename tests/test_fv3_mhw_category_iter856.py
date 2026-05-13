"""FV3_3D iter 856: marine_heatwave_category_fv3.

Hobday 2018 MHW category from 90th-percentile threshold multiples.

Tests
-----

1. ``test_no_mhw``: SST ≤ T_90 → cat = 0.
2. ``test_moderate_cat1``: 1× < intensity ≤ 2× → cat = 1.
3. ``test_strong_cat2``: 2× < intensity ≤ 3× → cat = 2.
4. ``test_severe_cat3``: 3× < intensity ≤ 4× → cat = 3.
5. ``test_extreme_cat4``: > 4× intensity → cat = 4 (clamped).
6. ``test_blob_2014_cat3``: NE Pacific Blob → cat 3 reachable.
7. ``test_at_threshold_zero``: SST = T_90 exactly → cat = 0.
8. ``test_below_clim_zero``: SST < clim → cat = 0.
9. ``test_delta_zero_floored``: clim=T_90 → finite via floor.
10. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import marine_heatwave_category_fv3


def test_no_mhw():
    """SST = clim (no anomaly) → cat = 0."""
    sst = jnp.array([20.0])
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [0.0], rtol=1e-12)


def test_moderate_cat1():
    """1× < x ≤ 2× → cat 1.  E.g. anomaly = 1.5× threshold."""
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    sst = jnp.array([21.5])  # anomaly=1.5, threshold-anom=1.0, x=1.5
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [1.0], rtol=1e-12)


def test_strong_cat2():
    """2× < x ≤ 3× → cat 2."""
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    sst = jnp.array([22.5])  # anomaly=2.5, x=2.5
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [2.0], rtol=1e-12)


def test_severe_cat3():
    """3× < x ≤ 4× → cat 3."""
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    sst = jnp.array([23.5])  # x=3.5
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [3.0], rtol=1e-12)


def test_extreme_cat4():
    """x > 4× → cat 4 (clamped to max)."""
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    sst = jnp.array([26.0])  # x=6, clamps to 4
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [4.0], rtol=1e-12)


def test_blob_2014_cat3():
    """NE Pacific Blob: anomaly ~3.5°C above clim, T_90 ~1°C → cat 3."""
    clim = jnp.array([14.0])
    t90 = jnp.array([15.0])
    sst = jnp.array([17.5])  # x=3.5
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [3.0], rtol=1e-12)


def test_at_threshold_zero():
    """SST = T_90 exactly → x=1, cat 0 (boundary, not strict exceedance)."""
    sst = jnp.array([21.0])
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [0.0], atol=1e-13)


def test_below_clim_zero():
    """SST < clim (cool spell) → cat 0."""
    sst = jnp.array([18.0])
    clim = jnp.array([20.0])
    t90 = jnp.array([21.0])
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    np.testing.assert_allclose(np.asarray(cat), [0.0], atol=1e-13)


def test_delta_zero_floored():
    """clim = T_90 (degenerate) → finite via floor."""
    sst = jnp.array([25.0])
    clim = jnp.array([20.0])
    t90 = jnp.array([20.0])
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    assert jnp.all(jnp.isfinite(cat))


def test_shapes_finite():
    """3-D shapes preserved, finite, in [0, 4]."""
    rng = np.random.default_rng(seed=856)
    n_x, n_y = 6, 8
    clim = jnp.asarray(rng.uniform(15.0, 25.0, size=(n_x, n_y)))
    t90 = clim + jnp.asarray(rng.uniform(0.5, 2.0, size=(n_x, n_y)))
    sst = clim + jnp.asarray(rng.uniform(-1.0, 6.0, size=(n_x, n_y)))
    cat = marine_heatwave_category_fv3(sst, clim, t90)
    assert cat.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cat))
    assert jnp.all(cat >= 0.0) and jnp.all(cat <= 4.0)
