"""FV3_3D iter 855: degree_heating_weeks_fv3.

DHW = Σ max(SST − MMM − 1°C, 0) over 12 wk window.

Tests
-----

1. ``test_no_anomaly_zero``: SST = MMM → DHW = 0.
2. ``test_below_hotspot_zero``: SST = MMM + 0.5 → DHW = 0 (no exceedance).
3. ``test_at_hotspot_zero``: SST = MMM + 1.0 → DHW = 0 (exactly threshold).
4. ``test_uniform_2c_anomaly``: SST = MMM + 2 over 12 wks → DHW = 12.
5. ``test_2016_gbr_bleaching_band``: realistic profile → DHW in 8–20 band.
6. ``test_severe_bleaching_threshold``: ≥8 °C·wk triggers severe.
7. ``test_cooling_below_mmm_zero``: SST < MMM → DHW = 0 (no contribution).
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import degree_heating_weeks_fv3


def test_no_anomaly_zero():
    """SST = MMM → DHW = 0."""
    sst = jnp.full((12,), 29.0)
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    np.testing.assert_allclose(np.asarray(dhw), 0.0, atol=1e-14)


def test_below_hotspot_zero():
    """SST = MMM + 0.5 → DHW = 0 (below hotspot threshold)."""
    sst = jnp.full((12,), 29.5)
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    np.testing.assert_allclose(np.asarray(dhw), 0.0, atol=1e-14)


def test_at_hotspot_zero():
    """SST = MMM + 1.0 → DHW = 0 (exactly at threshold)."""
    sst = jnp.full((12,), 30.0)
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    np.testing.assert_allclose(np.asarray(dhw), 0.0, atol=1e-14)


def test_uniform_2c_anomaly():
    """SST = MMM + 2 over 12 wks → DHW = 12 × (2-1) = 12 °C·wk."""
    sst = jnp.full((12,), 31.0)
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    np.testing.assert_allclose(np.asarray(dhw), 12.0, rtol=1e-12)


def test_2016_gbr_bleaching_band():
    """Realistic 2016 GBR bleaching profile → DHW in 8-20 band."""
    # Anomaly ramping up: +0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0
    anom = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                      2.5, 2.0, 1.5, 1.0, 0.5, 0.0])
    sst = 29.0 + anom
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    # HS = max(anom-1, 0): [0,0,0.5,1,1.5,2,1.5,1,0.5,0,0,0] = 8.0
    np.testing.assert_allclose(np.asarray(dhw), 8.0, rtol=1e-12)
    assert 6.0 < float(dhw) < 20.0


def test_severe_bleaching_threshold():
    """≥8 °C-wk triggers severe bleaching outcome."""
    # Construct DHW ≈ 10 (between 8-12, severe)
    sst = jnp.full((12,), 30.83)  # +1.83 K → HS = 0.83 × 12 = 9.96
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    assert float(dhw) >= 8.0


def test_cooling_below_mmm_zero():
    """SST < MMM (cool spell) → DHW = 0."""
    sst = jnp.full((12,), 28.0)
    mmm = jnp.array(29.0)
    dhw = degree_heating_weeks_fv3(sst, mmm)
    np.testing.assert_allclose(np.asarray(dhw), 0.0, atol=1e-14)


def test_shapes_finite():
    """3-D batched: (lat, lon, week) → (lat, lon)."""
    rng = np.random.default_rng(seed=855)
    n_lat, n_lon, n_wk = 4, 5, 12
    sst = jnp.asarray(rng.uniform(27.0, 32.0, size=(n_lat, n_lon, n_wk)))
    mmm = jnp.asarray(rng.uniform(28.0, 30.0, size=(n_lat, n_lon)))
    dhw = degree_heating_weeks_fv3(sst, mmm)
    assert dhw.shape == (n_lat, n_lon)
    assert jnp.all(jnp.isfinite(dhw))
    assert jnp.all(dhw >= 0.0)
