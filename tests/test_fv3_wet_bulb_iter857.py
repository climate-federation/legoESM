"""FV3_3D iter 857: wet_bulb_temperature_stull_fv3.

Stull 2011 empirical T_w fit (J. Appl. Meteor. Climatol.).

Tests
-----

1. ``test_saturated_air_equals_t``: RH=100% → T_w ≈ T.
2. ``test_dry_air_below_t``: RH=10% → T_w << T.
3. ``test_temperate_humid``: 30°C/70% → T_w ≈ 26°C.
4. ``test_persian_gulf_extreme``: 40°C/55% → T_w ≈ 31°C.
5. ``test_survivability_ceiling``: 45°C/50% → T_w near 35°C.
6. ``test_monotone_in_t``: ↑T at fixed RH → ↑T_w.
7. ``test_monotone_in_rh``: ↑RH at fixed T → ↑T_w.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import wet_bulb_temperature_stull_fv3


def test_saturated_air_equals_t():
    """At RH=100%, T_w ≈ T (saturated air, no evaporative cooling)."""
    t = jnp.array([25.0])
    rh = jnp.array([100.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    # Stull max error 0.3 K
    assert abs(float(tw[0]) - 25.0) < 0.5


def test_dry_air_below_t():
    """At RH=10%, dry air → T_w << T (strong evaporative cooling)."""
    t = jnp.array([30.0])
    rh = jnp.array([10.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    assert float(tw[0]) < 18.0  # T_w drops ≥12 K at low RH


def test_temperate_humid():
    """30°C / 70% RH → T_w ≈ 26°C (typical summer humid Earth)."""
    t = jnp.array([30.0])
    rh = jnp.array([70.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    assert 24.5 < float(tw[0]) < 27.0


def test_persian_gulf_extreme():
    """40°C / 55% RH → T_w near 31°C (Persian Gulf summer)."""
    t = jnp.array([40.0])
    rh = jnp.array([55.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    assert 30.0 < float(tw[0]) < 33.0


def test_survivability_ceiling():
    """45°C / 50% RH approaches Sherwood-Huber 35°C ceiling."""
    t = jnp.array([45.0])
    rh = jnp.array([50.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    assert float(tw[0]) > 32.0


def test_monotone_in_t():
    """↑T at fixed RH → ↑T_w."""
    rh = jnp.full((4,), 60.0)
    t = jnp.array([20.0, 25.0, 30.0, 35.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    diffs = jnp.diff(tw)
    assert jnp.all(diffs > 0.0)


def test_monotone_in_rh():
    """↑RH at fixed T → ↑T_w."""
    t = jnp.full((4,), 30.0)
    rh = jnp.array([20.0, 40.0, 60.0, 80.0])
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    diffs = jnp.diff(tw)
    assert jnp.all(diffs > 0.0)


def test_shapes_finite():
    """3-D shapes preserved, finite, T_w ≤ T."""
    rng = np.random.default_rng(seed=857)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(0.0, 45.0, size=(n_x, n_y)))
    rh = jnp.asarray(rng.uniform(10.0, 99.0, size=(n_x, n_y)))
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    assert tw.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(tw))
    # T_w must be ≤ T always (psychrometric inequality)
    assert jnp.all(tw <= t + 0.5)  # +0.5 K tolerance for Stull empirical
