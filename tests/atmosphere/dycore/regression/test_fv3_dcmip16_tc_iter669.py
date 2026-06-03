"""FV3_3D iter 669: dcmip16_tc_temperature + dcmip16_tc_pressure ports.

Faithful JAX ports of FV3 DCMIP16 TC temperature + pressure
(tools/test_cases.F90:7137-7167).

Tests
-----

1. ``test_tc_temperature_far_field_finite``.
2. ``test_tc_temperature_stratosphere_Tvt``.
3. ``test_tc_pressure_surface_finite``.
4. ``test_tc_pressure_decreases_with_z``.
5. ``test_tc_pressure_drops_at_center``.
6. ``test_tc_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dcmip16_tc_pressure,
    dcmip16_tc_temperature,
)


def test_tc_temperature_far_field_finite():
    """Far from TC center (r >> rp): T finite and positive."""
    T = dcmip16_tc_temperature(
        jnp.asarray(5000.0), jnp.asarray(1.0e7),
    )
    assert jnp.isfinite(T)
    assert float(T) > 0


def test_tc_temperature_stratosphere_Tvt():
    """Above zt = 15 km: T = Tvt (Tv0 - lapse·zt)."""
    Tv0 = 302.15 * (1.0 + 0.608 * 0.021)
    Tvt_expected = Tv0 - 0.007 * 15000.0
    T = dcmip16_tc_temperature(
        jnp.asarray(20000.0), jnp.asarray(1.0e6),
    )
    assert abs(float(T) - Tvt_expected) < 1e-10


def test_tc_pressure_surface_finite():
    """At z=0, r>>rp: p finite and near pb."""
    p = dcmip16_tc_pressure(
        jnp.asarray(0.0), jnp.asarray(1.0e7),
    )
    assert jnp.isfinite(p)
    # Far field at surface: p ≈ pb (background)
    assert 100000.0 < float(p) < 102000.0


def test_tc_pressure_decreases_with_z():
    """p monotonic in z (at far-field r=1e7)."""
    z = jnp.linspace(0.0, 25000.0, 30)
    r = jnp.full_like(z, 1.0e7)
    p = dcmip16_tc_pressure(z, r)
    diffs = p[1:] - p[:-1]
    assert jnp.all(diffs < 0), f"p not monotonic: max diff = {float(jnp.max(diffs))}"


def test_tc_pressure_drops_at_center():
    """At r=0 (TC center), p drops below background."""
    p_center = float(dcmip16_tc_pressure(
        jnp.asarray(0.0), jnp.asarray(0.0),
    ))
    p_far = float(dcmip16_tc_pressure(
        jnp.asarray(0.0), jnp.asarray(1.0e7),
    ))
    assert p_center < p_far, f"center p={p_center} should be < far p={p_far}"


def test_tc_finite():
    """No NaN/Inf on random (z, r) in test domain."""
    rng = np.random.default_rng(seed=669)
    z = jnp.asarray(rng.uniform(0, 25000, size=50))
    r = jnp.asarray(rng.uniform(1.0, 1.0e7, size=50))
    T = dcmip16_tc_temperature(z, r)
    p = dcmip16_tc_pressure(z, r)
    assert jnp.all(jnp.isfinite(T))
    assert jnp.all(jnp.isfinite(p))
