"""FV3_3D iter 684: get_height_given_pressure_fv3 port.

Faithful JAX port of FV3 ``get_height_given_pressure``
(tools/fv_diagnostics.F90:4366-4411).

Tests
-----

1. ``test_h_at_top``.
2. ``test_h_at_surface``.
3. ``test_h_monotonic_decrease``.
4. ``test_h_inverse_of_iter683``.
5. ``test_h_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_height_given_pressure_fv3,
    get_pressure_given_height_fv3,
)


def _setup_atm(km=8):
    wz = jnp.linspace(20000.0, 0.0, km + 1)
    peln = jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1)
    ts = jnp.asarray(290.0)
    return wz, peln, ts


def test_h_at_top():
    """At log_p = peln[0] (top): h = wz[0]."""
    wz, peln, _ = _setup_atm()
    h = get_height_given_pressure_fv3(wz, peln, peln[0])
    assert abs(float(h) - float(wz[0])) / float(wz[0]) < 1e-10


def test_h_at_surface():
    """At log_p = peln[km] (surface): h = wz[km] = 0."""
    wz, peln, _ = _setup_atm()
    h = get_height_given_pressure_fv3(wz, peln, peln[-1])
    assert abs(float(h) - float(wz[-1])) < 1e-10


def test_h_monotonic_decrease():
    """h decreases as log_p increases (higher p = lower z)."""
    wz, peln, _ = _setup_atm()
    log_p_vals = jnp.linspace(jnp.log(200.0), jnp.log(5.0e4), 20)
    h_vals = jnp.asarray([
        float(get_height_given_pressure_fv3(wz, peln, lp))
        for lp in log_p_vals
    ])
    diffs = h_vals[1:] - h_vals[:-1]
    assert jnp.all(diffs < 0)


def test_h_inverse_of_iter683():
    """h(p_given_h(h)) ≈ h (round-trip)."""
    wz, peln, ts = _setup_atm()
    h_target = 5000.0
    p = float(get_pressure_given_height_fv3(
        wz, peln, jnp.asarray(h_target), ts,
    ))
    h_recovered = float(get_height_given_pressure_fv3(
        wz, peln, jnp.log(p),
    ))
    assert abs(h_recovered - h_target) / h_target < 1e-8


def test_h_finite():
    """No NaN/Inf on reasonable log_p range."""
    wz, peln, _ = _setup_atm()
    rng = np.random.default_rng(seed=684)
    log_p_vals = jnp.asarray(rng.uniform(jnp.log(100.0), jnp.log(1.0e5), size=15))
    for lp in log_p_vals:
        h = float(get_height_given_pressure_fv3(wz, peln, lp))
        assert jnp.isfinite(h)
