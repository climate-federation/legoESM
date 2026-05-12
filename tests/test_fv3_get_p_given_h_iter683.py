"""FV3_3D iter 683: get_pressure_given_height_fv3 port.

Faithful JAX port of FV3 ``get_pressure_given_height``
(tools/fv_diagnostics.F90:4312-4365).

Tests
-----

1. ``test_p_at_surface``.
2. ``test_p_below_surface_higher``.
3. ``test_p_above_top_lower``.
4. ``test_p_monotonic_decrease_with_h``.
5. ``test_p_fac_multiplier``.
6. ``test_p_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import get_pressure_given_height_fv3


def _setup_atm(km=8):
    """Isothermal-ish hydrostatic profile: wz top→bottom, peln increasing."""
    wz = jnp.linspace(20000.0, 0.0, km + 1)
    peln = jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1)
    ts = jnp.asarray(290.0)
    return wz, peln, ts


def test_p_at_surface():
    """At height = wz[km] (surface): p ≈ exp(peln[km]) = p_s."""
    wz, peln, ts = _setup_atm()
    p = get_pressure_given_height_fv3(
        wz, peln, jnp.asarray(0.0), ts,
    )
    p_s = float(jnp.exp(peln[-1]))
    assert abs(float(p) - p_s) / p_s < 1e-6


def test_p_below_surface_higher():
    """Below surface (negative height): p > p_s."""
    wz, peln, ts = _setup_atm()
    p = get_pressure_given_height_fv3(
        wz, peln, jnp.asarray(-500.0), ts,
    )
    p_s = float(jnp.exp(peln[-1]))
    assert float(p) > p_s


def test_p_above_top_lower():
    """At top: p ≈ exp(peln[0]) = ptop."""
    wz, peln, ts = _setup_atm()
    p = get_pressure_given_height_fv3(
        wz, peln, wz[0], ts,
    )
    ptop = float(jnp.exp(peln[0]))
    assert abs(float(p) - ptop) / ptop < 1e-6


def test_p_monotonic_decrease_with_h():
    """p decreases as h increases."""
    wz, peln, ts = _setup_atm()
    heights = jnp.linspace(0.0, 20000.0, 30)
    # Vectorize: each height is a separate target
    p_list = [
        float(get_pressure_given_height_fv3(wz, peln, h, ts))
        for h in heights
    ]
    p_arr = jnp.asarray(p_list)
    diffs = p_arr[1:] - p_arr[:-1]
    assert jnp.all(diffs < 0)


def test_p_fac_multiplier():
    """fac=2 doubles output."""
    wz, peln, ts = _setup_atm()
    h = jnp.asarray(5000.0)
    p1 = get_pressure_given_height_fv3(wz, peln, h, ts)
    p2 = get_pressure_given_height_fv3(wz, peln, h, ts, fac=2.0)
    assert abs(float(p2) - 2.0 * float(p1)) < 1e-8


def test_p_finite():
    """No NaN/Inf for any reasonable h."""
    rng = np.random.default_rng(seed=683)
    wz = jnp.linspace(25000.0, 0.0, 10)
    peln = jnp.linspace(jnp.log(50.0), jnp.log(1.0e5), 10)
    ts = jnp.asarray(285.0)
    h = jnp.asarray(rng.uniform(-1000, 25000, size=20))
    p = jnp.asarray([
        float(get_pressure_given_height_fv3(wz, peln, h_, ts))
        for h_ in h
    ])
    assert jnp.all(jnp.isfinite(p))
    assert jnp.all(p > 0)
