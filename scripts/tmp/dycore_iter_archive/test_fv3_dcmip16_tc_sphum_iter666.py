"""FV3_3D iter 666: dcmip16_tc_sphum port.

Faithful JAX port of FV3 ``DCMIP16_TC_sphum`` (tools/test_cases.F90:
7198-7208, DCMIP16 Reed-Jablonowski tropical-cyclone humidity profile).

Tests
-----

1. ``test_sphum_surface_q0``.
2. ``test_sphum_tropopause_qt``.
3. ``test_sphum_above_tropopause_qt``.
4. ``test_sphum_monotonic_decrease_below``.
5. ``test_sphum_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import dcmip16_tc_sphum


def test_sphum_surface_q0():
    """At z=0, q = q0 = 0.021 kg/kg."""
    q = dcmip16_tc_sphum(jnp.asarray(0.0))
    assert abs(float(q) - 0.021) < 1e-10


def test_sphum_tropopause_qt():
    """At z = zt, q = qt (stratospheric background)."""
    q = dcmip16_tc_sphum(jnp.asarray(15000.0))
    assert abs(float(q) - 1.0e-11) < 1e-20


def test_sphum_above_tropopause_qt():
    """Above tropopause, q = qt regardless of z."""
    for z in (15001.0, 20000.0, 50000.0):
        q = dcmip16_tc_sphum(jnp.asarray(z))
        assert abs(float(q) - 1.0e-11) < 1e-20


def test_sphum_monotonic_decrease_below():
    """q decreases with z below tropopause."""
    z = jnp.linspace(0.0, 14000.0, 50)
    q = dcmip16_tc_sphum(z)
    diffs = q[1:] - q[:-1]
    assert jnp.all(diffs <= 0.0), (
        f"q not monotonically decreasing: max increase = {float(jnp.max(diffs))}"
    )


def test_sphum_finite():
    """No NaN/Inf across full z range."""
    z = jnp.linspace(0.0, 50000.0, 200)
    q = dcmip16_tc_sphum(z)
    assert jnp.all(jnp.isfinite(q))
    assert jnp.all(q > 0.0)
