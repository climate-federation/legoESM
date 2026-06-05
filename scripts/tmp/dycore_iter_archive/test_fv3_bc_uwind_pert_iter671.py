"""FV3_3D iter 671: dcmip16_bc_uwind_pert port.

Faithful JAX port of FV3 ``DCMIP16_BC_uwind_pert`` (tools/test_cases.F90:
6823-6838).  Localized Gaussian-in-x, Hermite-cubic-in-z wind
perturbation for BC test (DCMIP16 Test 410 trigger).

Tests
-----

1. ``test_pert_center_max``.
2. ``test_pert_far_field_zero``.
3. ``test_pert_above_zp_zero``.
4. ``test_pert_nonnegative``.
5. ``test_pert_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import dcmip16_bc_uwind_pert


def test_pert_center_max():
    """At z=0, perturbation focal point: pert ≈ up = 1 m/s."""
    pi = jnp.pi
    pert = dcmip16_bc_uwind_pert(
        jnp.asarray(0.0),
        lat=jnp.asarray(2 * pi / 9), lon=jnp.asarray(pi / 9),
    )
    assert abs(float(pert) - 1.0) < 1e-10


def test_pert_far_field_zero():
    """Far from center (opposite side of globe): pert ≈ 0."""
    pi = jnp.pi
    pert = dcmip16_bc_uwind_pert(
        jnp.asarray(0.0),
        lat=jnp.asarray(-2 * pi / 9), lon=jnp.asarray(pi + pi / 9),
    )
    assert abs(float(pert)) < 1e-10


def test_pert_at_zp_zero():
    """At z = zp = 15 km exactly: ZZ = 1 - 3 + 2 = 0 → pert = 0."""
    pi = jnp.pi
    pert = dcmip16_bc_uwind_pert(
        jnp.asarray(15000.0),
        lat=jnp.asarray(2 * pi / 9), lon=jnp.asarray(pi / 9),
    )
    assert abs(float(pert)) < 1e-10


def test_pert_nonnegative():
    """Pert >= 0 everywhere."""
    rng = np.random.default_rng(seed=671)
    z = jnp.asarray(rng.uniform(0, 20000, size=50))
    lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=50))
    lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=50))
    pert = dcmip16_bc_uwind_pert(z, lat, lon)
    assert jnp.all(pert >= 0)


def test_pert_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=672)
    z = jnp.asarray(rng.uniform(0, 20000, size=50))
    lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=50))
    lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=50))
    pert = dcmip16_bc_uwind_pert(z, lat, lon)
    assert jnp.all(jnp.isfinite(pert))
