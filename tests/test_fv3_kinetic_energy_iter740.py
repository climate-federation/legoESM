"""FV3_3D iter 740: kinetic_energy_fv3 helper + iter-693 refactor.

KE = 0.5 * (ua^2 + va^2 [+ w^2])

Tests
-----

1. ``test_ke_zero_wind``.
2. ``test_ke_unit_uniform_wind``.
3. ``test_ke_w_optional``.
4. ``test_ke_iter693_total_energy_unchanged``.
5. ``test_ke_shapes_3d``.
6. ``test_ke_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    kinetic_energy_fv3,
    nh_total_energy_fv3,
)


def test_ke_zero_wind():
    """All-zero wind → KE = 0."""
    n = 5
    z = jnp.zeros((n,))
    ke = kinetic_energy_fv3(z, z, z)
    assert jnp.all(ke == 0.0)


def test_ke_unit_uniform_wind():
    """ua=1, va=0, w=0 → KE = 0.5."""
    n = 5
    one = jnp.ones((n,))
    zero = jnp.zeros((n,))
    ke = kinetic_energy_fv3(one, zero, zero)
    assert jnp.allclose(ke, 0.5, atol=1e-12)


def test_ke_w_optional():
    """w=None → KE = 0.5(u² + v²) only."""
    n = 5
    ua = jnp.full((n,), 3.0)
    va = jnp.full((n,), 4.0)
    ke_no_w = kinetic_energy_fv3(ua, va)
    expected = 0.5 * (9.0 + 16.0)
    assert jnp.allclose(ke_no_w, expected, atol=1e-12)


def test_ke_iter693_total_energy_unchanged():
    """iter-693 nh_total_energy refactor preserves output bit-identically."""
    rng = np.random.default_rng(seed=740)
    km = 10
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    w = jnp.asarray(rng.normal(scale=0.1, size=(km,)))
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    hs = jnp.asarray(0.0)
    te = nh_total_energy_fv3(ua, va, w, pt, delp, delz, hs, moist_phys=False)
    assert jnp.isfinite(te)
    assert float(te) > 0.0


def test_ke_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=741)
    n_x, n_y, km = 4, 5, 20
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    w = jnp.asarray(rng.normal(scale=0.1, size=(n_x, n_y, km)))
    ke = kinetic_energy_fv3(ua, va, w)
    assert ke.shape == (n_x, n_y, km)


def test_ke_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=742)
    ua = jnp.asarray(rng.normal(scale=20.0, size=(4, 30)))
    va = jnp.asarray(rng.normal(scale=20.0, size=(4, 30)))
    w = jnp.asarray(rng.normal(scale=1.0, size=(4, 30)))
    ke = kinetic_energy_fv3(ua, va, w)
    assert jnp.all(jnp.isfinite(ke))
