"""FV3_3D iter 665: super_k_u_fv3 port.

Faithful JAX port of FV3 ``SuperK_u`` (tools/test_cases.F90:
6049-6082, MPAS branch).  Super-cell vertical wind-shear profile.

Tests
-----

1. ``test_super_k_u_shape``.
2. ``test_super_k_u_upper_constant``.
3. ``test_super_k_u_lower_linear``.
4. ``test_super_k_u_continuity_at_boundaries``.
5. ``test_super_k_u_uc_subtracted``.
6. ``test_super_k_u_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import super_k_u_fv3


def test_super_k_u_shape():
    """Output shapes match input."""
    zz = jnp.linspace(0.0, 10000.0, 50)
    um, dudz = super_k_u_fv3(zz)
    assert um.shape == zz.shape
    assert dudz.shape == zz.shape


def test_super_k_u_upper_constant():
    """For z > zs+1km, um is constant (us - uc); dudz = 0."""
    zz = jnp.asarray([7000.0, 8000.0, 10000.0])
    um, dudz = super_k_u_fv3(zz, zs=5000.0, us=30.0, uc=15.0)
    expected_um = 30.0 - 15.0    # us - uc
    assert jnp.allclose(um, expected_um, atol=1e-12)
    assert jnp.allclose(dudz, 0.0, atol=1e-12)


def test_super_k_u_lower_linear():
    """For z < zs-1km, um = us·z/zs - uc; dudz = us/zs."""
    zz = jnp.asarray([1000.0, 2000.0, 3000.0])
    zs, us, uc = 5000.0, 30.0, 15.0
    um, dudz = super_k_u_fv3(zz, zs=zs, us=us, uc=uc)
    expected_um = us * zz / zs - uc
    expected_dudz = jnp.full_like(zz, us / zs)
    assert jnp.allclose(um, expected_um, atol=1e-12)
    assert jnp.allclose(dudz, expected_dudz, atol=1e-12)


def test_super_k_u_continuity_at_boundaries():
    """Profile is continuous at z = zs - 1km and z = zs + 1km."""
    zs = 5000.0
    eps = 1.0
    # At z = zs - 1km
    z_left = zs - 1000.0
    um_below, _ = super_k_u_fv3(jnp.asarray([z_left - eps]))
    um_blend1, _ = super_k_u_fv3(jnp.asarray([z_left + eps]))
    assert abs(float(um_below[0]) - float(um_blend1[0])) < 0.02
    # At z = zs + 1km
    z_right = zs + 1000.0
    um_blend2, _ = super_k_u_fv3(jnp.asarray([z_right - eps]))
    um_above, _ = super_k_u_fv3(jnp.asarray([z_right + eps]))
    assert abs(float(um_blend2[0]) - float(um_above[0])) < 0.02


def test_super_k_u_uc_subtracted():
    """At z = 0, um = -uc (only the uc offset)."""
    um, _ = super_k_u_fv3(jnp.asarray([0.0]), uc=15.0)
    assert abs(float(um[0]) - (-15.0)) < 1e-12


def test_super_k_u_finite():
    """No NaN/Inf across full z range."""
    zz = jnp.linspace(0.0, 20000.0, 200)
    um, dudz = super_k_u_fv3(zz)
    assert jnp.all(jnp.isfinite(um))
    assert jnp.all(jnp.isfinite(dudz))
