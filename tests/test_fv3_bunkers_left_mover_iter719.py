"""FV3_3D iter 719: bunkers_vector_fv3 left-mover variant.

iter-689 returns Bunkers right-mover.  iter-719 adds
``right_mover=False`` flag for the left-mover variant (sign flip
on shear-perpendicular offset).

Tests
-----

1. ``test_bunkers_right_default_matches_iter689``.
2. ``test_bunkers_left_mover_zero_shear``.
3. ``test_bunkers_left_mover_linear_shear``.
4. ``test_bunkers_left_right_symmetric_around_mean``.
5. ``test_bunkers_left_mover_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import bunkers_vector_fv3


def test_bunkers_right_default_matches_iter689():
    """Default right_mover=True preserves iter-689 result exactly."""
    rng = np.random.default_rng(seed=719)
    km = 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    delz = jnp.full((km,), -500.0)
    uc_default, vc_default = bunkers_vector_fv3(ua, va, delz=delz)
    uc_explicit, vc_explicit = bunkers_vector_fv3(
        ua, va, delz=delz, right_mover=True,
    )
    assert jnp.allclose(uc_default, uc_explicit, atol=1e-12)
    assert jnp.allclose(vc_default, vc_explicit, atol=1e-12)


def test_bunkers_left_mover_zero_shear():
    """Uniform wind → shear = 0 → left/right both = mean wind."""
    km = 20
    ua = jnp.full((km,), 10.0)
    va = jnp.full((km,), 3.0)
    delz = jnp.full((km,), -500.0)
    uc_r, vc_r = bunkers_vector_fv3(ua, va, delz=delz, right_mover=True)
    uc_l, vc_l = bunkers_vector_fv3(ua, va, delz=delz, right_mover=False)
    assert abs(float(uc_r) - 10.0) < 1e-12
    assert abs(float(uc_l) - 10.0) < 1e-12
    assert abs(float(vc_r) - 3.0) < 1e-12
    assert abs(float(vc_l) - 3.0) < 1e-12


def test_bunkers_left_mover_linear_shear():
    """Linear westerly shear: u(z) = c·z, va=0, c=0.005/s.
    Right-mover: uc=15, vc=-7.5.
    Left-mover:  uc=15, vc=+7.5 (sign flip)."""
    km = 30
    delz = jnp.full((km,), -500.0)
    zh_mid = jnp.linspace(km * 500.0 - 250.0, 250.0, km)
    c = 0.005
    ua = c * zh_mid
    va = jnp.zeros((km,))
    uc_l, vc_l = bunkers_vector_fv3(ua, va, delz=delz, right_mover=False)
    # umn ≈ 15; +7.5 ushr offset
    assert abs(float(uc_l) - 15.0) < 0.5
    assert abs(float(vc_l) - 7.5) < 0.5


def test_bunkers_left_right_symmetric_around_mean():
    """(uc_right + uc_left)/2 = umn, (vc_right + vc_left)/2 = vmn."""
    rng = np.random.default_rng(seed=720)
    km = 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    delz = jnp.full((km,), -500.0)
    uc_r, vc_r = bunkers_vector_fv3(ua, va, delz=delz, right_mover=True)
    uc_l, vc_l = bunkers_vector_fv3(ua, va, delz=delz, right_mover=False)
    uc_avg = 0.5 * (uc_r + uc_l)
    vc_avg = 0.5 * (vc_r + vc_l)
    # Avg should equal mean wind (umn, vmn): bunkers offset cancels
    # Computed mean wind in 0-6 km layer
    dz = -delz
    cumsum = jnp.cumsum(dz[::-1])[::-1]
    zh_above = cumsum
    zh_below = zh_above - dz
    dz_eff = jnp.maximum(
        0.0, jnp.minimum(zh_above, 6000.0) - jnp.maximum(zh_below, 0.0)
    )
    total = jnp.sum(dz_eff)
    umn = jnp.sum(ua * dz_eff) / total
    vmn = jnp.sum(va * dz_eff) / total
    assert abs(float(uc_avg) - float(umn)) < 1e-10
    assert abs(float(vc_avg) - float(vmn)) < 1e-10


def test_bunkers_left_mover_finite():
    """No NaN/Inf on random 3-D inputs."""
    rng = np.random.default_rng(seed=721)
    km = 30
    ua = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    delz = jnp.full((4, 4, km), -300.0)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz, right_mover=False)
    assert jnp.all(jnp.isfinite(uc))
    assert jnp.all(jnp.isfinite(vc))
