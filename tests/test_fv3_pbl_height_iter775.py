"""FV3_3D iter 775: pbl_height_fv3 (Ri_b threshold crossing).

z_pbl = z[k*] where k* = argmin{k : Ri_b(k) > ri_crit}.

Composes iter-774 ``bulk_richardson_fv3``.

Tests
-----

1. ``test_pbl_monotone_crossing``: monotone-increasing Ri_b column,
   threshold caught at expected level.
2. ``test_pbl_no_crossing``: Ri_b all < ri_crit → top of column.
3. ``test_pbl_all_above``: Ri_b all > ri_crit → surface.
4. ``test_pbl_custom_crit``: ri_crit=0.5 vs 0.25 picks different level.
5. ``test_pbl_compose_with_iter774``: full (θ_v, u, v, z) → z_pbl chain.
6. ``test_pbl_shapes_batched``: batched 3-D columns return (...,) shape.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    bulk_richardson_fv3,
    pbl_height_fv3,
)


def test_pbl_monotone_crossing():
    """Crossing at index 2 (Ri_b = 0.3 > 0.25)."""
    ri_b = jnp.array([0.0, 0.1, 0.3, 0.5, 1.0])
    z = jnp.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
    z_pbl = pbl_height_fv3(ri_b, z, ri_crit=0.25)
    assert float(z_pbl) == 500.0


def test_pbl_no_crossing():
    """All Ri_b < ri_crit → top of column."""
    ri_b = jnp.array([0.0, 0.05, 0.1, 0.15, 0.2])
    z = jnp.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
    z_pbl = pbl_height_fv3(ri_b, z, ri_crit=0.25)
    assert float(z_pbl) == 2000.0


def test_pbl_all_above():
    """All Ri_b > ri_crit → surface."""
    ri_b = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    z = jnp.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
    z_pbl = pbl_height_fv3(ri_b, z, ri_crit=0.25)
    assert float(z_pbl) == 50.0


def test_pbl_custom_crit():
    """ri_crit=0.5 picks a higher level than ri_crit=0.25."""
    ri_b = jnp.array([0.0, 0.1, 0.3, 0.5, 1.0])
    z = jnp.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
    z_pbl_25 = pbl_height_fv3(ri_b, z, ri_crit=0.25)  # crosses at 0.3 (idx 2)
    z_pbl_50 = pbl_height_fv3(ri_b, z, ri_crit=0.50)  # crosses at 1.0 (idx 4)
    assert float(z_pbl_25) == 500.0
    assert float(z_pbl_50) == 2000.0


def test_pbl_compose_with_iter774():
    """Full (θ_v_surf, θ_v, u, v, z) → Ri_b → z_pbl chain."""
    theta_v_surf = jnp.array([295.0])
    # Stable θ_v profile (typical morning sounding)
    theta_v = jnp.array([[295.0, 296.0, 298.0, 302.0, 310.0]])
    z = jnp.array([[10.0, 100.0, 500.0, 1500.0, 3000.0]])
    u = jnp.array([[1.0, 3.0, 5.0, 8.0, 10.0]])
    v = jnp.zeros_like(u)
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    z_pbl = pbl_height_fv3(ri_b, z, ri_crit=0.25)
    # PBL should be physically plausible (somewhere in 100-3000 m)
    assert 100.0 <= float(z_pbl[0]) <= 3000.0


def test_pbl_shapes_batched():
    """3-D batched: ri_b shape (n_x, n_y, km) → z_pbl shape (n_x, n_y)."""
    rng = np.random.default_rng(seed=775)
    n_x, n_y, km = 4, 5, 20
    # Build monotone-z columns
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(50.0, 500.0, size=(n_x, n_y, km))), axis=-1
    )
    # Construct Ri_b columns with at least one crossing
    ri_b = jnp.linspace(-0.1, 1.0, km)
    ri_b = jnp.broadcast_to(ri_b, (n_x, n_y, km))
    z_pbl = pbl_height_fv3(ri_b, z, ri_crit=0.25)
    assert z_pbl.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(z_pbl))
