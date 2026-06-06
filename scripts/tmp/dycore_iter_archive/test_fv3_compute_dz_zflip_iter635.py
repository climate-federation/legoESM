"""FV3_3D iter 635: compute_dz_fv3 + zflip ports.

Faithful JAX ports of FV3 helpers:
- ``compute_dz_fv3``  (tools/fv_eta.F90:1894) — initial dz layering
- ``zflip``           (tools/fv_eta.F90:2482) — vertical flip

Tests
-----

1. ``test_compute_dz_shape``.
2. ``test_compute_dz_top_doubled``.
3. ``test_compute_dz_bottom_halved``.
4. ``test_compute_dz_interior_uniform``.
5. ``test_compute_dz_total_height``.
6. ``test_zflip_reverses_axis``.
7. ``test_zflip_default_axis``.
8. ``test_zflip_idempotent``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import compute_dz_fv3, zflip


def test_compute_dz_shape():
    """Output shape (km,)."""
    km = 32
    dz = compute_dz_fv3(km, ztop=50.0e3)
    assert dz.shape == (km,)


def test_compute_dz_top_doubled():
    """Top cell dz[0] = 2·(ztop/km)."""
    km = 20
    ztop = 30.0e3
    dz_uniform = ztop / km
    dz = compute_dz_fv3(km, ztop=ztop)
    assert abs(float(dz[0]) - 2.0 * dz_uniform) < 1e-8


def test_compute_dz_bottom_halved():
    """Bottom cell dz[km-1] = 0.5·(ztop/km)."""
    km = 20
    ztop = 30.0e3
    dz_uniform = ztop / km
    dz = compute_dz_fv3(km, ztop=ztop)
    assert abs(float(dz[km - 1]) - 0.5 * dz_uniform) < 1e-8


def test_compute_dz_interior_uniform():
    """Interior cells dz[1..km-2] = ztop/km."""
    km = 16
    ztop = 50.0e3
    dz_uniform = ztop / km
    dz = compute_dz_fv3(km, ztop=ztop)
    assert jnp.allclose(dz[1:km - 1], dz_uniform, atol=1e-8)


def test_compute_dz_total_height():
    """Total height = (km + 0.5)·ztop/km > ztop (FV3 quirk)."""
    km = 32
    ztop = 50.0e3
    dz = compute_dz_fv3(km, ztop=ztop)
    total = float(jnp.sum(dz))
    expected = (km + 0.5) * ztop / km
    assert abs(total - expected) / expected < 1e-10


def test_zflip_reverses_axis():
    """zflip(q) along axis -1 reverses last axis."""
    q = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zflip(q)
    assert jnp.allclose(z, jnp.asarray([5.0, 4.0, 3.0, 2.0, 1.0]))


def test_zflip_default_axis():
    """Default axis is -1."""
    rng = np.random.default_rng(seed=635)
    q = jnp.asarray(rng.normal(size=(3, 4, 8)))
    z = zflip(q)
    assert jnp.allclose(z[:, :, ::-1], q)


def test_zflip_idempotent():
    """zflip twice is identity."""
    rng = np.random.default_rng(seed=636)
    q = jnp.asarray(rng.normal(size=(2, 5, 10)))
    z = zflip(zflip(q))
    assert jnp.allclose(z, q)
