"""Guards for the D->A ua/va halo adapters (codex round 3).

The corner-divergence routine needs A-grid winds carrying a cross-panel
ring.  Building them by a four-corner average + scalar pad is NOT FV3's
operation -- FV3 passes ``d2a2c_vect``'s output (sw_core.F90:148-160).
These adapters expose that, sliced to one ring, with no extra halo message.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.fv3_sw_core import (
    d2a2c_global_fields,
    d2a2c_ua_va_halo,
    d2a2c_ua_va_halo_4d,
    sina_u_v_from_sin_sg,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(scope="module")
def cube():
    n = 8
    return create_cubed_sphere_cdgrid(create_cubed_sphere(n)), n


def _winds(n, nlev=None, seed=3):
    rng = np.random.default_rng(seed)
    shp_u = (6, n, n + 1) if nlev is None else (6, n, n + 1, nlev)
    shp_v = (6, n + 1, n) if nlev is None else (6, n + 1, n, nlev)
    return (jnp.asarray(rng.uniform(-4.0, 4.0, size=shp_u)),
            jnp.asarray(rng.uniform(-4.0, 4.0, size=shp_v)))


def test_ua_va_halo_shapes_are_one_ring(cube):
    cd, n = cube
    u_d, v_d = _winds(n)
    ua, va = d2a2c_ua_va_halo(u_d, v_d, cd)
    assert ua.shape == (6, n + 2, n + 2)
    assert va.shape == (6, n + 2, n + 2)


def test_ua_va_halo_interior_matches_the_h2_source(cube):
    """The ring slice must be a view of d2a2c's own h2 field, not a re-derivation."""
    cd, n = cube
    u_d, v_d = _winds(n)
    ref = d2a2c_global_fields(u_d, v_d, cd)
    ua, va = d2a2c_ua_va_halo(u_d, v_d, cd)
    np.testing.assert_array_equal(
        np.asarray(ua), np.asarray(ref.ua_pad[:, 1:-1, 1:-1]))
    np.testing.assert_array_equal(
        np.asarray(va), np.asarray(ref.va_pad[:, 1:-1, 1:-1]))


def test_ua_va_halo_is_not_a_four_corner_average(cube):
    """Non-vacuity: the adapter must DIFFER from the averaging it replaces.

    If these ever coincide, the adapter is not doing FV3's D->A algebra and
    the fidelity gap it exists to close is still open.
    """
    cd, n = cube
    u_d, v_d = _winds(n)
    ua, _ = d2a2c_ua_va_halo(u_d, v_d, cd)

    # the interior four-corner average of corner-staggered winds
    rng = np.random.default_rng(3)
    u_corner = jnp.asarray(rng.uniform(-4.0, 4.0, size=(6, n + 1, n + 1)))
    avg = 0.25 * (u_corner[:, :-1, :-1] + u_corner[:, 1:, :-1]
                  + u_corner[:, :-1, 1:] + u_corner[:, 1:, 1:])
    assert not np.allclose(np.asarray(ua[:, 1:-1, 1:-1]), np.asarray(avg))


def test_ua_va_halo_4d_matches_per_level(cube):
    """The all-levels path must equal the per-level path level by level."""
    cd, n = cube
    nlev = 3
    u4, v4 = _winds(n, nlev)
    ua4, va4 = d2a2c_ua_va_halo_4d(u4, v4, cd)
    assert ua4.shape == (6, n + 2, n + 2, nlev)
    for k in range(nlev):
        ua_k, va_k = d2a2c_ua_va_halo(u4[..., k], v4[..., k], cd)
        np.testing.assert_allclose(np.asarray(ua4[..., k]),
                                   np.asarray(ua_k), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(va4[..., k]),
                                   np.asarray(va_k), rtol=1e-12, atol=1e-12)


def test_ua_va_halo_is_differentiable(cube):
    """AD must flow: this sits inside the damping term of a live dycore."""
    cd, n = cube
    u_d, v_d = _winds(n)

    def loss(u):
        ua, va = d2a2c_ua_va_halo(u, v_d, cd)
        return jnp.sum(ua ** 2 + va ** 2)

    g = jax.grad(loss)(u_d)
    assert g.shape == u_d.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_sina_helper_is_public():
    """Promoted from ``_sina_u_v_from_sin_sg``: the divergence routine is in a
    different module, and cross-module private imports are banned here."""
    cd = create_cubed_sphere_cdgrid(create_cubed_sphere(8))
    sina_u, sina_v = sina_u_v_from_sin_sg(cd)
    assert sina_u.shape[0] == 6 and sina_v.shape[0] == 6
    assert np.all(np.isfinite(np.asarray(sina_u)))
    assert np.all(np.isfinite(np.asarray(sina_v)))
