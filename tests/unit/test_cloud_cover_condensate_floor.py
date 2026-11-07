"""Condensate-aware cover floor (``CloudConfig.cover_condensate_q_ref``).

Pins: off (0.0) is byte-identical to the RH-only cover; on, a layer holding
prognostic condensate gets cf >= q/(q + q_ref) even where the RH cover is 0;
the floor never lowers cf; misuse raises.  Each assertion fails with the
floor deleted from ``compute_cloud_properties`` (non-vacuity checked by hand,
2026-09-08).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.clouds.config import CloudConfig, build_cloud_config


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _column():
    ncol, nlev = 2, 4
    T = jnp.array([[290.0, 270.0, 240.0, 220.0]] * ncol)
    p = jnp.array([[95000.0, 70000.0, 40000.0, 20000.0]] * ncol)
    dp = jnp.full((ncol, nlev), 10000.0)
    q_v = jnp.array([[0.010, 0.002, 1.0e-5, 1.0e-6]] * ncol)  # dry aloft: RH cover 0
    q_c = jnp.zeros((ncol, nlev))
    q_i = jnp.array([[0.0, 0.0, 3.0e-5, 1.0e-4], [0.0] * nlev])
    return T, p, q_v, dp, q_c, q_i


@pytest.mark.parametrize("scheme", ["sundqvist", "xu_randall"])
def test_off_is_byte_identical(scheme):
    T, p, q_v, dp, q_c, q_i = _column()
    base = CloudConfig(scheme=scheme, rh_crit=0.85)
    a = compute_cloud_properties(T, p, q_v, dp, base, q_cloud=q_c, q_ice=q_i)
    b = compute_cloud_properties(T, p, q_v, dp, base._replace(cover_condensate_q_ref=0.0),
                                 q_cloud=q_c, q_ice=q_i)
    for x, y in zip(a, b):
        if x is not None:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_floor_makes_condensate_layers_cloudy_and_never_lowers_cf():
    T, p, q_v, dp, q_c, q_i = _column()
    cfg = CloudConfig(scheme="sundqvist", rh_crit=0.85)
    off = np.asarray(compute_cloud_properties(T, p, q_v, dp, cfg, q_cloud=q_c, q_ice=q_i).cloud_fraction)
    q_ref = 3.0e-5
    on = np.asarray(compute_cloud_properties(
        T, p, q_v, dp, cfg._replace(cover_condensate_q_ref=q_ref),
        q_cloud=q_c, q_ice=q_i).cloud_fraction)
    # dry, ice-bearing layers: RH cover 0, floor = q/(q+q_ref)
    assert off[0, 2] == 0.0 and off[0, 3] == 0.0
    np.testing.assert_allclose(on[0, 2], 3.0e-5 / (3.0e-5 + q_ref), rtol=1e-12)
    np.testing.assert_allclose(on[0, 3], 1.0e-4 / (1.0e-4 + q_ref), rtol=1e-12)
    assert np.all(on >= off)
    # condensate-free column untouched
    np.testing.assert_array_equal(on[1], off[1])
    # the radiative ice path now carries the prognostic ice (cf > 0 => visible)
    iwp_on = np.asarray(compute_cloud_properties(
        T, p, q_v, dp, cfg._replace(cover_condensate_q_ref=q_ref),
        q_cloud=q_c, q_ice=q_i).iwp)
    assert iwp_on[0, 3] > 0.0


def test_misuse_raises():
    T, p, q_v, dp, q_c, q_i = _column()
    with pytest.raises(ValueError, match="requires explicit q_cloud/q_ice"):
        compute_cloud_properties(T, p, q_v, dp,
                                 CloudConfig(scheme="sundqvist", cover_condensate_q_ref=1e-5))
    with pytest.raises(ValueError, match="applies to the RH-diagnosed"):
        compute_cloud_properties(T, p, q_v, dp,
                                 CloudConfig(scheme="resolved", cover_condensate_q_ref=1e-5),
                                 q_cloud=q_c, q_ice=q_i)


def test_build_cloud_config_threads_the_field():
    assert build_cloud_config("sundqvist").cover_condensate_q_ref == 0.0
    assert build_cloud_config("sundqvist", cover_condensate_q_ref=3e-5).cover_condensate_q_ref == 3e-5
