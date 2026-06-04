"""FV3_3D iter 585: FV3 iord=8 Lin (1996) monotonicity limiter.

Adds ``apply_hord8_limiter`` to ``legoesm.core.fv_tp_2d`` as an
alternative to the iord=9 ``_pert_ppm`` limiter.

Tests
-----

1. ``test_hord8_returns_finite``.
2. ``test_hord8_zero_dm_clips_to_zero`` — dm=0 → bl=br=0 (flat).
3. ``test_hord8_bounded_by_dm`` — |bl|, |br| ≤ 2|dm| always.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import apply_hord8_limiter


def test_hord8_returns_finite():
    rng = np.random.default_rng(seed=585)
    bl = jnp.asarray(rng.uniform(-2, 2, size=(10,)))
    br = jnp.asarray(rng.uniform(-2, 2, size=(10,)))
    dm = jnp.asarray(rng.uniform(-1, 1, size=(10,)))
    bl_out, br_out = apply_hord8_limiter(bl, br, dm)
    assert jnp.all(jnp.isfinite(bl_out))
    assert jnp.all(jnp.isfinite(br_out))


def test_hord8_zero_dm_clips_to_zero():
    """At flat dm=0, hord=8 clamps bl, br to 0 (no flux)."""
    bl = jnp.asarray([1.0, -0.5, 0.3, -2.0])
    br = jnp.asarray([0.5, -1.0, 0.7, 1.0])
    dm = jnp.zeros_like(bl)
    bl_out, br_out = apply_hord8_limiter(bl, br, dm)
    assert jnp.all(jnp.abs(bl_out) < 1e-15)
    assert jnp.all(jnp.abs(br_out) < 1e-15)


def test_hord8_bounded_by_dm():
    """|bl_out|, |br_out| ≤ 2·|dm|."""
    rng = np.random.default_rng(seed=586)
    bl = jnp.asarray(rng.uniform(-5, 5, size=(20,)))
    br = jnp.asarray(rng.uniform(-5, 5, size=(20,)))
    dm = jnp.asarray(rng.uniform(-1, 1, size=(20,)))
    bl_out, br_out = apply_hord8_limiter(bl, br, dm)
    assert jnp.all(jnp.abs(bl_out) <= 2.0 * jnp.abs(dm) + 1e-12)
    assert jnp.all(jnp.abs(br_out) <= 2.0 * jnp.abs(dm) + 1e-12)
