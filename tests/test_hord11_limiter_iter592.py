"""FV3_3D iter 592: FV3 iord=11 PPM limiter utility.

Tests
-----

1. ``test_hord11_returns_finite``.
2. ``test_hord11_default_factor_1p5`` — uses ppm_fac=1.5 default.
3. ``test_hord11_with_factor_2_matches_hord8`` — ppm_fac=2.0
   ⇒ identical to iord=8 (iter 585).
4. ``test_hord11_bounded_by_factor_times_dm``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import (
    apply_hord8_limiter,
    apply_hord11_limiter,
)


def test_hord11_returns_finite():
    rng = np.random.default_rng(seed=592)
    bl = jnp.asarray(rng.uniform(-2, 2, size=(10,)))
    br = jnp.asarray(rng.uniform(-2, 2, size=(10,)))
    dm = jnp.asarray(rng.uniform(-1, 1, size=(10,)))
    bl_out, br_out = apply_hord11_limiter(bl, br, dm)
    assert jnp.all(jnp.isfinite(bl_out))
    assert jnp.all(jnp.isfinite(br_out))


def test_hord11_default_factor_1p5():
    """ppm_fac=1.5 by default."""
    bl = jnp.asarray([3.0])
    br = jnp.asarray([-3.0])
    dm = jnp.asarray([1.0])  # 1.5*dm = 1.5; |bl|=3 > 1.5 → clipped to 1.5
    bl_out, br_out = apply_hord11_limiter(bl, br, dm)
    # bl: sign(xt)=sign(1.5)=+; -sign · min(1.5, 3.0) = -1.5
    assert abs(float(bl_out[0]) - (-1.5)) < 1e-10
    # br: +sign · min(1.5, 3.0) = +1.5
    assert abs(float(br_out[0]) - 1.5) < 1e-10


def test_hord11_with_factor_2_matches_hord8():
    """ppm_fac=2.0 = iord=8 from iter 585."""
    rng = np.random.default_rng(seed=593)
    bl = jnp.asarray(rng.uniform(-5, 5, size=(15,)))
    br = jnp.asarray(rng.uniform(-5, 5, size=(15,)))
    dm = jnp.asarray(rng.uniform(-1, 1, size=(15,)))
    bl_11, br_11 = apply_hord11_limiter(bl, br, dm, ppm_fac=2.0)
    bl_8, br_8 = apply_hord8_limiter(bl, br, dm)
    assert float(jnp.abs(bl_11 - bl_8).max()) < 1e-12
    assert float(jnp.abs(br_11 - br_8).max()) < 1e-12


def test_hord11_bounded_by_factor_times_dm():
    """|bl|, |br| ≤ ppm_fac · |dm|."""
    rng = np.random.default_rng(seed=594)
    bl = jnp.asarray(rng.uniform(-5, 5, size=(20,)))
    br = jnp.asarray(rng.uniform(-5, 5, size=(20,)))
    dm = jnp.asarray(rng.uniform(-1, 1, size=(20,)))
    for fac in [1.0, 1.5, 1.8]:
        bl_out, br_out = apply_hord11_limiter(bl, br, dm, ppm_fac=fac)
        assert jnp.all(jnp.abs(bl_out) <= fac * jnp.abs(dm) + 1e-12)
        assert jnp.all(jnp.abs(br_out) <= fac * jnp.abs(dm) + 1e-12)
