"""Direct unit tests for ``masked_background_vmix_coefficient`` (#517 item 6).

The helper factors the genuinely-shared MPAS implicit-vmix sub-part: build a
constant background vertical-mixing coefficient on the half-levels of a
partial-cell column and zero it at every interface below the deepest active
full level.  It is used by BOTH the MPAS per-cell tracer K_v build and the
per-edge momentum A_v build, replacing two byte-identical inline blocks.

Pins: the per-column seafloor masking, the active-half mask ``k < bottom_level``,
the dry-column (``bottom_level == -1``) all-zero behaviour, dtype/shape, and
differentiability (the coefficient must flow gradients through the background).

fp64 + CPU for deterministic comparison.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_tendency_common import (
    masked_background_vmix_coefficient,
)


def _reference(background, bottom_level, n_half):
    """Independent numpy reference for the active-half masked coefficient."""
    bl = np.asarray(bottom_level)
    k = np.arange(n_half, dtype=np.int32)[None, :]
    active = k < bl[:, None]
    coeff = np.where(active, float(background), 0.0)
    return coeff, active


def test_matches_reference_mixed_columns():
    """Coefficient + mask match the independent numpy reference on a mix of
    full, partial, single-interface, and dry columns."""
    bottom_level = jnp.asarray([4, 2, 1, 0, -1], dtype=jnp.int32)  # nlev=5
    n_half = 4
    bg = 1.0e-4
    coeff, active = masked_background_vmix_coefficient(bg, bottom_level, n_half)
    ref_coeff, ref_active = _reference(bg, bottom_level, n_half)
    np.testing.assert_array_equal(np.asarray(coeff), ref_coeff)
    np.testing.assert_array_equal(np.asarray(active), ref_active)
    assert active.dtype == jnp.bool_
    assert coeff.shape == (5, n_half)


def test_dry_column_all_zero():
    """A dry column (bottom_level == -1) yields zero coefficient everywhere and
    an all-False active mask (no interface is wet)."""
    coeff, active = masked_background_vmix_coefficient(
        5.0, jnp.asarray([-1], dtype=jnp.int32), 4)
    assert float(jnp.max(jnp.abs(coeff))) == 0.0
    assert not bool(jnp.any(active))


def test_full_column_all_active_equals_background():
    """A column wet to the bottom (bottom_level == nlev-1) has the background
    coefficient on EVERY interface and an all-True active mask."""
    nlev = 5
    coeff, active = masked_background_vmix_coefficient(
        3.0, jnp.asarray([nlev - 1], dtype=jnp.int32), nlev - 1)
    np.testing.assert_array_equal(
        np.asarray(coeff), np.full((1, nlev - 1), 3.0))
    assert bool(jnp.all(active))


def test_interface_k_couples_k_and_kplus1():
    """Interface k is active iff k < bottom_level, i.e. both coupled full
    levels k and k+1 are at or above the seafloor.  For bottom_level=2 the
    active interfaces are {0, 1} (couple 0-1 and 1-2) and {2, 3} are zero."""
    coeff, active = masked_background_vmix_coefficient(
        7.0, jnp.asarray([2], dtype=jnp.int32), 4)
    np.testing.assert_array_equal(
        np.asarray(active[0]), np.array([True, True, False, False]))
    np.testing.assert_array_equal(
        np.asarray(coeff[0]), np.array([7.0, 7.0, 0.0, 0.0]))


def test_differentiable_through_background():
    """The coefficient flows a gradient through the (traced) background on the
    active interfaces and zero on the masked ones — AD-safe."""
    bottom_level = jnp.asarray([3, 1], dtype=jnp.int32)
    n_half = 4

    def loss(bg):
        coeff, _ = masked_background_vmix_coefficient(bg, bottom_level, n_half)
        return jnp.sum(coeff)

    # column 0: 3 active interfaces; column 1: 1 active interface → d/dbg = 4.
    g = jax.grad(loss)(1.0e-4)
    assert np.isclose(float(g), 4.0)


def test_non_int32_bottom_level_is_cast():
    """A bottom_level in a wider int dtype is cast internally; result matches
    the int32 reference (the MPAS edge path passes a minimum-reduced index)."""
    bl64 = jnp.asarray([2, 0], dtype=jnp.int64)
    coeff, active = masked_background_vmix_coefficient(2.0, bl64, 3)
    ref_coeff, ref_active = _reference(2.0, bl64, 3)
    np.testing.assert_array_equal(np.asarray(coeff), ref_coeff)
    np.testing.assert_array_equal(np.asarray(active), ref_active)
