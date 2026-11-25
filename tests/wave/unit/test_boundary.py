"""Boundary updaters: absorbing edges, identity layout and prescribed spectra."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.wave.physics.boundary import (
    apply_boundary_condition,
    identity_boundary_update,
    update_absorb_boundary,
)


def _field():
    return jnp.ones((1, 5, 4, 2, 2), dtype=jnp.float32)


def test_absorb_boundary_zeroes_every_edge_cell():
    out = np.asarray(update_absorb_boundary(_field()))
    assert np.all(out[:, 0, :] == 0.0)
    assert np.all(out[:, -1, :] == 0.0)
    assert np.all(out[:, :, 0] == 0.0)
    assert np.all(out[:, :, -1] == 0.0)
    assert np.all(out[:, 1:-1, 1:-1] == 1.0)


def test_identity_boundary_retains_the_physical_edges():
    field = _field()
    np.testing.assert_array_equal(
        np.asarray(identity_boundary_update(field)), np.asarray(field))


def test_prescribed_zero_boundary_is_enforced():
    field = _field()
    zeros = jnp.zeros_like(field)
    boundary_mask = jnp.zeros(field.shape[:3], dtype=bool).at[0, 0, :].set(True)
    out = np.asarray(apply_boundary_condition(
        field, zeros, jnp.ones(field.shape[:3], dtype=bool),
        boundary_mask=boundary_mask))
    assert np.all(out[:, 0, :] == 0.0)
    assert np.all(out[:, 1:] == 1.0)


def test_boundary_mask_accepts_3d_and_5d_shapes():
    field = _field()
    zeros = jnp.zeros_like(field)
    mask3 = jnp.zeros(field.shape[:3], dtype=bool).at[0, 0, :].set(True)
    out3 = np.asarray(apply_boundary_condition(
        field, zeros, jnp.ones(field.shape[:3], dtype=bool),
        boundary_mask=mask3))
    out5 = np.asarray(apply_boundary_condition(
        field, zeros, jnp.ones(field.shape[:3], dtype=bool),
        boundary_mask=mask3[..., None, None]))
    np.testing.assert_array_equal(out3, out5)
    assert np.all(out3[:, 0, :] == 0.0)


def test_land_mask_zeroes_columns():
    field = _field()
    mask = jnp.ones(field.shape[:3], dtype=bool).at[0, 2, 2].set(False)
    out = np.asarray(apply_boundary_condition(
        field, jnp.zeros_like(field), mask))
    assert np.all(out[0, 2, 2] == 0.0)
    assert np.all(out[0, 0, 0] == 1.0)
