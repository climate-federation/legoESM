"""Unit tests for ``pad_halo_plane_4d`` in
``src/legoesm/atmosphere/dynamics/les/plane_operators.py``.

Covers shape correctness, round-trip recovery of the interior, edge
wrap, corner wrap (NE, NW, SE, SW), input validation, and
``jax.test_util.check_grads`` for AD safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.plane_operators import pad_halo_plane_4d


jax.config.update("jax_enable_x64", True)


def _random_field(batch=2, nlev=3, ny=6, nx=8, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((batch, nlev, ny, nx)))


# --------------------------------------------------------------------- #
# Shape + interior round-trip                                           #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("h", [1, 2, 3])
def test_output_shape(h):
    field = _random_field(ny=8, nx=10)
    out = pad_halo_plane_4d(field, h)
    assert out.shape == (2, 3, 8 + 2 * h, 10 + 2 * h)


@pytest.mark.parametrize("h", [1, 2, 3])
def test_interior_is_unchanged(h):
    field = _random_field(ny=8, nx=10)
    out = pad_halo_plane_4d(field, h)
    interior = out[:, :, h:h + field.shape[-2], h:h + field.shape[-1]]
    assert jnp.array_equal(interior, field)


# --------------------------------------------------------------------- #
# Edge wrap                                                             #
# --------------------------------------------------------------------- #


def test_left_halo_wraps_from_right():
    h = 2
    field = _random_field(ny=6, nx=8)
    out = pad_halo_plane_4d(field, h)
    # Left halo (columns 0..h-1) takes the rightmost h interior columns.
    expected = field[:, :, :, -h:]
    assert jnp.array_equal(out[:, :, h:h + 6, :h], expected)


def test_right_halo_wraps_from_left():
    h = 2
    field = _random_field(ny=6, nx=8)
    out = pad_halo_plane_4d(field, h)
    expected = field[:, :, :, :h]
    assert jnp.array_equal(out[:, :, h:h + 6, -h:], expected)


def test_top_halo_wraps_from_bottom():
    h = 2
    field = _random_field(ny=6, nx=8)
    out = pad_halo_plane_4d(field, h)
    expected = field[:, :, -h:, :]
    assert jnp.array_equal(out[:, :, :h, h:h + 8], expected)


def test_bottom_halo_wraps_from_top():
    h = 2
    field = _random_field(ny=6, nx=8)
    out = pad_halo_plane_4d(field, h)
    expected = field[:, :, :h, :]
    assert jnp.array_equal(out[:, :, -h:, h:h + 8], expected)


# --------------------------------------------------------------------- #
# Corner wrap                                                           #
# --------------------------------------------------------------------- #


def test_corner_halos_consistent_two_axis_wrap():
    """All four corners take the opposite-corner interior block."""
    h = 2
    field = _random_field(ny=6, nx=8)
    out = pad_halo_plane_4d(field, h)
    # NW corner of padded = SE corner of input.
    assert jnp.array_equal(out[:, :, :h, :h], field[:, :, -h:, -h:])
    # NE corner of padded = SW corner of input.
    assert jnp.array_equal(out[:, :, :h, -h:], field[:, :, -h:, :h])
    # SW corner of padded = NE corner of input.
    assert jnp.array_equal(out[:, :, -h:, :h], field[:, :, :h, -h:])
    # SE corner of padded = NW corner of input.
    assert jnp.array_equal(out[:, :, -h:, -h:], field[:, :, :h, :h])


# --------------------------------------------------------------------- #
# Input validation                                                      #
# --------------------------------------------------------------------- #


def test_rejects_non_4d_input():
    field_3d = jnp.zeros((4, 5, 6))
    with pytest.raises(ValueError, match="expects a 4D"):
        pad_halo_plane_4d(field_3d, 1)


def test_rejects_non_int_halo_width():
    field = _random_field()
    with pytest.raises(ValueError, match="must be a Python int"):
        pad_halo_plane_4d(field, 1.5)  # type: ignore[arg-type]


def test_rejects_non_positive_halo():
    field = _random_field()
    with pytest.raises(ValueError, match="must be positive"):
        pad_halo_plane_4d(field, 0)


def test_rejects_halo_exceeding_domain():
    field = _random_field(ny=4, nx=6)  # min = 4
    with pytest.raises(ValueError, match="strictly less than"):
        pad_halo_plane_4d(field, 4)


# --------------------------------------------------------------------- #
# Differentiability                                                     #
# --------------------------------------------------------------------- #


def test_pad_halo_check_grads():
    field = _random_field(batch=1, nlev=2, ny=6, nx=6)
    jax.test_util.check_grads(
        lambda f: pad_halo_plane_4d(f, 2).sum(), (field,),
        order=2, modes=["rev"], rtol=1.0e-6, atol=1.0e-6,
    )
