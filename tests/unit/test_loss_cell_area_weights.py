"""Cell-area weighting of the WB loss on the cubed sphere (user decision Q2)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.losses import grid_area_weights, lat_weighted_mean


def _cube():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(6)


def test_cube_default_weights_are_the_cell_areas():
    g = _cube()
    np.testing.assert_array_equal(np.asarray(grid_area_weights(g)),
                                  np.asarray(g.area))


def test_area_weighted_mean_of_a_column_field_with_levels():
    g = _cube()
    area = np.asarray(g.area)
    f = np.random.default_rng(0).normal(size=area.shape + (3,))
    want = np.sum(f * area[..., None]) / (np.sum(area) * 3)
    got = float(lat_weighted_mean(jnp.asarray(f), jnp.asarray(area)))
    # the grid stores fp32 areas: their sum carries ~1e-7 relative error
    assert got == pytest.approx(want, rel=1e-6)
    # Not the plain mean: the cube's cells differ in area.
    assert abs(got - float(np.mean(f))) > 1e-6


def test_shape_mismatch_raises():
    area = jnp.ones((6, 4, 4))
    with pytest.raises(ValueError):
        lat_weighted_mean(jnp.ones((6, 5, 5)), area)


def test_structured_grids_keep_their_latitude_weights():
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8, 16)
    assert grid_area_weights(g) is g.weights
