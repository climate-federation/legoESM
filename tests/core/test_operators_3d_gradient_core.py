"""Direct test of the ``gradient_{x,y}_3d_core`` leaves factored out of
``gradient_{x,y}_3d`` (the shared centred-difference stencil used by BOTH the
global 3D cubed-sphere operators and the tiled production thermodynamic stage).

The extraction must (a) leave the global op bit-identical and (b) reproduce the
global gradient at a sub-tile window when fed a slice of the global padded field
and the matching tile ``dx``/``dy`` — the property the tiled stage relies on.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import set_halo_backend, pad_halo_4d
from legoesm.core.operators_3d import (
    gradient_x_3d, gradient_y_3d, gradient_x_3d_core, gradient_y_3d_core,
)


def test_gradient_3d_core_reproduces_global_op():
    """core(padded, grid.dx/dy) == the global op (which now delegates to it)."""
    set_halo_backend("local")
    n, nlev = 12, 4
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    padded = pad_halo_4d(
        f, interp_offsets=grid.halo_interp_offsets, duogrid=None)
    np.testing.assert_array_equal(
        np.asarray(gradient_x_3d_core(padded, grid.dx)),
        np.asarray(gradient_x_3d(f, grid)))
    np.testing.assert_array_equal(
        np.asarray(gradient_y_3d_core(padded, grid.dy)),
        np.asarray(gradient_y_3d(f, grid)))


def test_gradient_3d_core_tile_window_matches_global():
    """A contiguous sub-tile window of the global padded field + the tile
    ``dx``/``dy`` reproduces the global gradient at those interior cells (the
    slice property the tiled stage's in-stage halo must match)."""
    set_halo_backend("local")
    n, nlev, nl = 12, 3, 6                     # kt=2 -> nl=6
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(1)
    f = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    padded = pad_halo_4d(
        f, interp_offsets=grid.halo_interp_offsets, duogrid=None)
    gx = np.asarray(gradient_x_3d(f, grid))
    gy = np.asarray(gradient_y_3d(f, grid))
    for ti in (0, 1):
        for tj in (0, 1):
            a_i, a_j = ti * nl, tj * nl
            win = padded[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2]
            dxt = grid.dx[:, a_i:a_i + nl, a_j:a_j + nl]
            dyt = grid.dy[:, a_i:a_i + nl, a_j:a_j + nl]
            np.testing.assert_array_equal(
                np.asarray(gradient_x_3d_core(win, dxt)),
                gx[:, a_i:a_i + nl, a_j:a_j + nl])
            np.testing.assert_array_equal(
                np.asarray(gradient_y_3d_core(win, dyt)),
                gy[:, a_i:a_i + nl, a_j:a_j + nl])
