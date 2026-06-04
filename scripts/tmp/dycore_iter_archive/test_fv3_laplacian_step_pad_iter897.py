"""FV3_3D iter 897: unit test fv3_laplacian_step_from_pad_h1.

Verify the extracted post-pad arithmetic produces bit-for-bit
identical output to ``fv3_corner_laplacian_iteration``'s default
(non-vector-fill) path.  Foundation for FV3-faithful expanding-halo
nord>=2 pattern (iter-898+).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_iteration,
    fv3_laplacian_step_from_pad_h1,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid
from legoesm.grids.halo import pad_halo


def _cdgrid(n):
    return _create_cdgrid(create_cubed_sphere(n))


def test_pad_h1_step_matches_iteration_random():
    """pad_halo(h=1) + step matches fv3_corner_laplacian_iteration
    bit-for-bit on random divg_d."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8970)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_extracted = fv3_laplacian_step_from_pad_h1(
        pad_halo(divg, halo=1), cdgrid,
    )
    out_reference = fv3_corner_laplacian_iteration(divg, cdgrid)
    np.testing.assert_array_equal(np.asarray(out_extracted),
                                   np.asarray(out_reference))


def test_pad_h1_step_matches_iteration_uniform():
    """Uniform divg_d → identical output (zero Laplacian on smooth field)."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.full((6, n + 1, n + 1), 2.5)
    out_extracted = fv3_laplacian_step_from_pad_h1(
        pad_halo(divg, halo=1), cdgrid,
    )
    out_reference = fv3_corner_laplacian_iteration(divg, cdgrid)
    np.testing.assert_array_equal(np.asarray(out_extracted),
                                   np.asarray(out_reference))


def test_output_shape_n_plus_1_squared():
    """Output shape: (6, n+1, n+1) for canonical corner-staggered field."""
    for n in (6, 8, 12):
        cdgrid = _cdgrid(n)
        divg = jnp.zeros((6, n + 1, n + 1))
        out = fv3_laplacian_step_from_pad_h1(
            pad_halo(divg, halo=1), cdgrid,
        )
        assert out.shape == (6, n + 1, n + 1)


def test_chained_two_steps_matches_nord_2():
    """pad once + extract twice should match the existing nord=2 path
    (which re-pads between iterations) up to JAX trace-reorder ULPs.

    This is the foundation for expanding-halo: future iter-898+ work
    will replace the re-pad pattern with a single wider pad + multiple
    extracted steps, eliminating cross-iteration corner-fill drift.
    """
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8971)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))

    # Method A: existing fv3_corner_laplacian_iteration ×2 (re-pad each).
    out_a = fv3_corner_laplacian_iteration(divg, cdgrid)
    out_a = fv3_corner_laplacian_iteration(out_a, cdgrid)

    # Method B: extracted helper ×2 (re-pad each, separated padding).
    out_b = fv3_laplacian_step_from_pad_h1(pad_halo(divg, halo=1), cdgrid)
    out_b = fv3_laplacian_step_from_pad_h1(pad_halo(out_b, halo=1), cdgrid)

    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))
