"""FV3_3D iter 892: unit tests for ``fv3_corner_laplacian_nord``.

The new wrapper extracts the nord-fold Laplacian-iteration loop
from inline PE/NH model code into a single named function.
Bit-for-bit equivalent to the inline loops; foundation for
iter-893's expanding-halo refactor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_iteration,
    fv3_corner_laplacian_nord,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid


def create_cubed_sphere_cdgrid(n):
    """Local helper: build CDGrid from n."""
    return _create_cdgrid(create_cubed_sphere(n))


def test_nord_0_passthrough():
    """nord=0 returns input unchanged (degenerate case)."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8920)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out = fv3_corner_laplacian_nord(divg, cdgrid, nord=0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(divg))


def test_nord_1_matches_single_iteration():
    """nord=1 bit-for-bit identical to single fv3_corner_laplacian_iteration."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8921)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_wrapper = fv3_corner_laplacian_nord(divg, cdgrid, nord=1)
    out_direct = fv3_corner_laplacian_iteration(divg, cdgrid)
    np.testing.assert_array_equal(np.asarray(out_wrapper),
                                   np.asarray(out_direct))


def test_nord_2_matches_inline_loop():
    """nord=2 bit-for-bit identical to inline ``for _ in range(2)`` loop."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8922)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_wrapper = fv3_corner_laplacian_nord(divg, cdgrid, nord=2)
    out_inline = divg
    for _ in range(2):
        out_inline = fv3_corner_laplacian_iteration(out_inline, cdgrid)
    np.testing.assert_array_equal(np.asarray(out_wrapper),
                                   np.asarray(out_inline))


def test_nord_3_matches_inline_loop():
    """nord=3 bit-for-bit identical to inline 3-iteration loop."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8923)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_wrapper = fv3_corner_laplacian_nord(divg, cdgrid, nord=3)
    out_inline = divg
    for _ in range(3):
        out_inline = fv3_corner_laplacian_iteration(out_inline, cdgrid)
    np.testing.assert_array_equal(np.asarray(out_wrapper),
                                   np.asarray(out_inline))


def test_vector_corner_fill_flag_propagated():
    """apply_vector_corner_fill=True propagates through every iteration."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8924)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_wrapper = fv3_corner_laplacian_nord(
        divg, cdgrid, nord=2, apply_vector_corner_fill=True,
    )
    out_inline = divg
    for _ in range(2):
        out_inline = fv3_corner_laplacian_iteration(
            out_inline, cdgrid, apply_vector_corner_fill=True,
        )
    np.testing.assert_array_equal(np.asarray(out_wrapper),
                                   np.asarray(out_inline))


def test_nord_monotone_change():
    """Successive nord values produce different outputs (each iteration
    contributes; Laplacian is not idempotent on smooth fields)."""
    n = 8
    cdgrid = create_cubed_sphere_cdgrid(n)
    rng = np.random.default_rng(seed=8925)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_1 = fv3_corner_laplacian_nord(divg, cdgrid, nord=1)
    out_2 = fv3_corner_laplacian_nord(divg, cdgrid, nord=2)
    out_3 = fv3_corner_laplacian_nord(divg, cdgrid, nord=3)
    # nord=2 differs from nord=1 (extra Laplacian iteration on
    # high-amplitude random field).
    assert float(jnp.max(jnp.abs(out_2 - out_1))) > 1e-12
    # nord=3 differs from nord=2 (Laplacian iteration is contracting
    # on smooth limits; for nord>=3 on random data the difference can
    # be small but should not be exactly zero — distinguishes silent
    # truncation from numerical contraction).
    diff_2_3 = float(jnp.max(jnp.abs(out_3 - out_2)))
    assert diff_2_3 > 0.0, (
        "nord=3 bit-for-bit identical to nord=2 — Laplacian iteration "
        "may be silently truncated at 2."
    )
