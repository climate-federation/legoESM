"""FV3_3D iter 310: ``pad_halo`` (scalar) preserves constants
and is linear.

The cubed-sphere halo padding for scalar fields is the
foundation of every FV3 helper that needs cross-panel
neighbours (interpolation, divergence, Laplacian, etc.).  The
operation is purely a copy + axis swap/reversal — no weighted
combination — so:

    * pad_halo(c-field) is a constant c-field at every padded
      index.
    * pad_halo(α f1 + β f2) = α pad_halo(f1) + β pad_halo(f2).

Pinning these foundational invariants ensures that EVERY
downstream helper that calls pad_halo (a2b_ord4, divergence,
laplacian, _interp_center_to_corner, etc.) inherits correct
constant-preservation and linearity.

Tests
-----

1. ``test_pad_halo_preserves_constant`` — constant scalar
   field stays constant after padding (rtol=1e-14).
2. ``test_pad_halo_linearity`` — linear combinations
   distribute through padding (rtol=1e-14).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.halo import pad_halo


def _setup(seed=310):
    n = 8
    rng = np.random.default_rng(seed=seed)
    f1 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    f2 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    return n, f1, f2


@pytest.mark.parametrize("c", [-3.5, 0.0, 1.0, 1e-6, 1e6])
@pytest.mark.parametrize("halo", [1, 2])
def test_pad_halo_preserves_constant(c, halo):
    """Scalar pad_halo preserves constants at every padded
    index (interior + halo cells from neighbour panels)."""
    n, _, _ = _setup()
    field = jnp.full((6, n, n), c, dtype=jnp.float64)
    padded = pad_halo(field, halo=halo)
    assert padded.shape == (6, n + 2 * halo, n + 2 * halo)
    expected = jnp.full(
        (6, n + 2 * halo, n + 2 * halo), c, dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(expected),
        rtol=1e-14, atol=1e-14,
        err_msg=(
            f"pad_halo(constant {c}, halo={halo}) must remain "
            f"constant at every padded index — neighbours are "
            f"the same constant on every panel."
        ),
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
@pytest.mark.parametrize("halo", [1, 2])
def test_pad_halo_linearity(ab, halo):
    """``pad_halo(α f1 + β f2) = α pad_halo(f1) + β pad_halo(f2)``."""
    alpha, beta = ab
    n, f1, f2 = _setup()

    lhs = pad_halo(alpha * f1 + beta * f2, halo=halo)
    rhs = alpha * pad_halo(f1, halo=halo) + beta * pad_halo(f2, halo=halo)

    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs),
        rtol=1e-14, atol=1e-14,
    )
