"""FV3_3D iter 294: translation (Galilean) invariance for the
Smagorinsky strain-magnitude helper.

The Smagorinsky form ``A_h_smag = c_s * dx² * |D|`` depends on
the **gradient** of (u, v).  Adding a constant offset to u and
v must therefore leave the strain magnitude unchanged::

    |D|(u + c1, v + c2) ≡ |D|(u, v)
    A_h_smag(u + c1, v + c2) ≡ A_h_smag(u, v)

This is a basic Galilean invariance property of any strain-
based viscosity.  The cubed-sphere ``pad_halo`` cross-panel
exchange is consistent (the constant is added on all 6 panels
including halos), so the centered finite differences cancel
the constant exactly — bit-for-bit invariance is achievable.

Tests
-----

1. ``test_smagorinsky_translation_invariant_uniform`` — adding
   the same constant to u and v leaves A_h_smag unchanged
   (rtol=1e-14).
2. ``test_smagorinsky_translation_invariant_separate`` —
   adding different constants c1 to u and c2 to v leaves
   A_h_smag unchanged.

iter-292 pinned linear-in-c_s; iter-293 pinned linear-in-(u, v)
magnitude; iter-294 pins translation invariance — third
orthogonal axis of helper-formula coverage.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._smagorinsky_visc import compute_smagorinsky_ah_2d
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=294)
    u = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    return cdgrid, u, v


@pytest.mark.parametrize("c", [-5.0, 1.0, 100.0])
def test_smagorinsky_translation_invariant_uniform(c):
    """A_h_smag(u + c, v + c) = A_h_smag(u, v) bit-for-bit.

    Adding a uniform Galilean offset to the velocity field
    must not change the strain magnitude.
    """
    cdgrid, u, v = _setup()
    cs = 0.20

    ah_orig = compute_smagorinsky_ah_2d(u, v, cdgrid, cs)
    ah_shift = compute_smagorinsky_ah_2d(u + c, v + c, cdgrid, cs)

    np.testing.assert_allclose(
        np.asarray(ah_shift), np.asarray(ah_orig), rtol=1e-14,
        err_msg=(
            f"Galilean translation by c={c} must leave A_h_smag "
            f"invariant.  If this fails, the helper has spurious "
            f"absolute-velocity dependence."
        ),
    )


@pytest.mark.parametrize("c1,c2", [(-3.0, 7.0), (10.0, -2.0)])
def test_smagorinsky_translation_invariant_separate(c1, c2):
    """A_h_smag(u + c1, v + c2) = A_h_smag(u, v) bit-for-bit.

    Independent constant offsets to each component still leave
    strain magnitude invariant (∂c1/∂x = 0, ∂c2/∂y = 0).
    """
    cdgrid, u, v = _setup()
    cs = 0.20

    ah_orig = compute_smagorinsky_ah_2d(u, v, cdgrid, cs)
    ah_shift = compute_smagorinsky_ah_2d(u + c1, v + c2, cdgrid, cs)

    np.testing.assert_allclose(
        np.asarray(ah_shift), np.asarray(ah_orig), rtol=1e-14,
        err_msg=(
            f"Translation by (c1={c1}, c2={c2}) must leave "
            f"A_h_smag invariant — separate-component shifts "
            f"are still constant-gradient = 0."
        ),
    )
