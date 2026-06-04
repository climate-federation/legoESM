"""FV3_3D iter 303: ``fv3_divergence_corner_3d`` linearity in
(u, v).

The FV3 corner divergence (sw_core.F90:divergence_corner port)
is a LINEAR operator on the (u, v) input pair.  Every term is
a fixed-weight finite difference times a fixed metric (sin_sg
edge factor, cosa cross-correction, area_corner normalization).
Therefore::

    div(α u1 + β u2, α v1 + β v2)
    = α * div(u1, v1) + β * div(u2, v2)

iter-301/302 tested ``a2b_ord4`` interpolation; iter-303 covers
the divergence helper used by the iter-16 (PE)/iter-168 (NH)
corner-div damping path.

Tests
-----

1. ``test_divergence_corner_zero_velocity`` — div(0, 0) = 0
   exactly.
2. ``test_divergence_corner_superposition`` — linear-
   combination identity at multiple (α, β).
3. ``test_divergence_corner_homogeneity`` — div(αu, αv) =
   α*div(u, v).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_divergence_corner_3d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=303):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    u1 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n + 1, nlev)))
    v1 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n + 1, nlev)))
    u2 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n + 1, nlev)))
    v2 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n + 1, nlev)))
    return cdgrid, u1, v1, u2, v2


def test_divergence_corner_zero_velocity():
    """div(0, 0) = 0 exactly — no spurious bias."""
    cdgrid, _, _, _, _ = _setup()
    n = cdgrid.n
    nlev = 5
    z = jnp.zeros((6, n + 1, n + 1, nlev))
    div = fv3_divergence_corner_3d(z, z, cdgrid)
    assert jnp.all(div == 0.0), (
        "fv3_divergence_corner_3d(0, 0) must be exactly 0 — "
        "every term is u or v times a metric, so zero input "
        "gives zero output regardless of metrics."
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_divergence_corner_superposition(ab):
    """``div`` distributes over linear combinations of (u, v)."""
    alpha, beta = ab
    cdgrid, u1, v1, u2, v2 = _setup()

    lhs = fv3_divergence_corner_3d(
        alpha * u1 + beta * u2,
        alpha * v1 + beta * v2,
        cdgrid,
    )
    rhs = (
        alpha * fv3_divergence_corner_3d(u1, v1, cdgrid) +
        beta * fv3_divergence_corner_3d(u2, v2, cdgrid)
    )
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-13, atol=1e-13,
        err_msg=(
            f"fv3_divergence_corner_3d not linear: alpha="
            f"{alpha}, beta={beta} superposition violated."
        ),
    )


@pytest.mark.parametrize("alpha", [-2.5, 0.5, 3.0, 1e-6])
def test_divergence_corner_homogeneity(alpha):
    """div(αu, αv) = α*div(u, v)."""
    cdgrid, u, v, _, _ = _setup()
    lhs = fv3_divergence_corner_3d(alpha * u, alpha * v, cdgrid)
    rhs = alpha * fv3_divergence_corner_3d(u, v, cdgrid)
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-13, atol=1e-13,
    )
