"""FV3_3D iter 292: linear-in-c_s scaling for the iter-180
Smagorinsky-adaptive A_h helper.

The legoESM Smagorinsky form (``_smagorinsky_visc.py:6``) is:

    A_h_smag(i, j) = c_s * dx² * sqrt(D11² + 2*D12² + D22²)

This is LINEAR in c_s (legoESM compresses the canonical FV3
``c_s² * dx² * |D|`` form by treating the user-supplied c_s as
the squared coefficient).  iter-180 has off/on/no-A_h tests but
no scaling test; iter-292 pins this linear-in-c_s contract by
direct numerical regression on the helper.

Tests
-----

1. ``test_smagorinsky_ah_linear_in_cs`` — at fixed (u, v),
   compute A_h_smag at c_s=0.10 and c_s=0.20, verify ratio is
   exactly 2.0 (rtol=1e-12).
2. ``test_smagorinsky_ah_zero_when_cs_zero`` — c_s=0 returns
   exactly zero (already pinned by iter-180 but explicit here).
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
    rng = np.random.default_rng(seed=292)
    u = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    return cdgrid, u, v


def test_smagorinsky_ah_linear_in_cs():
    """A_h_smag must be linear in c_s — 2x c_s → 2x A_h_smag."""
    cdgrid, u, v = _setup()
    cs1 = 0.10
    cs2 = 0.20

    ah1 = compute_smagorinsky_ah_2d(u, v, cdgrid, cs1)
    ah2 = compute_smagorinsky_ah_2d(u, v, cdgrid, cs2)

    # Where ah1 > 0, ratio must be exactly 2.0.  At zero strain,
    # both are 0 and ratio is undefined — mask out.
    mask = ah1 > 0.0
    assert jnp.any(mask), (
        "expected at least some non-zero A_h_smag from random "
        "(u, v) input"
    )
    ratio = ah2[mask] / ah1[mask]
    np.testing.assert_allclose(
        np.asarray(ratio), 2.0, rtol=1e-12,
        err_msg=(
            "FV3-faithful Smagorinsky form A_h_smag = c_s * dx² *"
            " |D| must scale LINEARLY with c_s.  Doubling c_s "
            "should exactly double A_h_smag everywhere |D|>0."
        ),
    )

    # And bit-for-bit: ah2 == 2 * ah1 (since the only c_s-
    # dependence is the prefactor multiply).
    np.testing.assert_allclose(
        np.asarray(ah2), 2.0 * np.asarray(ah1), rtol=1e-14,
    )


def test_smagorinsky_ah_zero_when_cs_zero():
    """c_s=0 → A_h_smag = 0 everywhere."""
    cdgrid, u, v = _setup()
    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, 0.0)
    assert jnp.all(ah == 0.0), (
        "c_s=0 must zero-out the Smagorinsky contribution "
        "(early-return branch in compute_smagorinsky_ah_2d)."
    )
