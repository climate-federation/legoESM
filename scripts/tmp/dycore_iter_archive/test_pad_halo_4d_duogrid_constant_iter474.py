"""FV3_3D iter 474: isolation test — does ``pad_halo_4d`` with
duogrid preserve a CONSTANT field at cube edges?

A perfect halo exchange MUST preserve a uniform constant
field (no spatial variation should appear in halo cells).
Any deviation from constancy in the halo region = numerical
artifact introduced by the halo operator.

This isolates iter-473's "duogrid increases edge std 4.77×"
finding from the dycore: if pad_halo_4d with duogrid produces
non-uniform halo cells on a constant input, the bug is in
the halo operator itself, not in the dycore-halo interaction.

Tests
-----

1. ``test_pad_halo_4d_no_duogrid_preserves_constant`` —
   baseline: pad_halo_4d without duogrid should preserve
   constant (interp_offsets path).
2. ``test_pad_halo_4d_duogrid_preserves_constant`` — the
   actual diagnostic.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d


def test_pad_halo_4d_no_duogrid_preserves_constant():
    """Constant scalar field → pad_halo_4d (no duogrid) gives
    constant halo too."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    const_value = 273.15
    field = jnp.full((6, n, n, nlev), const_value)
    padded = pad_halo_4d(
        field,
        halo=1,
        interp_offsets=grid.halo_interp_offsets,
        duogrid=None,
    )
    arr = np.asarray(padded)
    np.testing.assert_allclose(
        arr, const_value,
        rtol=1e-13, atol=1e-13,
        err_msg="pad_halo_4d (no duogrid) introduced non-"
                "constant values into halo of a constant input.",
    )


def test_pad_halo_4d_duogrid_preserves_constant():
    """Constant scalar field → pad_halo_4d (with duogrid)
    should also give constant halo.

    iter-474 diagnostic: if this fails, the duogrid halo
    operator itself introduces artifacts that explain the
    iter-473 edge-std 4.77× increase.
    """
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    const_value = 273.15
    field = jnp.full((6, n, n, nlev), const_value)
    padded = pad_halo_4d(
        field,
        halo=1,
        interp_offsets=None,
        duogrid=grid.duogrid,
    )
    arr = np.asarray(padded)
    # First check: just look at how non-constant the halo is.
    halo_max_dev = float(np.max(np.abs(arr - const_value)))
    print(
        f"\n[iter-474 pad_halo_4d duogrid constant test]"
        f"\n  max deviation in halo from {const_value}: "
        f"{halo_max_dev:.4e}"
    )
    np.testing.assert_allclose(
        arr, const_value,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            f"DUOGRID pad_halo_4d introduces "
            f"max deviation {halo_max_dev:.4e} in halo of "
            f"constant input — IMPLEMENTATION BUG! Halo "
            f"should preserve constants exactly."
        ),
    )
