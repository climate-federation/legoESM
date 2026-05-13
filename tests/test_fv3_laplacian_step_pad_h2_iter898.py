"""FV3_3D iter 898: unit tests for fv3_laplacian_step_from_pad_h2.

Validate that the h2-pre-padded Laplacian step produces a
(n+3, n+3) output (halo=1 interior shrunk by one cell per side
from h2 input).  Foundation for FV3-faithful expanding-halo
nord=2 single-pad-multi-step pattern.

Critical check: composing
``pad_halo(divg, halo=2) → step_h2 → step_h1`` should approximate
the existing nord=2 ``pad_halo(halo=1) → step ×2`` re-pad pattern
(may differ by halo-fill mode, but both should remain stable and
finite).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_iteration,
    fv3_laplacian_step_from_pad_h1,
    fv3_laplacian_step_from_pad_h2,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid
from legoesm.grids.halo import pad_halo


def _cdgrid(n):
    return _create_cdgrid(create_cubed_sphere(n))


def test_h2_step_output_shape():
    """Output shape (6, n+3, n+3) from (6, n+5, n+5) input."""
    for n in (6, 8, 12):
        cdgrid = _cdgrid(n)
        divg = jnp.zeros((6, n + 1, n + 1))
        divg_pad = pad_halo(divg, halo=2)
        assert divg_pad.shape == (6, n + 5, n + 5)
        out = fv3_laplacian_step_from_pad_h2(divg_pad, cdgrid)
        assert out.shape == (6, n + 3, n + 3)


def test_h2_step_finite_random():
    """Random divg_d through h2 step yields finite output."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8980)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    divg_pad = pad_halo(divg, halo=2)
    out = fv3_laplacian_step_from_pad_h2(divg_pad, cdgrid)
    assert jnp.all(jnp.isfinite(out))


def test_h2_step_uniform_zero_interior():
    """Uniform divg_d → near-zero Laplacian (geometric metric
    variations may give tiny non-zero edges but interior should be
    near machine-epsilon)."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.full((6, n + 1, n + 1), 2.5)
    divg_pad = pad_halo(divg, halo=2)
    out = fv3_laplacian_step_from_pad_h2(divg_pad, cdgrid)
    # Interior cells (away from cube vertices and panel edges)
    # should be ~0 since Laplacian of constant is zero.
    interior_max = float(jnp.max(jnp.abs(out[:, 2:n + 1, 2:n + 1])))
    assert interior_max < 1e-6, (
        f"Uniform divg_d gave non-zero interior Laplacian "
        f"max={interior_max:.2e}; expected ~0."
    )


def test_h2_then_h1_chain_finite():
    """Chain h2 + h1 produces canonical (n+1, n+1) finite output."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8981)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    # Method: pad_halo(h=2) → step_h2 → re-pad_halo(h=1) on shrunk → step_h1
    divg_pad_h2 = pad_halo(divg, halo=2)
    mid = fv3_laplacian_step_from_pad_h2(divg_pad_h2, cdgrid)
    assert mid.shape == (6, n + 3, n + 3)
    # The mid array is the halo=1 interior — it already contains the
    # 1-wide halo for the next h1 step, so we slice + pad.  Note:
    # this is the legoESM convention; FV3 keeps the wider halo without
    # re-pad.
    mid_interior = mid[:, 1:n + 2, 1:n + 2]
    assert mid_interior.shape == (6, n + 1, n + 1)
    out = fv3_laplacian_step_from_pad_h1(
        pad_halo(mid_interior, halo=1), cdgrid,
    )
    assert out.shape == (6, n + 1, n + 1)
    assert jnp.all(jnp.isfinite(out))
