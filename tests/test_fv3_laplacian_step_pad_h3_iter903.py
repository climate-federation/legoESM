"""FV3_3D iter 903: unit tests for fv3_laplacian_step_from_pad_h3.

Extends iter-897 h1 + iter-898 h2 step coverage with the h3 step
needed for nord=3 expanding-halo chain (h3 → h2 → h1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_nord,
    fv3_corner_laplacian_nord_expanding_halo,
    fv3_laplacian_step_from_pad_h3,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid
from legoesm.grids.halo import pad_halo


def _cdgrid(n):
    return _create_cdgrid(create_cubed_sphere(n))


def test_h3_step_shape():
    """h3 step consumes (6, n+7, n+7), returns (6, n+5, n+5)."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9030)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    divg_pad = pad_halo(divg, halo=3)
    assert divg_pad.shape == (6, n + 7, n + 7)
    out = fv3_laplacian_step_from_pad_h3(divg_pad, cdgrid)
    assert out.shape == (6, n + 5, n + 5)


def test_h3_step_finite():
    """h3 step output is finite for random input."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9031)
    divg = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n + 1, n + 1)))
    divg_pad = pad_halo(divg, halo=3)
    out = fv3_laplacian_step_from_pad_h3(divg_pad, cdgrid)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_h3_step_uniform_zero_lap():
    """Uniform divg_d → near-zero Laplacian at interior of h3 step."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.full((6, n + 1, n + 1), 1.0)
    divg_pad = pad_halo(divg, halo=3)
    out = fv3_laplacian_step_from_pad_h3(divg_pad, cdgrid)
    # interior of write region (away from edges + cube vertex adjusts)
    # should be ~0 since Laplacian of constant is 0
    interior_max = float(jnp.max(jnp.abs(out[:, 4:n + 1, 4:n + 1])))
    assert interior_max < 1e-6


def test_h3_step_grid_size_robust():
    """h3 step shape correct across n ∈ {4, 8, 16}."""
    for n in (4, 8, 16):
        cdgrid = _cdgrid(n)
        rng = np.random.default_rng(seed=9032 + n)
        divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
        divg_pad = pad_halo(divg, halo=3)
        out = fv3_laplacian_step_from_pad_h3(divg_pad, cdgrid)
        assert out.shape == (6, n + 5, n + 5), f"n={n}"


def test_nord_3_chain_finite():
    """fv3_corner_laplacian_nord_expanding_halo nord=3 returns finite."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9033)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=3)
    assert out.shape == (6, n + 1, n + 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_nord_3_approximates_repad():
    """nord=3 expanding-halo matches re-pad path within ULP tolerance.

    Same finding as iter-901 for nord=2: re-pad path numerically
    equivalent (max abs diff < 1e-20) to FV3 single-pad convention.
    Pinning here so future refactors that drift surface immediately.
    """
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9034)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_expand = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=3)
    out_repad = fv3_corner_laplacian_nord(divg, cdgrid, nord=3)
    max_diff = float(jnp.max(jnp.abs(out_expand - out_repad)))
    assert max_diff < 1e-20, (
        f"expand vs repad nord=3 max_diff={max_diff:.2e} "
        f"(expected < 1e-20).  Drift in h3/h2/h1 step or re-pad wrapper."
    )


def test_nord_4_still_raises():
    """nord ≥ 4 outside FV3 namelist range — still NotImplementedError."""
    import pytest
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.zeros((6, n + 1, n + 1))
    with pytest.raises(NotImplementedError, match=r"nord=4"):
        fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=4)
