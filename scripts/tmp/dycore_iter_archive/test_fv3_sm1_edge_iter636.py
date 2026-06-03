"""FV3_3D iter 636: sm1_edge_fv3 port.

Faithful JAX port of FV3 ``sm1_edge`` (tools/fv_eta.F90:2249-2284).
1D del-2 edge smoother on layer-interface heights.

Tests
-----

1. ``test_sm1_edge_shape``.
2. ``test_sm1_edge_ntimes_zero_noop``.
3. ``test_sm1_edge_bottom_preserved``.
4. ``test_sm1_edge_total_thickness_conserved``.
5. ``test_sm1_edge_smooths_oscillation``.
6. ``test_sm1_edge_uniform_unchanged``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import sm1_edge_fv3


def test_sm1_edge_shape():
    """Output shape matches input (km+1,)."""
    km = 20
    ze = jnp.linspace(50000.0, 0.0, km + 1)  # top→bottom
    out = sm1_edge_fv3(ze, ntimes=3)
    assert out.shape == (km + 1,)


def test_sm1_edge_ntimes_zero_noop():
    """ntimes=0 → no smoothing pass; output = input."""
    km = 16
    ze = jnp.linspace(40000.0, 0.0, km + 1)
    out = sm1_edge_fv3(ze, ntimes=0)
    assert jnp.allclose(out, ze, atol=1e-10)


def test_sm1_edge_bottom_preserved():
    """Bottom interface ze[km] unchanged after smoothing."""
    km = 20
    ze = jnp.asarray(np.random.default_rng(seed=636).uniform(
        500.0, 50000.0, size=km + 1
    ))
    ze = jnp.sort(ze)[::-1]              # top→bottom descending
    out = sm1_edge_fv3(ze, ntimes=4)
    assert abs(float(out[km]) - float(ze[km])) < 1e-10


def test_sm1_edge_total_thickness_conserved():
    """Sum of dz preserved (flux-form conservation)."""
    km = 20
    rng = np.random.default_rng(seed=637)
    # Generate a noisy descending profile
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 2000.0, size=km)))
    ze = jnp.concatenate([jnp.asarray([z[-1] + 100.0]), z[::-1]])
    # ze now (km+1,) descending top→bottom
    total_before = float(ze[0] - ze[km])
    out = sm1_edge_fv3(ze, ntimes=5)
    total_after = float(out[0] - out[km])
    assert abs(total_before - total_after) / abs(total_before) < 1e-10, (
        f"total before={total_before}, after={total_after}"
    )


def test_sm1_edge_smooths_oscillation():
    """Oscillating dz → smoother dz after sm1_edge."""
    km = 20
    # Build oscillating layer thicknesses
    dz = 1000.0 + 500.0 * jnp.asarray([(-1) ** k for k in range(km)])
    # ze[km] = 0, ze[k] = Σ_{j>=k} dz[j]
    cumsum_rev = jnp.cumsum(dz[::-1])[::-1]
    ze = jnp.concatenate([cumsum_rev, jnp.asarray([0.0])])
    out = sm1_edge_fv3(ze, ntimes=10)
    # Compute output dz
    dz_out = out[1:] - out[:-1]
    # Smoothed dz should have smaller std-dev than input dz
    # Note dz is positive (top-down ze descends, so dz = ze[k+1] - ze[k] is negative).
    # Sign doesn't matter for variance comparison.
    std_before = float(jnp.std(dz))
    std_after = float(jnp.std(dz_out))
    assert std_after < std_before, (
        f"std_before={std_before}, std_after={std_after}"
    )


def test_sm1_edge_uniform_unchanged():
    """Uniform layer thicknesses → no change under smoothing."""
    km = 16
    dz = 1000.0
    ze = jnp.asarray([dz * (km - k) for k in range(km + 1)],
                     dtype=jnp.float64)
    out = sm1_edge_fv3(ze, ntimes=5)
    assert jnp.allclose(out, ze, atol=1e-9)
