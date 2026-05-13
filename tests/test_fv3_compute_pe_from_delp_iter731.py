"""FV3_3D iter 731: compute_pe_from_delp_fv3 port.

Faithful JAX port of FV3's pe accumulation pattern used in IC
ingestion and vertical-remap init paths.

Tests
-----

1. ``test_pe_uniform_delp``.
2. ``test_pe_first_matches_p_top``.
3. ``test_pe_last_matches_total``.
4. ``test_pe_monotone_increasing``.
5. ``test_pe_composes_with_hybrid_inverse``.
6. ``test_pe_shapes_3d``.
7. ``test_pe_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    compute_hybrid_pressure_fv3,
    compute_pe_from_delp_fv3,
)


def test_pe_uniform_delp():
    """Uniform delp=10000 Pa, km=5 → pe = [p_top, p_top+1e4, ..., p_top+5e4]."""
    delp = jnp.full((5,), 1.0e4)
    pe = compute_pe_from_delp_fv3(delp, p_top=1.0e3)
    expected = jnp.array([1.0e3, 1.1e4, 2.1e4, 3.1e4, 4.1e4, 5.1e4])
    assert jnp.allclose(pe, expected, atol=1e-10)


def test_pe_first_matches_p_top():
    """pe[0] = p_top exactly."""
    delp = jnp.linspace(100.0, 1000.0, 10)
    pe = compute_pe_from_delp_fv3(delp, p_top=500.0)
    assert abs(float(pe[0]) - 500.0) < 1e-12


def test_pe_last_matches_total():
    """pe[km] = p_top + sum(delp)."""
    rng = np.random.default_rng(seed=731)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(20,)))
    p_top = 100.0
    pe = compute_pe_from_delp_fv3(delp, p_top=p_top)
    total = p_top + float(jnp.sum(delp))
    assert abs(float(pe[-1]) - total) < 1e-8


def test_pe_monotone_increasing():
    """pe monotone (positive delp guarantees this)."""
    rng = np.random.default_rng(seed=732)
    delp = jnp.asarray(rng.uniform(100.0, 2000.0, size=(30,)))
    pe = compute_pe_from_delp_fv3(delp, p_top=10.0)
    assert jnp.all(jnp.diff(pe) > 0.0)


def test_pe_composes_with_hybrid_inverse():
    """iter-724 hybrid produces (delp, pe).  Using delp here recovers pe."""
    km = 10
    ak = jnp.linspace(100.0, 0.0, km + 1)
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps = jnp.asarray(1.0e5)
    delp_hybrid, pe_hybrid, _ = compute_hybrid_pressure_fv3(ak, bk, ps)
    pe_via_pe_helper = compute_pe_from_delp_fv3(
        delp_hybrid, p_top=float(pe_hybrid[0]),
    )
    assert jnp.allclose(pe_via_pe_helper, pe_hybrid, atol=1e-8)


def test_pe_shapes_3d():
    """3-D (n_x, n_y, km) → (n_x, n_y, km+1)."""
    rng = np.random.default_rng(seed=733)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    pe = compute_pe_from_delp_fv3(delp, p_top=100.0)
    assert pe.shape == (n_x, n_y, km + 1)


def test_pe_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=734)
    delp = jnp.asarray(rng.uniform(100.0, 3000.0, size=(4, 4, 30)))
    pe = compute_pe_from_delp_fv3(delp, p_top=50.0)
    assert jnp.all(jnp.isfinite(pe))
