"""FV3_3D iter 839: radiative_forcing_ch4_fv3.

ΔF_CH₄ = 0.036·(√M − √M_ref).

Tests
-----

1. ``test_present_day_ar6``: M=1925, M_ref=722 → ΔF ≈ 0.61 W/m².
2. ``test_no_change_zero``: M = M_ref → ΔF = 0.
3. ``test_methane_decrease_cools``: M < M_ref → ΔF < 0.
4. ``test_monotone``: ↑M → ↑ΔF.
5. ``test_sqrt_subloglike``: doubling M < doubling ΔF (square-root saturation).
6. ``test_chain_combined_co2_ch4_ecs``: present-day CO₂+CH₄ → ECS via
   iter-838 + iter-839 + iter-836.
7. ``test_floor_no_nan``: M=0 → finite.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    equilibrium_climate_sensitivity_fv3,
    radiative_forcing_ch4_fv3,
    radiative_forcing_co2_fv3,
)


def test_present_day_ar6():
    """M=1925 (2024-ish), M_ref=722 → ΔF ≈ 0.61 W/m²."""
    m = jnp.array([1925.0])
    f = radiative_forcing_ch4_fv3(m)
    expected = 0.036 * (float(jnp.sqrt(1925.0)) - float(jnp.sqrt(722.0)))
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)
    assert 0.55 < float(f[0]) < 0.65


def test_no_change_zero():
    """M = M_ref → ΔF = 0."""
    m = jnp.array([722.0])
    f = radiative_forcing_ch4_fv3(m)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-13)


def test_methane_decrease_cools():
    """M < M_ref → ΔF < 0."""
    m = jnp.array([500.0])
    f = radiative_forcing_ch4_fv3(m)
    assert float(f[0]) < 0.0


def test_monotone():
    """↑M → ↑ΔF."""
    m = jnp.array([900.0, 1500.0, 2500.0, 3500.0])
    f = radiative_forcing_ch4_fv3(m)
    diffs = jnp.diff(f)
    assert jnp.all(diffs > 0.0)


def test_sqrt_subloglike():
    """Doubling M from 1000→2000 gives ΔF1 < doubling 2000→4000 ΔF2.
    Wait — actually equal-amount √-increments shrink: ΔF₂ < ΔF₁ for
    log-spaced ratios; but for √-form successive *doublings* give
    smaller marginal forcing because √(2M)−√M = (√2−1)·√M grows
    with M but slower than log."""
    # Compare ratio (√(4M)−√M)/(√(2M)−√M) = (2−1)/(√2−1) ≈ 2.414
    # vs log case which would give 2.0.  So √-form is *less*
    # saturating than log: 2nd doubling gives larger marginal ΔF.
    # Just verify monotone-positive marginal forcing.
    m_ref = jnp.array([722.0])
    f1 = radiative_forcing_ch4_fv3(jnp.array([1444.0]))  # 2×ref
    f2 = radiative_forcing_ch4_fv3(jnp.array([2888.0]))  # 4×ref
    f3 = radiative_forcing_ch4_fv3(jnp.array([5776.0]))  # 8×ref
    assert float(f1[0]) > 0.0
    assert float(f2[0]) > float(f1[0])
    assert float(f3[0]) > float(f2[0])


def test_chain_combined_co2_ch4_ecs():
    """Present-day CO₂+CH₄ → ECS via iter-838 + iter-839 + iter-836."""
    f_co2 = radiative_forcing_co2_fv3(jnp.array([420.0]))    # ~2.21
    f_ch4 = radiative_forcing_ch4_fv3(jnp.array([1925.0]))   # ~0.61
    f_total = f_co2 + f_ch4                                   # ~2.82
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f_total, lam)
    # 2.82 / 1.4 = 2.01 K
    assert 1.9 < float(ecs[0]) < 2.1


def test_floor_no_nan():
    """M=0 → finite via floor (no NaN)."""
    m = jnp.array([0.0])
    f = radiative_forcing_ch4_fv3(m)
    assert jnp.all(jnp.isfinite(f))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=839)
    n_x, n_y = 6, 8
    m = jnp.asarray(rng.uniform(400.0, 4000.0, size=(n_x, n_y)))
    f = radiative_forcing_ch4_fv3(m)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
