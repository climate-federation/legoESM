"""FV3_3D iter 840: radiative_forcing_n2o_fv3.

ΔF_N₂O = 0.12·(√N − √N_ref).

Tests
-----

1. ``test_present_day_ar6``: N=336, N_ref=270 → ΔF ≈ 0.23 W/m².
2. ``test_no_change_zero``: N = N_ref → ΔF = 0.
3. ``test_decrease_cools``: N < N_ref → ΔF < 0.
4. ``test_monotone``: ↑N → ↑ΔF.
5. ``test_all_three_ghg_chain``: full CO₂+CH₄+N₂O → ECS via
   iter-838 + iter-839 + iter-840 + iter-836.
6. ``test_n2o_alpha_stronger_than_ch4``: same Δ√, λ_N₂O > λ_CH₄.
7. ``test_floor_no_nan``.
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
    radiative_forcing_n2o_fv3,
)


def test_present_day_ar6():
    """N=336 (2024-ish), N_ref=270 → ΔF ≈ 0.23 W/m²."""
    n = jnp.array([336.0])
    f = radiative_forcing_n2o_fv3(n)
    expected = 0.12 * (float(jnp.sqrt(336.0)) - float(jnp.sqrt(270.0)))
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)
    assert 0.2 < float(f[0]) < 0.26


def test_no_change_zero():
    """N = N_ref → ΔF = 0."""
    n = jnp.array([270.0])
    f = radiative_forcing_n2o_fv3(n)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-13)


def test_decrease_cools():
    """N < N_ref → ΔF < 0."""
    n = jnp.array([200.0])
    f = radiative_forcing_n2o_fv3(n)
    assert float(f[0]) < 0.0


def test_monotone():
    """↑N → ↑ΔF."""
    n = jnp.array([280.0, 320.0, 360.0, 400.0])
    f = radiative_forcing_n2o_fv3(n)
    diffs = jnp.diff(f)
    assert jnp.all(diffs > 0.0)


def test_all_three_ghg_chain():
    """Full CO₂+CH₄+N₂O → ECS via iter-838+839+840+836."""
    f_co2 = radiative_forcing_co2_fv3(jnp.array([420.0]))    # ~2.21
    f_ch4 = radiative_forcing_ch4_fv3(jnp.array([1925.0]))   # ~0.61
    f_n2o = radiative_forcing_n2o_fv3(jnp.array([336.0]))    # ~0.23
    f_total = f_co2 + f_ch4 + f_n2o                           # ~3.05
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f_total, lam)
    # 3.05 / 1.4 ≈ 2.18
    assert 2.0 < float(ecs[0]) < 2.3


def test_n2o_alpha_stronger_than_ch4():
    """Same Δ√ argument, N₂O coefficient > CH₄ → larger ΔF."""
    # At same Δ√ form, α_N₂O=0.12 vs α_CH₄=0.036
    n = jnp.array([400.0])  # √400 − √270 = 20 − 16.43 ≈ 3.57
    m = jnp.array([400.0])  # but pre-ind ref different — use same N value
    # Compare per Δ-arg: identical inputs, ratio of forcings = 0.12/0.036 = 3.33
    f_n2o_test = radiative_forcing_n2o_fv3(n, n2o_ppb_ref=270.0)
    f_ch4_test = radiative_forcing_ch4_fv3(m, ch4_ppb_ref=270.0)
    ratio = float(f_n2o_test[0] / f_ch4_test[0])
    np.testing.assert_allclose(ratio, 0.12 / 0.036, rtol=1e-12)


def test_floor_no_nan():
    """N=0 → finite via floor (no NaN)."""
    n = jnp.array([0.0])
    f = radiative_forcing_n2o_fv3(n)
    assert jnp.all(jnp.isfinite(f))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=840)
    n_x, n_y = 6, 8
    n = jnp.asarray(rng.uniform(200.0, 500.0, size=(n_x, n_y)))
    f = radiative_forcing_n2o_fv3(n)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
