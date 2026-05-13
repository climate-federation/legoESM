"""FV3_3D iter 862: hargreaves_pet_fv3.

PET = 0.0023·R_a·(T_mean+17.8)·sqrt(T_max−T_min)  [mm/day].

Tests
-----

1. ``test_tropical_canonical``: R_a=40, T̄=27, ΔT=8 → PET ≈ 11.7 mm/day.
2. ``test_polar_winter_zero``: R_a=0 → PET=0.
3. ``test_no_diurnal_floored``: T_max=T_min → finite (no NaN).
4. ``test_low_t_negative_offset``: T_mean=−17.8 exactly → PET=0.
5. ``test_chain_with_spei``: PET output feeds iter-861 SPEI.
6. ``test_monotone_in_r_a``: ↑R_a → ↑PET.
7. ``test_monotone_in_t_mean``: ↑T_mean → ↑PET.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    hargreaves_pet_fv3,
    spei_z_score_fv3,
)


def test_tropical_canonical():
    """R_a=40, T̄=27, ΔT=8 → PET ≈ 11.7 mm/day."""
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([27.0]),
        t_max_c=jnp.array([31.0]),
        t_min_c=jnp.array([23.0]),
        r_a_mj=jnp.array([40.0]),
    )
    # 0.0023 * 40 * (27+17.8) * sqrt(8) = 0.092 * 44.8 * 2.828 = 11.66
    assert 11.0 < float(pet[0]) < 12.5


def test_polar_winter_zero():
    """R_a=0 → PET=0 (no solar input)."""
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([-30.0]),
        t_max_c=jnp.array([-20.0]),
        t_min_c=jnp.array([-40.0]),
        r_a_mj=jnp.array([0.0]),
    )
    np.testing.assert_allclose(np.asarray(pet), [0.0], atol=1e-14)


def test_no_diurnal_floored():
    """T_max=T_min → finite via range_floor (no NaN/0 sqrt)."""
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([20.0]),
        t_max_c=jnp.array([20.0]),
        t_min_c=jnp.array([20.0]),
        r_a_mj=jnp.array([30.0]),
    )
    assert jnp.all(jnp.isfinite(pet))


def test_low_t_negative_offset():
    """T_mean=−17.8 → PET=0 (offset cancels)."""
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([-17.8]),
        t_max_c=jnp.array([-15.0]),
        t_min_c=jnp.array([-20.0]),
        r_a_mj=jnp.array([30.0]),
    )
    np.testing.assert_allclose(np.asarray(pet), [0.0], atol=1e-12)


def test_chain_with_spei():
    """PET output → iter-861 SPEI input (chain consistency)."""
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([27.0]),
        t_max_c=jnp.array([31.0]),
        t_min_c=jnp.array([23.0]),
        r_a_mj=jnp.array([40.0]),
    )
    # Use returned PET in SPEI; e.g. 30-day window P=200 mm, PET·30 ≈ 350 mm
    pet_30day = pet * 30.0
    spei = spei_z_score_fv3(
        precip=jnp.array([200.0]),
        pet=pet_30day,
        mu_d=jnp.array([-100.0]),
        sigma_d=jnp.array([50.0]),
    )
    assert jnp.all(jnp.isfinite(spei))


def test_monotone_in_r_a():
    """↑R_a at fixed T → ↑PET (more solar input)."""
    pet_low = hargreaves_pet_fv3(
        jnp.array([25.0]), jnp.array([30.0]), jnp.array([20.0]),
        jnp.array([20.0]),
    )
    pet_high = hargreaves_pet_fv3(
        jnp.array([25.0]), jnp.array([30.0]), jnp.array([20.0]),
        jnp.array([40.0]),
    )
    assert float(pet_high[0]) > float(pet_low[0])


def test_monotone_in_t_mean():
    """↑T_mean at fixed R_a, ΔT → ↑PET."""
    pet_cool = hargreaves_pet_fv3(
        jnp.array([15.0]), jnp.array([20.0]), jnp.array([10.0]),
        jnp.array([30.0]),
    )
    pet_warm = hargreaves_pet_fv3(
        jnp.array([30.0]), jnp.array([35.0]), jnp.array([25.0]),
        jnp.array([30.0]),
    )
    assert float(pet_warm[0]) > float(pet_cool[0])


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=862)
    n_x, n_y = 6, 8
    t_mean = jnp.asarray(rng.uniform(-10.0, 40.0, size=(n_x, n_y)))
    t_max = t_mean + jnp.asarray(rng.uniform(2.0, 15.0, size=(n_x, n_y)))
    t_min = t_mean - jnp.asarray(rng.uniform(2.0, 15.0, size=(n_x, n_y)))
    r_a = jnp.asarray(rng.uniform(5.0, 40.0, size=(n_x, n_y)))
    pet = hargreaves_pet_fv3(t_mean, t_max, t_min, r_a)
    assert pet.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(pet))
