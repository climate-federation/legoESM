"""Per-bin ice terminal velocities (computed; oracle VR2..VR5 tables)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics.fast_sbm import mass_doubling_grid
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.ice_fall_speed import (
    ice_fall_speed,
)

jax.config.update("jax_enable_x64", True)


def test_monotone_increasing_in_mass():
    m = mass_doubling_grid()
    for cat in ("snow", "graupel"):
        v = np.asarray(ice_fall_speed(m, jnp.asarray(1.0), cat))
        assert np.all(np.diff(v) > 0.0)
        assert np.all(v > 0.0)


def test_graupel_faster_than_snow_in_precip_regime():
    # At equal MASS, low-density snow has a larger diameter than dense
    # graupel (D ∝ (m/ρ)^⅓), so fluffy snow's size wins at small masses;
    # dense graupel wins at large (precipitation-sized) masses. Crossover
    # ~170 µm graupel diameter (bin ~16). Assert the precip regime where
    # the category split matters: graupel falls faster in the upper bins
    # and reaches a much higher maximum.
    m = mass_doubling_grid()
    v_snow = np.asarray(ice_fall_speed(m, jnp.asarray(1.0), "snow"))
    v_graupel = np.asarray(ice_fall_speed(m, jnp.asarray(1.0), "graupel"))
    assert np.all(v_graupel[20:] > v_snow[20:])      # precip bins
    assert v_graupel.max() > 2.0 * v_snow.max()      # much faster tail


def test_physical_magnitudes_at_mm_sizes():
    # Tight Locatelli-Hobbs SI check (codex iter-2: catch coefficient/unit
    # errors, not just gross range). At D=1mm, ρ_air=ρ_ref (correction=1):
    # snow a·(1e-3)^b and graupel a·(1e-3)^b to 1% — pins the exact LH
    # coefficients, not a loose band.
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    from legoesm.atmosphere.physics.microphysics.fast_sbm.ice_fall_speed import (
        _SIX_OVER_PI)
    rho_ref = jnp.asarray(cfg.fall_rho_ref)        # correction == 1
    snow_1mm = cfg.fall_a_snow * (1.0e-3) ** cfg.fall_b_snow
    graupel_1mm = cfg.fall_a_graupel * (1.0e-3) ** cfg.fall_b_graupel
    assert 0.65 < snow_1mm < 0.75                  # LH aggregate ~0.69 m/s
    assert 1.25 < graupel_1mm < 1.35               # LH lump graupel ~1.30 m/s
    # The function reproduces those at the matching bin.
    D_g = (_SIX_OVER_PI * np.asarray(m) / cfg.rho_graupel) ** (1 / 3)
    k = int(np.argmin(np.abs(D_g - 1.0e-3)))
    vg = float(ice_fall_speed(m, rho_ref, "graupel")[k])
    D_g_k = float((_SIX_OVER_PI * np.asarray(m)[k] / cfg.rho_graupel) ** (1 / 3))
    assert vg == pytest.approx(
        cfg.fall_a_graupel * D_g_k ** cfg.fall_b_graupel, rel=1e-12)


def test_density_correction_speeds_thin_air():
    m = mass_doubling_grid()
    v_dense = ice_fall_speed(m, jnp.asarray(1.2), "graupel")
    v_thin = ice_fall_speed(m, jnp.asarray(0.4), "graupel")
    assert np.all(np.asarray(v_thin) > np.asarray(v_dense))
    # Exact (ρ0/ρ)^½ ratio.
    np.testing.assert_allclose(
        np.asarray(v_thin / v_dense), np.sqrt(1.2 / 0.4), rtol=1e-12)


def test_rho_air_array_broadcasts():
    m = mass_doubling_grid()
    rho = jnp.array([1.0, 0.8, 0.5])
    v = ice_fall_speed(m, rho, "snow")
    assert v.shape == (3, m.shape[0])


def test_unknown_category_raises():
    m = mass_doubling_grid()
    with pytest.raises(ValueError, match="category"):
        ice_fall_speed(m, jnp.asarray(1.0), "hailstone")


def test_differentiable_in_rho():
    m = mass_doubling_grid()
    g = jax.grad(lambda rho: jnp.sum(ice_fall_speed(m, rho, "graupel")))(
        jnp.asarray(0.9))
    assert np.isfinite(float(g))
    assert float(g) < 0.0     # lower ρ_air → faster fall
