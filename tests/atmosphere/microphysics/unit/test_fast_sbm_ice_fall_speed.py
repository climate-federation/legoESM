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
    # Locatelli-Hobbs range: snow ~0.5 m/s, graupel ~1-3 m/s at ~1-4 mm.
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    # Find a ~1 mm graupel-density particle.
    from legoesm.atmosphere.physics.microphysics.fast_sbm.ice_fall_speed import (
        _SIX_OVER_PI)
    D_graupel = (_SIX_OVER_PI * np.asarray(m) / cfg.rho_graupel) ** (1 / 3)
    k = int(np.argmin(np.abs(D_graupel - 1.0e-3)))
    vg = float(ice_fall_speed(m, jnp.asarray(1.0), "graupel")[k])
    assert 0.5 < vg < 5.0
    D_snow = (_SIX_OVER_PI * np.asarray(m) / cfg.rho_snow) ** (1 / 3)
    ks = int(np.argmin(np.abs(D_snow - 1.0e-3)))
    vs = float(ice_fall_speed(m, jnp.asarray(1.0), "snow")[ks])
    assert 0.1 < vs < 1.5


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
