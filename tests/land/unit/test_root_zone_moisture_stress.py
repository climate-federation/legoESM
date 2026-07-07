"""Direct tests for the shared root-zone soil-moisture stress used by BOTH the
multilayer land SEB and the coupler's atmospheric land tile.

Guards the fix for the multilayer-land over-evaporation: the atmosphere used to
re-derive the land surface humidity at ``beta = 1`` (a saturated swamp surface,
land hfls ~775 W/m^2, Bowen ~0.04) while the land model throttled internally.
``root_zone_moisture_stress`` / ``land_tile_beta_soil`` are the single source of
the soil-moisture throttle both sides now share.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.land.multilayer_land import (
    land_tile_beta_soil,
    root_zone_moisture_stress,
)
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import make_soil_grid


def _z_centers(n=10):
    # Monotone increasing node depths [m], surface (shallow) first.
    return jnp.linspace(0.05, 3.0, n)


def test_beta_floor_at_wilting_point():
    """theta <= theta_wp everywhere -> w_frac_rz = 0 -> beta = beta_min."""
    ncol, n = 4, 10
    beta_min, wp, fc = 0.1, 0.15, 0.30
    theta = jnp.full((ncol, n), wp)  # exactly at wilting point
    beta, _, beta_root, w = root_zone_moisture_stress(
        theta, beta_min, 1.0, wp, fc, _z_centers(n), ncol)
    np.testing.assert_allclose(np.asarray(beta), beta_min, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w), 0.0, atol=1e-12)
    assert np.all(np.asarray(beta_root) == 0.0)


def test_beta_saturates_at_field_capacity():
    """theta >= theta_fc everywhere -> w_frac_rz = 1 -> beta = 1."""
    ncol, n = 4, 10
    beta_min, wp, fc = 0.1, 0.15, 0.30
    theta = jnp.full((ncol, n), fc + 0.05)  # above field capacity -> clipped to 1
    beta, _, _, w = root_zone_moisture_stress(
        theta, beta_min, 1.0, wp, fc, _z_centers(n), ncol)
    np.testing.assert_allclose(np.asarray(beta), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w), 1.0, atol=1e-12)


def test_beta_monotonic_in_soil_moisture():
    """Wetter soil -> larger beta (bounded in [beta_min, 1])."""
    ncol, n = 1, 10
    beta_min, wp, fc = 0.1, 0.15, 0.30
    z = _z_centers(n)
    betas = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        theta = jnp.full((ncol, n), wp + frac * (fc - wp))
        b, _, _, _ = root_zone_moisture_stress(
            theta, beta_min, 1.0, wp, fc, z, ncol)
        betas.append(float(b[0]))
    assert betas == sorted(betas)                      # monotone non-decreasing
    assert betas[0] == pytest.approx(beta_min)          # dry -> floor
    assert betas[-1] == pytest.approx(1.0)              # wet -> potential
    assert all(beta_min - 1e-9 <= b <= 1.0 + 1e-9 for b in betas)


def test_degenerate_thresholds_do_not_blow_up():
    """theta_fc == theta_wp (pathological PFT row) -> 1e-3 floor keeps beta finite."""
    ncol, n = 2, 8
    theta = jnp.full((ncol, n), 0.2)
    beta, _, _, _ = root_zone_moisture_stress(
        theta, 0.1, 1.0, 0.2, 0.2, _z_centers(n), ncol)
    assert np.all(np.isfinite(np.asarray(beta)))
    assert np.all((np.asarray(beta) >= 0.1 - 1e-9) & (np.asarray(beta) <= 1.0 + 1e-9))


def test_land_tile_beta_soil_throttles_unsaturated_soil():
    """The coupler convenience returns beta in [beta_min, 1] and beta < 1 for
    unsaturated soil (so q_sfc = beta*q_sat < q_sat throttles land evaporation
    below the saturated-surface potential rate)."""
    cfg = MultiLayerLandConfig()   # beta_min=0.1, theta_wp=0.15, theta_fc=0.30
    grid = make_soil_grid(cfg.soil_grid)
    n = grid.z_node.shape[0]
    ncol = 5
    # Unsaturated but plant-available soil (the run's mean theta ~0.177).
    theta = jnp.full((ncol, n), 0.20)
    beta = land_tile_beta_soil(theta, cfg, land_params=None)
    beta = np.asarray(beta)
    assert beta.shape == (ncol,)
    assert np.all((beta >= cfg.beta_min - 1e-9) & (beta < 1.0))
    # Dry vs wet ordering.
    beta_dry = np.asarray(land_tile_beta_soil(
        jnp.full((ncol, n), cfg.theta_wp), cfg))
    beta_wet = np.asarray(land_tile_beta_soil(
        jnp.full((ncol, n), cfg.theta_fc), cfg))
    assert np.all(beta_dry <= beta) and np.all(beta <= beta_wet)
    np.testing.assert_allclose(beta_dry, cfg.beta_min, atol=1e-9)
    np.testing.assert_allclose(beta_wet, 1.0, atol=1e-9)
