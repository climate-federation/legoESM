"""Aridity-aware cold-start soil moisture for the multilayer (Richards) land tile.

``aridity_theta_init`` fixes the issue-#730 day-8 blowup: the legacy
moisture-uniform ``0.5 * theta_sat`` seed left subtropical deserts holding
rainforest-scale water, so a hot bare-soil skin drove runaway potential
evaporation.  The helper anchors the initial plant-available water to the
near-surface RH of the IC atmosphere, mapped into ``[theta_wp, theta_fc]``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.land import aridity_theta_init, init_multilayer_land_state
from legoesm.land.config import MultiLayerLandConfig


def test_endpoints_map_to_wilting_and_field_capacity():
    """RH=0 -> wilting point (arid); RH=1 -> field capacity (saturated air)."""
    wp, fc = 0.12, 0.30
    assert float(aridity_theta_init(0.0, wp, fc)) == pytest.approx(wp)
    assert float(aridity_theta_init(1.0, wp, fc)) == pytest.approx(fc)
    assert float(aridity_theta_init(0.5, wp, fc)) == pytest.approx(0.5 * (wp + fc))


def test_clipped_outside_unit_interval():
    """Out-of-range RH is clipped so theta stays within [theta_wp, theta_fc]."""
    wp, fc = 0.12, 0.30
    assert float(aridity_theta_init(-0.4, wp, fc)) == pytest.approx(wp)
    assert float(aridity_theta_init(1.7, wp, fc)) == pytest.approx(fc)


def test_monotonic_increasing_in_rh():
    rh = jnp.linspace(0.0, 1.0, 11)
    theta = np.asarray(aridity_theta_init(rh, 0.1, 0.35))
    assert np.all(np.diff(theta) > 0.0)


def test_per_column_arrays_broadcast():
    """Per-column RH and per-column thresholds combine element-wise."""
    rh = jnp.array([0.05, 0.5, 0.95])          # desert, mid, humid
    wp = jnp.array([0.10, 0.12, 0.14])
    fc = jnp.array([0.28, 0.30, 0.32])
    theta = np.asarray(aridity_theta_init(rh, wp, fc))
    assert theta.shape == (3,)
    # each column lands inside its own [wp, fc] band, and the arid column is the driest
    assert np.all((theta >= np.asarray(wp) - 1e-9) & (theta <= np.asarray(fc) + 1e-9))
    assert theta[0] < theta[1] < theta[2]


def test_arid_column_starts_near_wilting_not_legacy_seed():
    """The whole point: a desert now starts near wilting, so the tile's beta
    (=beta_min + (1-beta_min)*w_frac, w_frac the plant-available saturation) is
    at its floor from step 1 -- beta multiplies q_sat(T_skin), the runaway lever.
    The legacy uniform 0.5*theta_sat seed instead starts well up the beta ramp."""
    wp, fc, theta_sat, theta_r = 0.12, 0.30, 0.45, 0.05
    legacy = 0.5 * theta_sat                    # 0.225 -- the old uniform seed
    arid = float(aridity_theta_init(0.1, wp, fc))   # low-RH desert

    def w_frac(theta):                          # plant-available saturation -> beta ramp
        return np.clip((theta - wp) / (fc - wp), 0.0, 1.0)

    assert w_frac(arid) < 0.2                    # desert beta pinned near the floor
    assert w_frac(legacy) > 0.5                   # legacy seed half-way up the ramp
    # extractable water (what the evap limiter caps against) is also strictly lower
    depth, rho_w = 3.0, 1000.0
    assert (arid - theta_r) * depth * rho_w < (legacy - theta_r) * depth * rho_w


def test_seed_is_a_valid_richards_state():
    """theta_init in [wp, fc] stays within [theta_r, theta_sat] -> finite psi."""
    cfg = MultiLayerLandConfig()
    ncol = 4
    wp = jnp.full((ncol, 1), 0.12)
    fc = jnp.full((ncol, 1), 0.30)
    rh = jnp.array([0.0, 0.3, 0.7, 1.0]).reshape(ncol, 1)
    theta_init = aridity_theta_init(rh, wp, fc)
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, theta_init=theta_init)
    theta = np.asarray(state.theta_soil)
    assert theta.shape == (ncol, cfg.soil_grid.n_layers)
    assert np.all(np.isfinite(np.asarray(state.psi_soil)))
    assert np.all((theta >= cfg.hydraulics.theta_r) & (theta <= cfg.hydraulics.theta_sat))
    # row 0 (RH=0) drier than row 3 (RH=1)
    assert theta[0, 0] < theta[3, 0]


def test_differentiable():
    """Pure + differentiable (the tile trains through the land init in calibration)."""
    g = jax.grad(lambda rh: jnp.sum(aridity_theta_init(rh, 0.12, 0.30)))
    # d(theta)/d(rh) = (fc - wp) inside (0,1)
    assert float(g(jnp.array(0.5))) == pytest.approx(0.30 - 0.12)
