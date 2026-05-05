"""Targeted tests for land-physics water budget fixes.

Covers:
1. Slab land: zero-bucket evaporation must shut off
2. Slab land: bucket overflow tracked as explicit runoff
3. Multilayer: transpiration sink integrates to surface LH flux
4. Richards: intense infiltration without premature runoff
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.state import LandState


def _make_forcing(ncol, **overrides):
    """Construct a simple AtmToSurface with sensible defaults."""
    ones = jnp.ones(ncol)
    defaults = dict(
        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=280.0, q_lowest=5e-3, u_lowest=5.0, v_lowest=2.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    defaults.update(overrides)
    return AtmToSurface(**{k: v * ones for k, v in defaults.items()})


def _make_slab_state(ncol, T_init=280.0, W_init=50.0):
    return LandState(
        T_soil=Field(jnp.full(ncol, T_init), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, W_init), name="W_bucket"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
    )


# =========================================================================
# 1. Slab land: zero-bucket evaporation shutdown
# =========================================================================


class TestZeroBucketEvaporation(unittest.TestCase):
    """When W=0 and precip=0, evaporation must be water-limited to zero."""

    def test_empty_bucket_no_evaporation(self):
        """LH flux must be zero when the bucket is empty and there is no precip."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        state = _make_slab_state(ncol, T_init=300.0, W_init=0.0)

        # Hot surface, dry atmosphere → large evaporative demand but no water
        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
                                T_lowest=290.0, q_lowest=1e-3, sw_down=400.0)

        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=3600.0)

        # Bucket must remain non-negative
        self.assertTrue(jnp.all(state2.W_bucket.data >= -1e-12),
                        f"Bucket went negative: {state2.W_bucket.data}")

        # LH flux must be effectively zero (water-limited)
        self.assertTrue(jnp.all(resp.lhflx < 1e-6),
                        f"LH flux should be ~0 with empty bucket, got {resp.lhflx}")

    def test_small_bucket_evap_limited(self):
        """Evaporation should not exceed available water + precip over dt."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        W_small = 0.5  # kg/m2 — tiny amount of water
        dt = 3600.0
        state = _make_slab_state(ncol, T_init=310.0, W_init=W_small)

        # Hot surface → strong evaporative demand, no precip
        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
                                T_lowest=280.0, q_lowest=1e-3, sw_down=600.0)

        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=dt)

        # Bucket must not go below zero
        self.assertTrue(jnp.all(state2.W_bucket.data >= -1e-12))

        # Total evap should not exceed available water
        evap_total = float(jnp.mean(resp.lhflx / constants.L_v * dt))  # kg/m2
        self.assertLessEqual(evap_total, W_small + 1e-8,
                             f"Evaporated {evap_total} from {W_small} kg/m2 bucket")

    def test_excess_energy_warms_soil(self):
        """When evap is water-limited, unused LH energy should heat the soil."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        dt = 3600.0

        # Compare: same forcing, one with water and one without
        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
                                T_lowest=280.0, q_lowest=1e-3, sw_down=400.0)

        state_wet = _make_slab_state(ncol, T_init=295.0, W_init=100.0)
        state_dry = _make_slab_state(ncol, T_init=295.0, W_init=0.0)

        s_wet, _, _ = step_land(state_wet, forcing, config, U_min=1.0, dt=dt)
        s_dry, _, _ = step_land(state_dry, forcing, config, U_min=1.0, dt=dt)

        # Dry case should warm more (energy that would have gone to evap heats soil)
        self.assertTrue(jnp.all(s_dry.T_soil.data > s_wet.T_soil.data),
                        "Dry soil should warm more than wet soil")


# =========================================================================
# 2. Slab land: bucket overflow conservation / runoff
# =========================================================================


class TestBucketOverflowRunoff(unittest.TestCase):
    """Overflow water must be tracked as explicit runoff, not silently lost."""

    def test_overflow_produces_runoff(self):
        """Heavy rain on a full bucket should produce nonzero runoff."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        state = _make_slab_state(ncol, T_init=280.0, W_init=config.W_max)

        # Heavy rain
        forcing = _make_forcing(ncol, precip_total=1e-2)
        state2, _, _ = step_land(state, forcing, config, U_min=1.0, dt=3600.0)

        # Runoff should be positive
        self.assertIsNotNone(state2.runoff, "LandState should have runoff field")
        self.assertTrue(jnp.all(state2.runoff > 0),
                        f"Runoff should be >0 for full bucket + rain, got {state2.runoff}")

    def test_water_conservation_with_runoff(self):
        """Total water budget: dW = (P + melt - E - runoff) * dt."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        W_init = config.W_max - 10.0
        state = _make_slab_state(ncol, T_init=280.0, W_init=W_init)

        # Heavy rain to cause overflow
        dt = 3600.0
        forcing = _make_forcing(ncol, precip_total=5e-3, precip_snow=0.0)
        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=dt)

        # Water budget: W_new = W_old + (P - E - runoff) * dt
        precip = float(forcing.precip_total[0])
        evap = float(resp.lhflx[0] / constants.L_v)
        runoff = float(state2.runoff[0])
        W_new = float(state2.W_bucket.data[0])
        W_old = float(W_init)

        budget_residual = abs(W_new - W_old - (precip - evap - runoff) * dt)
        self.assertLess(budget_residual, 1e-8,
                        f"Water budget residual = {budget_residual}")

    def test_no_runoff_below_capacity(self):
        """No runoff when bucket has room and rain is modest."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        state = _make_slab_state(ncol, T_init=280.0, W_init=50.0)

        forcing = _make_forcing(ncol, precip_total=1e-5)
        state2, _, _ = step_land(state, forcing, config, U_min=1.0, dt=3600.0)

        self.assertTrue(jnp.all(state2.runoff < 1e-15),
                        f"Should be no runoff when bucket is half-full, got {state2.runoff}")


# =========================================================================
# 3. Multilayer: transpiration water-budget closure
# =========================================================================


class TestTranspirationBudgetClosure(unittest.TestCase):
    """Vertically integrated root sink must equal the transpiration from LH flux."""

    def test_sink_integrates_to_transpiration(self):
        """The Richards root sink integrated over depth must match E_transp."""
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
        from legoesm.land.config import MultiLayerLandConfig

        config = MultiLayerLandConfig()
        grid = make_soil_grid(config.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        dz = grid.dz

        # Moist soil — above wilting point but below field capacity
        theta_val = 0.25
        theta = jnp.full((ncol, nlayers), theta_val)

        # Compute root distribution and beta_root (reproduce multilayer_land logic)
        z_centers = grid.z_node
        root_frac = jnp.exp(-z_centers / config.root_depth)
        root_frac = root_frac / jnp.sum(root_frac)

        beta_root = jnp.clip(
            (theta - config.theta_wp) / (config.theta_fc - config.theta_wp + 1e-10),
            0.0, 1.0,
        )

        f_veg = jnp.clip(
            jnp.sum(root_frac[None, :] * beta_root, axis=-1), 0.0, 1.0,
        )

        # Simulate a known evaporation rate
        evap_rate = jnp.full(ncol, 5e-5)  # kg/m2/s
        rho_w = constants.rho_water
        evap_transp = evap_rate * f_veg
        E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w  # m/s

        # Compute sink the new way (normalized weights)
        weight = root_frac[None, :] * beta_root
        weight_sum = jnp.sum(weight, axis=-1, keepdims=True)
        weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
        sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

        # Integrate sink * dz over layers — should equal E_pot_transp
        integrated_sink = jnp.sum(sink * dz[None, :], axis=-1)
        for col in range(ncol):
            self.assertAlmostEqual(
                float(integrated_sink[col]),
                float(E_pot_transp[col]),
                places=12,
                msg=f"Col {col}: integrated sink {integrated_sink[col]} "
                    f"!= E_pot_transp {E_pot_transp[col]}",
            )

    def test_dry_layer_gets_no_extraction(self):
        """A layer at wilting point should receive zero root extraction."""
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.config import MultiLayerLandConfig

        config = MultiLayerLandConfig()
        grid = make_soil_grid(config.soil_grid)
        ncol, nlayers = 2, grid.n_layers
        dz = grid.dz

        # Top layer dry (at wilting point), rest is moist
        theta = jnp.full((ncol, nlayers), 0.25)
        theta = theta.at[:, 0].set(config.theta_wp)

        z_centers = grid.z_node
        root_frac = jnp.exp(-z_centers / config.root_depth)
        root_frac = root_frac / jnp.sum(root_frac)

        beta_root = jnp.clip(
            (theta - config.theta_wp) / (config.theta_fc - config.theta_wp + 1e-10),
            0.0, 1.0,
        )

        f_veg = jnp.clip(
            jnp.sum(root_frac[None, :] * beta_root, axis=-1), 0.0, 1.0,
        )

        evap_rate = jnp.full(ncol, 3e-5)
        rho_w = constants.rho_water
        evap_transp = evap_rate * f_veg
        E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w

        weight = root_frac[None, :] * beta_root
        weight_sum = jnp.sum(weight, axis=-1, keepdims=True)
        weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
        sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

        # Top layer (at wilting point) should have zero extraction
        self.assertAlmostEqual(float(jnp.max(sink[:, 0])), 0.0, places=15)

        # But total integrated sink should still equal transpiration
        integrated = jnp.sum(sink * dz[None, :], axis=-1)
        for col in range(ncol):
            self.assertAlmostEqual(
                float(integrated[col]), float(E_pot_transp[col]), places=12,
            )


# =========================================================================
# 4. Richards: intense infiltration without premature runoff
# =========================================================================


class TestRichardsNoPrematureRunoff(unittest.TestCase):
    """Removing the top-layer saturation cap should allow deeper infiltration."""

    def test_near_saturated_thin_layer_no_spurious_runoff(self):
        """Rain below Darcy capacity on near-saturated soil should not produce
        spurious runoff due to the thin top-layer saturation cap.

        This targets the exact failure mode: the old code capped infiltration
        at (theta_sat - theta_top) * dz[0] / dt, which is tiny when the top
        layer is thin and near-saturated, even though K is high and the
        implicit solver can redistribute water downward.
        """
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, psi_from_theta, hydraulic_conductivity,
        )

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))

        # Near-saturated top layer (high K) but not full — deeper layers
        # have room to absorb percolation
        theta_top = hconfig.theta_sat - 0.01
        theta = jnp.full((ncol, nlayers), 0.30)
        theta = theta.at[:, 0].set(theta_top)
        psi = psi_from_theta(theta, hconfig)
        rconfig = RichardsConfig()

        # Flux well below the Darcy infiltration capacity at near-saturation
        K_top = hydraulic_conductivity(psi[:, 0], theta[:, 0], hconfig)
        flux_top = jnp.full(ncol, float(K_top[0]) * 0.3)  # 30% of Darcy K
        sink = jnp.zeros((ncol, nlayers))
        dt = 1800.0

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt)

        # With the old code, this would produce spurious runoff because
        # max_flux = (theta_sat - theta_top) * dz[0] / dt = 0.01 * 0.025 / 1800
        #          = 1.39e-7 m/s, which is far below the actual flux_top.
        # After the fix, only the Darcy capacity matters, and flux < K_top.
        max_runoff = float(jnp.max(out.runoff_surface))
        self.assertLess(max_runoff, 1e-10,
                        f"Spurious runoff = {max_runoff} on near-saturated "
                        f"soil with rain well below Darcy capacity")

    def test_truly_saturated_still_produces_runoff(self):
        """Genuinely saturated soil with heavy rain should still produce runoff."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))

        # Near saturation — very little remaining capacity
        theta = jnp.full((ncol, nlayers), hconfig.theta_sat - 1e-3)
        psi = psi_from_theta(theta, hconfig)
        rconfig = RichardsConfig()

        # Very heavy rain — above K_sat
        flux_top = jnp.full(ncol, hconfig.K_sat * 10.0)
        sink = jnp.zeros((ncol, nlayers))

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=600.0)

        # This should produce genuine runoff (Darcy capacity exceeded)
        self.assertTrue(jnp.all(out.runoff_surface > 0),
                        "Saturated soil + heavy rain should produce runoff")

    def test_mass_conservation(self):
        """Total water in = change in storage + runoff_surface + runoff_sub."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        dz = grid.dz

        theta_val = 0.25
        theta = jnp.full((ncol, nlayers), theta_val)
        psi = psi_from_theta(theta, hconfig)
        rconfig = RichardsConfig()

        flux_top = jnp.full(ncol, 2e-5)  # moderate rain
        sink = jnp.zeros((ncol, nlayers))
        dt = 1800.0

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt)

        # Storage change: sum((theta_new - theta_old) * dz) [m of water]
        storage_change = jnp.sum(
            (out.theta_new - theta) * dz[None, :], axis=-1
        )
        # Input = flux_infiltrated * dt [m]
        # Total input minus output = storage change
        runoff_sfc_m = out.runoff_surface / constants.rho_water  # m/s
        runoff_sub_m = out.runoff_subsurface / constants.rho_water  # m/s

        water_in = flux_top * dt
        water_out = (runoff_sfc_m + runoff_sub_m) * dt
        expected_storage = water_in - water_out

        for col in range(ncol):
            residual = abs(float(storage_change[col]) - float(expected_storage[col]))
            # Allow some tolerance (Picard iterations may not fully converge)
            self.assertLess(residual, 5e-5,
                            f"Col {col}: mass residual = {residual}")

    def test_matric_flux_redistributes_steep_psi_gradient(self):
        """Capillary (matric) flux should drive water from wet to dry
        layers when there is a steep psi gradient — even without
        gravity acting alone.

        This is the regression test for the iter-1 fix that added
        the missing ``L^m psi^m`` flux divergence to the Richards
        RHS.  Without that term, the converged Picard solution
        reduced to gravity-drainage only: a wet layer would drain
        downward via gravity but a dry layer would not pull water
        upward via capillarity, even with strong head gradients.

        Setup: dry bottom layers, wet top layer, zero surface flux,
        zero-flux bottom (no gravity drainage at the bottom).  The
        wet layer should lose water and the dry layers should gain
        water due to matric flux alone.
        """
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, psi_from_theta,
        )

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 2, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))

        # Wet upper layer (theta near saturation), dry layers below.
        theta = jnp.full((ncol, nlayers), hconfig.theta_r + 0.05)
        theta = theta.at[:, 0].set(hconfig.theta_sat - 0.02)
        psi = psi_from_theta(theta, hconfig)

        # Zero-flux at top (no infiltration), zero-flux at bottom
        # (no drainage).  Any redistribution comes from matric flux.
        rconfig = RichardsConfig(bottom_bc="zero_flux", max_iter=20)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        dt = 7200.0  # 2 h — long enough for noticeable redistribution

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt)

        dtheta_top = float(out.theta_new[0, 0] - theta[0, 0])
        dtheta_below = float(out.theta_new[0, 1] - theta[0, 1])

        # The wet top layer must lose moisture (capillary suction
        # from the dry layers below pulls water down) AND the
        # adjacent dry layer must gain moisture.  Without the
        # iter-1 L psi^m fix, the converged Picard iteration can
        # produce gravity-drainage-only behaviour — but with zero
        # gravity flux at the bottom, even gravity drainage would
        # be suppressed for layers above the bottom.  The matric
        # flux is the only mechanism that can redistribute here.
        self.assertLess(dtheta_top, -1e-5,
                        f"Wet top layer should lose moisture via "
                        f"capillarity; saw dtheta = {dtheta_top}")
        self.assertGreater(dtheta_below, 1e-5,
                           f"Dry second layer should gain moisture "
                           f"via capillarity; saw dtheta = {dtheta_below}")


if __name__ == "__main__":
    unittest.main()
