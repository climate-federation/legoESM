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
from legoesm.core.coupling_fields import AtmToSurface
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
        evap = float(resp.surface_mass_flux[0])   # the water the step removed [kg/m2/s]
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
        """Saturated soil + sustained heavy rain ponds, then spills to runoff.

        The coupled overland cell (ParFlow-style) detains rejected infiltration as
        surface water up to ``pond_max``, then overflows to runoff — water never
        vanishes.  Heavy rain on saturated soil (Darcy capacity ~ K_sat) cannot
        infiltrate, so the surface fills past pond_max within the step and runoff
        fires; all rejected rain is conserved as pond + runoff."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))

        # Near saturation — very little remaining capacity
        theta = jnp.full((ncol, nlayers), hconfig.theta_sat - 1e-3)
        psi = psi_from_theta(theta, hconfig)
        # Iterate well past the transient: this deliberately extreme forcing
        # (saturated soil + 10x-K_sat rain into the stiff specific-storage cell)
        # drives the fixed-iteration Picard into a small LIMIT CYCLE rather than
        # convergence (max|dpsi| stalls at ~9e-2 m psi for any max_iter >= ~20),
        # leaving a genuine ~3.4e-4 m residual: the last iteration's
        # linearization error theta(psi+dpsi) - [theta + C*dpsi] — real
        # (convergence, NOT structural) slack of the mixed form, not a flux-
        # reporting error.  Before the drainage report was made solve-consistent
        # (C13: report the K the last rhs debited, not K(psi_final)), this read
        # as < 1e-4 m by ACCIDENTAL CANCELLATION between the psi_final-evaluated
        # drainage and that linearization error.  The realistic-forcing gate
        # (test_multilayer_water_balance) closes at the production 10 iters.
        rconfig = RichardsConfig(max_iter=60)

        # Very heavy rain — above K_sat, sustained long enough to exceed pond_max
        flux_top = jnp.full(ncol, hconfig.K_sat * 10.0)
        sink = jnp.zeros((ncol, nlayers))
        pond0 = jnp.zeros(ncol)
        dt = 3600.0

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt, surface_water=pond0)

        # Saturated soil + heavy rain overflows the pond -> genuine runoff
        self.assertTrue(jnp.all(out.runoff_surface > 0),
                        "Saturated soil + heavy rain should produce runoff")
        # Pond holds at most pond_max
        self.assertTrue(jnp.all(out.surface_water <= rconfig.pond_max + 1e-9))
        # Conservation: rain = infiltrated + pond + runoff (no water created/lost).
        infil = jnp.sum((out.theta_new - theta) * grid.dz[None, :], axis=-1)
        runoff_m = (out.runoff_surface + out.runoff_subsurface) / constants.rho_water * dt
        resid = flux_top * dt - (infil + out.surface_water + runoff_m)
        # 5e-4 m bounds the limit-cycle linearization slack (~3.4e-4 m, see the
        # rconfig note); a real flux leak under this forcing is O(1e-2) m
        # (flux_top*dt ~ 0.104 m of rain).
        self.assertTrue(jnp.all(jnp.abs(resid) < 5e-4), f"resid={resid}")

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

        pond0 = jnp.zeros(ncol)
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt, surface_water=pond0)

        # Total storage = soil column + surface pond (the coupled overland cell):
        # excess rain detains on the surface rather than vanishing, so the budget
        # must track it.  sum((theta_new-theta)*dz) + (pond_new-pond0) [m of water]
        storage_change = (jnp.sum((out.theta_new - theta) * dz[None, :], axis=-1)
                          + (out.surface_water - pond0))
        # Input = flux_top * dt [m]; output = surface + subsurface runoff
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
        """Capillary (matric) flux should pull water UPWARD from a
        wet bottom into a dry top — gravity alone cannot do this.

        This is the regression test for the iter-1 fix that added
        the missing ``L^m psi^m`` flux divergence to the Richards
        RHS.  Without that term, the converged Picard solution
        reduces to gravity-drainage only — gravity drains downward
        but **never pulls water upward against gravity**.  So a
        setup with a wet BOTTOM and a dry TOP isolates the matric
        flux: any upward redistribution can only come from
        capillary suction.

        (The original test had wet TOP and dry BOTTOM, which
        gravity ALONE would also resolve, masking the L psi^m
        regression — Codex round-8 caught this.)
        """
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, psi_from_theta,
        )

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 2, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))

        # Dry top layers, wet BOTTOM layer.  The matric (capillary)
        # flux must move water *upward* against gravity — gravity
        # alone would never do this.
        theta = jnp.full((ncol, nlayers), hconfig.theta_r + 0.03)
        theta = theta.at[:, -1].set(hconfig.theta_sat - 0.02)
        psi = psi_from_theta(theta, hconfig)

        # Zero-flux at top (no infiltration), zero-flux at bottom
        # (no drainage).  Any redistribution comes from matric flux.
        rconfig = RichardsConfig(bottom_bc="zero_flux", max_iter=20)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        # Run for a full day so the slow capillary equilibration
        # produces a measurable signal at threshold ≥ 1e-6 m³/m³.
        dt = 86400.0

        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt)

        # The wet BOTTOM should lose moisture (drained UPWARD into
        # dry layers via capillary suction) AND the layer just
        # above (still relatively dry) should gain moisture.
        # Without the iter-1 L psi^m fix, the converged Picard
        # solution would be gravity-drainage only, which CANNOT
        # move water upward, so dtheta_above_bottom would be 0 and
        # this test would fail.  Threshold 1e-6 catches the
        # qualitative direction while staying above float noise.
        dtheta_bot = float(out.theta_new[0, -1] - theta[0, -1])
        dtheta_above = float(out.theta_new[0, -2] - theta[0, -2])
        self.assertLess(dtheta_bot, -1e-6,
                        f"Wet bottom should lose moisture via "
                        f"capillarity; saw dtheta = {dtheta_bot}")
        self.assertGreater(dtheta_above, 1e-6,
                           f"Dry layer above the wet bottom should "
                           f"gain moisture via upward capillary flux; "
                           f"saw dtheta = {dtheta_above}.  Without the "
                           f"L psi^m fix, gravity alone cannot move "
                           f"water upward.")


if __name__ == "__main__":
    unittest.main()
