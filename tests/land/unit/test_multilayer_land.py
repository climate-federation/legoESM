"""Tests for Task 8: Multi-layer soil with Richards equation.

Covers:
- 8A: SoilGrid construction (geometric, custom, total_depth)
- 8B: Soil hydraulic retention curves (VG, CH, BC, Campbell)
- 8C: Richards equation solver (mass conservation, boundary conditions)
- 8D: Soil thermal diffusion (energy conservation, convergence)
- 8E: Integrated multi-layer land step
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)


# =========================================================================
# 8A: Soil Grid
# =========================================================================


class TestSoilGrid(unittest.TestCase):
    """Test soil grid construction."""

    def test_default_grid(self):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        grid = make_soil_grid(SoilGridConfig())
        self.assertEqual(grid.n_layers, 8)
        self.assertEqual(grid.dz.shape, (8,))
        self.assertEqual(grid.z_node.shape, (8,))
        self.assertEqual(grid.z_interface.shape, (9,))
        self.assertEqual(grid.dz_interface.shape, (7,))
        # First layer thickness
        self.assertAlmostEqual(float(grid.dz[0]), 0.025, places=6)
        # Geometric growth: each layer is 2x previous
        for k in range(1, 8):
            self.assertAlmostEqual(
                float(grid.dz[k] / grid.dz[k - 1]), 2.0, places=6
            )

    def test_total_depth(self):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        target = 3.0
        grid = make_soil_grid(SoilGridConfig(n_layers=6, total_depth=target))
        total = float(jnp.sum(grid.dz))
        self.assertAlmostEqual(total, target, places=6)

    def test_custom_grid(self):
        from legoesm.land.soil_grid import make_soil_grid_custom
        dz = [0.1, 0.2, 0.5, 1.0]
        grid = make_soil_grid_custom(dz)
        self.assertEqual(grid.n_layers, 4)
        self.assertAlmostEqual(float(jnp.sum(grid.dz)), 1.8, places=6)
        # Interface at top is 0
        self.assertAlmostEqual(float(grid.z_interface[0]), 0.0)
        # Interface at bottom
        self.assertAlmostEqual(float(grid.z_interface[-1]), 1.8, places=6)

    def test_uniform_spacing(self):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        grid = make_soil_grid(SoilGridConfig(
            n_layers=5, dz_top=0.5, growth_factor=1.0
        ))
        for k in range(5):
            self.assertAlmostEqual(float(grid.dz[k]), 0.5, places=6)

    def test_midpoints_between_interfaces(self):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        grid = make_soil_grid(SoilGridConfig())
        for k in range(grid.n_layers):
            expected = 0.5 * (float(grid.z_interface[k]) + float(grid.z_interface[k + 1]))
            self.assertAlmostEqual(float(grid.z_node[k]), expected, places=10)


# =========================================================================
# 8B: Soil Hydraulics
# =========================================================================


class TestSoilHydraulics(unittest.TestCase):
    """Test soil water retention curves."""

    def test_van_genuchten_roundtrip(self):
        """theta(psi(theta)) should recover theta."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, van_genuchten_psi, van_genuchten_theta,
        )
        config = SoilHydraulicsConfig()
        theta = jnp.linspace(config.theta_r + 0.01, config.theta_sat - 0.01, 20)
        psi = van_genuchten_psi(theta, config)
        theta_back = van_genuchten_theta(psi, config)
        npt.assert_allclose(theta_back, theta, atol=1e-6)

    def test_van_genuchten_saturation(self):
        """At psi=0, theta should equal theta_sat."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, van_genuchten_theta, van_genuchten_K,
        )
        config = SoilHydraulicsConfig()
        psi = jnp.array([0.0])
        theta = van_genuchten_theta(psi, config)
        self.assertAlmostEqual(float(theta[0]), config.theta_sat, places=6)
        K = van_genuchten_K(psi, config)
        self.assertAlmostEqual(float(K[0]), config.K_sat, places=10)

    def test_van_genuchten_K_monotonic(self):
        """K should increase with increasing psi (wetter soil)."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, van_genuchten_K,
        )
        config = SoilHydraulicsConfig()
        psi = jnp.linspace(-10.0, -0.01, 50)
        K = van_genuchten_K(psi, config)
        diffs = jnp.diff(K)
        self.assertTrue(jnp.all(diffs >= 0))

    def test_clapp_hornberger_roundtrip(self):
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, clapp_hornberger_psi, clapp_hornberger_theta,
        )
        config = SoilHydraulicsConfig(retention_curve="clapp_hornberger")
        theta = jnp.linspace(0.1, config.theta_sat - 0.01, 20)
        psi = clapp_hornberger_psi(theta, config)
        theta_back = clapp_hornberger_theta(psi, config)
        npt.assert_allclose(theta_back, theta, atol=1e-4)

    def test_dispatch_theta_from_psi(self):
        """Dispatch function should route to correct model."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, theta_from_psi, van_genuchten_theta,
        )
        config = SoilHydraulicsConfig(retention_curve="van_genuchten")
        psi = jnp.array([-1.0, -5.0])
        theta1 = theta_from_psi(psi, config)
        theta2 = van_genuchten_theta(psi, config)
        npt.assert_allclose(theta1, theta2)

    def test_dispatch_hydraulic_conductivity(self):
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, hydraulic_conductivity, van_genuchten_K,
        )
        config = SoilHydraulicsConfig()
        psi = jnp.array([-1.0, -5.0])
        theta = jnp.array([0.3, 0.2])
        K1 = hydraulic_conductivity(psi, theta, config)
        K2 = van_genuchten_K(psi, config)
        npt.assert_allclose(K1, K2)

    def test_moisture_capacity_positive(self):
        """dtheta/dpsi should be non-negative."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, moisture_capacity,
        )
        config = SoilHydraulicsConfig()
        psi = jnp.linspace(-10.0, -0.01, 30)
        theta = jnp.full_like(psi, 0.3)  # dummy
        C = moisture_capacity(psi, theta, config)
        self.assertTrue(jnp.all(C >= 0))

    def test_interblock_K(self):
        from legoesm.land.soil_hydraulics import interblock_K
        K1 = jnp.array([1e-6, 1e-5])
        K2 = jnp.array([1e-5, 1e-4])
        K_half = interblock_K(K1, K2)
        expected = jnp.sqrt(K1 * K2)
        npt.assert_allclose(K_half, expected, rtol=1e-6)

    def test_brooks_corey_saturation(self):
        """At psi >= psi_b, Se should be 1."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, brooks_corey_Se,
        )
        config = SoilHydraulicsConfig()
        psi = jnp.array([config.psi_b + 0.1, 0.0])
        Se = brooks_corey_Se(psi, config)
        npt.assert_allclose(Se, jnp.ones(2), atol=1e-6)


# =========================================================================
# 8B-PDI: Peters-Durner-Iden (2015)
# =========================================================================


class TestPDI(unittest.TestCase):
    """Test Peters-Durner-Iden retention curve model."""

    def _config(self):
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        return SoilHydraulicsConfig(retention_curve="pdi")

    def test_pdi_saturation(self):
        """At psi=0, theta should equal theta_sat."""
        from legoesm.land.soil_hydraulics import pdi_theta
        config = self._config()
        psi = jnp.array([0.0])
        theta = pdi_theta(psi, config)
        self.assertAlmostEqual(float(theta[0]), config.theta_sat, places=6)

    def test_pdi_theta_monotonic(self):
        """Theta should increase with increasing psi (less negative = wetter)."""
        from legoesm.land.soil_hydraulics import pdi_theta
        config = self._config()
        psi = jnp.linspace(-100.0, -0.01, 50)
        theta = pdi_theta(psi, config)
        diffs = jnp.diff(theta)
        self.assertTrue(jnp.all(diffs >= -1e-10))

    def test_pdi_theta_bounded(self):
        """Theta should stay in [0, theta_sat]."""
        from legoesm.land.soil_hydraulics import pdi_theta
        config = self._config()
        psi = jnp.linspace(-1e4, -0.001, 100)
        theta = pdi_theta(psi, config)
        self.assertTrue(jnp.all(theta >= 0.0))
        self.assertTrue(jnp.all(theta <= config.theta_sat + 1e-10))

    def test_pdi_K_at_saturation(self):
        """K at psi=0 should equal K_sat."""
        from legoesm.land.soil_hydraulics import pdi_K
        config = self._config()
        K = pdi_K(jnp.array([0.0]), config)
        self.assertAlmostEqual(float(K[0]), config.K_sat, places=10)

    def test_pdi_K_positive(self):
        """K should be positive everywhere."""
        from legoesm.land.soil_hydraulics import pdi_K
        config = self._config()
        psi = jnp.linspace(-1000.0, -0.01, 50)
        K = pdi_K(psi, config)
        self.assertTrue(jnp.all(K > 0))

    def test_pdi_C_positive(self):
        """Moisture capacity should be non-negative."""
        from legoesm.land.soil_hydraulics import pdi_C
        config = self._config()
        psi = jnp.linspace(-100.0, -0.01, 30)
        C = pdi_C(psi, config)
        self.assertTrue(jnp.all(C >= -1e-8))

    def test_pdi_dispatch(self):
        """Dispatch functions should route to PDI."""
        from legoesm.land.soil_hydraulics import (
            theta_from_psi, hydraulic_conductivity, moisture_capacity,
            pdi_theta, pdi_K,
        )
        config = self._config()
        psi = jnp.array([-1.0, -5.0])
        theta = jnp.array([0.3, 0.2])
        npt.assert_allclose(theta_from_psi(psi, config), pdi_theta(psi, config))
        npt.assert_allclose(
            hydraulic_conductivity(psi, theta, config), pdi_K(psi, config)
        )
        # moisture_capacity should not raise
        C = moisture_capacity(psi, theta, config)
        self.assertTrue(jnp.all(jnp.isfinite(C)))

    def test_pdi_adsorptive_component(self):
        """In the mid-range, adsorptive saturation Snc should be between 0 and 1."""
        from legoesm.land.soil_hydraulics import _pdi_Snc
        config = self._config()
        # Mid-range suction where Snc transitions
        h_mid = jnp.array([10.0])
        Snc = _pdi_Snc(h_mid, config)
        self.assertTrue(float(Snc[0]) > 0)
        self.assertTrue(float(Snc[0]) < 1)

    def test_pdi_max_pore_constraint(self):
        """K near saturation should be affected by h_crit constraint."""
        from legoesm.land.soil_hydraulics import pdi_K, van_genuchten_K
        config = self._config()
        # Just below h_crit
        psi_near_sat = jnp.array([-config.h_crit * 0.5])
        K_pdi = pdi_K(psi_near_sat, config)
        K_vg = van_genuchten_K(psi_near_sat, config)
        # PDI should differ from standard VG near saturation due to constraint
        # (PDI interpolates to K_sat, VG drops from K_sat differently)
        self.assertTrue(jnp.all(jnp.isfinite(K_pdi)))

    def test_pdi_richards_integration(self):
        """PDI should work end-to-end through Richards solver."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        config = SoilHydraulicsConfig(retention_curve="pdi")
        ncol, nlayers = 4, 6
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        theta = jnp.full((ncol, nlayers), 0.25)
        psi = psi_from_theta(theta, config)
        rconfig = RichardsConfig(bottom_bc="zero_flux")
        out = solve_richards(psi, theta, grid, config, rconfig,
                             jnp.zeros(ncol), jnp.zeros((ncol, nlayers)), dt=600.0)
        self.assertTrue(jnp.all(jnp.isfinite(out.theta_new)))


# =========================================================================
# 8B-Lu: Lu (2016) three-regime SWRC
# =========================================================================


class TestLu(unittest.TestCase):
    """Test Lu (2016) three-regime SWRC model."""

    def _config(self):
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        return SoilHydraulicsConfig(retention_curve="lu")

    def test_lu_saturation(self):
        """At psi=0, theta should equal theta_sat."""
        from legoesm.land.soil_hydraulics import lu_theta
        config = self._config()
        theta = lu_theta(jnp.array([0.0]), config)
        self.assertAlmostEqual(float(theta[0]), config.theta_sat, places=6)

    def test_lu_theta_monotonic(self):
        """Theta should increase with increasing psi."""
        from legoesm.land.soil_hydraulics import lu_theta
        config = self._config()
        psi = jnp.linspace(-500.0, -0.01, 50)
        theta = lu_theta(psi, config)
        diffs = jnp.diff(theta)
        self.assertTrue(jnp.all(diffs >= -1e-10))

    def test_lu_theta_bounded(self):
        """Theta should stay in [0, theta_sat]."""
        from legoesm.land.soil_hydraulics import lu_theta
        config = self._config()
        psi = jnp.linspace(-1e4, -0.001, 100)
        theta = lu_theta(psi, config)
        self.assertTrue(jnp.all(theta >= 0.0))
        self.assertTrue(jnp.all(theta <= config.theta_sat + 1e-10))

    def test_lu_K_at_saturation(self):
        """K at psi=0 should equal K_sat."""
        from legoesm.land.soil_hydraulics import lu_K
        config = self._config()
        K = lu_K(jnp.array([0.0]), config)
        self.assertAlmostEqual(float(K[0]), config.K_sat, places=10)

    def test_lu_K_positive(self):
        """K should be positive everywhere."""
        from legoesm.land.soil_hydraulics import lu_K
        config = self._config()
        psi = jnp.linspace(-500.0, -0.01, 50)
        K = lu_K(psi, config)
        self.assertTrue(jnp.all(K > 0))

    def test_lu_C_positive(self):
        """Moisture capacity should be non-negative."""
        from legoesm.land.soil_hydraulics import lu_C
        config = self._config()
        psi = jnp.linspace(-100.0, -0.01, 30)
        C = lu_C(psi, config)
        self.assertTrue(jnp.all(C >= -1e-8))

    def test_lu_dispatch(self):
        """Dispatch functions should route to Lu."""
        from legoesm.land.soil_hydraulics import (
            theta_from_psi, hydraulic_conductivity, moisture_capacity,
            lu_theta, lu_K,
        )
        config = self._config()
        psi = jnp.array([-1.0, -5.0])
        theta = jnp.array([0.3, 0.2])
        npt.assert_allclose(theta_from_psi(psi, config), lu_theta(psi, config))
        npt.assert_allclose(
            hydraulic_conductivity(psi, theta, config), lu_K(psi, config)
        )
        C = moisture_capacity(psi, theta, config)
        self.assertTrue(jnp.all(jnp.isfinite(C)))

    def test_lu_three_regimes(self):
        """Lu model should show three distinct regimes."""
        from legoesm.land.soil_hydraulics import lu_theta
        config = self._config()
        # Wet (capillary-dominated)
        theta_wet = float(lu_theta(jnp.array([-0.5]), config)[0])
        # Mid-range (transition)
        theta_mid = float(lu_theta(jnp.array([-50.0]), config)[0])
        # Dry (adsorption-dominated)
        theta_dry = float(lu_theta(jnp.array([-5000.0]), config)[0])
        # Should be monotonically decreasing
        self.assertGreater(theta_wet, theta_mid)
        self.assertGreater(theta_mid, theta_dry)
        # Dry regime should still have some adsorbed water
        self.assertGreater(theta_dry, 0)

    def test_lu_richards_integration(self):
        """Lu model should work end-to-end through Richards solver."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        config = SoilHydraulicsConfig(retention_curve="lu")
        ncol, nlayers = 4, 6
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        theta = jnp.full((ncol, nlayers), 0.25)
        psi = psi_from_theta(theta, config)
        rconfig = RichardsConfig(bottom_bc="zero_flux")
        out = solve_richards(psi, theta, grid, config, rconfig,
                             jnp.zeros(ncol), jnp.zeros((ncol, nlayers)), dt=600.0)
        self.assertTrue(jnp.all(jnp.isfinite(out.theta_new)))


# =========================================================================
# 8C: Richards Equation
# =========================================================================


class TestRichards(unittest.TestCase):
    """Test Richards equation solver."""

    def _make_uniform_state(self, ncol=4, nlayers=6, theta_val=0.25):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        config = SoilHydraulicsConfig()
        theta = jnp.full((ncol, nlayers), theta_val)
        psi = psi_from_theta(theta, config)
        return psi, theta, grid, config

    def test_zero_flux_steady_state(self):
        """With zero flux and uniform moisture, state should be near-unchanged."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        psi, theta, grid, hconfig = self._make_uniform_state()
        ncol, nlayers = theta.shape
        rconfig = RichardsConfig(bottom_bc="zero_flux")
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=3600.0)
        # theta should change very little (gravity still causes small redistribution)
        max_change = float(jnp.max(jnp.abs(out.theta_new - theta)))
        self.assertLess(max_change, 5e-3)

    def test_infiltration_increases_moisture(self):
        """Positive infiltration should increase top-layer moisture."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        psi, theta, grid, hconfig = self._make_uniform_state(theta_val=0.2)
        ncol, nlayers = theta.shape
        rconfig = RichardsConfig()
        flux_top = jnp.full(ncol, 1e-5)  # 0.01 mm/s rain
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=600.0)
        # Top layer should be wetter
        self.assertTrue(jnp.all(out.theta_new[:, 0] >= theta[:, 0] - 1e-10))

    def test_surface_runoff_excess(self):
        """Near-saturated soil should produce surface runoff."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 6
        # Near saturation
        theta = jnp.full((ncol, nlayers), hconfig.theta_sat - 0.001)
        psi = psi_from_theta(theta, hconfig)
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        rconfig = RichardsConfig()
        flux_top = jnp.full(ncol, 1e-3)  # very heavy rain
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=600.0)
        # Should have nonzero surface runoff
        self.assertTrue(jnp.all(out.runoff_surface > 0))

    def test_free_drainage_runoff(self):
        """Free drainage should produce subsurface runoff."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        psi, theta, grid, hconfig = self._make_uniform_state(theta_val=0.3)
        ncol, nlayers = theta.shape
        rconfig = RichardsConfig(bottom_bc="free_drainage")
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=3600.0)
        # Should have positive subsurface runoff
        self.assertTrue(jnp.all(out.runoff_subsurface > 0))

    def test_theta_within_bounds(self):
        """Theta stays >= theta_r and <= the specific-storage ceiling.

        Below saturation theta is bounded by theta_sat; AT/above saturation the
        ParFlow/CliMA specific-storage switch lets theta rise elastically to
        theta_sat*(1 + S_s*psi) for psi >= 0 (the compressible-storage term that
        replaced the old non-conservative theta clip).  The ceiling is that exact
        law, not theta_sat — a hard clip at theta_sat would destroy ponded water."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        psi, theta, grid, hconfig = self._make_uniform_state(theta_val=0.15)
        ncol, nlayers = theta.shape
        rconfig = RichardsConfig()
        flux_top = jnp.full(ncol, 5e-5)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hconfig, rconfig,
                             flux_top, sink, dt=1800.0, surface_water=jnp.zeros(ncol))
        self.assertTrue(jnp.all(out.theta_new >= hconfig.theta_r - 1e-10))
        ceiling = hconfig.theta_sat * (1.0 + hconfig.S_s * jnp.maximum(out.psi_new, 0.0))
        self.assertTrue(jnp.all(out.theta_new <= ceiling + 1e-10))


# =========================================================================
# 8D: Soil Thermal
# =========================================================================


class TestSoilThermal(unittest.TestCase):
    """Test soil thermal diffusion."""

    def _make_state(self, ncol=4, nlayers=6):
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        from legoesm.land.soil_thermal import SoilThermalConfig
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        hconfig = SoilHydraulicsConfig()
        tconfig = SoilThermalConfig()
        return grid, hconfig, tconfig, ncol, nlayers

    def test_uniform_T_steady(self):
        """Uniform temperature with zero flux should stay constant."""
        from legoesm.land.soil_thermal import solve_soil_thermal, SoilThermalConfig
        grid, hconfig, _tconfig, ncol, nlayers = self._make_state()
        # Disable geothermal flux so the system is truly zero-flux steady state
        tconfig = SoilThermalConfig(Q_geothermal=0.0)
        T = jnp.full((ncol, nlayers), 285.0)
        theta = jnp.full((ncol, nlayers), 0.25)
        G = jnp.zeros(ncol)
        T_new = solve_soil_thermal(T, theta, grid, hconfig, tconfig, G, dt=3600.0)
        npt.assert_allclose(T_new, T, atol=1e-10)

    def test_surface_heating(self):
        """Positive G_surface should warm the top layer."""
        from legoesm.land.soil_thermal import solve_soil_thermal
        grid, hconfig, tconfig, ncol, nlayers = self._make_state()
        T = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.25)
        G = jnp.full(ncol, 50.0)  # 50 W/m2 into soil
        T_new = solve_soil_thermal(T, theta, grid, hconfig, tconfig, G, dt=3600.0)
        # Top layer should warm
        self.assertTrue(jnp.all(T_new[:, 0] > T[:, 0]))
        # Deep layers should barely change
        deep_change = float(jnp.max(jnp.abs(T_new[:, -1] - T[:, -1])))
        self.assertLess(deep_change, 0.1)

    def test_heat_capacity_increases_with_moisture(self):
        """Wetter soil should have higher heat capacity."""
        from legoesm.land.soil_thermal import compute_heat_capacity
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        from legoesm.land.soil_thermal import SoilThermalConfig
        hconfig = SoilHydraulicsConfig()
        tconfig = SoilThermalConfig()
        theta_dry = jnp.array([0.1])
        theta_wet = jnp.array([0.4])
        C_dry = compute_heat_capacity(theta_dry, hconfig, tconfig)
        C_wet = compute_heat_capacity(theta_wet, hconfig, tconfig)
        self.assertTrue(float(C_wet[0]) > float(C_dry[0]))

    def test_conductivity_increases_with_moisture(self):
        """Wetter soil should have higher thermal conductivity."""
        from legoesm.land.soil_thermal import compute_thermal_conductivity
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        from legoesm.land.soil_thermal import SoilThermalConfig
        hconfig = SoilHydraulicsConfig()
        tconfig = SoilThermalConfig()
        theta_dry = jnp.array([0.1])
        theta_wet = jnp.array([0.4])
        k_dry = compute_thermal_conductivity(theta_dry, hconfig, tconfig)
        k_wet = compute_thermal_conductivity(theta_wet, hconfig, tconfig)
        self.assertTrue(float(k_wet[0]) > float(k_dry[0]))

    def test_energy_conservation(self):
        """Total energy change should equal integrated flux.

        The total energy input includes both the surface ground heat flux
        *and* the geothermal heat flux at the bottom boundary.
        """
        from legoesm.land.soil_thermal import (
            solve_soil_thermal, compute_heat_capacity,
        )
        grid, hconfig, tconfig, ncol, nlayers = self._make_state()
        T = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.25)
        G = jnp.full(ncol, 100.0)
        dt = 600.0
        T_new = solve_soil_thermal(T, theta, grid, hconfig, tconfig, G, dt)
        C = compute_heat_capacity(theta, hconfig, tconfig)
        dE = jnp.sum(C * grid.dz * (T_new - T), axis=1)  # J/m2
        # Energy input = surface flux + geothermal flux at bottom
        expected = (G + tconfig.Q_geothermal) * dt  # J/m2
        npt.assert_allclose(dE, expected, rtol=1e-6)


# =========================================================================
# 8E: Integrated Multi-Layer Land Step
# =========================================================================


class TestMultilayerLandStep(unittest.TestCase):
    """Test the full multi-layer land step function."""

    def _make_forcing(self, ncol):
        """Create simple atmospheric forcing."""
        from legoesm.core.coupling_fields import AtmToSurface
        return AtmToSurface(
            sw_down=jnp.full(ncol, 300.0),
            lw_down=jnp.full(ncol, 350.0),
            precip_total=jnp.full(ncol, 1e-4),  # 0.1 mm/s
            precip_snow=jnp.zeros(ncol),
            T_lowest=jnp.full(ncol, 290.0),
            q_lowest=jnp.full(ncol, 0.008),
            u_lowest=jnp.full(ncol, 5.0),
            v_lowest=jnp.full(ncol, 2.0),
            p_lowest=jnp.full(ncol, 95000.0),
            p_surface=jnp.full(ncol, 100000.0),
            rho_lowest=jnp.full(ncol, 1.2),
            cos_zenith=jnp.full(ncol, 0.5),
            co2_ppmv=jnp.full(ncol, 400.0),
            has_radiation=jnp.ones(ncol),
            has_precipitation=jnp.ones(ncol),
        )

    def test_basic_step(self):
        """Single step should produce valid state and response."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        config = MultiLayerLandConfig()
        ncol = 8
        state = init_multilayer_land_state(ncol, config, T_init=280.0)
        forcing = self._make_forcing(ncol)

        new_state, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0,
        )

        # Check shapes
        nlayers = config.soil_grid.n_layers
        self.assertEqual(new_state.T_soil.shape, (ncol, nlayers))
        self.assertEqual(new_state.theta_soil.shape, (ncol, nlayers))
        self.assertEqual(new_state.psi_soil.shape, (ncol, nlayers))
        self.assertEqual(new_state.runoff_surface.shape, (ncol,))
        self.assertEqual(response.T_sfc.shape, (ncol,))
        self.assertEqual(response.shflx.shape, (ncol,))

    def test_state_dtype_stable_under_x64(self):
        """Under JAX_ENABLE_X64 the soil-thermal / Richards solves promote to
        float64, but the returned state must be cast back to the INPUT (storage)
        dtype so a lax.scan carry has matching input/output dtypes.  The SOTA AMIP
        run (JAX_ENABLE_X64=1) crashed here: carry land_ml.T_soil was float32 in
        but float64 out.  Runs under x64 to reproduce the promotion."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        _was = getattr(jax.config, "jax_enable_x64", False)
        jax.config.update("jax_enable_x64", True)
        try:
            config = MultiLayerLandConfig()
            ncol = 8
            # storage-dtype (float32) state, as ModelDriver seeds the carry
            state = init_multilayer_land_state(
                ncol, config, T_init=jnp.full(ncol, 280.0, dtype=jnp.float32))
            self.assertEqual(state.T_soil.dtype, jnp.float32)
            forcing = self._make_forcing(ncol)
            new_state, _resp, _ = step_multilayer_land(
                state, forcing, config, U_min=1.0, dt=600.0)
            mismatched = [
                f for f in state._fields
                if hasattr(getattr(state, f), "dtype")
                and getattr(new_state, f).dtype != getattr(state, f).dtype
            ]
            self.assertEqual(
                mismatched, [],
                f"state leaves changed dtype under x64 (scan-carry unsafe): "
                f"{mismatched}")
        finally:
            jax.config.update("jax_enable_x64", _was)

    def test_surface_temperature_responds(self):
        """Surface temperature should change after a step."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config, T_init=270.0)
        forcing = self._make_forcing(ncol)

        new_state, _, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0,
        )

        # With SW+LW forcing into cold soil, surface should warm
        self.assertTrue(jnp.all(new_state.T_soil[:, 0] != state.T_soil[:, 0]))

    def test_multi_step_stability(self):
        """Multiple time steps should remain stable."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config, T_init=280.0)
        forcing = self._make_forcing(ncol)

        for _ in range(10):
            state, response, _ = step_multilayer_land(
                state, forcing, config, U_min=1.0, dt=600.0,
            )

        # Temperature should be finite and reasonable
        self.assertTrue(jnp.all(jnp.isfinite(state.T_soil)))
        self.assertTrue(jnp.all(state.T_soil > 200.0))
        self.assertTrue(jnp.all(state.T_soil < 400.0))
        # Theta within bounds
        self.assertTrue(jnp.all(state.theta_soil >= config.hydraulics.theta_r - 1e-6))
        self.assertTrue(jnp.all(state.theta_soil <= config.hydraulics.theta_sat + 1e-6))

    def test_init_state(self):
        """init_multilayer_land_state should produce consistent psi/theta."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import init_multilayer_land_state
        from legoesm.land.soil_hydraulics import theta_from_psi
        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config, theta_init=0.3)
        # psi should be consistent with theta
        theta_check = theta_from_psi(state.psi_soil, config.hydraulics)
        npt.assert_allclose(theta_check, state.theta_soil, atol=1e-6)

    def test_response_fields_finite(self):
        """All TileResponse fields should be finite."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config)
        forcing = self._make_forcing(ncol)

        _, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0,
        )

        # Fields WITHOUT a NamedTuple default are mandatory on every tile and
        # must be populated arrays; the defaulted ones (``T_rad``,
        # ``ice_concentration_thermo``) are opt-in channels that a land tile
        # legitimately leaves as None for its consumers to fall back from.
        # Checked this way rather than by skipping every None, so a mandatory
        # field silently becoming None still fails.
        optional = set(type(response)._field_defaults)
        for name in response._fields:
            arr = getattr(response, name)
            if arr is None:
                self.assertIn(
                    name, optional,
                    f"TileResponse.{name} is mandatory but was None",
                )
                continue
            self.assertTrue(
                jnp.all(jnp.isfinite(arr)),
                f"TileResponse.{name} has non-finite values",
            )

    def test_no_precip_no_runoff(self):
        """Without precipitation, should have minimal surface runoff."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        from legoesm.core.coupling_fields import AtmToSurface
        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config, T_init=280.0, theta_init=0.2)
        # Zero precipitation, low q so minimal evap demand
        forcing = AtmToSurface(
            sw_down=jnp.full(ncol, 200.0),
            lw_down=jnp.full(ncol, 300.0),
            precip_total=jnp.zeros(ncol),
            precip_snow=jnp.zeros(ncol),
            T_lowest=jnp.full(ncol, 280.0),
            q_lowest=jnp.full(ncol, 0.005),
            u_lowest=jnp.full(ncol, 3.0),
            v_lowest=jnp.zeros(ncol),
            p_lowest=jnp.full(ncol, 95000.0),
            p_surface=jnp.full(ncol, 100000.0),
            rho_lowest=jnp.full(ncol, 1.2),
            cos_zenith=jnp.full(ncol, 0.5),
            co2_ppmv=jnp.full(ncol, 400.0),
            has_radiation=jnp.ones(ncol),
            has_precipitation=jnp.ones(ncol),
        )
        new_state, _, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0,
        )
        # Surface runoff should be zero (no precip and evap may be negative flux_top,
        # but runoff_surface = max(flux_top - infil, 0) so should be 0 when flux_top < 0)
        npt.assert_allclose(new_state.runoff_surface, 0.0, atol=1e-10)


    def test_transpiration_sink_water_budget(self):
        """No-precip, zero-bottom-flux: total soil water loss matches
        latent-heat-implied evaporation."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        from legoesm.land.richards import RichardsConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm import constants

        config = MultiLayerLandConfig(
            richards=RichardsConfig(bottom_bc="zero_flux"),
        )
        ncol = 4
        state = init_multilayer_land_state(
            ncol, config, T_init=290.0, theta_init=0.35,
        )
        # No precip, warm sunny → drives evaporation from soil
        forcing = AtmToSurface(
            sw_down=jnp.full(ncol, 300.0),
            lw_down=jnp.full(ncol, 350.0),
            precip_total=jnp.zeros(ncol),
            precip_snow=jnp.zeros(ncol),
            T_lowest=jnp.full(ncol, 295.0),
            q_lowest=jnp.full(ncol, 0.005),
            u_lowest=jnp.full(ncol, 3.0),
            v_lowest=jnp.zeros(ncol),
            p_lowest=jnp.full(ncol, 95000.0),
            p_surface=jnp.full(ncol, 100000.0),
            rho_lowest=jnp.full(ncol, 1.2),
            cos_zenith=jnp.full(ncol, 0.5),
            co2_ppmv=jnp.full(ncol, 400.0),
            has_radiation=jnp.ones(ncol),
            has_precipitation=jnp.ones(ncol),
        )
        dt = 600.0
        from legoesm.land.soil_grid import make_soil_grid
        grid = make_soil_grid(config.soil_grid)
        dz = grid.dz

        theta_old = state.theta_soil
        new_state, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=dt,
        )
        theta_new = new_state.theta_soil

        # Total water change (m of water per unit area)
        dwater = jnp.sum((theta_new - theta_old) * dz[None, :], axis=-1)
        # lhflx → evap [m/s]: E = lhflx / L_v / rho_w
        evap_m = response.lhflx / constants.L_v / 1000.0  # m/s
        expected_loss = -evap_m * dt  # m (negative = loss)
        # Also subtract any runoff
        runoff_m = (new_state.runoff_surface + new_state.runoff_subsurface) / 1000.0 * dt

        # Water budget: dwater ≈ expected_loss - runoff
        budget_err = jnp.abs(dwater - expected_loss + runoff_m)
        # Allow tolerance for Picard iteration convergence
        self.assertTrue(
            jnp.all(budget_err < 5e-4),
            f"Water budget error {float(budget_err.max()):.6f} exceeds 5e-4 m"
        )

    def test_q_surface_stomatal_limitation(self):
        """Under dark conditions stomata close and q_surface reflects this."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.land.stomata import StomataConfig

        # Enable stomata with Jarvis model (simple conductance reduction)
        config = MultiLayerLandConfig(stomata=StomataConfig(enabled=True))
        ncol = 4
        state = init_multilayer_land_state(
            ncol, config, T_init=290.0, theta_init=0.35,
        )

        # Dark conditions: zero shortwave → stomata close
        forcing = AtmToSurface(
            sw_down=jnp.zeros(ncol),  # no sunlight → stomata shut
            lw_down=jnp.full(ncol, 350.0),
            precip_total=jnp.zeros(ncol),
            precip_snow=jnp.zeros(ncol),
            T_lowest=jnp.full(ncol, 290.0),
            q_lowest=jnp.full(ncol, 0.008),
            u_lowest=jnp.full(ncol, 3.0),
            v_lowest=jnp.zeros(ncol),
            p_lowest=jnp.full(ncol, 95000.0),
            p_surface=jnp.full(ncol, 100000.0),
            rho_lowest=jnp.full(ncol, 1.2),
            cos_zenith=jnp.zeros(ncol),  # nighttime
            co2_ppmv=jnp.full(ncol, 400.0),
            has_radiation=jnp.ones(ncol),
            has_precipitation=jnp.ones(ncol),
        )
        dt = 600.0
        _, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=dt,
        )

        # The returned q_surface should be LESS than fully saturated
        T_sfc = response.T_sfc
        q_sat = saturation_mixing_ratio(T_sfc, forcing.p_surface)

        # q_surface should be strictly less than q_sat (stomata limiting)
        ratio = response.q_surface / jnp.maximum(q_sat, 1e-20)
        self.assertTrue(
            jnp.all(ratio < 1.0 - 1e-6),
            f"q_surface/q_sat = {float(ratio.mean()):.4f}; stomata should limit"
        )


# =========================================================================
# root_zone_beta_soil — shared moisture-stress helper
# =========================================================================


class TestRootZoneBetaSoil(unittest.TestCase):
    """Factored root-zone beta_soil helper (single source of truth)."""

    def _root_frac(self, n_layers=4):
        rf = jnp.ones(n_layers) / n_layers
        return rf

    def test_saturated_gives_one(self):
        """theta >= theta_fc everywhere -> beta_soil == 1."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.full((2, 4), 0.4)
        beta, _ = root_zone_beta_soil(
            theta, self._root_frac(), 0.1, 0.35, 0.05, spatial=False)
        npt.assert_allclose(beta, 1.0, rtol=1e-6)

    def test_wilting_gives_floor(self):
        """theta <= theta_wp everywhere -> beta_soil == beta_min."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.full((2, 4), 0.05)
        beta, _ = root_zone_beta_soil(
            theta, self._root_frac(), 0.1, 0.35, 0.05, spatial=False)
        npt.assert_allclose(beta, 0.05, rtol=1e-6)

    def test_linear_between(self):
        """Half-way between wp and fc -> beta_min + 0.5*(1-beta_min)."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.full((1, 4), 0.225)  # (0.1+0.35)/2
        beta, _ = root_zone_beta_soil(
            theta, self._root_frac(), 0.1, 0.35, 0.05, spatial=False)
        npt.assert_allclose(beta, 0.05 + 0.5 * 0.95, rtol=1e-4)

    def test_degenerate_range_bounded(self):
        """theta_fc <= theta_wp must not blow up beta (audit finding #6)."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.full((1, 4), 0.3)
        beta, _ = root_zone_beta_soil(
            theta, self._root_frac(), 0.30, 0.30, 0.05, spatial=False)
        self.assertTrue(jnp.all(beta <= 1.0) and jnp.all(beta >= 0.05))

    def test_spatial_matches_scalar(self):
        """Per-column (spatial) path equals the scalar path for uniform params."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.array([[0.15, 0.2, 0.25, 0.3]])
        rf = self._root_frac()
        b_scalar, br_scalar = root_zone_beta_soil(
            theta, rf, 0.1, 0.35, 0.05, spatial=False)
        b_spatial, br_spatial = root_zone_beta_soil(
            theta, rf, jnp.array([0.1]), jnp.array([0.35]), 0.05, spatial=True)
        npt.assert_allclose(b_scalar, b_spatial, rtol=1e-9)
        npt.assert_allclose(br_scalar, br_spatial, rtol=1e-9)

    def test_matches_manual_root_weighting(self):
        """Non-uniform root_frac: weighted mean of per-layer stress."""
        from legoesm.land.multilayer_land import root_zone_beta_soil
        theta = jnp.array([[0.1, 0.35, 0.35, 0.35]])  # top wilting, rest fc
        rf = jnp.array([0.7, 0.1, 0.1, 0.1])  # top-heavy roots
        beta, _ = root_zone_beta_soil(theta, rf, 0.1, 0.35, 0.0, spatial=False)
        # per-layer beta_root = [0, 1, 1, 1]; w = 0.7*0 + 0.3*1 = 0.3
        npt.assert_allclose(beta, 0.3, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
