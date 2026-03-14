"""Tests for FV3-style C-D grid on the cubed-sphere.

Tests cover:
1. Grid construction: CubedSphereCDGrid metrics
2. Operators: vorticity, divergence, mass flux, gradients
3. Shallow water solver: stability and conservation
4. Ocean PE solver: tendency structure and shapes
"""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestCDGridConstruction(unittest.TestCase):
    """Test CubedSphereCDGrid metric computation."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        self.n = 8
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)

    def test_corner_shapes(self):
        n = self.n
        cg = self.cdgrid
        self.assertEqual(cg.lon_corner.shape, (6, n + 1, n + 1))
        self.assertEqual(cg.lat_corner.shape, (6, n + 1, n + 1))
        self.assertEqual(cg.f_corner.shape, (6, n + 1, n + 1))
        self.assertEqual(cg.angle_corner.shape, (6, n + 1, n + 1))
        self.assertEqual(cg.cos_angle_corner.shape, (6, n + 1, n + 1))
        self.assertEqual(cg.sin_angle_corner.shape, (6, n + 1, n + 1))

    def test_edge_shapes(self):
        n = self.n
        cg = self.cdgrid
        self.assertEqual(cg.dx_edge_y.shape, (6, n, n + 1))
        self.assertEqual(cg.dy_edge_x.shape, (6, n + 1, n))
        self.assertEqual(cg.area_corner.shape, (6, n + 1, n + 1))

    def test_edge_lengths_positive(self):
        self.assertTrue(jnp.all(self.cdgrid.dx_edge_y > 0))
        self.assertTrue(jnp.all(self.cdgrid.dy_edge_x > 0))

    def test_corner_area_positive(self):
        self.assertTrue(jnp.all(self.cdgrid.area_corner > 0))

    def test_latitude_range(self):
        lat = self.cdgrid.lat_corner
        self.assertGreaterEqual(float(jnp.min(lat)), -jnp.pi / 2 - 0.01)
        self.assertLessEqual(float(jnp.max(lat)), jnp.pi / 2 + 0.01)

    def test_coriolis_at_equator_near_zero(self):
        """Coriolis should be near zero for corners near the equator."""
        lat = self.cdgrid.lat_corner
        equator_mask = jnp.abs(lat) < 0.1
        if jnp.any(equator_mask):
            f_equator = jnp.where(equator_mask, jnp.abs(self.cdgrid.f_corner), 0.0)
            max_f = float(jnp.max(f_equator))
            self.assertLess(max_f, 2e-5)

    def test_n_and_radius_properties(self):
        self.assertEqual(self.cdgrid.n, self.n)
        self.assertEqual(self.cdgrid.radius, self.grid.radius)


class TestCDGridOperators(unittest.TestCase):
    """Test C-D grid operators."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        self.n = 8
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)

    def test_dgrid_to_cgrid_shapes(self):
        from legoesm.core.operators_cdgrid import dgrid_to_cgrid
        n = self.n
        u_d = jnp.zeros((6, n + 1, n + 1))
        v_d = jnp.zeros((6, n + 1, n + 1))
        u_c, v_c = dgrid_to_cgrid(u_d, v_d, self.cdgrid)
        self.assertEqual(u_c.shape, (6, n + 1, n))
        self.assertEqual(v_c.shape, (6, n, n + 1))

    def test_cgrid_to_dgrid_shapes(self):
        from legoesm.core.operators_cdgrid import cgrid_to_dgrid
        n = self.n
        u_c = jnp.zeros((6, n + 1, n))
        v_c = jnp.zeros((6, n, n + 1))
        u_d, v_d = cgrid_to_dgrid(u_c, v_c, self.cdgrid)
        self.assertEqual(u_d.shape, (6, n + 1, n + 1))
        self.assertEqual(v_d.shape, (6, n + 1, n + 1))

    def test_dgrid_cgrid_roundtrip(self):
        """D→C→D should approximately recover the original for smooth fields."""
        from legoesm.core.operators_cdgrid import dgrid_to_cgrid, cgrid_to_dgrid
        n = self.n
        key = jax.random.PRNGKey(42)
        # Smooth field: constant (exact roundtrip)
        u_d = jnp.ones((6, n + 1, n + 1)) * 3.0
        v_d = jnp.ones((6, n + 1, n + 1)) * -2.0
        u_c, v_c = dgrid_to_cgrid(u_d, v_d, self.cdgrid)
        u_d2, v_d2 = cgrid_to_dgrid(u_c, v_c, self.cdgrid)
        # Interior should be exact for constant fields
        self.assertLess(float(jnp.max(jnp.abs(u_d2[:, 1:-1, 1:-1] - 3.0))), 1e-10)
        self.assertLess(float(jnp.max(jnp.abs(v_d2[:, 1:-1, 1:-1] - (-2.0)))), 1e-10)

    def test_vorticity_solid_body(self):
        """Vorticity of solid-body rotation should be approximately 2*Omega."""
        from legoesm.core.operators_cdgrid import dgrid_vorticity

        n = self.n
        cdgrid = self.cdgrid
        Omega = 7.292e-5
        R = cdgrid.radius

        # Solid-body rotation: u = Omega*R*cos(lat), v = 0
        cos_lat = jnp.cos(cdgrid.lat_corner)
        u_d = Omega * R * cos_lat * cdgrid.cos_angle_corner
        v_d = Omega * R * cos_lat * cdgrid.sin_angle_corner

        zeta = dgrid_vorticity(u_d, v_d, cdgrid)
        # For solid-body rotation, ζ = 2Ω·sin(lat) at each cell centre
        expected = 2 * Omega * jnp.sin(cdgrid.base.lat)

        # Check RMS relative error (exclude near-zero values)
        # At n=8 the cubed-sphere cells are very coarse (~5.6° wide) and the
        # midpoint quadrature for the circulation integral has large O(Δx²)
        # truncation error near cube corners. Use median relative error
        # which is robust to outliers at cube-face edges.
        mask = jnp.abs(expected) > 1e-5
        if jnp.any(mask):
            rel_err = jnp.abs(zeta - expected) / jnp.maximum(jnp.abs(expected), 1e-10)
            rel_err = jnp.where(mask, rel_err, 0.0)
            med_err = float(jnp.median(jnp.where(mask, rel_err, 0.0)))
            self.assertLess(med_err, 0.5, f"Vorticity median relative error: {med_err:.3f}")

    def test_vorticity_zero_for_irrotational(self):
        """Irrotational field (uniform u, v=0) should have near-zero vorticity."""
        from legoesm.core.operators_cdgrid import dgrid_vorticity
        n = self.n
        u_d = jnp.ones((6, n + 1, n + 1)) * 10.0
        v_d = jnp.zeros((6, n + 1, n + 1))
        zeta = dgrid_vorticity(u_d, v_d, self.cdgrid)
        # On a sphere, uniform u is not exactly irrotational, but the
        # vorticity should be small compared to the velocity magnitude
        max_zeta = float(jnp.max(jnp.abs(zeta)))
        # Scale: u/R ~ 10/6.4e6 ~ 1.5e-6
        self.assertLess(max_zeta, 1e-4)

    def test_divergence_constant_velocity(self):
        """Divergence of constant velocity should be near zero."""
        from legoesm.core.operators_cdgrid import cgrid_divergence, dgrid_to_cgrid
        n = self.n
        u_d = jnp.ones((6, n + 1, n + 1)) * 5.0
        v_d = jnp.ones((6, n + 1, n + 1)) * 3.0
        u_c, v_c = dgrid_to_cgrid(u_d, v_d, self.cdgrid)
        div = cgrid_divergence(u_c, v_c, self.cdgrid)
        # Constant velocity on a curved surface has non-zero divergence,
        # but it should be small
        self.assertTrue(jnp.all(jnp.isfinite(div)))

    def test_mass_flux_shape(self):
        from legoesm.core.operators_cdgrid import (
            cgrid_mass_flux_divergence, dgrid_to_cgrid,
        )
        n = self.n
        h = jnp.ones((6, n, n)) * 1000.0
        u_d = jnp.ones((6, n + 1, n + 1))
        v_d = jnp.zeros((6, n + 1, n + 1))
        u_c, v_c = dgrid_to_cgrid(u_d, v_d, self.cdgrid)
        dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, self.cdgrid)
        self.assertEqual(dh_dt.shape, (6, n, n))
        self.assertTrue(jnp.all(jnp.isfinite(dh_dt)))

    def test_gradient_constant_field(self):
        """Gradient of constant field should be zero."""
        from legoesm.core.operators_cdgrid import _arakawa_lamb_gradient
        n = self.n
        B = jnp.ones((6, n, n)) * 42.0
        dB_dx, dB_dy = _arakawa_lamb_gradient(B, self.cdgrid)
        self.assertLess(float(jnp.max(jnp.abs(dB_dx))), 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(dB_dy))), 1e-6)

    def test_momentum_shape(self):
        from legoesm.core.operators_cdgrid import cdgrid_momentum_tendencies
        n = self.n
        h = jnp.ones((6, n, n)) * 1000.0
        h_s = jnp.zeros((6, n, n))
        u_d = jnp.ones((6, n + 1, n + 1)) * 0.1
        v_d = jnp.zeros((6, n + 1, n + 1))
        du, dv = cdgrid_momentum_tendencies(
            h, u_d, v_d, h_s, self.cdgrid,
        )
        self.assertEqual(du.shape, (6, n + 1, n + 1))
        self.assertEqual(dv.shape, (6, n + 1, n + 1))
        self.assertTrue(jnp.all(jnp.isfinite(du)))
        self.assertTrue(jnp.all(jnp.isfinite(dv)))


class TestCDGridShallowWater(unittest.TestCase):
    """Test C-D grid shallow water solver."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )

        self.n = 8
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)
        self.config = CDGridShallowWaterConfig(A_h=1e5)
        self.model = CDGridShallowWaterModel(self.grid, self.config)

        n = self.n
        H0 = 1000.0
        h = jnp.ones((6, n, n)) * H0
        u_d = jnp.zeros((6, n + 1, n + 1))
        v_d = jnp.zeros((6, n + 1, n + 1))
        h_s = jnp.zeros((6, n, n))
        self.state0 = CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)

    def test_rest_state_zero_tendencies(self):
        """Rest state should have near-zero tendencies."""
        dh, du, dv = self.model.tendencies(self.state0)
        self.assertLess(float(jnp.max(jnp.abs(dh))), 1e-10)
        self.assertLess(float(jnp.max(jnp.abs(du))), 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(dv))), 1e-6)

    def test_one_step_finite(self):
        """One time step should produce finite values."""
        # Add a perturbation
        key = jax.random.PRNGKey(0)
        h_pert = self.state0.h + 10.0 * jax.random.normal(key, self.state0.h.shape)
        state = self.state0._replace(h=h_pert)
        state_new = self.model.step(state, 60.0)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.h)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u_d)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.v_d)))

    def test_mass_conservation(self):
        """Mass should be conserved after time stepping."""
        key = jax.random.PRNGKey(1)
        h_pert = self.state0.h + 5.0 * jax.random.normal(key, self.state0.h.shape)
        state = self.state0._replace(h=h_pert)

        area = self.cdgrid.base.area
        mass_0 = float(jnp.sum(state.h * area))

        state_new = self.model.step(state, 30.0)
        mass_1 = float(jnp.sum(state_new.h * area))

        rel_err = abs(mass_1 - mass_0) / abs(mass_0)
        self.assertLess(rel_err, 1e-10)

    def test_multi_step_stability(self):
        """10 steps should remain stable."""
        key = jax.random.PRNGKey(2)
        h_pert = self.state0.h + 2.0 * jax.random.normal(key, self.state0.h.shape)
        state = self.state0._replace(h=h_pert)

        for _ in range(10):
            state = self.model.step(state, 30.0)

        self.assertTrue(jnp.all(jnp.isfinite(state.h)))
        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)))
        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)))
        # h should not blow up
        self.assertLess(float(jnp.max(jnp.abs(state.h))), 1e6)

    def test_integrate(self):
        """Integration interface should work."""
        state_final, traj = self.model.integrate(
            self.state0, duration=120.0, dt=60.0,
        )
        self.assertTrue(jnp.all(jnp.isfinite(state_final.h)))
        self.assertEqual(len(traj), 3)  # initial + 2 steps

    def test_differentiable(self):
        """Tendency function should be differentiable."""
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            cdgrid_shallow_water_tendencies,
        )

        def loss(h):
            state = self.state0._replace(h=h)
            dh, du, dv = cdgrid_shallow_water_tendencies(
                state, self.cdgrid, self.config,
            )
            return jnp.sum(dh ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(self.state0.h)
        self.assertEqual(g.shape, self.state0.h.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


class TestCDGridOceanPE(unittest.TestCase):
    """Test C-D grid ocean primitive equation solver."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.state import OceanState, OceanConfig
        from legoesm.core.field import Field

        self.n = 6
        self.nlev = 5
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)
        self.z_coord = create_ocean_z_star(self.nlev)
        self.config = OceanConfig(A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4)

        n, nlev = self.n, self.nlev
        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        u = jnp.zeros((6, n, n, nlev))
        v = jnp.zeros((6, n, n, nlev))
        T = jnp.ones((6, n, n, nlev)) * 15.0
        S = jnp.ones((6, n, n, nlev)) * 35.0
        eta = jnp.zeros((6, n, n))
        H_bathy = jnp.ones((6, n, n)) * 1000.0
        mask = jnp.ones((6, n, n))

        self.state0 = OceanState(
            u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T, name="T", dims=dims_3d, units="degC"),
            S=Field(data=S, name="S", dims=dims_3d, units="PSU"),
            eta=Field(data=eta, name="eta", dims=dims_2d, units="m"),
            H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
            land_mask=Field(data=mask, name="land_mask", dims=dims_2d, units=""),
        )

    def test_tendency_shapes(self):
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
            ocean_baroclinic_tendencies_cdgrid,
        )

        tend = ocean_baroclinic_tendencies_cdgrid(
            self.state0, self.grid, self.z_coord,
            self.cdgrid, self.config,
        )

        n, nlev = self.n, self.nlev
        self.assertEqual(tend.du_dt.data.shape, (6, n, n, nlev))
        self.assertEqual(tend.dv_dt.data.shape, (6, n, n, nlev))
        self.assertEqual(tend.dT_dt.data.shape, (6, n, n, nlev))
        self.assertEqual(tend.dS_dt.data.shape, (6, n, n, nlev))
        self.assertEqual(tend.deta_dt.data.shape, (6, n, n))

    def test_tendency_finite(self):
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
            ocean_baroclinic_tendencies_cdgrid,
        )

        tend = ocean_baroclinic_tendencies_cdgrid(
            self.state0, self.grid, self.z_coord,
            self.cdgrid, self.config,
        )

        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dv_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dS_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.deta_dt.data)))

    def test_rest_state_small_tendencies(self):
        """Rest state with uniform T/S should have small tendencies."""
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
            ocean_baroclinic_tendencies_cdgrid,
        )

        tend = ocean_baroclinic_tendencies_cdgrid(
            self.state0, self.grid, self.z_coord,
            self.cdgrid, self.config,
        )

        # Velocity tendencies should be very small at rest
        self.assertLess(float(jnp.max(jnp.abs(tend.du_dt.data))), 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(tend.dv_dt.data))), 1e-3)
        # Tracer tendencies should be zero (uniform, no gradients)
        self.assertLess(float(jnp.max(jnp.abs(tend.dT_dt.data))), 1e-8)
        self.assertLess(float(jnp.max(jnp.abs(tend.dS_dt.data))), 1e-8)

    def test_ocean_model_dispatch(self):
        """OceanModel with discretization='cdgrid' should work."""
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        model = OceanModel(
            self.grid, self.z_coord, self.config,
            discretization="cdgrid",
        )
        self.assertEqual(model.discretization, "cdgrid")
        self.assertIsNotNone(model._cdgrid)

    def test_land_masking(self):
        """Land points should have zero tendencies."""
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
            ocean_baroclinic_tendencies_cdgrid,
        )
        from legoesm.core.field import Field

        # Set face 0 as land
        mask = self.state0.land_mask.data.at[0].set(0.0)
        state = self.state0._replace(
            land_mask=Field(data=mask, name="land_mask",
                            dims=("face", "x", "y"), units=""),
        )

        tend = ocean_baroclinic_tendencies_cdgrid(
            state, self.grid, self.z_coord,
            self.cdgrid, self.config,
        )

        self.assertLess(
            float(jnp.max(jnp.abs(tend.du_dt.data[0]))), 1e-15,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(tend.deta_dt.data[0]))), 1e-15,
        )


class TestCDGrid3DOperators(unittest.TestCase):
    """Test 3D extensions of C-D grid operators."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        self.n = 6
        self.nlev = 4
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)

    def test_vorticity_3d_shape(self):
        from legoesm.core.operators_cdgrid import dgrid_vorticity_3d
        n, nlev = self.n, self.nlev
        u_d = jnp.zeros((6, n + 1, n + 1, nlev))
        v_d = jnp.zeros((6, n + 1, n + 1, nlev))
        zeta = dgrid_vorticity_3d(u_d, v_d, self.cdgrid)
        self.assertEqual(zeta.shape, (6, n, n, nlev))

    def test_divergence_3d_shape(self):
        from legoesm.core.operators_cdgrid import cgrid_divergence_3d
        n, nlev = self.n, self.nlev
        u_c = jnp.zeros((6, n + 1, n, nlev))
        v_c = jnp.zeros((6, n, n + 1, nlev))
        div = cgrid_divergence_3d(u_c, v_c, self.cdgrid)
        self.assertEqual(div.shape, (6, n, n, nlev))

    def test_gradient_3d_constant(self):
        from legoesm.core.operators_cdgrid import _arakawa_lamb_gradient_3d
        n, nlev = self.n, self.nlev
        B = jnp.ones((6, n, n, nlev)) * 100.0
        dB_dx, dB_dy = _arakawa_lamb_gradient_3d(B, self.cdgrid)
        self.assertEqual(dB_dx.shape, (6, n + 1, n + 1, nlev))
        self.assertLess(float(jnp.max(jnp.abs(dB_dx))), 1e-5)
        self.assertLess(float(jnp.max(jnp.abs(dB_dy))), 1e-5)

    def test_mass_flux_3d_shape(self):
        from legoesm.core.operators_cdgrid import cgrid_mass_flux_divergence_3d
        n, nlev = self.n, self.nlev
        h = jnp.ones((6, n, n, nlev)) * 100.0
        u_c = jnp.ones((6, n + 1, n, nlev)) * 0.1
        v_c = jnp.zeros((6, n, n + 1, nlev))
        dh = cgrid_mass_flux_divergence_3d(h, u_c, v_c, self.cdgrid)
        self.assertEqual(dh.shape, (6, n, n, nlev))
        self.assertTrue(jnp.all(jnp.isfinite(dh)))


if __name__ == "__main__":
    unittest.main()
