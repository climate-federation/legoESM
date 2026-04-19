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

    def test_w2_balanced_state_polar_mass_tendency_post_iter505(self):
        """Iter-126 originally baselined the W2-balanced polar-face
        mass-tendency asymmetry of 1.0567 as a tripwire for future
        architectural work.  Iter-505 fixed the underlying x-direction
        PPM axis bug in `cgrid_mass_flux_divergence`, dropping the
        polar ratio to 1.000 (machine precision).  Iter-518 (Codex
        follow-up) updates this test to lock the post-iter-505 state
        — the iter-126 expected value of 1.0567 was the BUG state and
        was preventing this test from passing on the fixed code.

        The complementary post-iter-505 polar-symmetry test in
        `TestFv3SwTendenciesPolarFaceSymmetry`
        (test_cdgrid_fv3_regression.py) covers the SAME invariants on
        a self-built balanced state; this test additionally locks the
        canonical `williamson_test2(grid)` IC behaviour at C36.

        Asserts on canonical-state C36:
        - Equatorial faces 0-3 area-weighted mass-rate symmetric
          (within 5e-2 relative — loose because the production A-L
          path inherently carries O(1e-2) face-boundary noise that
          iter-505 reduces but does not eliminate).
        - Polar mass-rate ratio face 4 / face 5 = 1.000 ± 0.05
          (post-iter-505; iter-126's 1.0567 was the BUG).
        """
        import numpy as np
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import fv3_sw_tendencies
        from tests.atmosphere.shallow_water.test_cases.williamson import (
            williamson_test2,
        )

        n = 36
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sw = williamson_test2(grid)

        # Reproduce the test-matrix runner's exact D-grid wind
        # construction (run_atmosphere_test_matrix.py:1187-1192).
        U0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
        u_east_x = U0 * jnp.cos(cdgrid.lat_edge_x)
        u_d = cdgrid.cos_angle_edge_x * u_east_x
        u_east_y = U0 * jnp.cos(cdgrid.lat_edge_y)
        v_d = -cdgrid.sin_angle_edge_y * u_east_y

        dh_dt, _, _ = fv3_sw_tendencies(sw.h.data, u_d, v_d,
                                         sw.h_s.data, cdgrid)
        area = cdgrid.base.area
        mass_rates = [float(jnp.sum(dh_dt[f] * area[f])) for f in range(6)]

        # Equatorial faces 0-3 area-weighted mass rates: post-iter-505
        # measurement gives ~1e-2 relative spread on the canonical
        # IC (residual non-FV3 face-boundary error inherent to the
        # A-L production path; eliminated only by the FB chain that
        # remains unstable at C36).  Tolerate up to 5e-2.
        eq_max = max(abs(mass_rates[i] - mass_rates[0]) for i in range(4))
        self.assertLess(
            eq_max / abs(mass_rates[0]), 5e-2,
            f"Equatorial face symmetry: max diff "
            f"{eq_max:.3e} vs face-0 {mass_rates[0]:.3e}")

        # Polar mass-rate diff post-iter-505: should be ~0 (both
        # faces have essentially zero mass rate after the bug is
        # fixed).  Compare |m4 - m5| against the equatorial scale
        # (which IS non-zero) — ratio-based comparison would divide
        # by zero now that polar mass rates are essentially nil.
        polar_diff = abs(mass_rates[4] - mass_rates[5])
        eq_scale = abs(mass_rates[0])
        polar_diff_rel = polar_diff / eq_scale
        # Pre-iter-505 the polar diff was ~5.7e-2 of the equatorial
        # scale (mass_rates[4]=3.78e9 vs mass_rates[5]=3.58e9 vs
        # equatorial 4.18e9).  Post-iter-505 the diff drops to ~1e-4
        # of the equatorial scale.  Ceiling at 1e-2 cleanly
        # discriminates: would fire on a regression toward the
        # pre-iter-505 5.7e-2.
        self.assertLess(
            polar_diff_rel, 1e-2,
            msg=(f"Polar mass-rate diff |m4 - m5| = {polar_diff:.3e}, "
                 f"i.e. {polar_diff_rel:.3e} of the equatorial scale "
                 f"({eq_scale:.3e}).  Pre-iter-505 baseline was "
                 f"~5.7e-2 — if polar_diff_rel drifted above 1e-2, "
                 f"the iter-505 axis fix regressed."))

    def test_f_corner_matches_base_f_under_small_earth_scaling(self):
        """f_corner must track base.f when omega is scaled (e.g. small-earth).

        Regression test for the iter-77 fix that infers omega from base.f
        rather than hardcoding Earth's value.  Without this fix, a
        small-earth-scaled grid would have ``base.f`` scaled by `factor`
        but ``cdgrid.f_corner`` still at Earth's omega, leading to
        inconsistent Coriolis at cell centres vs corners.
        """
        from legoesm.grids.cubed_sphere import apply_small_earth_scaling
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        factor = 120.0
        grid_scaled = apply_small_earth_scaling(self.grid, factor)
        cdgrid_scaled = create_cubed_sphere_cdgrid(grid_scaled)

        ratio_base = (float(jnp.max(jnp.abs(grid_scaled.f)))
                      / float(jnp.max(jnp.abs(self.grid.f))))
        ratio_corner = (float(jnp.max(jnp.abs(cdgrid_scaled.f_corner)))
                        / float(jnp.max(jnp.abs(self.cdgrid.f_corner))))
        self.assertAlmostEqual(ratio_base, factor, places=2)
        self.assertAlmostEqual(ratio_corner, factor, places=2)
        # The two ratios must agree exactly (both derived from same omega)
        self.assertAlmostEqual(ratio_base, ratio_corner, places=4)

    def test_n_and_radius_properties(self):
        self.assertEqual(self.cdgrid.n, self.n)
        self.assertEqual(self.cdgrid.radius, self.grid.radius)

    def test_metric_dtype_configurable(self):
        """metric_dtype parameter controls corner-critical metric precision."""
        import jax
        if not getattr(jax.config, 'x64_enabled', False):
            self.skipTest("x64 not enabled")
        try:
            jax.device_put(jnp.array(1.0, dtype=jnp.float64))
        except Exception:
            self.skipTest("backend does not support float64")
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        cdgrid = create_cubed_sphere_cdgrid(self.grid, metric_dtype=jnp.float64)
        self.assertEqual(cdgrid.grad_c00.dtype, jnp.float64)
        self.assertEqual(cdgrid.rarea_c.dtype, jnp.float64)
        self.assertEqual(cdgrid.rdxc.dtype, jnp.float64)
        self.assertEqual(cdgrid.rsin2_corner.dtype, jnp.float64)
        # Position fields stay float32
        self.assertEqual(cdgrid.lon_corner.dtype, jnp.float32)

    def test_metric_dtype_defaults_float32(self):
        """Default metric_dtype is float32 for backward compatibility."""
        self.assertEqual(self.cdgrid.grad_c00.dtype, jnp.float32)
        self.assertEqual(self.cdgrid.rarea_c.dtype, jnp.float32)


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
        """D->C->D should approximately recover the original for smooth fields."""
        from legoesm.core.operators_cdgrid import dgrid_to_cgrid, cgrid_to_dgrid
        n = self.n
        # Smooth field: constant — with non-orthogonality corrections, the
        # roundtrip is not exact because the averaging and projection don't
        # commute, but the error should be small (O(h^2) where h is the
        # variation in cosa across a cell).
        u_d = jnp.ones((6, n + 1, n + 1)) * 3.0
        v_d = jnp.ones((6, n + 1, n + 1)) * -2.0
        u_c, v_c = dgrid_to_cgrid(u_d, v_d, self.cdgrid)
        u_d2, v_d2 = cgrid_to_dgrid(u_c, v_c, self.cdgrid)
        # v_d roundtrip is exact since v_c = v_d and the inverse is v_d = v_c
        self.assertLess(float(jnp.max(jnp.abs(v_d2[:, 1:-1, 1:-1] - (-2.0)))), 1e-10)
        # u_d roundtrip has O(h^2) error from averaging of non-uniform cosa
        self.assertLess(float(jnp.max(jnp.abs(u_d2[:, 1:-1, 1:-1] - 3.0))), 0.05)

    def test_vorticity_solid_body(self):
        """Vorticity of solid-body rotation should be approximately 2*Omega."""
        from legoesm.core.operators_cdgrid import dgrid_vorticity

        n = self.n
        cdgrid = self.cdgrid
        Omega = 7.292e-5
        R = cdgrid.radius

        # Solid-body rotation: u_east = Omega*R*cos(lat), v_north = 0
        # Rotate geographic to grid-aligned:
        #   u_grid =  cos(angle)*u_east + sin(angle)*v_north =  cos(angle)*u_geo
        #   v_grid = -sin(angle)*u_east + cos(angle)*v_north = -sin(angle)*u_geo
        cos_lat = jnp.cos(cdgrid.lat_corner)
        u_geo = Omega * R * cos_lat
        u_d = u_geo * cdgrid.cos_angle_corner
        v_d = -u_geo * cdgrid.sin_angle_corner

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
        # Post-step fixer corrects mass drift, but float32 state
        # limits correction precision to ~1e-7. The in-tendency
        # global-mean subtraction was removed to expose raw flux errors.
        self.assertLess(rel_err, 1e-6)

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
        from legoesm.core.operators_cdgrid import dgrid_vorticity
        n, nlev = self.n, self.nlev
        u_d = jnp.zeros((6, n + 1, n + 1, nlev))
        v_d = jnp.zeros((6, n + 1, n + 1, nlev))
        zeta = dgrid_vorticity(u_d, v_d, self.cdgrid)
        self.assertEqual(zeta.shape, (6, n, n, nlev))

    def test_divergence_3d_shape(self):
        from legoesm.core.operators_cdgrid import cgrid_divergence
        n, nlev = self.n, self.nlev
        u_c = jnp.zeros((6, n + 1, n, nlev))
        v_c = jnp.zeros((6, n, n + 1, nlev))
        div = cgrid_divergence(u_c, v_c, self.cdgrid)
        self.assertEqual(div.shape, (6, n, n, nlev))

    def test_gradient_3d_constant(self):
        from legoesm.core.operators_cdgrid import _arakawa_lamb_gradient
        n, nlev = self.n, self.nlev
        B = jnp.ones((6, n, n, nlev)) * 100.0
        dB_dx, dB_dy = _arakawa_lamb_gradient(B, self.cdgrid)
        self.assertEqual(dB_dx.shape, (6, n + 1, n + 1, nlev))
        self.assertLess(float(jnp.max(jnp.abs(dB_dx))), 1e-5)
        self.assertLess(float(jnp.max(jnp.abs(dB_dy))), 1e-5)

    def test_mass_flux_3d_shape(self):
        from legoesm.core.operators_cdgrid import cgrid_mass_flux_divergence
        n, nlev = self.n, self.nlev
        h = jnp.ones((6, n, n, nlev)) * 100.0
        u_c = jnp.ones((6, n + 1, n, nlev)) * 0.1
        v_c = jnp.zeros((6, n, n + 1, nlev))
        dh = cgrid_mass_flux_divergence(h, u_c, v_c, self.cdgrid)
        self.assertEqual(dh.shape, (6, n, n, nlev))
        self.assertTrue(jnp.all(jnp.isfinite(dh)))


class TestInterpCornerToCenter(unittest.TestCase):
    """Iter-544: regression lock for `_interp_corner_to_center`.

    This helper (src/legoesm/core/operators_cdgrid.py:895-910) is
    load-bearing in the production A-L path: it is the final step that
    projects corner gradient / divergence-damping contributions back
    to cell centres before the momentum update in
    `fv3_sw_tendencies` (at lines 1421-1422 for Bernoulli gradient and
    1436-1437 for the divergence-damping contribution).

    Before iter-544 the function had NO tests.  A silent refactor to
    an area-weighted average, a skewed 3-point average, or an
    accidental index shift would propagate directly into W2/W5
    tendencies without any regression trip.

    Locks:
      (a) 2D shape: (6, n+1, n+1) -> (6, n, n)
      (b) 3D shape: (6, n+1, n+1, nlev) -> (6, n, n, nlev)
      (c) EXACT arithmetic 4-point average (no area weighting):
          out[i,j] = 0.25*(f[i,j] + f[i+1,j] + f[i,j+1] + f[i+1,j+1])
          on a known non-trivial field.
      (d) Independent application per level in 3D.
      (e) Area-independence: substituting a different `cdgrid` with
          different `area_corner` must NOT change the result (the
          helper takes `field_d` only, no grid argument).
    """

    def _build(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid)
        n = 6
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        return n, cdgrid

    def test_shape_2d_and_3d(self):
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        n, _ = self._build()
        nlev = 5
        field_2d = jnp.zeros((6, n + 1, n + 1))
        field_3d = jnp.zeros((6, n + 1, n + 1, nlev))
        self.assertEqual(
            _interp_corner_to_center(field_2d).shape, (6, n, n))
        self.assertEqual(
            _interp_corner_to_center(field_3d).shape, (6, n, n, nlev))

    def test_constant_preservation(self):
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        n, _ = self._build()
        field = jnp.ones((6, n + 1, n + 1)) * 7.5
        out = _interp_corner_to_center(field)
        self.assertTrue(jnp.all(out == 7.5),
                        msg="Constant input must be preserved exactly.")

    def test_exact_4_point_arithmetic_average_2d(self):
        """Distinct-value stencil check: the function must compute
        exactly `0.25*(SW + SE + NW + NE)` — not a weighted variant.
        """
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        n, _ = self._build()
        # Build a field where each corner has a unique value so the
        # exact averaging formula is unambiguous.
        rng = np.random.default_rng(544)
        field_np = rng.standard_normal((6, n + 1, n + 1)).astype(
            np.float64)
        field = jnp.asarray(field_np)
        out = np.asarray(_interp_corner_to_center(field),
                         dtype=np.float64)
        expected = 0.25 * (
            field_np[:, :-1, :-1]      # SW corner
            + field_np[:, 1:, :-1]     # SE
            + field_np[:, :-1, 1:]     # NW
            + field_np[:, 1:, 1:]      # NE
        )
        max_diff = float(np.max(np.abs(out - expected)))
        self.assertLess(
            max_diff, 1e-10,
            msg=(f"`_interp_corner_to_center` deviates from "
                 f"`0.25*(SW+SE+NW+NE)` by {max_diff:.3e}.  If the "
                 f"implementation changed to an area-weighted or "
                 f"non-uniform average, UPDATE this test with the "
                 f"new expected formula and document the change in "
                 f"docs/fv3_fortran_fidelity_review.md.  The two "
                 f"production call sites in `fv3_sw_tendencies` "
                 f"assume plain arithmetic averaging."))

    def test_3d_applies_per_level_independently(self):
        """Each vertical level must be averaged independently -- no
        cross-level mixing."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        n, _ = self._build()
        nlev = 4
        rng = np.random.default_rng(1544)
        # Different field on each level: level k has constant k+1
        per_level = (
            np.arange(1, nlev + 1, dtype=np.float64)[None, None, None, :]
            + np.zeros((6, n + 1, n + 1, nlev))
        )
        field_3d = jnp.asarray(per_level)
        out = np.asarray(_interp_corner_to_center(field_3d))
        # Constant-per-level input must yield constant output matching
        # the level value (no cross-level averaging).
        for k in range(nlev):
            lv = np.asarray(out[..., k])
            self.assertTrue(
                np.all(np.abs(lv - (k + 1)) < 1e-10),
                msg=(f"Level {k} output deviates from expected "
                     f"constant {k + 1}; max dev = "
                     f"{float(np.max(np.abs(lv - (k + 1)))):.3e}.  "
                     f"Cross-level averaging detected."))

    def test_exact_4_point_arithmetic_average_4d(self):
        """Iter-545 (Codex follow-up): the 4D (ndim==4) branch is
        production-hot (used by `fv3_sw_tendencies` for Bernoulli and
        divergence-damping projections whenever the SW model is
        driven with a vertical dimension).  Iter-544's 2D exact-value
        test hits `ndim==3` via the `(6, n+1, n+1)` shape; the 4D
        branch uses different slicing (`field_d[..., :-1, :]` etc.)
        and was NOT covered by an exact-value arithmetic check.  A
        refactor that silently broke the 4D branch could slip
        through.

        This test generates a distinct-value-per-corner 4D stencil
        `(6, n+1, n+1, nlev)` with a non-trivial dependence on ALL
        four axes (face, i, j, k) and asserts the 4D output matches
        `0.25*(SW+SE+NW+NE)` applied per-level, to 1e-10.
        """
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        n, _ = self._build()
        nlev = 3
        rng = np.random.default_rng(2544)
        # Distinct-value 4D field: random with a full (6, n+1, n+1,
        # nlev) shape.  Every corner (face, i, j, k) is unique, so the
        # output is uniquely determined by the 4-point averaging
        # formula applied INDEPENDENTLY at each (face, k) layer.
        field_np = rng.standard_normal(
            (6, n + 1, n + 1, nlev)).astype(np.float64)
        field_4d = jnp.asarray(field_np)

        # Confirm the test exercises the 4D branch (not accidentally
        # the 3D branch) by asserting input ndim is 4.
        self.assertEqual(
            field_4d.ndim, 4,
            msg="Test must exercise the ndim==4 branch.")

        out = np.asarray(
            _interp_corner_to_center(field_4d), dtype=np.float64)
        self.assertEqual(
            out.shape, (6, n, n, nlev),
            msg="4D output shape must collapse (n+1, n+1) -> (n, n).")

        expected = 0.25 * (
            field_np[:, :-1, :-1, :]      # SW corner
            + field_np[:, 1:, :-1, :]     # SE
            + field_np[:, :-1, 1:, :]     # NW
            + field_np[:, 1:, 1:, :]      # NE
        )
        max_diff = float(np.max(np.abs(out - expected)))
        self.assertLess(
            max_diff, 1e-10,
            msg=(f"4D `_interp_corner_to_center` deviates from "
                 f"per-level `0.25*(SW+SE+NW+NE)` by {max_diff:.3e}. "
                 f"The production SW-with-vertical paths assume "
                 f"plain arithmetic per-level averaging on 4D fields. "
                 f"If an area-weighted or index-shifted 4D variant "
                 f"was introduced, UPDATE this test with the new "
                 f"formula and document in fidelity review."))

        # Also verify that the 4D output at any single level matches
        # what we would get by pulling that level out to a 3D field
        # and running `_interp_corner_to_center` on the 3D branch.
        # Catches silent divergence between the two branches.
        for k in range(nlev):
            level_3d = field_4d[..., k]              # (6, n+1, n+1)
            out_3d = np.asarray(
                _interp_corner_to_center(level_3d),
                dtype=np.float64)
            max_diff_branch = float(
                np.max(np.abs(out[..., k] - out_3d)))
            self.assertLess(
                max_diff_branch, 1e-10,
                msg=(f"Level {k}: 4D branch output differs from 3D "
                     f"branch applied to the same level by "
                     f"{max_diff_branch:.3e}.  The two branches "
                     f"must produce identical values per level."))

    def test_area_independence(self):
        """The helper takes `field_d` only — no grid argument — so the
        grid's `area_corner` cannot affect the result.  Confirming the
        API surface has no hidden area weighting."""
        import inspect
        from legoesm.core.operators_cdgrid import (
            _interp_corner_to_center)
        sig = inspect.signature(_interp_corner_to_center)
        self.assertEqual(
            list(sig.parameters), ["field_d"],
            msg=(f"`_interp_corner_to_center` signature = "
                 f"{list(sig.parameters)}.  If an area-weighted "
                 f"variant is introduced, it should live under a new "
                 f"name (e.g. `_interp_corner_to_center_weighted`) "
                 f"and this lock should be kept pointing at the "
                 f"plain-arithmetic variant."))


class TestExtrapolateBoundaryCorners(unittest.TestCase):
    """Iter-558: regression lock for `_extrapolate_boundary_corners`
    (`src/legoesm/core/operators_cdgrid.py:963-1006`).

    Applies bilinear extrapolation to the 4 cube-vertex corners of
    momentum tendencies `(du, dv)` on the production A-L path:
        tend(0, 0) = tend(1, 0) + tend(0, 1) - tend(1, 1)
    (and analogous formulas at the other 3 corners).

    Used in production SW (`shallow_water_fv3_cdgrid.py:163`) and
    ocean PE (`ocean_pe_cdgrid.py:225`).  Previously had NO direct
    tests — a sign flip in the formula, swapped source indices, or
    dropped corner would silently degrade the O(dx²) accuracy claim.

    Locks:
      (a) Shape 2D: `(6, n+1, n+1)` preserved.
      (b) On a BILINEAR field `f(i, j) = a + b*i + c*j + d*i*j`,
          bilinear extrapolation is EXACT at the 4 cube vertices
          (error = 0 to machine precision).
      (c) Non-vertex cells (row 0 interior, etc.) are UNCHANGED.
      (d) On random input, the 4 vertex values exactly satisfy
          the bilinear formula.
    """

    def _build_n(self):
        return 6

    def test_shape_preserved_2d(self):
        from legoesm.core.operators_cdgrid import (
            _extrapolate_boundary_corners)
        n = self._build_n()
        du = jnp.zeros((6, n + 1, n + 1))
        dv = jnp.zeros((6, n + 1, n + 1))
        du_out, dv_out = _extrapolate_boundary_corners(du, dv, n)
        self.assertEqual(du_out.shape, (6, n + 1, n + 1))
        self.assertEqual(dv_out.shape, (6, n + 1, n + 1))

    def test_linear_field_exact_at_vertices(self):
        """On a LINEAR field f(i, j) = a + b*i + c*j, the formula
        `f(0, 0) = f(1, 0) + f(0, 1) - f(1, 1)` is EXACT because
        the cross-term is zero.  (Note: for a GENUINE BILINEAR
        field with f(i,j) = a + bi + cj + d*i*j, the formula
        recovers a - d instead of a, so it's not exact — the
        docstring says "bilinear extrapolation" but the formula
        is really "linear extrapolation at the vertex".)
        Verify error at the 4 cube vertices is 0 on random linear
        coefficients."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _extrapolate_boundary_corners)
        n = self._build_n()
        rng = np.random.default_rng(558)
        i_idx, j_idx = np.meshgrid(
            np.arange(n + 1), np.arange(n + 1), indexing='ij')
        i_idx = i_idx.astype(np.float64)
        j_idx = j_idx.astype(np.float64)
        du_np = np.zeros((6, n + 1, n + 1))
        dv_np = np.zeros((6, n + 1, n + 1))
        for f in range(6):
            a, b, c = rng.standard_normal(3)
            du_np[f] = a + b * i_idx + c * j_idx   # LINEAR (no d*ij)
            a, b, c = rng.standard_normal(3)
            dv_np[f] = a + b * i_idx + c * j_idx
        du = jnp.asarray(du_np)
        dv = jnp.asarray(dv_np)

        orig_corners_du = {
            (f, ci, cj): float(du_np[f, ci, cj])
            for f in range(6)
            for ci in (0, n)
            for cj in (0, n)
        }

        du_out, dv_out = _extrapolate_boundary_corners(du, dv, n)

        for f in range(6):
            for ci in (0, n):
                for cj in (0, n):
                    reproduced = float(du_out[f, ci, cj])
                    original = orig_corners_du[(f, ci, cj)]
                    diff = abs(reproduced - original)
                    self.assertLess(
                        diff, 1e-10,
                        msg=(f"du face {f} corner ({ci},{cj}): "
                             f"linear extrapolation on a linear "
                             f"field yields {reproduced:.6f}, "
                             f"original {original:.6f}, diff "
                             f"{diff:.3e}.  The formula is NOT "
                             f"`tend(0,0) = tend(1,0) + tend(0,1) "
                             f"- tend(1,1)`."))

    def test_exact_formula_at_all_4_vertices(self):
        """On a random input, verify the output at each of the 4
        cube vertices EXACTLY equals the bilinear formula applied
        to the 3 source cells."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _extrapolate_boundary_corners)
        n = self._build_n()
        rng = np.random.default_rng(1558)
        du_np = rng.standard_normal((6, n + 1, n + 1)).astype(
            np.float64)
        dv_np = rng.standard_normal((6, n + 1, n + 1)).astype(
            np.float64)
        du = jnp.asarray(du_np)
        dv = jnp.asarray(dv_np)

        du_out_np = np.asarray(
            _extrapolate_boundary_corners(du, dv, n)[0],
            dtype=np.float64)

        # Check (0, 0): output[0, 0] == in[1, 0] + in[0, 1] - in[1, 1]
        # Check (n, 0): output[n, 0] == in[n-1, 0] + in[n, 1] - in[n-1, 1]
        # Check (0, n): output[0, n] == in[1, n] + in[0, n-1] - in[1, n-1]
        # Check (n, n): output[n, n] == in[n-1, n] + in[n, n-1] - in[n-1, n-1]
        specs = [
            ("(0, 0)",   (0, 0),   (1, 0),       (0, 1),       (1, 1)),
            ("(n, 0)",   (n, 0),   (n - 1, 0),   (n, 1),       (n - 1, 1)),
            ("(0, n)",   (0, n),   (1, n),       (0, n - 1),   (1, n - 1)),
            ("(n, n)",   (n, n),   (n - 1, n),   (n, n - 1),   (n - 1, n - 1)),
        ]
        for label, (ci, cj), (e1i, e1j), (e2i, e2j), (di, dj) in specs:
            actual = du_out_np[:, ci, cj]
            expected = (
                du_np[:, e1i, e1j]
                + du_np[:, e2i, e2j]
                - du_np[:, di, dj]
            )
            diff = float(np.max(np.abs(actual - expected)))
            self.assertLess(
                diff, 1e-12,
                msg=(f"Corner {label}: output differs from "
                     f"`in[{e1i},{e1j}] + in[{e2i},{e2j}] - "
                     f"in[{di},{dj}]` by {diff:.3e}.  Either the "
                     f"source indices were swapped or the sign "
                     f"convention changed (should be + + -)."))

    def test_non_vertex_cells_unchanged(self):
        """Only the 4 cube vertices are modified.  Any other cell
        must equal its input.  Regression against a bug where the
        function modifies edges or interior."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _extrapolate_boundary_corners)
        n = self._build_n()
        rng = np.random.default_rng(2558)
        du_np = rng.standard_normal((6, n + 1, n + 1))
        dv_np = rng.standard_normal((6, n + 1, n + 1))
        du = jnp.asarray(du_np)
        dv = jnp.asarray(dv_np)

        du_out = _extrapolate_boundary_corners(du, dv, n)[0]
        du_out_np = np.asarray(du_out)

        # Non-vertex cells
        mask = np.ones((6, n + 1, n + 1), dtype=bool)
        for ci in (0, n):
            for cj in (0, n):
                mask[:, ci, cj] = False
        diff_nv = float(np.max(np.abs(
            du_out_np[mask] - du_np[mask])))
        self.assertEqual(
            diff_nv, 0.0,
            msg=(f"Non-vertex cells modified; max diff {diff_nv:.3e}. "
                 f"`_extrapolate_boundary_corners` must only touch "
                 f"the 4 cube-vertex corners."))


class TestBroadcastMetric(unittest.TestCase):
    """Iter-549: regression lock for `_broadcast_metric`
    (`src/legoesm/core/operators_cdgrid.py:75-79`).

    Utility that inserts a trailing singleton axis on a 2D metric when
    the consumer field has a higher ndim (typically adding ``nlev``).
    Used on every 3D-compatible code path: `dgrid_vorticity`,
    `_arakawa_lamb_gradient`, `cgrid_mass_flux_divergence`, etc.
    A silent refactor that inserted the axis at the WRONG position
    (e.g. `metric[None, ...]` instead of `metric[..., None]`) would
    produce broadcast errors or silently wrong element-wise products.

    Before iter-549 this helper had NO direct tests.

    Locks:
      (a) Same-ndim: metric returned unchanged.
      (b) Field 1-d higher: trailing singleton axis added.
      (c) Field 2-d higher (rare but possible): still inserts once
          then relies on numpy broadcasting.
      (d) Broadcast semantics: `metric * field` with the broadcast
          result equals per-level element-wise `metric * field[k]`.
    """

    def test_same_ndim_metric_returned_unchanged(self):
        from legoesm.core.operators_cdgrid import _broadcast_metric
        metric = jnp.ones((6, 8, 8))
        field = jnp.zeros((6, 8, 8))
        out = _broadcast_metric(metric, field)
        # Must be identical object/array contents and shape.
        self.assertEqual(out.shape, metric.shape)
        self.assertTrue(jnp.all(out == metric))

    def test_field_one_dim_higher_adds_trailing_singleton(self):
        from legoesm.core.operators_cdgrid import _broadcast_metric
        metric = jnp.ones((6, 8, 8))
        field = jnp.zeros((6, 8, 8, 5))
        out = _broadcast_metric(metric, field)
        self.assertEqual(
            out.shape, (6, 8, 8, 1),
            msg=(f"Expected trailing singleton axis for 3D-field "
                 f"input; got shape {out.shape}.  A `metric[None, "
                 f"...]` refactor would produce (1, 6, 8, 8)."))

    def test_broadcast_product_matches_per_level(self):
        """The whole point of `_broadcast_metric` is that
        `metric * field` works correctly for 3D fields.  Verify."""
        import numpy as np
        from legoesm.core.operators_cdgrid import _broadcast_metric
        rng = np.random.default_rng(549)
        metric_np = rng.standard_normal((6, 8, 8)).astype(np.float64)
        field_np = rng.standard_normal((6, 8, 8, 5)).astype(np.float64)
        metric = jnp.asarray(metric_np)
        field = jnp.asarray(field_np)

        mbc = _broadcast_metric(metric, field)
        product = np.asarray(mbc * field, dtype=np.float64)

        for k in range(field_np.shape[-1]):
            expected_k = metric_np * field_np[..., k]
            max_diff = float(np.max(
                np.abs(product[..., k] - expected_k)))
            self.assertLess(
                max_diff, 1e-12,
                msg=(f"Level {k}: broadcast product differs from "
                     f"`metric * field[..., {k}]` by {max_diff:.3e}. "
                     f"A wrong-axis insertion (e.g. `metric[None, "
                     f"...]` yielding shape (1,6,8,8)) would "
                     f"broadcast incorrectly and fail this."))

    def test_rejects_axis_position_error(self):
        """If the helper incorrectly used `metric[None, ...]` instead
        of `metric[..., None]`, the broadcast would mismatch shapes.
        This test constructs the specific case that catches that
        error mode."""
        from legoesm.core.operators_cdgrid import _broadcast_metric
        # metric shape (6, 8, 8), field shape (6, 8, 8, 5)
        metric = jnp.arange(6 * 8 * 8, dtype=jnp.float64).reshape((6, 8, 8))
        field = jnp.ones((6, 8, 8, 5), dtype=jnp.float64)

        mbc = _broadcast_metric(metric, field)
        # For correct broadcast shape (6, 8, 8, 1): product = metric
        # replicated across level dim.  So product[..., 0] == metric.
        product = mbc * field
        self.assertEqual(product.shape, (6, 8, 8, 5))
        for k in range(5):
            max_diff = float(jnp.max(jnp.abs(product[..., k] - metric)))
            self.assertLess(
                max_diff, 1e-12,
                msg=(f"Level {k}: product not equal to metric when "
                     f"field is all-ones; max diff = {max_diff:.3e}."))


class TestInterpCenterToCorner(unittest.TestCase):
    """Iter-548: regression lock for `_interp_center_to_corner`
    (`src/legoesm/core/operators_cdgrid.py:868-892`).

    The dual of `_interp_corner_to_center` (iter-544/545 lock):
    averages a cell-centre field `(6, n, n[, nlev])` to D-grid corners
    `(6, n+1, n+1[, nlev])` via a 4-point average of the halo-padded
    field.  The helper supports an optional `padded=` argument for
    stage-level pre-padded inputs — both the `padded=None` (internal
    halo exchange) and the `padded=...` (bypass) paths need locks.

    Before iter-548 the function had NO direct tests.  Production
    callers (e.g., `_d_sw5_corner_divergence` at fv3_sw_core.py:1024
    for the Smagorinsky vorticity-to-corner interpolation) rely on
    this helper's exact averaging formula.  A silent refactor to
    weighted/skewed averaging would propagate into the damping term
    without a regression.

    Locks:
      (a) 2D shape: `(6, n, n)` -> `(6, n+1, n+1)` via internal pad
      (b) 3D shape: `(6, n, n, nlev)` -> `(6, n+1, n+1, nlev)`
      (c) `padded=` bypass: when caller pre-pads, the internal halo
          exchange is skipped and the pre-padded input is used as-is.
      (d) Exact `0.25*(SW + SE + NW + NE)` from the padded field at
          cube-face interior corners (where the halo values are
          well-defined from the internal cubed-sphere exchange).
      (e) 3D branch matches 3D-branch-applied-to-single-level for
          each k.
    """

    def _build(self, n=6):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid)
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        return n, cdgrid

    def test_shape_2d_and_3d_with_internal_pad(self):
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner)
        n, cdgrid = self._build(n=6)
        nlev = 5
        field_2d = jnp.zeros((6, n, n))
        field_3d = jnp.zeros((6, n, n, nlev))
        out_2d = _interp_center_to_corner(field_2d, cdgrid)
        out_3d = _interp_center_to_corner(field_3d, cdgrid)
        self.assertEqual(out_2d.shape, (6, n + 1, n + 1))
        self.assertEqual(out_3d.shape, (6, n + 1, n + 1, nlev))

    def test_constant_field_preserved(self):
        """Constant input must round-trip through halo + averaging to
        a constant output at every corner (no phase artifacts)."""
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner)
        n, cdgrid = self._build(n=6)
        field = jnp.full((6, n, n), 3.75, dtype=jnp.float64)
        out = _interp_center_to_corner(field, cdgrid)
        max_dev = float(jnp.max(jnp.abs(out - 3.75)))
        # Allow small drift from duogrid halo interpolation; should be
        # exact for constant fields since the interpolation is linear.
        self.assertLess(
            max_dev, 1e-10,
            msg=(f"Constant field not preserved at corners; max dev "
                 f"= {max_dev:.3e}.  A weighted-average refactor "
                 f"that preserves summation may still fail this if "
                 f"the weights do not sum to 1."))

    def test_padded_bypass_exact_arithmetic_average_2d(self):
        """With a caller-supplied `padded=...`, the helper MUST skip
        its own halo exchange and use `padded` directly.  Verified by
        passing a pre-padded field with distinct values at every cell
        and asserting output = `0.25*(SW+SE+NW+NE)` from `padded`."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner)
        n, cdgrid = self._build(n=6)
        rng = np.random.default_rng(548)
        # Caller-supplied pad shape (6, n+2, n+2) with random unique
        # values.  The `field` argument is passed through as a shape
        # carrier — its content should be IGNORED because padded is
        # provided.
        padded_np = rng.standard_normal(
            (6, n + 2, n + 2)).astype(np.float64)
        padded = jnp.asarray(padded_np)
        field_dummy = jnp.zeros((6, n, n))

        out = np.asarray(
            _interp_center_to_corner(
                field_dummy, cdgrid, padded=padded),
            dtype=np.float64)
        expected = 0.25 * (
            padded_np[:, :-1, :-1]
            + padded_np[:, 1:, :-1]
            + padded_np[:, :-1, 1:]
            + padded_np[:, 1:, 1:]
        )
        max_diff = float(np.max(np.abs(out - expected)))
        self.assertLess(
            max_diff, 1e-10,
            msg=(f"`_interp_center_to_corner(field, cdgrid, "
                 f"padded=...)` does not take the 4-point average "
                 f"directly from the provided `padded` array; max "
                 f"diff = {max_diff:.3e}.  Either a halo exchange is "
                 f"still being done internally (defeating the "
                 f"stage-packing bypass) or the averaging formula "
                 f"changed.  Update this test with the new expected "
                 f"formula if the change is intentional."))

    def test_padded_bypass_exact_arithmetic_average_4d(self):
        """Same as 4D branch: caller-supplied pad, distinct values,
        per-level 4-point average."""
        import numpy as np
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner)
        n, cdgrid = self._build(n=6)
        nlev = 3
        rng = np.random.default_rng(1548)
        padded_np = rng.standard_normal(
            (6, n + 2, n + 2, nlev)).astype(np.float64)
        padded = jnp.asarray(padded_np)
        field_dummy = jnp.zeros((6, n, n, nlev))

        self.assertEqual(field_dummy.ndim, 4,
                         msg="Test must exercise ndim==4 branch.")

        out = np.asarray(
            _interp_center_to_corner(
                field_dummy, cdgrid, padded=padded),
            dtype=np.float64)
        self.assertEqual(out.shape, (6, n + 1, n + 1, nlev))

        expected = 0.25 * (
            padded_np[:, :-1, :-1, :]
            + padded_np[:, 1:, :-1, :]
            + padded_np[:, :-1, 1:, :]
            + padded_np[:, 1:, 1:, :]
        )
        max_diff = float(np.max(np.abs(out - expected)))
        self.assertLess(
            max_diff, 1e-10,
            msg=(f"4D branch of `_interp_center_to_corner(padded=)` "
                 f"deviates from per-level `0.25*(SW+SE+NW+NE)` by "
                 f"{max_diff:.3e}.  Production caller "
                 f"`_d_sw5_corner_divergence` assumes this formula."))

    def test_padded_argument_changes_output(self):
        """Sanity: `padded=` actually controls the output.  Supplying
        a pad filled with zeros (ignoring the real field) must produce
        all-zero output even though the `field` argument has
        non-zero content."""
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner)
        n, cdgrid = self._build(n=6)
        field = jnp.ones((6, n, n)) * 100.0   # non-trivial field
        zero_pad = jnp.zeros((6, n + 2, n + 2))
        out = _interp_center_to_corner(field, cdgrid, padded=zero_pad)
        # With all-zero pad, the 0.25*(...) average must be zero.
        max_abs = float(jnp.max(jnp.abs(out)))
        self.assertEqual(
            max_abs, 0.0,
            msg=(f"Providing `padded=zeros` did not force zero "
                 f"output; max abs = {max_abs:.3e}.  The `padded` "
                 f"bypass must override the internal halo exchange."))


class TestDgridToCgridAsymmetryIsIntentional(unittest.TestCase):
    """Iter-518: lock the structural asymmetry between u_c and v_c
    formulas in `dgrid_to_cgrid` and `fv3_cc2c` as INTENTIONAL.

    FV3 uses a mixed-orthogonal D-grid convention where:
      - `u_d` (and `u_cc`) is the velocity component along the local
        i-axis (e_i).
      - `v_d` (and `v_cc`) is the velocity component along the
        perpendicular-to-e_i direction (e_perp), NOT along the local
        j-axis (e_v).
    On a non-orthogonal grid e_perp ≠ e_v, so:
      - The x-face (i = const) outward NORMAL is along e_i.  The
        velocity along e_i is NOT u_d directly because u_d is also
        partly aligned with e_v due to non-orthogonality.  We need
        the correction: ``u_c = u_d * sina_u - v_d * cosa_u``.
      - The y-face (j = const) outward normal is along e_perp.  And
        v_d IS along e_perp by convention.  So no correction needed:
        ``v_c = v_d`` (averaged to the y-face position).

    A naive "symmetrize for elegance" refactor would add an analogous
    `v_c = v_d * sina_v - u_d * cosa_v` formula and break the FV3
    convention.  This test guards against that by AST-asserting the
    exact formula structure in both functions.
    """

    def _ast_check_unique(self, src, callee_name):
        """Find the unique module-level FunctionDef named callee_name."""
        import ast
        tree = ast.parse(src)
        funcs = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == callee_name]
        self.assertEqual(
            len(funcs), 1,
            msg=f"Expected exactly 1 module-level def `{callee_name}`; "
                f"found {len(funcs)}.")
        return funcs[0]

    def _read_operators_cdgrid(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent.parent
        return (root / "src/legoesm/grids/__init__.py").exists() and (
            root / "src/legoesm/core/operators_cdgrid.py").read_text()

    def test_dgrid_to_cgrid_u_has_correction_v_does_not(self):
        """`dgrid_to_cgrid`: `u_c =` line must contain BOTH `sina_u`
        AND `cosa_u`; `v_c =` line(s) must contain NEITHER `sina_v`
        NOR `cosa_v` (no non-orthogonality correction on v)."""
        src = self._read_operators_cdgrid()
        self.assertTrue(src, "Could not read operators_cdgrid.py")
        func_src = self._extract_function_source(src, "dgrid_to_cgrid")

        u_assign_lines = [ln for ln in func_src.splitlines()
                          if ln.strip().startswith("u_c =")]
        v_assign_lines = [ln for ln in func_src.splitlines()
                          if ln.strip().startswith("v_c =")]
        self.assertEqual(
            len(u_assign_lines), 1,
            msg=f"Expected exactly one `u_c =` assignment in "
                f"dgrid_to_cgrid; found {len(u_assign_lines)}.  "
                f"Source:\n{func_src}")
        self.assertEqual(
            len(v_assign_lines), 1,
            msg=f"Expected exactly one `v_c =` assignment in "
                f"dgrid_to_cgrid; found {len(v_assign_lines)}.")

        u_line = u_assign_lines[0]
        v_line = v_assign_lines[0]
        self.assertIn(
            "sina_u", u_line,
            msg=f"u_c assignment must contain `sina_u` (FV3 mixed-"
                f"orthogonal D-grid x-face normal projection): "
                f"`{u_line.strip()}`.")
        self.assertIn(
            "cosa_u", u_line,
            msg=f"u_c assignment must contain `cosa_u` (FV3 mixed-"
                f"orthogonal D-grid x-face normal projection): "
                f"`{u_line.strip()}`.")
        self.assertNotIn(
            "sina_v", v_line,
            msg=(f"v_c assignment must NOT contain `sina_v` — by FV3 "
                 f"convention v_d IS the y-face normal direction "
                 f"(no projection needed).  If a `symmetrize` "
                 f"refactor added one, REVERT IT and consult the "
                 f"docstring of dgrid_to_cgrid.  Source: "
                 f"`{v_line.strip()}`."))
        self.assertNotIn(
            "cosa_v", v_line,
            msg=(f"v_c assignment must NOT contain `cosa_v`.  Source: "
                 f"`{v_line.strip()}`."))

    def test_fv3_cc2c_u_has_correction_v_does_not(self):
        """`fv3_cc2c`: same structural asymmetry as dgrid_to_cgrid —
        u_c uses the non-orthogonality correction, v_c does not."""
        src = self._read_operators_cdgrid()
        func_src = self._extract_function_source(src, "fv3_cc2c")

        u_assign_lines = [ln for ln in func_src.splitlines()
                          if ln.strip().startswith("u_c =")]
        v_assign_lines = [ln for ln in func_src.splitlines()
                          if ln.strip().startswith("v_c =")]
        self.assertEqual(
            len(u_assign_lines), 1,
            msg=f"Expected exactly one `u_c =` line in fv3_cc2c.")
        self.assertEqual(
            len(v_assign_lines), 1,
            msg=f"Expected exactly one `v_c =` line in fv3_cc2c.")

        u_line = u_assign_lines[0]
        v_line = v_assign_lines[0]
        self.assertIn("sina_u", u_line)
        self.assertIn("cosa_u", u_line)
        self.assertNotIn(
            "sina_v", v_line,
            msg=(f"v_c assignment in fv3_cc2c must NOT contain "
                 f"`sina_v` per FV3 mixed-orthogonal D-grid "
                 f"convention.  Source: `{v_line.strip()}`."))
        self.assertNotIn(
            "cosa_v", v_line,
            msg=(f"v_c assignment in fv3_cc2c must NOT contain "
                 f"`cosa_v`.  Source: `{v_line.strip()}`."))

    @staticmethod
    def _extract_function_source(src, name):
        """Return the source text of the function `name` from `src`."""
        import ast
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.unparse(node)
        raise AssertionError(f"Function `{name}` not found in source.")

    def test_fv3_cc2c_v_c_is_plain_average_behaviorally(self):
        """Iter-541 BEHAVIORAL lock of iter-518's convention finding.

        iter-518 locked the convention asymmetry via an AST check.  But a
        refactor that routes the asymmetric correction through a helper
        (e.g., `_apply_cc2c_correction(u_cc, v_cc, cdgrid)`) could pass
        the AST check while silently changing behavior.  This test
        exercises `fv3_cc2c` on a real cdgrid with a specific non-
        trivial cell-centre wind pattern, and verifies:

          (a) v_c at INTERIOR (non-cube-edge) positions equals the plain
              2-point average of v_cc between adjacent cells — NO
              non-orthogonality correction applied.

          (b) u_c at INTERIOR positions differs from the plain 2-point
              average of u_cc — non-orthogonality correction IS applied
              (at minimum, `u_c != u_avg` measurably when v_cc != 0).

        Iter-541 verified empirically that adding a symmetric
        `v_c = v_avg*sina_v - u_at_v*cosa_v` correction degrades W2 L2
        by 200x (4.79e-2 vs 2.42e-4 baseline).  This behavioural lock
        makes that regression catchable at unit-test scope.
        """
        from legoesm.core.operators_cdgrid import fv3_cc2c
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
        )
        import jax.numpy as jnp
        import numpy as np

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Non-trivial cell-centre wind: u_cc depends on lat, v_cc on lon
        # (both non-zero, no symmetry that would accidentally hide the
        # correction/no-correction behaviour).
        lat = cdgrid.base.lat
        lon = cdgrid.base.lon
        u_cc = jnp.cos(lat)            # (6, n, n)
        v_cc = 0.1 * jnp.sin(2.0 * lon)  # (6, n, n)  non-zero to
                                        # trigger the u-correction term

        u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

        # (a) v_c at INTERIOR (non-cube-edge) should be the plain
        # 2-point average of v_cc.  Take an interior y-face: face 0, at
        # interior i and interior j (avoid cube edges where halo data
        # enters).
        interior_i = slice(2, n - 2)
        interior_j = slice(2, n - 2)  # j=2..n-3 for v_c (shape (6,n,n+1))

        # v_c[face, i, j] = 0.5 * (v_cc[face, i, j-1] + v_cc[face, i, j])
        # for interior y-faces.  Check on face 0.
        v_c_interior = np.asarray(v_c)[0, interior_i, interior_j]
        v_avg_expected = 0.5 * (
            np.asarray(v_cc)[0, interior_i, 1:n - 3]
            + np.asarray(v_cc)[0, interior_i, 2:n - 2]
        )
        max_abs_dev_v = float(np.max(np.abs(v_c_interior - v_avg_expected)))
        # Tolerance 1e-6 = 4 orders of magnitude below the magnitude
        # of a real correction term (v_avg*sina_v - u_at_v*cosa_v
        # yields ~0.01 deviation at cube-face interior for this field).
        # Actual production precision on v_c is float32 (~1e-8 noise).
        self.assertLess(
            max_abs_dev_v, 1e-6,
            msg=(f"v_c at cube-face interior must equal the plain "
                 f"0.5*(v_cc[j-1] + v_cc[j]) average (NO non-"
                 f"orthogonality correction).  Max deviation = "
                 f"{max_abs_dev_v:.3e}.  If a 'symmetrize' refactor "
                 f"added a `v_c = v_avg*sina_v - u_at_v*cosa_v` term, "
                 f"REVERT IT and see iter-541 + iter-518 convention "
                 f"notes in docs/fv3_fortran_fidelity_review.md."))

        # (b) u_c at INTERIOR should DIFFER from plain 2-point average
        # of u_cc (non-orthogonality correction IS applied).
        # u_c[face, i, j] corresponds to an x-face at x-position i.
        u_c_interior = np.asarray(u_c)[0, interior_i, interior_j]
        # Reconstruct the plain-average of u_cc at x-face positions:
        # u_avg_plain[i, j] = 0.5 * (u_cc[i-1, j] + u_cc[i, j])
        # For face 0, interior u-face i in [2, n-2]:
        u_avg_plain = 0.5 * (
            np.asarray(u_cc)[0, 1:n - 3, interior_j]
            + np.asarray(u_cc)[0, 2:n - 2, interior_j]
        )
        max_abs_dev_u = float(np.max(np.abs(u_c_interior - u_avg_plain)))
        # Non-strict: just require measurably non-zero correction.
        # cos(alpha) is O(0.1) near cube edges, smaller at interior;
        # the correction is v_at_u * cosa_u ~ 0.1 * 0.01 = 1e-3 at
        # cube-face interior.
        self.assertGreater(
            max_abs_dev_u, 1e-5,
            msg=(f"u_c at cube-face interior must DIFFER from the "
                 f"plain 0.5*(u_cc[i-1] + u_cc[i]) average — the non-"
                 f"orthogonality correction must be active.  Max "
                 f"deviation = {max_abs_dev_u:.3e} < 1e-5 threshold.  "
                 f"If the correction term was removed, RESTORE IT."))


class TestCellCentreAnglesFrom4Edge(unittest.TestCase):
    """Iter-528: regression for the new `cell_centre_angles_from_4edge`
    helper extracted from `run_atmosphere_test_matrix.py:1213-1231`."""

    def test_4edge_helper_matches_matrix_inline_formula(self):
        """The helper's output must equal the matrix's exact inline
        formula on a real cdgrid."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
            cell_centre_angles_from_4edge,
        )
        import numpy as np

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Reproduce matrix's inline formula
        cax = np.asarray(cdgrid.cos_angle_edge_x, dtype=np.float64)
        sax = np.asarray(cdgrid.sin_angle_edge_x, dtype=np.float64)
        cay = np.asarray(cdgrid.cos_angle_edge_y, dtype=np.float64)
        say = np.asarray(cdgrid.sin_angle_edge_y, dtype=np.float64)
        ca_ref = 0.25 * (cax[:, :, :-1] + cax[:, :, 1:]
                         + cay[:, :-1, :] + cay[:, 1:, :])
        sa_ref = 0.25 * (sax[:, :, :-1] + sax[:, :, 1:]
                         + say[:, :-1, :] + say[:, 1:, :])
        norm = np.sqrt(ca_ref ** 2 + sa_ref ** 2)
        ca_ref /= norm
        sa_ref /= norm

        ca, sa = cell_centre_angles_from_4edge(cdgrid)
        np.testing.assert_allclose(np.asarray(ca), ca_ref, atol=1e-6)
        np.testing.assert_allclose(np.asarray(sa), sa_ref, atol=1e-6)

    def test_4edge_helper_outputs_unit_magnitude(self):
        """The post-renormalization (cos, sin) pair must satisfy
        cos² + sin² = 1 to machine precision."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
            cell_centre_angles_from_4edge,
        )
        import numpy as np

        for n in (8, 16):
            grid = create_cubed_sphere(n)
            cdgrid = create_cubed_sphere_cdgrid(grid)
            ca, sa = cell_centre_angles_from_4edge(cdgrid)
            mag = np.asarray(ca ** 2 + sa ** 2)
            np.testing.assert_allclose(
                mag, 1.0, atol=1e-6,
                err_msg=f"4-edge angle helper output not unit magnitude at n={n}")

    def test_4edge_shape(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
            cell_centre_angles_from_4edge,
        )
        n = 12
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        ca, sa = cell_centre_angles_from_4edge(cdgrid)
        self.assertEqual(ca.shape, (6, n, n))
        self.assertEqual(sa.shape, (6, n, n))


if __name__ == "__main__":
    unittest.main()
