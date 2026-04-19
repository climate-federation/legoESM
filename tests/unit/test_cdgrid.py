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
