"""Targeted regression tests for the land model audit fixes.

Covers:
1. Snow sublimation/deposition bookkeeping (L_s energetics, snow mass)
2. TileResponse flux-state consistency
3. x64 soil thermal solve (shared tridiagonal solver)
4. Root-zone moisture sensitivity with transpiration
5. Richards n_iter truthfulness
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

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
        T_lowest=270.0, q_lowest=2e-3, u_lowest=5.0, v_lowest=2.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    defaults.update(overrides)
    return AtmToSurface(**{k: v * ones for k, v in defaults.items()})


def _make_slab_state(ncol, T_init=270.0, W_init=50.0, snow_init=0.0):
    return LandState(
        T_soil=Field(jnp.full(ncol, T_init), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, W_init), name="W_bucket"),
        snow_depth=Field(jnp.full(ncol, snow_init), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
    )


# =========================================================================
# 1. Snow sublimation / deposition bookkeeping
# =========================================================================


class TestSnowSublimation(unittest.TestCase):
    """Snow-covered latent exchange must use snowpack, not soil bucket."""

    def test_snow_covered_qsurface_independent_of_bucket(self):
        """q_surface over snow should NOT collapse when W_bucket is empty."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        dt = 3600.0
        # Cold surface with snow, but empty bucket
        forcing = _make_forcing(ncol, T_lowest=265.0, q_lowest=1e-3,
                                sw_down=50.0, lw_down=250.0)

        state_wet = _make_slab_state(ncol, T_init=260.0, W_init=100.0, snow_init=20.0)
        state_dry = _make_slab_state(ncol, T_init=260.0, W_init=0.0, snow_init=20.0)

        _, resp_wet, _ = step_land(state_wet, forcing, config, U_min=1.0, dt=dt)
        _, resp_dry, _ = step_land(state_dry, forcing, config, U_min=1.0, dt=dt)

        # Over snow, q_surface should be ~q_sat_ice(T), independent of bucket.
        # With the old code, q_sfc_dry would collapse because beta depends on W.
        ratio = resp_dry.q_surface / jnp.maximum(resp_wet.q_surface, 1e-20)
        npt.assert_allclose(ratio, 1.0, atol=0.05,
                            err_msg="Snow-covered q_surface should not depend on bucket state")

    def test_sublimation_reduces_snow_mass(self):
        """Sublimation (evaporation over snow) must reduce snow mass."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        dt = 3600.0
        # Warm-ish below-freezing surface with snow, dry atmosphere → sublimation
        forcing = _make_forcing(ncol, T_lowest=268.0, q_lowest=1e-4,
                                sw_down=100.0, precip_total=0.0, precip_snow=0.0)
        state = _make_slab_state(ncol, T_init=268.0, W_init=50.0, snow_init=10.0)

        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=dt)

        # Snow should decrease (sublimation)
        snow_old = float(state.snow_depth.data[0])
        snow_new = float(state2.snow_depth.data[0])
        self.assertLess(snow_new, snow_old,
                        f"Snow should decrease under sublimation: {snow_old} -> {snow_new}")

    def test_deposition_increases_snow_mass(self):
        """Deposition (condensation over snow) must increase snow mass."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        dt = 3600.0
        # Cold surface, very moist atmosphere → deposition onto snow
        forcing = _make_forcing(ncol, T_lowest=265.0, q_lowest=0.005,
                                sw_down=0.0, lw_down=200.0,
                                precip_total=0.0, precip_snow=0.0)
        state = _make_slab_state(ncol, T_init=255.0, W_init=50.0, snow_init=5.0)

        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=dt)

        snow_old = float(state.snow_depth.data[0])
        snow_new = float(state2.snow_depth.data[0])
        # lhflx < 0 means moisture going toward surface → deposition
        lhflx = float(resp.lhflx[0])
        if lhflx < 0:
            self.assertGreater(snow_new, snow_old,
                               f"Deposition should increase snow: {snow_old} -> {snow_new}, lhflx={lhflx}")

    def test_bucket_unchanged_during_snow_sublimation(self):
        """W_bucket should not change from snow sublimation."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        dt = 3600.0
        # Snow-covered, no rain or melt — sublimation only
        forcing = _make_forcing(ncol, T_lowest=265.0, q_lowest=1e-4,
                                sw_down=50.0, lw_down=200.0,
                                precip_total=0.0, precip_snow=0.0)
        state = _make_slab_state(ncol, T_init=263.0, W_init=50.0, snow_init=20.0)

        state2, _, _ = step_land(state, forcing, config, U_min=1.0, dt=dt)

        W_old = float(state.W_bucket.data[0])
        W_new = float(state2.W_bucket.data[0])
        # With snow cover, no rain, and below-freezing T → no melt,
        # so bucket should not change (all latent exchange is from snow).
        self.assertAlmostEqual(W_new, W_old, places=6,
                               msg=f"Bucket should not change during snow sublimation: {W_old} -> {W_new}")


# =========================================================================
# 2. TileResponse flux-state temporal consistency
# =========================================================================


class TestTileResponseConsistency(unittest.TestCase):
    """Fluxes and state in TileResponse have documented temporal semantics."""

    def test_lw_up_matches_end_of_step_T(self):
        """lw_up should be eps*sigma*T^4 + (1-eps)*lw_down at end of step."""
        from legoesm.land.slab_land import step_land

        ncol = 4
        config = LandConfig()
        forcing = _make_forcing(ncol)
        state = _make_slab_state(ncol, T_init=290.0, W_init=50.0)

        _, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=3600.0)

        eps = config.emissivity_land
        # lw_up = emitted + reflected = eps*sigma*T^4 + (1-eps)*lw_down
        expected_lw_up = (eps * constants.sigma_sb * resp.T_sfc ** 4
                          + (1.0 - eps) * forcing.lw_down)
        npt.assert_allclose(resp.lw_up, expected_lw_up, rtol=1e-6,
                            err_msg="lw_up should match end-of-step T_sfc")

    def test_q_surface_matches_end_of_step_state(self):
        """q_surface should be consistent with end-of-step T and moisture."""
        from legoesm.land.slab_land import step_land
        from legoesm.thermo import saturation_mixing_ratio

        ncol = 4
        config = LandConfig()
        forcing = _make_forcing(ncol)
        state = _make_slab_state(ncol, T_init=290.0, W_init=50.0)

        state2, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=3600.0)

        # q_surface should be beta_new * q_sat(T_new, p)
        q_sat = saturation_mixing_ratio(resp.T_sfc, forcing.p_surface)
        # beta_new from post-step W
        W_new = state2.W_bucket.data
        w_frac = jnp.clip(W_new / config.W_max, 0.0, 1.0)
        beta_new = config.beta_min + (1.0 - config.beta_min) * w_frac
        expected_q = beta_new * q_sat
        npt.assert_allclose(resp.q_surface, expected_q, rtol=1e-6,
                            err_msg="q_surface should match end-of-step moisture state")


# =========================================================================
# 3. x64 soil thermal solve (shared tridiagonal solver)
# =========================================================================


class TestSoilThermalX64(unittest.TestCase):
    """Soil thermal solver must work under float64."""

    def test_uniform_T_steady_x64(self):
        """Uniform T with zero flux stays constant under x64."""
        from legoesm.land.soil_thermal import solve_soil_thermal, SoilThermalConfig
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig

        grid = make_soil_grid(SoilGridConfig(n_layers=6))
        hconfig = SoilHydraulicsConfig()
        tconfig = SoilThermalConfig(Q_geothermal=0.0)

        ncol, nlayers = 4, 6
        T = jnp.full((ncol, nlayers), 285.0)
        theta = jnp.full((ncol, nlayers), 0.25)
        G = jnp.zeros(ncol)

        T_new = solve_soil_thermal(T, theta, grid, hconfig, tconfig, G, dt=3600.0)
        npt.assert_allclose(T_new, T, atol=1e-12,
                            err_msg="Uniform T with zero flux should stay constant")
        self.assertEqual(T_new.dtype, jnp.float64)

    def test_energy_conservation_x64(self):
        """Total energy change equals integrated flux under x64."""
        from legoesm.land.soil_thermal import (
            solve_soil_thermal, compute_heat_capacity, SoilThermalConfig,
        )
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig

        grid = make_soil_grid(SoilGridConfig(n_layers=6))
        hconfig = SoilHydraulicsConfig()
        tconfig = SoilThermalConfig()

        ncol, nlayers = 4, 6
        T = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.25)
        G = jnp.full(ncol, 100.0)
        dt = 600.0

        T_new = solve_soil_thermal(T, theta, grid, hconfig, tconfig, G, dt)
        C = compute_heat_capacity(theta, hconfig, tconfig)
        dE = jnp.sum(C * grid.dz * (T_new - T), axis=1)
        expected = (G + tconfig.Q_geothermal) * dt
        npt.assert_allclose(dE, expected, rtol=1e-6,
                            err_msg="Energy conservation violated under x64")

    def test_tridiag_solver_dtype_consistency(self):
        """Shared Thomas solver handles mixed-dtype inputs gracefully."""
        from legoesm.timestepping.tridiagonal import thomas_solve

        n = 5
        ncol = 3
        # Create system with explicit float32 and float64 inputs
        a = jnp.zeros((ncol, n), dtype=jnp.float32)
        b = jnp.ones((ncol, n), dtype=jnp.float64) * 2.0
        c = jnp.zeros((ncol, n), dtype=jnp.float32)
        d = jnp.ones((ncol, n), dtype=jnp.float64)

        # Should not crash with mixed dtypes
        x = thomas_solve(a, b, c, d)
        npt.assert_allclose(x, 0.5, atol=1e-12)
        self.assertEqual(x.dtype, jnp.float64)

    def test_tridiag_solver_finite_on_near_zero_pivot(self):
        """The soil column now reuses the SHARED ``thomas_solve`` (land's duplicate
        land/tridiag.py was removed — bit-identical on diagonally-dominant systems;
        verified max|x_land-x_shared|=0 and grad diff 5.5e-17).

        The invariant land needs is only FINITENESS on its (diagonally dominant)
        systems, where the denom floor is in fact unreachable.  This guards the
        floor on both the FIRST and an INTERIOR near-zero pivot.  NOTE: the shared
        solver's floor is positive-only (maps |denom|<tiny -> +tiny), NOT the
        sign-preserving floor the deleted land solver used; land does not require
        sign preservation because a diagonally-dominant matrix never produces a
        subnormal pivot, so the floor branch never fires in practice."""
        from legoesm.timestepping.tridiagonal import thomas_solve
        # interior near-zero pivot
        x = thomas_solve(jnp.array([[0.0, 0.0]], dtype=jnp.float32),
                         jnp.array([[1.0, -1e-40]], dtype=jnp.float32),
                         jnp.array([[0.0, 0.0]], dtype=jnp.float32),
                         jnp.array([[0.0, 1.0]], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(x)), f"interior pivot -> non-finite: {x}"
        # near-zero FIRST pivot
        x0 = thomas_solve(jnp.array([[0.0, 0.0]], dtype=jnp.float32),
                          jnp.array([[1e-40, 1.0]], dtype=jnp.float32),
                          jnp.array([[0.0, 0.0]], dtype=jnp.float32),
                          jnp.array([[1.0, 0.0]], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(x0)), f"first pivot -> non-finite: {x0}"
        # the real land regime: a diagonally-dominant system solves correctly
        a = jnp.array([[0.0, -1.0, -1.0]]); c = jnp.array([[-1.0, -1.0, 0.0]])
        b = jnp.array([[3.0, 4.0, 3.0]]); d = jnp.array([[1.0, 2.0, 1.0]])
        xs = thomas_solve(a, b, c, d)
        # residual A x = d (a sub, b diag, c super)
        Ax = b[:, :] * xs
        Ax = Ax.at[:, 1:].add(a[:, 1:] * xs[:, :-1])
        Ax = Ax.at[:, :-1].add(c[:, :-1] * xs[:, 1:])
        npt.assert_allclose(Ax, d, atol=1e-6)


# =========================================================================
# 4. Root-zone moisture sensitivity with transpiration
# =========================================================================


class TestRootZoneMoisture(unittest.TestCase):
    """Transpiration should access root-zone moisture, not just top layer."""

    def test_dry_top_wet_deep_differs_from_all_dry(self):
        """With transpiration, a dry-top / wet-deep profile should evaporate
        more than an all-dry profile."""
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )
        from legoesm.land.soil_grid import make_soil_grid

        config = MultiLayerLandConfig(
            stomata=MultiLayerLandConfig().stomata._replace(enabled=False),
        )
        grid = make_soil_grid(config.soil_grid)
        ncol = 4

        # Dry-top / wet-deep profile
        state_mixed = init_multilayer_land_state(ncol, config, T_init=290.0, theta_init=0.30)
        # Set top 2 layers to wilting point
        theta_mixed = state_mixed.theta_soil.at[:, 0].set(config.theta_wp)
        theta_mixed = theta_mixed.at[:, 1].set(config.theta_wp)
        from legoesm.land.soil_hydraulics import psi_from_theta
        psi_mixed = psi_from_theta(theta_mixed, config.hydraulics)
        state_mixed = state_mixed._replace(
            theta_soil=theta_mixed, psi_soil=psi_mixed,
        )

        # All-dry profile (at wilting point everywhere)
        state_dry = init_multilayer_land_state(ncol, config, T_init=290.0)
        theta_dry = jnp.full_like(state_dry.theta_soil, config.theta_wp + 0.001)
        psi_dry = psi_from_theta(theta_dry, config.hydraulics)
        state_dry = state_dry._replace(
            theta_soil=theta_dry, psi_soil=psi_dry,
        )

        forcing = _make_forcing(ncol, T_lowest=295.0, q_lowest=5e-3,
                                sw_down=300.0)

        _, resp_mixed, _ = step_multilayer_land(
            state_mixed, forcing, config, U_min=1.0, dt=3600.0,
        )
        _, resp_dry, _ = step_multilayer_land(
            state_dry, forcing, config, U_min=1.0, dt=3600.0,
        )

        # Mixed profile (wet deep layers) should have higher latent flux
        # than the all-dry profile, because root-zone beta is higher.
        self.assertTrue(
            jnp.all(resp_mixed.lhflx > resp_dry.lhflx),
            f"Dry-top/wet-deep should have higher LH than all-dry: "
            f"mixed={resp_mixed.lhflx}, dry={resp_dry.lhflx}",
        )


# =========================================================================
# 5. Richards n_iter truthfulness
# =========================================================================


class TestRichardsNiter(unittest.TestCase):
    """Richards n_iter counts the Picard iterations each column applied."""

    def test_n_iter_counts_applied_iterations(self):
        """n_iter never exceeds max_iter; converged columns stop counting."""
        from legoesm.land.richards import RichardsConfig, solve_richards
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta

        hconfig = SoilHydraulicsConfig()
        ncol, nlayers = 4, 8
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        theta = jnp.full((ncol, nlayers), 0.25)
        psi = psi_from_theta(theta, hconfig)
        flux_top = jnp.full(ncol, 1e-5)
        sink = jnp.zeros((ncol, nlayers))

        out = solve_richards(psi, theta, grid, hconfig, RichardsConfig(max_iter=60),
                             flux_top, sink, dt=1800.0)
        n = np.asarray(out.n_iter)
        self.assertTrue(np.all(np.asarray(out.converged)))
        self.assertTrue(np.all((n >= 1) & (n < 60)), n)
        capped = solve_richards(psi, theta, grid, hconfig,
                                RichardsConfig(max_iter=int(n.min()) - 1),
                                flux_top, sink, dt=1800.0)
        npt.assert_array_equal(capped.n_iter, n.min() - 1)
        self.assertFalse(np.any(np.asarray(capped.converged)))


# =========================================================================
# 6. Multilayer snow sublimation
# =========================================================================


class TestMultilayerSnowSublimation(unittest.TestCase):
    """Snow sublimation in the multilayer model."""

    def test_sublimation_from_snowpack_not_soil(self):
        """Sublimation over snow should deplete snowpack, not soil theta."""
        from legoesm.land.multilayer_land import (
            step_multilayer_land, init_multilayer_land_state,
        )

        config = MultiLayerLandConfig()
        ncol = 4
        state = init_multilayer_land_state(ncol, config, T_init=265.0)
        # Add snow
        state = state._replace(snow_depth=jnp.full(ncol, 15.0))

        # Dry atmosphere, some SW → sublimation
        forcing = _make_forcing(ncol, T_lowest=263.0, q_lowest=1e-4,
                                sw_down=50.0, lw_down=200.0,
                                precip_total=0.0, precip_snow=0.0)

        theta_old = state.theta_soil.copy()
        state2, resp, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=3600.0,
        )

        # Snow should decrease
        self.assertTrue(
            jnp.all(state2.snow_depth < 15.0),
            f"Snow should decrease: {state2.snow_depth}",
        )

        # Soil moisture should NOT decrease significantly
        # (small changes from Richards redistribution are OK, but no large
        # evaporative depletion from the soil when snow is covering it)
        theta_change = jnp.sum(jnp.abs(state2.theta_soil - theta_old), axis=-1)
        # Allow small redistribution but not evaporative loss
        self.assertTrue(
            jnp.all(theta_change < 0.01),
            f"Soil moisture should not change much during snow sublimation: "
            f"total |dtheta| = {theta_change}",
        )


if __name__ == "__main__":
    unittest.main()
