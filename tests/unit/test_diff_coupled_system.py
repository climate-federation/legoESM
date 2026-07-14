"""CATEGORY 8 — Coupled-system end-to-end gradient chains.

These tests verify that gradients flow correctly through the FULL coupled
system — the most important tests for 4D-Var and parameter estimation.

  8a) Atmosphere dynamics + physics (held_suarez)
  8b) Atmosphere -> coupler -> land (T_lowest -> T_soil)
  8c) Atmosphere -> coupler -> ocean (SST sensitivity, correct sign)
  8d) Atmosphere -> coupler -> sea ice -> albedo feedback
  8e) Full AMIP-like chain (dynamics + physics + land)
  8f) DA cost function with a coupled atmosphere+coupler forward model

Run with:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/unit/test_diff_coupled_system.py -v --tb=short

Small grids only (C4, 5 levels, 32 columns) so the suite runs in seconds.
Differentiation is always w.r.t. the ``.data`` array inside Field-wrapped
states (see CLAUDE.md / the test-differentiability "Key pattern").
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    """Assert a gradient is finite and has the expected non-zero fraction."""
    grad_array = jnp.asarray(grad_array)
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac * 100:.1f}% non-zero "
        f"(need {min_nonzero_frac * 100:.0f}%)"
    )


# ---------------------------------------------------------------------------
# Shared builders for the cubed-sphere hydrostatic PE state / model
# ---------------------------------------------------------------------------

_N = 4      # C4 cubed sphere
_NLEV = 5


def _build_cdgrid_pe():
    """Build a small cubed-sphere hydrostatic PE model + isothermal state."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )
    from legoesm.core.state import FV3HydrostaticState

    n, nlev = _N, _NLEV
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    model = CDGridPrimitiveEquationModel(grid, sigma)

    key = jax.random.PRNGKey(100)
    T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 1.0 * jax.random.normal(
        key, (6, n, n, nlev)
    )
    state = FV3HydrostaticState(
        u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
        v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
        T=Field(T_data, name="T"),
        p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
    )
    return model, state, grid, sigma


def _hs_temperature_tendency(fv3_state, grid, sigma):
    """Held-Suarez dT/dt for an FV3 hydrostatic state (A-grid, zero winds).

    Held-Suarez only reads T, p_s, phis (and lat from the grid) for its
    temperature relaxation, so a zero-wind A-grid HydrostaticState is a
    faithful input for the thermal tendency.
    """
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing
    from legoesm.core.state import HydrostaticState

    n, nlev = _N, _NLEV
    hs_state = HydrostaticState(
        u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
        v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
        T=fv3_state.T,
        p_s=fv3_state.p_s,
        phis=fv3_state.phis,
    )
    return held_suarez_forcing(hs_state, grid, sigma).dT_dt.data


# ============================================================================
# 8a  Atmosphere dynamics + Held-Suarez physics
# ============================================================================

class TestAtmDynPlusPhysics:

    def test_grad_dynamics_plus_hs_one_step(self):
        model, state, grid, sigma = _build_cdgrid_pe()
        dt = 60.0

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            s = model.step(s, dt)                       # dynamics
            dT = _hs_temperature_tendency(s, grid, sigma)  # physics
            T_final = s.T.data + dt * dT
            return jnp.sum(T_final ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "8a dynamics + Held-Suarez (1 step)")
        assert not jnp.all(grad == grad.ravel()[0]), (
            "8a: gradient has no spatial structure"
        )

    def test_grad_dynamics_plus_hs_three_steps(self):
        model, state, grid, sigma = _build_cdgrid_pe()
        dt = 60.0

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            for _ in range(3):
                s = model.step(s, dt)
                dT = _hs_temperature_tendency(s, grid, sigma)
                s = s._replace(T=s.T.replace(data=s.T.data + dt * dT))
            return jnp.sum(s.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "8a dynamics + Held-Suarez (3 steps)")
        assert not jnp.all(grad == grad.ravel()[0]), (
            "8a (3-step): gradient has no spatial structure"
        )


# ============================================================================
# 8b  Atmosphere -> coupler -> land (T_lowest -> T_soil)
# ============================================================================

class TestAtmCouplerLand:

    @staticmethod
    def _build():
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState

        ncol = 32
        config = LandConfig()
        state = LandState(
            T_soil=Field(280.0 * jnp.ones(ncol), name="T_soil"),
            W_bucket=Field(50.0 * jnp.ones(ncol), name="W_bucket"),
            snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
            snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        )
        return ncol, config, state

    @staticmethod
    def _forcing(ncol, T_lowest):
        from legoesm.core.coupling_fields import AtmToSurface

        ones = jnp.ones(ncol)
        return AtmToSurface(
            sw_down=200.0 * ones, lw_down=300.0 * ones,
            precip_total=1e-5 * ones, precip_snow=0.0 * ones,
            T_lowest=T_lowest, q_lowest=5e-3 * ones,
            u_lowest=5.0 * ones, v_lowest=2.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )

    def test_grad_T_lowest_to_T_soil(self):
        from legoesm.land.slab_land import step_land

        ncol, config, state = self._build()

        def loss(T_lowest):
            forcing = self._forcing(ncol, T_lowest)
            out, _, _ = step_land(state, forcing, config, U_min=1.0, dt=300.0)
            return jnp.sum(out.T_soil.data ** 2)

        T_lowest = 280.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T_lowest)
        assert_gradient_ok(grad, "8b T_lowest -> T_soil")
        # Warmer overlying air -> warmer soil: positive sensitivity.
        assert jnp.mean(grad) > 0, (
            "8b: expected positive gradient (warmer air -> warmer soil)"
        )

    def test_grad_sw_down_to_T_soil(self):
        """Cross-component sensitivity: surface insolation -> soil temp."""
        from legoesm.land.slab_land import step_land
        from legoesm.core.coupling_fields import AtmToSurface

        ncol, config, state = self._build()
        ones = jnp.ones(ncol)

        def loss(sw_down):
            forcing = AtmToSurface(
                sw_down=sw_down, lw_down=300.0 * ones,
                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
                T_lowest=280.0 * ones, q_lowest=5e-3 * ones,
                u_lowest=5.0 * ones, v_lowest=2.0 * ones,
                p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
                rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
            )
            out, _, _ = step_land(state, forcing, config, U_min=1.0, dt=300.0)
            return jnp.sum(out.T_soil.data ** 2)

        sw = 200.0 * ones
        grad = jax.grad(loss)(sw)
        assert_gradient_ok(grad, "8b sw_down -> T_soil")
        # More absorbed shortwave -> warmer soil: positive sensitivity.
        assert jnp.mean(grad) > 0, (
            "8b: expected positive gradient (more sunlight -> warmer soil)"
        )


# ============================================================================
# 8c  Atmosphere -> coupler -> ocean (SST sensitivity, via ocean_tile_response)
# ============================================================================

class TestSSTToOceanFlux:

    @staticmethod
    def _build_forcing(shape, T_atm):
        from legoesm.core.coupling_fields import AtmToSurface

        ones = jnp.ones(shape)
        return AtmToSurface(
            sw_down=200.0 * ones, lw_down=350.0 * ones,
            precip_total=1e-5 * ones, precip_snow=0.0 * ones,
            T_lowest=T_atm, q_lowest=5e-3 * ones,
            u_lowest=6.0 * ones, v_lowest=3.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )

    def test_grad_sst_to_shflx_correct_sign(self):
        """d(shflx)/d(SST) through the coupler ocean tile.

        shflx is UPWARD-POSITIVE; in the unstable regime (SST > T_atm)
        warmer SST -> larger upward sensible heat -> positive sensitivity.
        """
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.config import CouplerConfig

        shape = (32,)
        config = CouplerConfig(bulk_scheme="coare3", bulk_n_iter=5)
        T_atm = 290.0 * jnp.ones(shape)         # cooler air than SST
        forcing = self._build_forcing(shape, T_atm)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        def loss(sst):
            resp = ocean_tile_response(forcing, sst, ocean_u, ocean_v, config)
            return jnp.sum(resp.shflx)

        sst = 296.0 * jnp.ones(shape)           # SST > T_atm (unstable)
        grad = jax.grad(loss)(sst)
        assert_gradient_ok(grad, "8c SST -> shflx (coupler ocean tile)")
        # Correct physical sign: warmer SST -> more upward sensible heat.
        assert jnp.mean(grad) > 0, (
            "8c: expected positive d(shflx)/d(SST) when SST > T_atm "
            f"(got mean {jnp.mean(grad).item():.4g})"
        )

    @pytest.mark.parametrize("n_iter", [1, 3, 5, 10])
    def test_grad_sst_through_most_iterations(self, n_iter):
        """Gradient survives the MOST jax.lax.fori_loop for all iter counts."""
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.config import CouplerConfig

        shape = (16,)
        config = CouplerConfig(bulk_scheme="coare3", bulk_n_iter=n_iter)
        forcing = self._build_forcing(shape, 290.0 * jnp.ones(shape))
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        def loss(sst):
            resp = ocean_tile_response(forcing, sst, ocean_u, ocean_v, config)
            return jnp.sum(resp.shflx ** 2)

        sst = 296.0 * jnp.ones(shape)
        grad = jax.grad(loss)(sst)
        assert_gradient_ok(grad, f"8c SST -> shflx (n_iter={n_iter})")


# ============================================================================
# 8d  Atmosphere -> coupler -> sea ice -> albedo feedback
# ============================================================================

class TestIceAlbedoChain:

    def test_sw_down_ice_albedo_feedback(self):
        """sw_down -> ice_step -> T_ice change -> albedo change -> absorbed SW.

        Captures the (positive) ice-albedo feedback: with a temperature
        dependent albedo, warming reduces albedo and increases absorption.
        """
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.core.coupling_fields import AtmToSurface

        config = SeaIceConfig(dynamics="none", temp_dependent_albedo=True)
        shape = (6, 4, 4)
        ones = jnp.ones(shape)

        state = SeaIceState(
            h_ice=Field(1.0 * ones, name="h_ice"),
            T_ice=Field(268.0 * ones, name="T_ice"),
            concentration=Field(0.8 * ones, name="concentration"),
        )

        def loss(sw_down):
            forcing = AtmToSurface(
                sw_down=sw_down, lw_down=250.0 * ones,
                precip_total=0.0 * ones, precip_snow=0.0 * ones,
                T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
                u_lowest=5.0 * ones, v_lowest=2.0 * ones,
                p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
                rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
            )
            _, response = step_sea_ice(
                state, forcing, constants.T_freeze_ocean * ones,
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            # Absorbed SW = (1 - albedo) * sw_down
            return jnp.sum((1.0 - response.albedo) * sw_down)

        sw = 100.0 * ones
        grad = jax.grad(loss)(sw)
        assert jnp.all(jnp.isfinite(grad)), "8d: ice albedo chain grad not finite"
        assert jnp.any(grad != 0), "8d: ice albedo chain gradient all zero"


# ============================================================================
# 8e  Full AMIP-like chain (small): dynamics + physics + land
# ============================================================================

class TestFullAMIPChain:

    def test_dynamics_plus_physics_plus_land(self):
        """dynamics.step -> Held-Suarez -> land step -> scalar loss.

        Gradient through the entire small atmosphere + surface chain.
        """
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.core.coupling_fields import AtmToSurface

        model, state, grid, sigma = _build_cdgrid_pe()
        dt = 60.0
        n = _N

        land_config = LandConfig()
        shape2d = (6, n, n)
        ones = jnp.ones(shape2d)
        land_state = LandState(
            T_soil=Field(280.0 * ones, name="T_soil"),
            W_bucket=Field(50.0 * ones, name="W_bucket"),
            snow_depth=Field(jnp.zeros(shape2d), name="snow_depth"),
            snow_age=Field(jnp.zeros(shape2d), name="snow_age"),
        )

        def loss(T_init):
            s = state._replace(T=state.T.replace(data=T_init))
            # 1. dynamics step
            s = model.step(s, dt)
            # 2. physics (Held-Suarez tendency)
            dT = _hs_temperature_tendency(s, grid, sigma)
            T_after_phys = s.T.data + dt * dT
            # 3. lowest-level temperature drives the land surface
            T_lowest = T_after_phys[:, :, :, -1]
            forcing = AtmToSurface(
                sw_down=200.0 * ones, lw_down=300.0 * ones,
                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
                T_lowest=T_lowest, q_lowest=5e-3 * ones,
                u_lowest=5.0 * ones, v_lowest=2.0 * ones,
                p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
                rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
            )
            land_out, _, _ = step_land(
                land_state, forcing, land_config, U_min=1.0, dt=dt
            )
            return jnp.sum(T_after_phys ** 2) + jnp.sum(land_out.T_soil.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "8e full AMIP chain: dynamics + physics + land")
        assert not jnp.all(grad == grad.ravel()[0]), (
            "8e: gradient has no spatial structure"
        )


# ============================================================================
# 8f  DA cost function with a coupled atmosphere + physics forward model
# ============================================================================

class _CoupledHSModel:
    """Forward model with a coupled dynamics + Held-Suarez physics step.

    Exposes the ``.step(state, dt)`` contract expected by the DA cost
    builder while running an extra physics coupling (Held-Suarez thermal
    relaxation) every step.  This makes the 4D-Var forward operator the
    full coupled atmosphere column, not pure dynamics.
    """

    def __init__(self, dyn_model, grid, sigma):
        self._dyn = dyn_model
        self._grid = grid
        self._sigma = sigma

    def step(self, state, dt):
        s = self._dyn.step(state, dt)
        dT = _hs_temperature_tendency(s, self._grid, self._sigma)
        return s._replace(T=s.T.replace(data=s.T.data + dt * dT))


class TestDACostCoupledModel:

    def _setup(self, n_steps):
        from legoesm.da.control_vector import build_control_spec, state_to_control
        from legoesm.da.background_error import DiagonalB
        from legoesm.da.observation import Observation, DirectObsOperator
        from legoesm.da.cost_function import build_cost_fn

        model, truth_state, grid, sigma = _build_cdgrid_pe()
        coupled = _CoupledHSModel(model, grid, sigma)
        dt = 60.0

        # Control vector: temperature field only (keeps the problem small
        # and well-conditioned for the cost-decrease check).
        spec = build_control_spec(truth_state, fields=("T",))

        # Synthetic truth trajectory -> observations of T at a few points.
        traj = [truth_state]
        s = truth_state
        for _ in range(n_steps):
            s = coupled.step(s, dt)
            traj.append(s)

        # Observe a handful of T points at the final time.
        idx = (jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]))
        op = DirectObsOperator("T", idx)
        obs_values = op(traj[n_steps])
        obs = Observation(
            values=obs_values,
            errors=0.5 * jnp.ones_like(obs_values),
            time_index=n_steps - 1,   # index into the n_steps-long trajectory
            operator=op,
        )

        # Background = truth perturbed in T; control vector matches spec.
        key = jax.random.PRNGKey(7)
        T_bg = truth_state.T.data + 2.0 * jax.random.normal(
            key, truth_state.T.data.shape
        )
        bg_state = truth_state._replace(T=truth_state.T.replace(data=T_bg))
        background = state_to_control(bg_state, spec)

        B = DiagonalB(sigma=4.0 * jnp.ones(spec.total_size))

        cost_fn = build_cost_fn(
            coupled, background, (obs,), B, spec, truth_state,
            dt=dt, n_steps=n_steps, checkpoint=True,
        )
        return cost_fn, background, spec

    def test_cost_gradient_finite(self):
        cost_fn, background, _ = self._setup(n_steps=3)
        grad = jax.grad(cost_fn)(background)
        assert_gradient_ok(grad, "8f coupled 4D-Var cost gradient")

    def test_lbfgs_cost_decreases(self):
        """5 L-BFGS iterations on the coupled-model cost must reduce J."""
        from legoesm.da.cost_function import build_cost_and_grad_fn
        from legoesm.da.background_error import DiagonalB
        from legoesm.da.control_vector import build_control_spec, state_to_control
        from legoesm.da.observation import Observation, DirectObsOperator
        from legoesm.da.minimizer import minimize_lbfgs

        model, truth_state, grid, sigma = _build_cdgrid_pe()
        coupled = _CoupledHSModel(model, grid, sigma)
        dt = 60.0
        n_steps = 3

        spec = build_control_spec(truth_state, fields=("T",))

        s = truth_state
        for _ in range(n_steps):
            s = coupled.step(s, dt)
        final_truth = s

        idx = (jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]),
               jnp.array([0, 1, 2, 3]))
        op = DirectObsOperator("T", idx)
        obs_values = op(final_truth)
        obs = Observation(
            values=obs_values,
            errors=0.5 * jnp.ones_like(obs_values),
            time_index=n_steps - 1,
            operator=op,
        )

        key = jax.random.PRNGKey(11)
        T_bg = truth_state.T.data + 2.0 * jax.random.normal(
            key, truth_state.T.data.shape
        )
        bg_state = truth_state._replace(T=truth_state.T.replace(data=T_bg))
        background = state_to_control(bg_state, spec)
        B = DiagonalB(sigma=4.0 * jnp.ones(spec.total_size))

        cost_and_grad = build_cost_and_grad_fn(
            coupled, background, (obs,), B, spec, truth_state,
            dt=dt, n_steps=n_steps, checkpoint=True,
        )

        J0, g0 = cost_and_grad(background)
        assert jnp.isfinite(J0), "8f: initial cost not finite"
        assert jnp.all(jnp.isfinite(g0)), "8f: initial gradient not finite"

        result = minimize_lbfgs(cost_and_grad, background, max_iter=5)
        assert jnp.isfinite(result.fun), "8f: final cost not finite"
        # The optimizer must make progress on the coupled-model cost.
        assert result.fun < J0, (
            f"8f: L-BFGS did not reduce cost (J0={J0.item():.6g}, "
            f"J_final={result.fun.item():.6g})"
        )
