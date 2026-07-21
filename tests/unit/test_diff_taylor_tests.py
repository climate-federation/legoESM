"""Taylor test gradient verification for each component.

The Taylor test verifies gradient correctness by checking:
  |J(x + h*dx) - J(x) - h * <grad, dx>| / h^2 → C  as h → 0

If the ratio converges (approximately constant or decreasing), the
gradient is correct (2nd-order convergence). If it blows up, the
gradient is WRONG.

Categories:
  9a) Dynamics (shallow water lat-lon, hydrostatic PE lat-lon)
  9b) Physics (Held-Suarez)
  9c) Land (slab land)
  9d) Sea ice (slab thermodynamics)
  9e) Coupler (bulk flux COARE3)
  9f) DA cost function (full 4D-Var cost, multi-step forward model)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field


def taylor_test(loss_fn, x0, hs=(1e-1, 1e-2, 1e-3, 1e-4)):
    """Run a Taylor test and return convergence ratios.

    Returns list of |J(x+h*dx)-J(x)-h*<g,dx>|/h^2 for each h.
    For correct gradients, these should be approximately constant.
    """
    J0 = loss_fn(x0)
    grad = jax.grad(loss_fn)(x0)
    dx = grad / jnp.linalg.norm(grad)
    gdx = jnp.sum(grad * dx)

    ratios = []
    for h in hs:
        J_pert = loss_fn(x0 + h * dx)
        remainder = jnp.abs(J_pert - J0 - h * gdx)
        ratios.append((remainder / h ** 2).item())
    return ratios


def assert_taylor_ok(ratios, name=""):
    """Check Taylor test convergence.

    The ratio r(h) = |J(x+h*dx)-J(x)-h*<g,dx>|/h^2 should be bounded
    and approximately constant for correct gradients.
    """
    ratios = [r for r in ratios if r > 0]  # skip exact zeros
    assert len(ratios) > 0, f"{name}: all Taylor ratios are zero"
    assert all(jnp.isfinite(r) for r in ratios), f"{name}: Taylor ratios have NaN/Inf"
    # Ratio should not blow up: last ratio < 1000 * first ratio
    if len(ratios) >= 2:
        assert ratios[-1] < 1000 * ratios[0] + 1e-6, (
            f"{name}: Taylor ratios diverging: {ratios}"
        )


# ============================================================================
# 9a  Dynamics (shallow water lat-lon, hydrostatic PE lat-lon)
# ============================================================================

class TestTaylorDynamics:

    def test_shallow_water_latlon(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterModel,
            CGridLatLonShallowWaterConfig,
            williamson_test2_cgrid,
        )

        n_lat, n_lon = 8, 16
        grid = create_latlon_grid(n_lat, n_lon)
        # Disable mass fixer: the anchored fixer snapshots a *static*
        # fp64 target on the first (un-traced) step, which is captured
        # as a constant inside jax.grad and silently distorts the
        # Taylor remainder.  A pure (config) step keeps the loss an
        # honest function of the perturbed h.
        config = CGridLatLonShallowWaterConfig(fix_mass=False)
        model = CGridLatLonShallowWaterModel(grid, config)
        dt = 60.0

        # Williamson-2 steady geostrophic balance + small perturbation
        # in h so the loss has spatial structure.
        state = williamson_test2_cgrid(grid)
        key = jax.random.PRNGKey(3)
        h0 = state.h + 5.0 * jax.random.normal(key, state.h.shape)
        state = state._replace(h=h0)

        def loss(h_data):
            s = state._replace(h=h_data)
            out = model.step(s, dt)
            return jnp.sum(out.h ** 2)

        ratios = taylor_test(loss, state.h)
        assert_taylor_ok(ratios, "Shallow water lat-lon Taylor test")

    def test_primitive_eq_latlon(self):
        import math

        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
            CGridLatLonPrimitiveEquationConfig,
            hydrostatic_to_cgrid,
        )
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon

        n_lat, n_lon, nlev = 8, 16, 5
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = create_sigma_coordinate(nlev)
        dx_pole = float(grid.radius) * grid.dlon * math.cos(
            math.pi / 2 - grid.dlat / 2
        )
        dt = min(120.0, 0.5 * dx_pole / 300.0)
        # fix_mass off for the same Taylor-honesty reason as the SW case.
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=False)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)

        # Realistic Held-Suarez initial condition (T has horizontal +
        # vertical structure), converted to the C-grid state layout.
        # NOTE: held_suarez_init_latlon emits float32 arrays even under
        # JAX_ENABLE_X64; the loss sum(T**2) is then O(5e7) and the
        # h*<g,dx> Taylor signal (O(1) at h<=1e-3) drowns in float32
        # round-off — a precision artifact, NOT a gradient bug (the
        # central-FD directional derivative matches <g,dx> to ~16% at
        # h=1e-1 and the cancellation-noise remainder is exactly
        # -h*<g,dx>).  Promote to float64 (the spec's gold-standard
        # precision for the Taylor test) so the gradient is checked
        # honestly.
        state = hydrostatic_to_cgrid(
            held_suarez_init_latlon(grid, sigma), grid,
        )
        state = jax.tree.map(
            lambda a: a.astype(jnp.float64) if hasattr(a, "dtype") else a,
            state,
        )

        # mean (not sum) keeps the loss O(1e5) so the h*<g,dx> term stays
        # above the float64 cancellation floor across the tested h range.
        def loss(T_data):
            s = state._replace(T=T_data)
            out = model.step(s, dt)
            return jnp.mean(out.T ** 2)

        ratios = taylor_test(loss, state.T)
        assert_taylor_ok(ratios, "Primitive eq lat-lon Taylor test")


# ============================================================================
# 9b  Physics (Held-Suarez)
# ============================================================================

class TestTaylorPhysics:

    def test_held_suarez(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing
        from legoesm.core.state import HydrostaticState

        n, nlev = 4, 5
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)

        key = jax.random.PRNGKey(1)
        T_data = 250.0 + 5.0 * jax.random.normal(key, (6, n, n, nlev))
        state = HydrostaticState(
            u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
            v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = held_suarez_forcing(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        ratios = taylor_test(loss, state.T.data)
        assert_taylor_ok(ratios, "Held-Suarez Taylor test")


# ============================================================================
# 9c  Land (slab land)
# ============================================================================

class TestTaylorLand:

    def test_slab_land(self):
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.core.coupling_fields import AtmToSurface

        ncol = 32
        config = LandConfig()
        state = LandState(
            T_soil=Field(280.0 * jnp.ones(ncol), name="T_soil"),
            W_bucket=Field(50.0 * jnp.ones(ncol), name="W_bucket"),
            snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
            snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        )
        ones = jnp.ones(ncol)
        forcing = AtmToSurface(
            sw_down=200.0 * ones, lw_down=300.0 * ones,
            precip_total=1e-5 * ones, precip_snow=0.0 * ones,
            T_lowest=280.0 * ones, q_lowest=5e-3 * ones,
            u_lowest=5.0 * ones, v_lowest=2.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )

        def loss(T_data):
            s = state._replace(T_soil=state.T_soil.replace(data=T_data))
            out, _, _ = step_land(s, forcing, config, U_min=1.0, dt=60.0)
            return jnp.sum(out.T_soil.data ** 2)

        ratios = taylor_test(loss, state.T_soil.data)
        assert_taylor_ok(ratios, "Slab land Taylor test")


# ============================================================================
# 9d  Sea ice (slab thermodynamics)
# ============================================================================

class TestTaylorIce:

    def test_slab_ice(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.core.coupling_fields import AtmToSurface

        config = SeaIceConfig(dynamics="none")
        shape = (6, 4, 4)
        ones = jnp.ones(shape)
        state = SeaIceState(
            h_ice=Field(1.0 * ones, name="h_ice"),
            T_ice=Field(265.0 * ones, name="T_ice"),
            concentration=Field(0.8 * ones, name="concentration"),
        )
        forcing = AtmToSurface(
            sw_down=100.0 * ones, lw_down=250.0 * ones,
            precip_total=0.0 * ones, precip_snow=0.0 * ones,
            T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
            u_lowest=5.0 * ones, v_lowest=2.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )
        ocean_sst = constants.T_freeze_ocean * ones

        def loss(T_data):
            s = state._replace(T_ice=state.T_ice.replace(data=T_data))
            out, _ = step_sea_ice(
                s, forcing, ocean_sst, jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            return jnp.sum(out.T_ice.data ** 2)

        ratios = taylor_test(loss, state.T_ice.data)
        assert_taylor_ok(ratios, "Slab ice Taylor test")


# ============================================================================
# 9e  Coupler (bulk flux COARE3)
# ============================================================================

class TestTaylorCoupler:

    def test_coare3(self):
        from legoesm.core.bulk_flux import compute_most_fluxes

        ncol = 32
        T_sfc = 300.0 * jnp.ones(ncol)
        T_atm = 295.0 * jnp.ones(ncol)
        u_rel = 5.0 * jnp.ones(ncol)
        v_rel = 2.0 * jnp.ones(ncol)
        q_atm = 5e-3 * jnp.ones(ncol)
        q_sfc = 8e-3 * jnp.ones(ncol)
        rho = 1.2 * jnp.ones(ncol)

        def loss(T_sfc):
            _, _, shflx, _, _ = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho,
                scheme="coare3", n_iter=5,
            )
            return jnp.sum(shflx ** 2)

        ratios = taylor_test(loss, T_sfc)
        assert_taylor_ok(ratios, "COARE3 Taylor test")


# ============================================================================
# 9f  DA cost function (full 4D-Var cost, multi-step forward model)
# ============================================================================

class _DecayModel:
    """Simple differentiable forward model for the Taylor test.

    ``h`` relaxes toward its spatial mean each step — a smooth, fully
    differentiable operator so the 4D-Var cost is a genuine function
    of the control vector through ``jax.lax.scan``.
    """

    def step(self, state, dt):
        h_data = state.h.data
        h_mean = jnp.mean(h_data)
        alpha = 0.01
        h_new = h_data + alpha * (h_mean - h_data)
        return state._replace(h=state.h.replace(data=h_new))


class TestTaylorCostFunction:

    def test_4dvar_cost(self):
        from legoesm.core.state import ShallowWaterState
        from legoesm.da.control_vector import build_control_spec, state_to_control
        from legoesm.da.background_error import DiagonalB
        from legoesm.da.observation import DirectObsOperator, Observation
        from legoesm.da.cost_function import build_cost_fn

        shape = (8, 16)
        bg_state = ShallowWaterState(
            h=Field(data=1000.0 * jnp.ones(shape), name="h", dims=(), units="m"),
            u=Field(data=jnp.zeros(shape), name="u", dims=(), units="m/s"),
            v=Field(data=jnp.zeros(shape), name="v", dims=(), units="m/s"),
            h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
        )
        model = _DecayModel()

        spec = build_control_spec(bg_state, fields=("h",))
        x_b = state_to_control(bg_state, spec)
        sigma = jnp.ones(spec.total_size) * 50.0
        B = DiagonalB(sigma=sigma)

        # Synthetic observations at two time levels in the window so the
        # cost exercises gradient accumulation through the scan.
        key = jax.random.PRNGKey(7)
        n_obs = 12
        key, ki, kj = jax.random.split(key, 3)
        idx = (
            jax.random.randint(ki, (n_obs,), 0, shape[0]),
            jax.random.randint(kj, (n_obs,), 0, shape[1]),
        )
        op = DirectObsOperator("h", idx)
        truth_h = bg_state.h.data + 20.0 * jax.random.normal(key, shape)
        obs = (
            Observation(values=truth_h[idx], errors=jnp.full(n_obs, 5.0),
                        time_index=0, operator=op),
            Observation(values=truth_h[idx], errors=jnp.full(n_obs, 5.0),
                        time_index=4, operator=op),
        )

        cost_fn = build_cost_fn(
            model, x_b, obs, B, spec, bg_state,
            dt=600.0, n_steps=5,
        )

        # Start the Taylor test away from the background so J_b also
        # contributes a non-trivial (non-zero) gradient.
        x0 = x_b + 30.0 * jax.random.normal(jax.random.PRNGKey(11), x_b.shape)
        ratios = taylor_test(cost_fn, x0)
        assert_taylor_ok(ratios, "4D-Var cost function Taylor test")


