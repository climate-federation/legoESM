"""Shallow water 4D-Var twin experiment on lat-lon grid.

Setup:
1. Truth: Williamson TC2 geostrophic flow + small height perturbation
2. Run truth forward N steps
3. Sample synthetic height observations at random grid points
4. Background: unperturbed Williamson TC2
5. Run incremental 4D-Var
6. Verify: analysis RMSE(h) < background RMSE(h)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids import create_latlon_grid
from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    FVShallowWaterLatLonConfig,
)
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation, generate_synthetic_obs
from legoesm.da.incremental import incremental_4dvar, IncrementalConfig

# Import test case
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "atmosphere", "shallow_water", "test_cases"))
from williamson_latlon import williamson_test2_latlon


@pytest.fixture(scope="module")
def grid():
    return create_latlon_grid(16, 32)


@pytest.fixture(scope="module")
def model(grid):
    config = FVShallowWaterLatLonConfig(
        use_conservation_fixer=False,
        fix_mass=False,
        fix_energy=False,
    )
    return FVShallowWaterLatLonModel(grid, config)


class TestSWTwinExperiment:
    def test_4dvar_reduces_rmse(self, grid, model):
        """Twin experiment: 4D-Var analysis should be closer to truth than background."""
        # Background: Williamson TC2
        bg_state = williamson_test2_latlon(grid)

        # Truth: background + small h perturbation
        key = jax.random.PRNGKey(42)
        h_pert = jax.random.normal(key, bg_state.h.data.shape) * 10.0  # 10m perturbation
        truth_state = bg_state._replace(
            h=bg_state.h.replace(data=bg_state.h.data + h_pert)
        )

        # Forward integrate truth for 5 steps
        dt = 600.0
        n_steps = 5

        def scan_step(carry, _):
            s = model.step(carry, dt)
            return s, s

        _, truth_traj = jax.lax.scan(scan_step, truth_state, jnp.arange(n_steps))

        # Create observations: observe h at 50 random grid points every other step
        n_obs = 50
        key, subkey = jax.random.split(key)
        obs_lat_idx = jax.random.randint(subkey, (n_obs,), 0, grid.n_lat)
        key, subkey = jax.random.split(key)
        obs_lon_idx = jax.random.randint(subkey, (n_obs,), 0, grid.n_lon)
        idx = (obs_lat_idx, obs_lon_idx)

        obs_times = [1, 3]  # Observe at step 1 and step 3
        observations = []
        for t_idx in obs_times:
            state_t = jax.tree.map(lambda a: a[t_idx], truth_traj)
            h_obs = state_t.h.data[idx]
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, h_obs.shape) * 5.0
            op = DirectObsOperator("h", idx)
            obs = Observation(
                values=h_obs + noise,
                errors=jnp.full(n_obs, 5.0),
                time_index=t_idx,
                operator=op,
            )
            observations.append(obs)

        # Build control spec and B
        spec = build_control_spec(bg_state, fields=("h", "u", "v"))
        x_b = state_to_control(bg_state, spec)
        sigma = jnp.ones(spec.total_size) * 20.0
        B = DiagonalB(sigma=sigma)

        # Run incremental 4D-Var
        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-5,
            inner_method="lbfgs", use_preconditioning=False,
            checkpoint_every=1,
        )

        analysis, diag = incremental_4dvar(
            model, bg_state, tuple(observations), B, spec,
            dt=dt, n_steps=n_steps, config=config,
        )

        # Compare RMSE
        rmse_bg = float(jnp.sqrt(jnp.mean((bg_state.h.data - truth_state.h.data) ** 2)))
        rmse_ana = float(jnp.sqrt(jnp.mean((analysis.h.data - truth_state.h.data) ** 2)))

        assert rmse_ana < rmse_bg, (
            f"Analysis RMSE ({rmse_ana:.4f}) should be less than "
            f"background RMSE ({rmse_bg:.4f})"
        )

    def test_cost_decreases_over_outer_loops(self, grid, model):
        """Cost function should decrease with each outer iteration."""
        bg_state = williamson_test2_latlon(grid)
        spec = build_control_spec(bg_state, fields=("h",))
        sigma = jnp.ones(spec.total_size) * 20.0
        B = DiagonalB(sigma=sigma)

        # Single obs
        idx = (jnp.array([8]), jnp.array([16]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=bg_state.h.data[idx] + 50.0,  # 50m offset
            errors=jnp.array([10.0]),
            time_index=0, operator=op,
        )

        config = IncrementalConfig(
            n_outer=3, n_inner=20, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        _, diag = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=600.0, n_steps=1, config=config,
        )

        # Cost should decrease
        for i in range(1, len(diag.cost_history)):
            assert diag.cost_history[i] <= diag.cost_history[i - 1] + 1e-4
