"""Hydrostatic PE 4D-Var twin experiment on lat-lon grid.

Setup:
1. Truth: isothermal rest state + localized temperature perturbation
2. Forward integration
3. Observe T at grid points
4. Background: unperturbed state
5. 4D-Var with DiagonalB
6. Verify: analysis T closer to truth than background
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.incremental import incremental_4dvar, IncrementalConfig


class _SimpleHydroModel:
    """A very simple hydrostatic model: temperature diffuses toward mean."""

    def step(self, state, dt):
        # Simple relaxation: T -> T + alpha*(T_mean - T)
        T_data = state.T.data
        T_mean = jnp.mean(T_data, axis=(0, 1), keepdims=True)
        alpha = 0.01
        T_new = T_data + alpha * (T_mean - T_data)
        return state._replace(T=state.T.replace(data=T_new))


def _make_pe_state(n_lat=8, n_lon=16, nlev=5, T_base=280.0):
    """Create a simple hydrostatic state on lat-lon grid."""
    shape_2d = (n_lat, n_lon)
    shape_3d = (n_lat, n_lon, nlev)
    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=(), units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * T_base, name="T", dims=(), units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=(), units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=(), units="m2/s2"),
        tracers=None,
    )


class TestPETwinExperiment:
    def test_temperature_analysis(self):
        """Analysis T should be closer to truth than background."""
        n_lat, n_lon, nlev = 8, 16, 3
        bg_state = _make_pe_state(n_lat, n_lon, nlev)

        # Truth: background + localized T perturbation (+5K Gaussian blob)
        lat_idx, lon_idx = 4, 8  # center of perturbation
        pert = jnp.zeros((n_lat, n_lon, nlev))
        for i in range(n_lat):
            for j in range(n_lon):
                dist2 = (i - lat_idx) ** 2 + (j - lon_idx) ** 2
                pert = pert.at[i, j, :].set(5.0 * jnp.exp(-dist2 / 4.0))

        truth_state = bg_state._replace(
            T=bg_state.T.replace(data=bg_state.T.data + pert)
        )

        model = _SimpleHydroModel()
        dt = 300.0
        n_steps = 3

        # Run truth forward
        def scan_step(carry, _):
            s = model.step(carry, dt)
            return s, s

        _, truth_traj = jax.lax.scan(scan_step, truth_state, jnp.arange(n_steps))

        # Observe T at 20 random locations
        key = jax.random.PRNGKey(0)
        n_obs = 20
        obs_i = jax.random.randint(key, (n_obs,), 0, n_lat)
        key, subkey = jax.random.split(key)
        obs_j = jax.random.randint(subkey, (n_obs,), 0, n_lon)
        key, subkey = jax.random.split(key)
        obs_k = jax.random.randint(subkey, (n_obs,), 0, nlev)
        idx = (obs_i, obs_j, obs_k)

        # Observations at step 1
        state_t = jax.tree.map(lambda a: a[1], truth_traj)
        T_obs = state_t.T.data[idx]
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, T_obs.shape) * 0.5
        op = DirectObsOperator("T", idx)
        obs = Observation(
            values=T_obs + noise,
            errors=jnp.full(n_obs, 0.5),
            time_index=1, operator=op,
        )

        # DA setup
        spec = build_control_spec(bg_state, fields=("T",))
        sigma = jnp.ones(spec.total_size) * 5.0
        B = DiagonalB(sigma=sigma)

        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        analysis, diag = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=dt, n_steps=n_steps, config=config,
        )

        # RMSE comparison
        rmse_bg = float(jnp.sqrt(jnp.mean((bg_state.T.data - truth_state.T.data) ** 2)))
        rmse_ana = float(jnp.sqrt(jnp.mean((analysis.T.data - truth_state.T.data) ** 2)))

        assert rmse_ana < rmse_bg, (
            f"Analysis RMSE ({rmse_ana:.4f}) >= background RMSE ({rmse_bg:.4f})"
        )
