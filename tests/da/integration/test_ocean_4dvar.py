"""Ocean 4D-Var twin experiment.

Setup:
1. Truth: rest state + warm SST anomaly
2. Forward integration with simple model
3. Observe T at surface
4. Background: rest state (no anomaly)
5. 4D-Var with DiagonalB
6. Verify: analysis SST RMSE < background SST RMSE
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.ocean.state import OceanState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.incremental import incremental_4dvar, IncrementalConfig


class _SimpleOceanModel:
    """Simple ocean model: temperature relaxes toward vertical mean."""

    def step(self, state, dt):
        T_data = state.T.data
        T_mean = jnp.mean(T_data, axis=-1, keepdims=True)
        alpha = 0.01
        T_new = T_data + alpha * (T_mean - T_data)
        return state._replace(T=state.T.replace(data=T_new))


def _make_ocean_state(n=4, nlev=3, T_base=15.0):
    """Create a simple cubed-sphere ocean state."""
    shape_2d = (6, n, n)
    shape_3d = (6, n, n, nlev)
    return OceanState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=(), units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * T_base, name="T", dims=(), units="degC"),
        S=Field(data=jnp.ones(shape_3d) * 35.0, name="S", dims=(), units="PSU"),
        eta=Field(data=jnp.zeros(shape_2d), name="eta", dims=(), units="m"),
        H_bathy=Field(data=jnp.ones(shape_2d) * 4000.0, name="H_bathy", dims=(), units="m"),
        land_mask=Field(data=jnp.ones(shape_2d), name="land_mask", dims=(), units=""),
    )


class TestOceanTwinExperiment:
    def test_sst_analysis(self):
        """Analysis SST should be closer to truth than background."""
        n, nlev = 4, 3
        bg_state = _make_ocean_state(n, nlev)

        # Truth: warm SST anomaly on face 0 (+2K)
        T_truth = bg_state.T.data.copy()
        T_truth = T_truth.at[0, :, :, 0].set(T_truth[0, :, :, 0] + 2.0)  # surface only
        truth_state = bg_state._replace(T=bg_state.T.replace(data=T_truth))

        model = _SimpleOceanModel()
        dt = 3600.0
        n_steps = 3

        # Run truth forward
        def scan_step(carry, _):
            s = model.step(carry, dt)
            return s, s

        _, truth_traj = jax.lax.scan(scan_step, truth_state, jnp.arange(n_steps))

        # Observe surface T on face 0 at step 1
        n_obs = n * n  # all points on face 0
        face_idx = jnp.zeros(n_obs, dtype=jnp.int32)
        ii, jj = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        i_idx = ii.ravel()
        j_idx = jj.ravel()
        k_idx = jnp.zeros(n_obs, dtype=jnp.int32)  # surface level

        idx = (face_idx, i_idx, j_idx, k_idx)
        state_t = jax.tree.map(lambda a: a[1], truth_traj)
        T_obs = state_t.T.data[idx]
        key = jax.random.PRNGKey(7)
        noise = jax.random.normal(key, T_obs.shape) * 0.5
        op = DirectObsOperator("T", idx)
        obs = Observation(
            values=T_obs + noise,
            errors=jnp.full(n_obs, 0.5),
            time_index=1, operator=op,
        )

        # DA
        spec = build_control_spec(bg_state, fields=("T",))
        sigma = jnp.ones(spec.total_size) * 2.0
        B = DiagonalB(sigma=sigma)

        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        analysis, _ = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=dt, n_steps=n_steps, config=config,
        )

        # RMSE on surface T
        rmse_bg = float(jnp.sqrt(jnp.mean(
            (bg_state.T.data[:, :, :, 0] - truth_state.T.data[:, :, :, 0]) ** 2
        )))
        rmse_ana = float(jnp.sqrt(jnp.mean(
            (analysis.T.data[:, :, :, 0] - truth_state.T.data[:, :, :, 0]) ** 2
        )))

        assert rmse_ana < rmse_bg, (
            f"Analysis SST RMSE ({rmse_ana:.4f}) >= background RMSE ({rmse_bg:.4f})"
        )
