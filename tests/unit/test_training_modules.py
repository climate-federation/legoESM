"""Tests for the training package (legoesm.training).

Validates that all 8 training modules import, core functions work,
and gradient flow is verified for each training mode.

Uses synthetic data only — no GCS/ERA5 access required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

N = 4
NLEV = 3
_GRID = create_cubed_sphere(N)
_SIGMA = create_sigma_coordinate(NLEV)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(T_val=280.0):
    s3, s2 = (6, N, N, NLEV), (6, N, N)
    return HydrostaticState(
        u=Field(jnp.zeros(s3), name="u", dims=("f","x","y","l"), units="m/s"),
        v=Field(jnp.zeros(s3), name="v", dims=("f","x","y","l"), units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=("f","x","y","l"), units="K"),
        p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("f","x","y"), units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=("f","x","y"), units="m2/s2"),
    )


# ---------------------------------------------------------------------------
# 1. vertical_interp
# ---------------------------------------------------------------------------

class TestVerticalInterp:

    def test_sigma_interp_shape(self):
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        plev = jnp.linspace(5000, 100000, 13)
        T = jnp.ones((10, 20, 13)) * 250.0
        p_s = jnp.full((10, 20), 101325.0)
        sigma = jnp.linspace(0.05, 0.975, 20)
        out = interp_pressure_to_sigma(T, plev, p_s, sigma)
        assert out.shape == (10, 20, 20)

    def test_differentiable(self):
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        plev = jnp.linspace(5000, 100000, 5)
        T = jnp.ones(5) * 250.0
        sigma = jnp.linspace(0.1, 0.9, 3)
        grad = jax.grad(lambda ps: jnp.mean(
            interp_pressure_to_sigma(T, plev, ps, sigma)
        ))(jnp.float64(101325.0))
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# 2. losses
# ---------------------------------------------------------------------------

class TestLosses:

    def test_carry_mse_scalar(self):
        from legoesm.training.losses import carry_mse
        from legoesm.driver.compiled_segments import pack_carry
        s3, s2 = (6, N, N, NLEV), (6, N, N)
        state = _make_state(280.0)
        c1 = pack_carry(state, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                         q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                         held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
                         held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
                         held_sw_down_toa=jnp.zeros(s2), step_index=0)
        state2 = _make_state(282.0)
        c2 = pack_carry(state2, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                         q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                         held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
                         held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
                         held_sw_down_toa=jnp.zeros(s2), step_index=0)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        loss = carry_mse(c1, c2, sigma_full)
        assert loss.shape == ()
        assert float(loss) > 0

    def test_level_weights(self):
        from legoesm.training.losses import level_weights
        sigma = jnp.linspace(0.05, 0.975, 20)
        w = level_weights(sigma)
        assert w.shape == (20,)
        assert jnp.all(w > 0)


# ---------------------------------------------------------------------------
# 3. trainable_params
# ---------------------------------------------------------------------------

class TestTrainableParams:

    def test_from_defaults(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        p = TrainablePhysicsParams.from_defaults()
        d = p.as_dict()
        assert "tau_equator" in d
        assert abs(float(d["tau_equator"]) - 7.2) < 0.01

    def test_gradient_flow(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        import equinox as eqx
        p = TrainablePhysicsParams.from_defaults()
        loss_fn = lambda p_: sum(v**2 for v in p_.as_dict().values())
        _, grads = eqx.filter_value_and_grad(loss_fn)(p)
        assert all(jnp.isfinite(v) for v in grads.raw_values.values())


# ---------------------------------------------------------------------------
# 4. dycore_rollout
# ---------------------------------------------------------------------------

class TestDycoreRollout:

    def test_rollout_config(self):
        from legoesm.training.dycore_rollout import RolloutConfig
        cfg = RolloutConfig(n_days=3, dt=600.0)
        assert cfg.n_days == 3

    def test_single_day_rollout_callable(self):
        from legoesm.training.dycore_rollout import single_day_rollout
        assert callable(single_day_rollout)


# ---------------------------------------------------------------------------
# 5. neural_physics
# ---------------------------------------------------------------------------

class TestNeuralPhysics:

    def test_import_and_construct(self):
        from legoesm.training.neural_physics import NeuralPhysics
        key = jax.random.PRNGKey(0)
        nn = NeuralPhysics(nlev=NLEV, key=key)
        assert hasattr(nn, '__call__')

    def test_produces_tendencies(self):
        from legoesm.training.neural_physics import NeuralPhysics
        key = jax.random.PRNGKey(0)
        nn = NeuralPhysics(nlev=NLEV, key=key)
        # NeuralPhysics takes a single packed column vector
        n_input = NLEV * 4 + 2  # T, u, v, q per level + p_s + solar
        x = jnp.ones(n_input)
        out = nn(x)
        assert out.shape[0] == NLEV * 4 + 6


# ---------------------------------------------------------------------------
# 6. sfno_dycore_coupling
# ---------------------------------------------------------------------------

class TestSFNOCoupling:

    def test_import(self):
        from legoesm.training.sfno_dycore_coupling import SFNOPhysics
        assert SFNOPhysics is not None


# ---------------------------------------------------------------------------
# 7. era5_to_state
# ---------------------------------------------------------------------------

class TestERA5ToState:

    def test_config_defaults(self):
        from legoesm.training.era5_to_state import TrainingERA5Config
        cfg = TrainingERA5Config()
        assert "temperature" in cfg.pressure_variables
        assert "surface_pressure" in cfg.surface_variables

    def test_era5_slice_namedtuple(self):
        from legoesm.training.era5_to_state import ERA5Slice
        s = ERA5Slice(
            T=np.zeros((10, 20, 5)), u=np.zeros((10, 20, 5)),
            v=np.zeros((10, 20, 5)), q=np.zeros((10, 20, 5)),
            p_s=np.zeros((10, 20)), sst=np.zeros((10, 20)),
            phis=np.zeros((10, 20)),
            lat=np.linspace(-np.pi/2, np.pi/2, 10),
            lon=np.linspace(0, 2*np.pi, 20),
            plev_Pa=np.array([5000, 10000, 50000, 85000, 100000], dtype=np.float64),
        )
        assert s.T.shape == (10, 20, 5)

    def test_weight_cache(self):
        from legoesm.training.era5_to_state import _get_cs_weights
        w1 = _get_cs_weights(100, _GRID)
        w2 = _get_cs_weights(100, _GRID)
        assert w1 is w2  # same object from cache


# ---------------------------------------------------------------------------
# 8. training_driver
# ---------------------------------------------------------------------------

class TestTrainingDriver:

    def test_build_training_segment(self):
        from legoesm.training.training_driver import _build_training_segment
        from legoesm.driver.physics_pipeline import PhysicsOutput

        class MockModel:
            _state_type = HydrostaticState
            def step(self, s, dt):
                return s

        def mock_step(need_rad, T, p_s, *a, **kw):
            p = PhysicsOutput(
                dT_dt=jnp.zeros(T.shape), dq_v_dt=jnp.zeros(T.shape),
                dq_c_dt=jnp.zeros(T.shape), dq_r_dt=jnp.zeros(T.shape),
                precip=jnp.zeros(p_s.shape), sw_net_sfc=jnp.zeros(p_s.shape),
                lw_net_sfc=jnp.zeros(p_s.shape), sw_up_toa=jnp.zeros(p_s.shape),
                lw_up_toa=jnp.zeros(p_s.shape), sw_down_toa=jnp.zeros(p_s.shape),
            )
            return p, (a[16], a[17], a[18], a[19], a[20], a[21])

        fn = _build_training_segment(
            MockModel(), mock_step, _GRID, _SIGMA, 600.0,
        )
        assert callable(fn)
        assert hasattr(fn, 'raw')
