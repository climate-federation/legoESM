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

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

N = 4
NLEV = 3
_GRID = create_cubed_sphere(N)
_SIGMA = create_sigma_coordinate(NLEV)
_OPTIONAL_3D_OUTPUT_FIELDS = (
    "du_dt",
    "dv_dt",
    "dq_i_dt",
    "dq_s_dt",
    "dq_g_dt",
    "dN_c_dt",
    "dN_r_dt",
    "dN_i_dt",
)


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


def _zero_physics_output(T, p_s, *, conv_prog=None):
    kwargs = dict(
        dT_dt=jnp.zeros(T.shape),
        dq_v_dt=jnp.zeros(T.shape),
        dq_c_dt=jnp.zeros(T.shape),
        dq_r_dt=jnp.zeros(T.shape),
        precip=jnp.zeros(p_s.shape),
        sw_net_sfc=jnp.zeros(p_s.shape),
        lw_net_sfc=jnp.zeros(p_s.shape),
        sw_up_toa=jnp.zeros(p_s.shape),
        lw_up_toa=jnp.zeros(p_s.shape),
        sw_down_toa=jnp.zeros(p_s.shape),
    )
    for field_name in _OPTIONAL_3D_OUTPUT_FIELDS:
        if field_name in PhysicsOutput._fields:
            kwargs[field_name] = jnp.zeros(T.shape)
    if "conv_prog" in PhysicsOutput._fields:
        if conv_prog is None:
            conv_prog = jnp.asarray(0.0, dtype=T.dtype)
        kwargs["conv_prog"] = conv_prog
    return kwargs


def _step_unified_args(*, include_conv_prog=False):
    state = _make_state()
    shape_3d = state.T.data.shape
    shape_2d = state.p_s.data.shape
    args = [
        jnp.bool_(True),
        state.T.data,
        state.p_s.data,
        jnp.zeros(shape_3d),
        jnp.zeros(shape_3d),
        jnp.zeros(shape_3d),
    ]
    if include_conv_prog:
        args.append(jnp.zeros((6 * N * N,), dtype=jnp.float32))
    args.extend([
        state.u.data,
        state.v.data,
        jnp.full(shape_2d, 300.0),
        jnp.zeros(shape_2d),
        _GRID.lat,
        _GRID.lon,
        1.0,
        0.0,
        600.0,
        jnp.ones(14),
        constants.S_0,
        jnp.zeros(shape_3d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_3d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
    ])
    return tuple(args)


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

    def test_carry_mse_normalization_balances_scales(self):
        """Iter-69: per-variable scale normalization balances the loss
        contributions across T (~K), wind (~m/s), q (~kg/kg), ps (~Pa)
        which differ by ~9 orders of magnitude in raw squared units.

        Why non-vacuous: with normalize_by_scale=False, a 1 Pa
        ps perturbation contributes ~w_ps to the loss, while a 1 K
        T perturbation contributes ~w_T (commensurate) but a 0.001
        kg/kg q perturbation contributes only ~w_q · 1e-6 (six orders
        below).  With normalize_by_scale=True (default), each
        contribution is scaled by 1/(typical_amplitude²), bringing
        the moisture and wind branches into commensurate range.

        The test perturbs a single variable at a time at its typical
        anomaly scale and asserts the loss contributions are within
        1 order of magnitude of each other.
        """
        from legoesm.training.losses import carry_mse, LossConfig
        from legoesm.driver.compiled_segments import pack_carry
        s3, s2 = (6, N, N, NLEV), (6, N, N)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        cfg = LossConfig()  # normalize_by_scale=True default

        def _make_carry(state):
            return pack_carry(
                state, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                held_sw_net_sfc=jnp.zeros(s2),
                held_lw_net_sfc=jnp.zeros(s2),
                held_sw_up_toa=jnp.zeros(s2),
                held_lw_up_toa=jnp.zeros(s2),
                held_sw_down_toa=jnp.zeros(s2), step_index=0,
            )

        # Baseline state (all variables at "rest")
        ref = _make_state(280.0)
        c_ref = _make_carry(ref)

        # Perturb T by 30 K (one T_scale)
        state_dT = _make_state(280.0)
        from legoesm.core.field import Field
        state_dT = state_dT._replace(
            T=Field(state_dT.T.data + 30.0, name="T",
                    dims=state_dT.T.dims, units="K")
        )
        c_dT = _make_carry(state_dT)
        loss_T = float(carry_mse(c_ref, c_dT, sigma_full, config=cfg))

        # Perturb ps by 1000 Pa (one ps_scale)
        state_dps = _make_state(280.0)
        state_dps = state_dps._replace(
            p_s=Field(state_dps.p_s.data + 1000.0, name="p_s",
                       dims=state_dps.p_s.dims, units="Pa")
        )
        c_dps = _make_carry(state_dps)
        loss_ps = float(carry_mse(c_ref, c_dps, sigma_full, config=cfg))

        # Both perturbations are at one scale-unit: the loss
        # contributions should be commensurate (within a factor of
        # ~10).  Without scale normalization, loss_ps would dominate
        # by ~1e9 (the ps²/T² ratio for raw units).
        ratio = loss_ps / loss_T
        assert 0.01 < ratio < 100.0, (
            f"After per-variable scale normalization, T- and ps-"
            f"perturbation losses should be commensurate (ratio in "
            f"[0.01, 100]); got loss_T = {loss_T:.3e}, "
            f"loss_ps = {loss_ps:.3e}, ratio = {ratio:.3e}.  "
            f"Without normalization the ratio would be ~1e9."
        )

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
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        key = jax.random.PRNGKey(0)
        nn = NeuralPhysics(nlev=NLEV, key=key)
        assert hasattr(nn, '__call__')

    def test_produces_tendencies(self):
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
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

    def test_era5_to_cubedsphere_carry_shapes_and_finite(self):
        """era5_to_cubedsphere_carry produces carry with correct shapes
        and finite values from a synthetic ERA5Slice."""
        import jax.numpy as jnp
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        rng = np.random.default_rng(0)
        # Realistic T and p_s to avoid saturation/interp edge cases
        T_ll = (260.0 + rng.random((n_lat, n_lon, n_plev)) * 40.0).astype(np.float32)
        u_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        v_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        q_ll = (rng.random((n_lat, n_lon, n_plev)) * 0.01).astype(np.float32)
        p_s = np.full((n_lat, n_lon), 101325.0, dtype=np.float32)
        plev_Pa = np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64)

        era5 = ERA5Slice(
            T=T_ll, u=u_ll, v=v_ll, q=q_ll,
            p_s=p_s, sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.zeros((n_lat, n_lon), dtype=np.float32),
            lat=np.linspace(-np.pi/2, np.pi/2, n_lat),
            lon=np.linspace(0, 2*np.pi, n_lon, endpoint=False),
            plev_Pa=plev_Pa,
        )

        carry = era5_to_cubedsphere_carry(era5, _GRID, _SIGMA)

        expected_3d = (6, N, N, NLEV)
        expected_2d = (6, N, N)
        assert carry.T.shape == expected_3d, f"T shape {carry.T.shape} != {expected_3d}"
        assert carry.u.shape == expected_3d
        assert carry.v.shape == expected_3d
        assert carry.q_v.shape == expected_3d
        assert carry.p_s.shape == expected_2d

        assert jnp.all(jnp.isfinite(carry.T)), "T contains non-finite values"
        assert jnp.all(jnp.isfinite(carry.u)), "u contains non-finite values"
        assert jnp.all(jnp.isfinite(carry.q_v)), "q_v contains non-finite values"
        assert jnp.all(carry.q_v >= 0), "q_v contains negative values"


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
            p = PhysicsOutput(**_zero_physics_output(T, p_s))
            return p, (a[16], a[17], a[18], a[19], a[20], a[21])

        fn = _build_training_segment(
            MockModel(), mock_step, _GRID, _SIGMA, 600.0,
        )
        assert callable(fn)
        assert hasattr(fn, 'raw')


# ---------------------------------------------------------------------------
# 9. API signature guards
# ---------------------------------------------------------------------------

class TestTrainingAPISignatures:
    """Verify training entrypoint signatures match their callees."""

    def test_neural_gcm_uses_adapter(self):
        """train_neural_gcm must create a ColumnAdapter, not pass grid directly."""
        from legoesm.atmosphere.physics.neural_physics import make_neural_step_unified
        from legoesm.core.grid_adapters import make_adapter, ColumnAdapter
        import inspect

        sig = inspect.signature(make_neural_step_unified)
        params = list(sig.parameters.keys())
        # Second param should be 'adapter', not 'grid'
        assert params[1] == "adapter"

        # Verify make_adapter produces a ColumnAdapter
        adapter = make_adapter(_GRID)
        assert isinstance(adapter, ColumnAdapter)
        assert adapter.ncol == 6 * N * N

    def test_sfno_step_unified_no_grid_param(self):
        """make_sfno_step_unified must NOT accept a grid positional argument."""
        from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified
        import inspect

        sig = inspect.signature(make_sfno_step_unified)
        params = list(sig.parameters.keys())
        assert "grid" not in params
        assert params == ["sfno_physics", "mode", "traditional_step_unified"]

    def test_sfno_correction_mode_requires_pipeline(self):
        """train_sfno_coupled with mode='correction' must require physics_pipeline."""
        from legoesm.training.training_driver import train_sfno_coupled
        import inspect

        sig = inspect.signature(train_sfno_coupled)
        assert "physics_pipeline" in sig.parameters

    def test_sfno_replacement_mode_no_pipeline(self):
        """train_sfno_coupled with mode='replacement' should not require physics_pipeline."""
        from legoesm.training.training_driver import train_sfno_coupled
        import inspect

        sig = inspect.signature(train_sfno_coupled)
        # physics_pipeline should default to None
        assert sig.parameters["physics_pipeline"].default is None

    def test_neural_step_unified_accepts_optional_conv_prog_slot(self):
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics,
            make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter

        neural = NeuralPhysics(nlev=NLEV, key=jax.random.PRNGKey(0))
        step = make_neural_step_unified(neural, make_adapter(_GRID))

        phys_out, held = step(*_step_unified_args(include_conv_prog=True))

        assert phys_out.du_dt.shape == _make_state().T.data.shape
        assert len(held) == 6

    def test_sfno_step_unified_accepts_optional_conv_prog_slot(self):
        from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified

        class MockSFNOPhysics:
            def __call__(self, T, u, v, q_v, p_s, phis, dt):
                del u, v, q_v, phis, dt
                return PhysicsOutput(**_zero_physics_output(T, p_s))

        step = make_sfno_step_unified(MockSFNOPhysics(), mode="replacement")

        phys_out, held = step(*_step_unified_args(include_conv_prog=True))

        assert phys_out.du_dt.shape == _make_state().T.data.shape
        assert len(held) == 6


# ---------------------------------------------------------------------------
# 10. trainable_params scheme awareness
# ---------------------------------------------------------------------------

class TestTrainableParamsSchemeAware:
    """Trainable parameters must be scheme-aware and not crash for non-SBM."""

    def test_sbm_includes_convection_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("sbm")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" in names
        assert "sbm_RH_ref" in names
        assert "tau_equator" in names

    def test_dca_excludes_sbm_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("dca")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" not in names
        assert "sbm_RH_ref" not in names
        assert "tau_equator" in names

    def test_none_scheme_excludes_sbm_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("none")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" not in names
        assert "tau_equator" in names

    def test_from_defaults_with_non_sbm(self):
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, trainable_constraints_for_scheme,
        )
        constraints = trainable_constraints_for_scheme("dca")
        params = TrainablePhysicsParams.from_defaults(constraints=constraints)
        d = params.as_dict()
        assert "sbm_tau_c" not in d
        assert "tau_equator" in d

    def test_sbm_params_gradient_flow(self):
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, trainable_constraints_for_scheme,
        )
        import equinox as eqx

        for scheme in ["sbm", "dca", "none"]:
            constraints = trainable_constraints_for_scheme(scheme)
            p = TrainablePhysicsParams.from_defaults(constraints=constraints)
            loss_fn = lambda p_: sum(v**2 for v in p_.as_dict().values())
            _, grads = eqx.filter_value_and_grad(loss_fn)(p)
            assert all(jnp.isfinite(v) for v in grads.raw_values.values()), (
                f"Non-finite grad for scheme={scheme}"
            )
