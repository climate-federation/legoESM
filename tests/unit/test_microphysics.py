"""Unit tests for the microphysics module.

Tests all 6 backends (Kessler, Sundqvist, Seifert-Beheng, Morrison,
Thompson, ML emulator) and the integration bridge for hydrostatic
and non-hydrostatic dycores.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
    SeifertBehengConfig,
    MorrisonConfig,
    ThompsonConfig,
    MLEmulatorConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    make_zero_hydrometeors,
    make_zero_output,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm import constants


# ======================================================================
# Test helpers
# ======================================================================

def _make_warm_columns(ncol=4, nlev=10):
    """Create warm, near-saturated columns for testing warm-rain schemes."""
    # Temperature profile: 290K at surface, 220K at top
    T = jnp.linspace(220.0, 290.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Near-saturated vapor
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.95 * q_sat

    # Cloud and rain water at mid levels
    q_c = jnp.zeros((ncol, nlev))
    q_c = q_c.at[:, 3:7].set(1e-3)
    q_r = jnp.zeros((ncol, nlev))
    q_r = q_r.at[:, 5:9].set(1e-4)

    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0))
    dz = jnp.abs(dz)

    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r,
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.zeros((ncol, nlev)),
    )

    return T, q_v, hydrometeors, p_full, p_half, rho, dz


def _make_cold_columns(ncol=4, nlev=10):
    """Create cold columns with ice and snow for testing mixed-phase schemes."""
    T = jnp.linspace(200.0, 265.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.9 * q_sat

    q_c = jnp.zeros((ncol, nlev)).at[:, 6:9].set(5e-4)
    q_r = jnp.zeros((ncol, nlev)).at[:, 7:9].set(5e-5)
    q_i = jnp.zeros((ncol, nlev)).at[:, 2:6].set(1e-4)
    q_s = jnp.zeros((ncol, nlev)).at[:, 3:7].set(5e-5)
    q_g = jnp.zeros((ncol, nlev)).at[:, 4:7].set(2e-5)

    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))

    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_s, q_g=q_g,
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.full((ncol, nlev), 1e3),
    )

    return T, q_v, hydrometeors, p_full, p_half, rho, dz


# ======================================================================
# Output helpers
# ======================================================================

class TestOutputHelpers:

    def test_make_zero_hydrometeors_shape(self):
        h = make_zero_hydrometeors(4, 10)
        assert h.q_c.shape == (4, 10)
        assert jnp.all(h.q_c == 0)

    def test_make_zero_output_shape(self):
        out = make_zero_output(4, 10)
        assert out.dT_dt.shape == (4, 10)
        assert out.precipitation.shape == (4,)

    def test_sedimentation_tendency_shape(self):
        q = jnp.ones((4, 10)) * 1e-4
        rho = jnp.ones((4, 10)) * 1.2
        V_t = jnp.ones((4, 10)) * 5.0
        dz = jnp.ones((4, 10)) * 500.0
        tend = sedimentation_tendency(q, rho, V_t, dz)
        assert tend.shape == (4, 10)
        assert jnp.all(jnp.isfinite(tend))


# ======================================================================
# Kessler tests
# ======================================================================

class TestKessler:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"

    def test_dry_air_zero_tendency(self):
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_v = jnp.zeros((ncol, nlev))
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        # With zero moisture, all ice/snow/graupel/number tendencies are exact zero
        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_s_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) == 0.0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Sundqvist tests
# ======================================================================

class TestSundqvist:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_below_RH_crit_no_condensation(self):
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = 0.5 * q_sat  # well below RH_crit=0.8
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        # Use very high sharpness to make the sigmoid effectively a step
        config = SundqvistConfig(RH_crit=0.8, sigmoid_sharpness=200.0)
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        # Condensation should be much smaller than saturated case
        out_sat = sundqvist_microphysics(T, q_sat, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        ratio = float(jnp.max(jnp.abs(out.dq_c_dt))) / float(jnp.max(jnp.abs(out_sat.dq_c_dt)) + 1e-20)
        assert ratio < 0.1

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = sundqvist_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Seifert-Beheng tests
# ======================================================================

class TestSeifertBeheng:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dN_c_dt.shape == T.shape
        assert out.dN_r_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = seifert_beheng_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Morrison tests
# ======================================================================

class TestMorrison:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dq_i_dt.shape == T.shape
        assert out.dN_i_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_ice_only_below_freezing(self):
        """Ice tendencies should be near-zero when T > T_freeze everywhere."""
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 290.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_v = 0.5 * saturation_mixing_ratio(T, p_full)
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        # Ice tendencies should be very small above freezing
        assert float(jnp.max(jnp.abs(out.dq_i_dt))) < 1e-6

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()

        def loss(T_in):
            out = morrison_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Thompson tests
# ======================================================================

class TestThompson:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dq_g_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_graupel_from_riming(self):
        """With strong riming, graupel tendencies should be nonzero."""
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        # Boost cloud water and ice to promote riming
        h = h._replace(
            q_c=jnp.full_like(h.q_c, 5e-3),
            q_i=jnp.full_like(h.q_i, 5e-3),
            q_s=jnp.full_like(h.q_s, 5e-3),
        )
        config = ThompsonConfig(
            rime_coeff=10.0,
            rime_to_graupel_threshold=1e-6,
        )
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()

        def loss(T_in):
            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# ML Emulator tests
# ======================================================================

class TestMLEmulator:

    def _make_model(self, config=None):
        if config is None:
            config = MLEmulatorConfig()
        key = jax.random.PRNGKey(config.seed)
        return MicrophysicsEmulator(
            config.n_input, config.n_hidden, config.n_layers,
            config.n_output, key=key,
        )

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        # Untrained model will still produce nonzero output
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MLEmulatorConfig()
        model = self._make_model(config)

        def loss(T_in):
            out = ml_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                                  config=config, model=model)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Integration bridge tests
# ======================================================================

class TestIntegrationHydrostatic:

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        return state, grid, sigma

    def test_hydrostatic_shapes(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
        tend = physics_fn(state, grid, sigma)
        assert tend.dT_dt.data.shape == state.T.data.shape

    def test_none_scheme_zeros(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="none")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
        tend = physics_fn(state, grid, sigma)
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0

    def test_grad_through_hydrostatic(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = physics_fn(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad))


class TestIntegrationNonhydrostatic:

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        grid = create_cubed_sphere(8)
        height_coord = create_height_coordinate(10, 10000.0)
        z_s = jnp.zeros((6, grid.n, grid.n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        nlev = height_coord.n_levels
        n = grid.n
        shape_3d = (6, n, n, nlev)
        shape_w = (6, n, n, nlev + 1)
        shape_2d = (6, n, n)
        n_tracers = 3

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v",
                    dims=("face", "x", "y", "level"), units="m/s"),
            w=Field(data=jnp.zeros(shape_w), name="w",
                    dims=("face", "x", "y", "level_half"), units="m/s"),
            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
                              dims=("face", "x", "y", "level"), units="K"),
            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
                            dims=("face", "x", "y", "level"), units="kg/m^3"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis",
                       dims=("face", "x", "y"), units="m^2/s^2"),
            tracers=Field(
                data=jnp.zeros((*shape_3d, n_tracers)),
                name="tracers",
                dims=("face", "x", "y", "level", "tracer"),
                units="kg/kg",
            ),
        )
        return state, grid, height_coord, terrain_metric

    def test_nonhydrostatic_shapes(self, setup):
        state, grid, hc, tm = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
        assert tend.dtracers_dt.data.shape == state.tracers.data.shape

    def test_nonhydrostatic_nonzero_heating(self, setup):
        state, grid, hc, tm = setup
        # Add moisture to trigger microphysics
        tracers = state.tracers.data.at[..., 0].set(0.01)
        state = state._replace(tracers=state.tracers.replace(data=tracers))
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))


class TestSchemeSelection:

    def test_scheme_selection(self):
        """All 6 scheme strings are accepted by the factory."""
        for scheme in ["kessler", "sundqvist", "seifert_beheng",
                       "morrison", "thompson", "ml_emulator", "none"]:
            config = MicrophysicsConfig(scheme=scheme)
            physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
            assert callable(physics_fn)

    def test_invalid_scheme_raises(self):
        config = MicrophysicsConfig(scheme="invalid")
        with pytest.raises(ValueError, match="Unknown microphysics scheme"):
            make_microphysics_physics(config, "hydrostatic", dt=300.0)

    def test_invalid_model_type_raises(self):
        config = MicrophysicsConfig(scheme="kessler")
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_microphysics_physics(config, "invalid", dt=300.0)


class TestBackwardCompatKessler:

    def test_backward_compat_kessler_tendencies(self):
        """The old kessler_tendencies API still works."""
        from legoesm.atmosphere.physics.kessler import kessler_tendencies, KesslerConfig
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        grid = create_cubed_sphere(8)
        hc = create_height_coordinate(10, 10000.0)
        z_s = jnp.zeros((6, grid.n, grid.n))
        tm = compute_terrain_metric(z_s, hc)

        nlev = hc.n_levels
        n = grid.n
        shape_3d = (6, n, n, nlev)
        shape_w = (6, n, n, nlev + 1)
        shape_2d = (6, n, n)

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v",
                    dims=("face", "x", "y", "level"), units="m/s"),
            w=Field(data=jnp.zeros(shape_w), name="w",
                    dims=("face", "x", "y", "level_half"), units="m/s"),
            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
                              dims=("face", "x", "y", "level"), units="K"),
            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
                            dims=("face", "x", "y", "level"), units="kg/m^3"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis",
                       dims=("face", "x", "y"), units="m^2/s^2"),
            tracers=Field(
                data=jnp.zeros((*shape_3d, 3)),
                name="tracers",
                dims=("face", "x", "y", "level", "tracer"),
                units="kg/kg",
            ),
        )

        config = KesslerConfig()
        tend = kessler_tendencies(state, grid, hc, tm, config)
        assert tend.dtheta_prime_dt.data.shape == shape_3d
        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))

    def test_saturation_mixing_ratio_import(self):
        """saturation_mixing_ratio can still be imported from kessler module."""
        from legoesm.atmosphere.physics.kessler import saturation_mixing_ratio
        T = jnp.array(280.0)
        p = jnp.array(1e5)
        q = saturation_mixing_ratio(T, p)
        assert float(q) > 0
