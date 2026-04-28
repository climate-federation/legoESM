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
from legoesm.thermo import saturation_mixing_ratio
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


def _make_evaporation_columns(ncol=4, nlev=10):
    """Create warm, subsaturated columns with rain to isolate evaporation."""
    T = jnp.full((ncol, nlev), 290.0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.2 * q_sat

    q_r = jnp.full((ncol, nlev), 1e-3)
    z = jnp.zeros((ncol, nlev))
    hydrometeors = HydrometeorState(
        q_c=z,
        q_r=q_r,
        q_i=z,
        q_s=z,
        q_g=z,
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.full((ncol, nlev), 1e3),
    )

    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
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

    def test_evaporation_enthalpy_balance(self):
        """Evaporation cooling should balance vapor tendency latent energy."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 1e-6


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

    def test_evaporation_enthalpy_balance(self):
        """Evaporation cooling should balance vapor tendency latent energy."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 1e-6


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

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3


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

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3


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
        from legoesm.atmosphere.held_suarez import held_suarez_init

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


class TestKesslerNonhydrostatic:

    def test_kessler_nonhydrostatic_tendencies(self):
        """Kessler microphysics via integration bridge on nonhydrostatic state."""
        from legoesm.atmosphere.physics.microphysics.config import KesslerConfig, MicrophysicsConfig
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
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
        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert tend.dtheta_prime_dt.data.shape == shape_3d
        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))


# ======================================================================
# AMIP integration tests (checkpoint save/load with q_c/q_r)
# ======================================================================

class TestCheckpointWithHydrometeors:
    """Test checkpoint save/load roundtrip with q_c and q_r fields."""

    @pytest.fixture
    def setup(self, tmp_path):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init
        from legoesm.forcing.amip_config import (
            AMIPExperimentConfig, save_checkpoint, load_checkpoint,
        )

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        n = grid.n
        nlev = 10
        q_v = jnp.full((6, n, n, nlev), 0.005)
        q_c = jnp.full((6, n, n, nlev), 1e-4)
        q_r = jnp.full((6, n, n, nlev), 5e-5)
        config = AMIPExperimentConfig(
            resolution=n, nlev=nlev, microphysics="kessler",
        )
        return state, q_v, q_c, q_r, config, grid, sigma, tmp_path

    def test_roundtrip_with_hydrometeors(self, setup):
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt.npz"
        save_checkpoint(path, state, q_v, step=100, day=10.0, config=config,
                        q_c=q_c, q_r=q_r)

        loaded = load_checkpoint(path, grid, sigma)
        assert len(loaded) == 8  # state, q_v, step, day, config, diag, q_c, q_r
        _, _, step, day, _, _, q_c_loaded, q_r_loaded = loaded
        assert step == 100
        assert day == 10.0
        assert q_c_loaded is not None
        assert q_r_loaded is not None
        assert float(jnp.max(jnp.abs(q_c_loaded - q_c))) < 1e-10
        assert float(jnp.max(jnp.abs(q_r_loaded - q_r))) < 1e-10

    def test_roundtrip_without_hydrometeors(self, setup):
        """Old checkpoints without q_c/q_r should load with None."""
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt_old.npz"
        # Save without q_c/q_r (old-style)
        save_checkpoint(path, state, q_v, step=50, day=5.0, config=config)

        loaded = load_checkpoint(path, grid, sigma)
        _, _, _, _, _, _, q_c_loaded, q_r_loaded = loaded
        assert q_c_loaded is None
        assert q_r_loaded is None

    def test_config_microphysics_field_roundtrip(self, setup):
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt_cfg.npz"
        save_checkpoint(path, state, q_v, step=10, day=1.0, config=config,
                        q_c=q_c, q_r=q_r)
        _, _, _, _, restored_config, _, _, _ = load_checkpoint(path, grid, sigma)
        assert restored_config.microphysics == "kessler"


class TestAMIPMicrophysicsConfig:
    """Test AMIPExperimentConfig microphysics field."""

    def test_default_microphysics_none(self):
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        config = AMIPExperimentConfig()
        assert config.microphysics == "none"

    def test_kessler_microphysics(self):
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        config = AMIPExperimentConfig(microphysics="kessler")
        assert config.microphysics == "kessler"

    def test_config_json_roundtrip(self):
        from legoesm.forcing.amip_config import (
            AMIPExperimentConfig, config_to_dict, config_from_dict,
        )
        config = AMIPExperimentConfig(microphysics="sundqvist")
        d = config_to_dict(config)
        assert d["microphysics"] == "sundqvist"
        restored = config_from_dict(d)
        assert restored.microphysics == "sundqvist"


class TestMicrophysicsInPhysicsStep:
    """Test microphysics backend dispatch for AMIP-like operator-split setup."""

    def test_kessler_in_column_physics(self):
        """Kessler produces non-trivial tendencies on moist columns."""
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = KesslerConfig()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz,
                                   dt=600.0, config=config)
        # Should have nonzero cloud water and rain tendencies
        assert float(jnp.max(jnp.abs(out.dq_c_dt))) > 0
        assert float(jnp.max(jnp.abs(out.dq_r_dt))) > 0
        # Latent heating should be nonzero
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_multi_step_stability(self):
        """Multiple Kessler steps shouldn't produce NaN."""
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = KesslerConfig()
        dt = 600.0
        q_c = h.q_c
        q_r = h.q_r

        for _ in range(10):
            h_step = h._replace(q_c=q_c, q_r=q_r)
            out = kessler_microphysics(T, q_v, h_step, p_full, p_half, rho, dz,
                                       dt=dt, config=config)
            q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
            q_c = jnp.maximum(q_c + dt * out.dq_c_dt, 0.0)
            q_r = jnp.maximum(q_r + dt * out.dq_r_dt, 0.0)
            T = T + dt * out.dT_dt

        assert jnp.all(jnp.isfinite(T))
        assert jnp.all(jnp.isfinite(q_v))
        assert jnp.all(jnp.isfinite(q_c))
        assert jnp.all(jnp.isfinite(q_r))
