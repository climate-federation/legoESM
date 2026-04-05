"""Differentiability tests for atmosphere physics parameterizations.

Tests that jax.grad produces finite, non-zero gradients through each
physics scheme independently and through the combined physics pipeline.

Categories:
  2a) Held-Suarez forcing
  2b) Gray radiation
  2c) Convection schemes (sbm, dca, kuo)
  2d) Turbulence schemes (smagorinsky, louis)
  2e) Microphysics (kessler, sundqvist)
  2f) Combined physics (make_physics)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


def make_hydrostatic_state(n, nlev, key=None):
    """Create a minimal C-n hydrostatic state with realistic values."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 2.0 * jax.random.normal(k1, (6, n, n, nlev))
    u_data = 10.0 * jax.random.normal(k2, (6, n, n, nlev))
    v_data = 3.0 * jax.random.normal(k3, (6, n, n, nlev))
    q_v_data = 1e-3 * jnp.ones((6, n, n, nlev))
    q_c_data = 1e-5 * jnp.ones((6, n, n, nlev))
    q_r_data = 1e-6 * jnp.ones((6, n, n, nlev))
    return HydrostaticState(
        u=Field(u_data, name="u"),
        v=Field(v_data, name="v"),
        T=Field(T_data, name="T"),
        p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
        tracers={
            "q_v": Field(q_v_data, name="q_v"),
            "q_c": Field(q_c_data, name="q_c"),
            "q_r": Field(q_r_data, name="q_r"),
        },
    )


# ============================================================================
# 2a  Held-Suarez forcing
# ============================================================================

class TestHeldSuarezGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    def test_grad_wrt_T(self):
        from tests.test_cases.held_suarez import held_suarez_forcing
        grid, sigma, state = self.grid, self.sigma, self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = held_suarez_forcing(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Held-Suarez dT_dt w.r.t. T")


# ============================================================================
# 2b  Gray radiation
# ============================================================================

class TestGrayRadiationGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        self.gray_radiation = gray_radiation
        self.config = GrayRadiationConfig()
        ncol, nlev = 32, 5
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        self.T = 250.0 + 20.0 * jax.random.normal(k1, (ncol, nlev))
        # Pressure decreasing with height
        p_half = jnp.linspace(1e5, 100.0, nlev + 1)
        self.p_half = jnp.broadcast_to(p_half, (ncol, nlev + 1))
        self.p_full = 0.5 * (self.p_half[:, :-1] + self.p_half[:, 1:])
        self.sfc_temp = 290.0 * jnp.ones(ncol)
        self.lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, ncol)
        self.q_v = 1e-3 * jnp.ones((ncol, nlev))
        self.insolation = 400.0 * jnp.ones(ncol)

    def test_grad_wrt_sfc_temp(self):
        """Gray LW fluxes depend on surface temperature."""
        config = self.config

        def loss(sfc_temp):
            out = self.gray_radiation(self.T, self.p_full, self.p_half,
                                       sfc_temp, self.lat, self.q_v,
                                       self.insolation, config)
            return jnp.sum(out.lw_flux_up ** 2)

        grad = jax.grad(loss)(self.sfc_temp)
        assert_gradient_ok(grad, "Gray radiation w.r.t. sfc_temp")


# ============================================================================
# 2c  Convection schemes
# ============================================================================

class TestConvectionGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    @pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo"])
    def test_grad_wrt_T(self, scheme):
        from legoesm.atmosphere.physics.convection.integration import make_convection_physics
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig

        config = ConvectionConfig(scheme=scheme)
        conv_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = conv_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, f"Convection({scheme}) w.r.t. T", min_nonzero_frac=0.01)


# ============================================================================
# 2d  Turbulence schemes
# ============================================================================

class TestTurbulenceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    @pytest.mark.parametrize("scheme", ["smagorinsky", "louis"])
    def test_grad_wrt_T(self, scheme):
        from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

        config = TurbulenceConfig(scheme=scheme)
        turb_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = turb_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, f"Turbulence({scheme}) w.r.t. T", min_nonzero_frac=0.01)


# ============================================================================
# 2e  Microphysics
# ============================================================================

class TestMicrophysicsGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    @pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
    def test_grad_wrt_qv(self, scheme):
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        config = MicrophysicsConfig(scheme=scheme)
        micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state

        def loss(qv_data):
            tracers_new = {**state.tracers, "q_v": state.tracers["q_v"].replace(data=qv_data)}
            s = state._replace(tracers=tracers_new)
            tend = micro_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.tracers["q_v"].data)
        assert_gradient_ok(grad, f"Microphysics({scheme}) w.r.t. q_v", min_nonzero_frac=0.01)


# ============================================================================
# 2f  Combined physics (make_physics)
# ============================================================================

class TestCombinedPhysicsGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

        phys_config = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
            microphysics=MicrophysicsConfig(scheme="none"),
        )
        self.physics_fn = make_physics(phys_config, model_type="hydrostatic", dt=300.0)
        self.physics_fn.set_time(80.0, 43200.0)

    def test_grad_wrt_T(self):
        physics_fn, state = self.physics_fn, self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = physics_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Combined physics w.r.t. T")
