"""Unit tests for the spectral hydrostatic primitive equation model.

Tests cover:
- 3D SH transform wrappers (roundtrip, shape, solid-body velocity)
- SpectralHydrostaticState structure and JIT compatibility
- Spectral PE tendency computation (rest state, shapes, finite values)
- Geopotential integration on Gaussian grid
- SpectralPrimitiveEquationModel single/multi-step integration
- JAX differentiability
- Solver axis resolution
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    spectral_pe_tendencies,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
    _compute_geopotential_gaussian,
)
from legoesm.core.field import Field
from legoesm import constants


# Enable float64 for spectral transforms
jax.config.update("jax_enable_x64", True)


def _proper_hyperdiff(grid):
    """Resolution-appropriate hyperdiffusion (4-hour damping at truncation)."""
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max**2)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def grid():
    """T21 Gaussian grid for fast tests."""
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    """10-level sigma coordinate."""
    return create_sigma_coordinate(10)


@pytest.fixture(scope="module")
def rest_state(grid, sigma_coord):
    """Isothermal rest-state initial condition (no perturbation for exact tests)."""
    return isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )


@pytest.fixture(scope="module")
def config(grid):
    """Config with weak hyperdiffusion for stability."""
    return SpectralPEConfig(
        hyperdiff_coeff=_proper_hyperdiff(grid),
        hyperdiff_order=2,
    )


# =============================================================================
# 3D SH Transform Tests
# =============================================================================

class TestSH3DTransforms:
    """Tests for 3D vmap SH transform wrappers."""

    def test_analysis_synthesis_roundtrip(self, grid):
        """sh_analysis_3d -> sh_synthesis_3d roundtrip preserves field."""
        nlev = 5
        # Create smooth field: constant per level with different values
        field = jnp.ones((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        for k in range(nlev):
            field = field.at[:, :, k].set(float(k + 1) * 100.0)

        coeffs = sh_analysis_3d(grid, field)
        recovered = sh_synthesis_3d(grid, coeffs)
        np.testing.assert_allclose(
            np.array(recovered), np.array(field), atol=1e-10,
            err_msg="3D analysis-synthesis roundtrip failed",
        )

    def test_shapes(self, grid):
        """Verify output shapes of 3D transforms."""
        nlev = 8
        field = jnp.zeros((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        coeffs = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)

        assert sh_analysis_3d(grid, field).shape == (grid.n_sh, nlev)
        assert sh_synthesis_3d(grid, coeffs).shape == (grid.n_lat, grid.n_lon, nlev)
        assert sh_analysis_oc2_3d(grid, field).shape == (grid.n_sh, nlev)
        assert sh_analysis_dmu_3d(grid, field).shape == (grid.n_sh, nlev)

    def test_uv_from_vordiv_3d_shapes(self, grid):
        """uv_from_vordiv_3d should return two (n_lat, n_lon, nlev) arrays."""
        nlev = 4
        vor_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
        div_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
        u_cos, v_cos = uv_from_vordiv_3d(grid, vor_hat, div_hat)
        assert u_cos.shape == (grid.n_lat, grid.n_lon, nlev)
        assert v_cos.shape == (grid.n_lat, grid.n_lon, nlev)

    def test_solid_body_rotation_3d(self, grid):
        """Solid-body rotation: vor=2*Omega*sin(lat), div=0 -> u*cos=Omega*a*cos^2."""
        nlev = 3
        Omega = constants.Omega
        a = grid.radius

        # For solid-body rotation u = Omega*a*cos(lat), v = 0,
        # the relative vorticity is 2*Omega*sin(lat).
        vor_grid = 2.0 * Omega * jnp.sin(grid.lat[:, None]) * jnp.ones(
            (grid.n_lat, grid.n_lon), dtype=jnp.float64,
        )
        vor_hat_2d = sh_analysis(grid, vor_grid)
        vor_hat = jnp.broadcast_to(
            vor_hat_2d[:, None], (grid.n_sh, nlev),
        ).copy()
        div_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)

        u_cos, v_cos = uv_from_vordiv_3d(grid, vor_hat, div_hat)

        # Expected: u_cos = Omega * a * cos^2(lat)
        expected_u_cos = Omega * a * grid.cos_lat[:, None, None] ** 2
        mask = grid.cos_lat > 0.1
        np.testing.assert_allclose(
            np.array(u_cos[mask, :]),
            np.array(jnp.broadcast_to(expected_u_cos, u_cos.shape)[mask, :]),
            rtol=1e-3,
            err_msg="Solid-body u*cos reconstruction failed",
        )

    def test_hyperdiffusion_3d_shape(self, grid):
        """spectral_hyperdiffusion_3d should broadcast correctly."""
        nlev = 5
        coeffs = jnp.ones((grid.n_sh, nlev), dtype=jnp.complex128)
        result = spectral_hyperdiffusion_3d(grid, coeffs, 1e10, order=2)
        assert result.shape == (grid.n_sh, nlev)
        # n=0 mode should have zero damping
        assert float(jnp.abs(result[0, 0])) == 0.0


# =============================================================================
# State Tests
# =============================================================================

class TestSpectralPEState:
    """Tests for SpectralHydrostaticState."""

    def test_state_structure(self, rest_state, grid, sigma_coord):
        """State should have correct field names and shapes."""
        nlev = sigma_coord.n_levels
        n_sh = grid.n_sh

        assert rest_state.vor_hat.data.shape == (n_sh, nlev)
        assert rest_state.div_hat.data.shape == (n_sh, nlev)
        assert rest_state.T_hat.data.shape == (n_sh, nlev)
        assert rest_state.lnps_hat.data.shape == (n_sh,)
        assert rest_state.phis_hat.data.shape == (n_sh,)

    def test_state_jit_compatible(self, rest_state):
        """State should be compatible with jax.jit."""
        @jax.jit
        def identity(s):
            return s

        result = identity(rest_state)
        np.testing.assert_allclose(
            np.array(result.vor_hat.data),
            np.array(rest_state.vor_hat.data),
        )

    def test_rest_state_zero_winds(self, rest_state, grid):
        """Rest state should have zero vorticity and divergence."""
        assert float(jnp.max(jnp.abs(rest_state.vor_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(rest_state.div_hat.data))) == 0.0

    def test_rest_state_uniform_temperature(self, rest_state, grid):
        """Rest state should have uniform 300K temperature."""
        T = sh_synthesis_3d(grid, rest_state.T_hat.data)
        np.testing.assert_allclose(
            np.array(T), 300.0, atol=1e-10,
            err_msg="Rest state temperature should be 300 K everywhere",
        )

    def test_rest_state_uniform_pressure(self, rest_state, grid):
        """Rest state surface pressure should be 1e5 Pa."""
        lnps = sh_synthesis(grid, rest_state.lnps_hat.data)
        p_s = jnp.exp(lnps)
        np.testing.assert_allclose(
            np.array(p_s), 1e5, atol=1e-6,
            err_msg="Rest state surface pressure should be 1e5 Pa",
        )

    def test_state_tracers_default_none(self, rest_state):
        """``isothermal_rest_state_spectral`` produces a state with
        ``tracers=None`` by default — preserves backward-compat for
        existing constructors that don't pass tracers."""
        assert rest_state.tracers is None

    @pytest.mark.parametrize("integrator", ["ssp_rk3", "ssp_rk34", "ssp_rk54"])
    def test_step_with_tracers_does_not_break_pytree(
        self, rest_state, grid, sigma_coord, integrator,
    ):
        """Regression: when state has a non-None tracers dict, the dycore
        RHS must propagate the tracer pytree structure as zeros into
        the tendency state.  Otherwise ``jax.tree.map(state, tendency)``
        in the RK step fails with "Expected dict, got None"."""
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig,
            SpectralPrimitiveEquationModel,
        )
        from legoesm.core.field import Field

        # Attach a synthetic q_v tracer.
        nlev = sigma_coord.n_levels
        qv_grid = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.01, dtype=jnp.float64,
        )
        qv_field = Field(
            data=qv_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )
        state_w_tracers = rest_state._replace(tracers={"q_v": qv_field})

        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator=integrator,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        new_state = model.step(state_w_tracers, dt=300.0)

        # Tracer is preserved across the step (the dycore time-loop does
        # not yet apply tracer tendencies — they're zero-filled).
        assert new_state.tracers is not None
        assert "q_v" in new_state.tracers
        assert bool(jnp.allclose(new_state.tracers["q_v"].data, qv_grid))

    @pytest.mark.parametrize("integrator", ["ssp_rk3", "ssp_rk34", "ssp_rk54"])
    def test_step_with_raw_array_tracers(
        self, rest_state, grid, sigma_coord, integrator,
    ):
        """Regression: tracer dict values may be raw JAX arrays (not
        ``Field``-wrapped).  The dycore tendency path must build a
        zero-tendency container that mirrors whichever shape the input
        used — calling ``.replace(data=...)`` on a raw array crashes
        with ``AttributeError: DynamicJaxprTracer has no attribute
        replace``.
        """
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig,
            SpectralPrimitiveEquationModel,
        )

        nlev = sigma_coord.n_levels
        qv_array = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.01, dtype=jnp.float64,
        )
        # Raw-array tracer (no Field wrapper).
        state_raw = rest_state._replace(tracers={"q_v": qv_array})

        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator=integrator,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        new_state = model.step(state_raw, dt=300.0)

        assert new_state.tracers is not None
        assert "q_v" in new_state.tracers
        # Raw array preserved through the step.
        out = new_state.tracers["q_v"]
        assert not hasattr(out, "data"), (
            "Raw-array tracer should remain a raw array, not get "
            "promoted to a Field"
        )
        assert bool(jnp.allclose(out, qv_array))

    def test_step_with_mixed_field_and_raw_tracers(
        self, rest_state, grid, sigma_coord,
    ):
        """A ``tracers`` dict mixing ``Field`` and raw-array values
        should round-trip through the dycore step without crashing.
        Each value preserves its original container."""
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig,
            SpectralPrimitiveEquationModel,
        )
        from legoesm.core.field import Field

        nlev = sigma_coord.n_levels
        qv_array = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.012, dtype=jnp.float64,
        )
        qc_field = Field(
            data=jnp.full((grid.n_lat, grid.n_lon, nlev), 1e-5, dtype=jnp.float64),
            name="q_c", dims=("lat", "lon", "level"), units="kg/kg",
        )
        state_mixed = rest_state._replace(
            tracers={"q_v": qv_array, "q_c": qc_field},
        )

        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        new_state = model.step(state_mixed, dt=300.0)

        assert not hasattr(new_state.tracers["q_v"], "data")
        assert hasattr(new_state.tracers["q_c"], "data")
        assert bool(jnp.allclose(new_state.tracers["q_v"], qv_array))
        assert bool(jnp.allclose(new_state.tracers["q_c"].data, qc_field.data))

    def test_step_with_full_orchestrator_and_tracers(
        self, rest_state, grid, sigma_coord,
    ):
        """End-to-end: spectral PE stepping with the multi-physics
        orchestrator (radiation + convection both active) AND a
        tracer-aware state.  Exercises the full pytree-stepping +
        orchestrator combine + dycore RHS + tracer-propagation chain
        on a single step.  Pre-fix the orchestrator's combined
        tendency had ``tracers=None`` which broke any downstream
        tree.map that expected matching pytree structure.
        """
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig,
            SpectralPrimitiveEquationModel,
        )
        from legoesm.atmosphere.physics.combined import (
            PhysicsConfig, make_physics,
        )
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        from legoesm.atmosphere.physics.physics_state import init_physics_state
        from legoesm.core.field import Field

        nlev = sigma_coord.n_levels
        qv_grid = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.01, dtype=jnp.float64,
        )
        qv_field = Field(
            data=qv_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )
        state_w_tracers = rest_state._replace(tracers={"q_v": qv_field})

        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        ncol = grid.n_lat * grid.n_lon
        ps = init_physics_state(ncol, nlev, cfg)
        physics_fn = make_physics(cfg, model_type="spectral_pe", dt=300.0)

        # The orchestrator returns (combined, phys_state) — the dycore
        # ``step_with_physics`` indexes [0] to grab the tendency.
        # We exercise it directly here as well to lock in the fix:
        # the combined tendency must carry a tracers field that
        # mirrors the input state.  Convection's SBM scheme produces
        # a tiny q_v sink at the boundary layer (smooth trigger ε
        # behavior even in a quiescent column), so the q_v tendency
        # is non-zero but bounded.
        combined, _ = physics_fn(state_w_tracers, grid, sigma_coord, phys_state=ps)
        assert combined.tracers is not None
        assert "q_v" in combined.tracers
        assert hasattr(combined.tracers["q_v"], "data")
        # SBM's smooth trigger gives O(1e-7) K/s tendencies in the
        # rest state — bounded but non-zero.  We cap it to confirm no
        # numerical blow-up rather than insisting on exact zero.
        assert float(jnp.max(jnp.abs(combined.tracers["q_v"].data))) < 1e-3

        # Now wire it through the actual model.step() path.
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, pe_config)
        new_state = model.step(
            state_w_tracers, dt=300.0, physics_fn=physics_fn,
        )

        # Tracer field survives the step (advection of uniform-q_v
        # is analytically zero; convection's smooth-trigger ε kicks
        # in at the surface).  Pin: finite, positive, and bounded
        # change relative to a reasonable convective drying rate
        # (< 5% of the initial field per 300s step).
        assert new_state.tracers is not None
        assert "q_v" in new_state.tracers
        new_qv = new_state.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(new_qv)))
        # Tracer remains positive (q_v >= 0).
        assert float(jnp.min(new_qv)) > -1e-12
        max_change = float(jnp.max(jnp.abs(new_qv - qv_grid)))
        assert max_change < 0.05 * float(jnp.max(jnp.abs(qv_grid))), (
            f"q_v change {max_change} exceeded 5% of initial field"
        )


# =============================================================================
# Geopotential Tests
# =============================================================================

class TestGeopotentialGaussian:
    """Tests for inline geopotential on Gaussian grid."""

    def test_isothermal_geopotential_decreases_upward(self, sigma_coord):
        """Geopotential should increase with height (decrease with level index)."""
        n_lat, n_lon = 10, 20
        T = jnp.full((n_lat, n_lon, sigma_coord.n_levels), 300.0, dtype=jnp.float64)
        p_s = jnp.full((n_lat, n_lon), 1e5, dtype=jnp.float64)
        phis = jnp.zeros((n_lat, n_lon), dtype=jnp.float64)

        Phi = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis)
        assert Phi.shape == (n_lat, n_lon, sigma_coord.n_levels)

        # Geopotential should increase upward: Phi[:,:,0] > Phi[:,:,-1]
        assert float(Phi[0, 0, 0]) > float(Phi[0, 0, -1]), \
            "Geopotential should be larger at upper levels"

    def test_geopotential_with_topography(self, sigma_coord):
        """Geopotential at surface level should reflect topography."""
        n_lat, n_lon = 5, 10
        T = jnp.full((n_lat, n_lon, sigma_coord.n_levels), 300.0, dtype=jnp.float64)
        p_s = jnp.full((n_lat, n_lon), 1e5, dtype=jnp.float64)

        phis_flat = jnp.zeros((n_lat, n_lon), dtype=jnp.float64)
        phis_mountain = jnp.full((n_lat, n_lon), constants.g * 2000.0, dtype=jnp.float64)

        Phi_flat = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis_flat)
        Phi_mountain = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis_mountain)

        # Mountain case should have higher geopotential at all levels
        assert jnp.all(Phi_mountain > Phi_flat), \
            "Mountain topography should raise geopotential"

    def test_geopotential_finite(self, sigma_coord):
        """Geopotential should be finite everywhere."""
        n_lat, n_lon = 8, 16
        T = jnp.full((n_lat, n_lon, sigma_coord.n_levels), 250.0, dtype=jnp.float64)
        p_s = jnp.full((n_lat, n_lon), 1e5, dtype=jnp.float64)
        phis = jnp.zeros((n_lat, n_lon), dtype=jnp.float64)

        Phi = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis)
        assert jnp.all(jnp.isfinite(Phi)), "Geopotential has non-finite values"


# =============================================================================
# Tendency Tests
# =============================================================================

class TestSpectralPETendencies:
    """Tests for spectral PE tendency computation."""

    def test_rest_state_tendencies_small(self, rest_state, grid, sigma_coord, config):
        """Isothermal rest state should produce near-zero tendencies."""
        tend = spectral_pe_tendencies(rest_state, grid, sigma_coord, config)

        # All tendencies should be very small (near machine precision)
        vor_max = float(jnp.max(jnp.abs(tend.vor_hat.data)))
        div_max = float(jnp.max(jnp.abs(tend.div_hat.data)))
        T_max = float(jnp.max(jnp.abs(tend.T_hat.data)))
        lnps_max = float(jnp.max(jnp.abs(tend.lnps_hat.data)))

        assert vor_max < 1e-10, f"Rest state vor tendency too large: {vor_max}"
        assert div_max < 1e-10, f"Rest state div tendency too large: {div_max}"
        assert T_max < 1e-8, f"Rest state T tendency too large: {T_max}"
        assert lnps_max < 1e-10, f"Rest state lnps tendency too large: {lnps_max}"

    def test_tendencies_shapes_match_state(self, rest_state, grid, sigma_coord, config):
        """Tendencies should have the same pytree structure and shapes as state."""
        tend = spectral_pe_tendencies(rest_state, grid, sigma_coord, config)

        assert tend.vor_hat.data.shape == rest_state.vor_hat.data.shape
        assert tend.div_hat.data.shape == rest_state.div_hat.data.shape
        assert tend.T_hat.data.shape == rest_state.T_hat.data.shape
        assert tend.lnps_hat.data.shape == rest_state.lnps_hat.data.shape
        assert tend.phis_hat.data.shape == rest_state.phis_hat.data.shape

    def test_tendencies_finite(self, rest_state, grid, sigma_coord, config):
        """All tendency values should be finite."""
        tend = spectral_pe_tendencies(rest_state, grid, sigma_coord, config)

        for field_name in ['vor_hat', 'div_hat', 'T_hat', 'lnps_hat', 'phis_hat']:
            data = getattr(tend, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"Tendency {field_name} has non-finite values"

    def test_phis_tendency_zero(self, rest_state, grid, sigma_coord, config):
        """Surface geopotential tendency should always be zero (static)."""
        tend = spectral_pe_tendencies(rest_state, grid, sigma_coord, config)
        assert float(jnp.max(jnp.abs(tend.phis_hat.data))) == 0.0


# =============================================================================
# Model Tests
# =============================================================================

class TestSpectralPEModel:
    """Tests for SpectralPrimitiveEquationModel."""

    def test_model_creation(self, grid, sigma_coord, config):
        """Model should be creatable."""
        model = SpectralPrimitiveEquationModel(
            grid, sigma_coord, config,
            allow_unsupported_backend=True,
        )
        assert model.config == config

    def test_single_step_finite(self, grid, sigma_coord, config):
        """Single time step should produce finite state."""
        model = SpectralPrimitiveEquationModel(
            grid, sigma_coord, config,
            allow_unsupported_backend=True,
        )
        state = isothermal_rest_state_spectral(
            grid, sigma_coord, perturbation_amplitude=0.0,
        )
        dt = 600.0
        new_state = model.step(state, dt)

        for field_name in ['vor_hat', 'div_hat', 'T_hat', 'lnps_hat']:
            data = getattr(new_state, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"After 1 step, {field_name} has non-finite values"

    def test_rest_state_stability(self, grid, sigma_coord, config):
        """Rest state should remain stable over 50 time steps."""
        model = SpectralPrimitiveEquationModel(
            grid, sigma_coord, config,
            allow_unsupported_backend=True,
        )
        state = isothermal_rest_state_spectral(
            grid, sigma_coord, perturbation_amplitude=0.0,
        )
        dt = 300.0

        for _ in range(50):
            state = model.step(state, dt)

        # Temperature should still be close to 300K
        T = sh_synthesis_3d(grid, state.T_hat.data)
        np.testing.assert_allclose(
            np.array(T), 300.0, atol=1.0,
            err_msg="Temperature drifted from 300K after 50 steps",
        )

        # All fields should be finite
        for field_name in ['vor_hat', 'div_hat', 'T_hat', 'lnps_hat']:
            data = getattr(state, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"After 50 steps, {field_name} has non-finite values"

    def test_integrate(self, grid, sigma_coord, config):
        """integrate() should produce a trajectory."""
        model = SpectralPrimitiveEquationModel(
            grid, sigma_coord, config,
            allow_unsupported_backend=True,
        )
        state = isothermal_rest_state_spectral(grid, sigma_coord)
        dt = 600.0
        duration = 3600.0  # 1 hour
        final, trajectory = model.integrate(state, duration, dt, save_every=3)

        assert len(trajectory) == 3  # initial + 6/3=2 saves
        assert jnp.all(jnp.isfinite(final.T_hat.data))

    def test_invalid_si_substeps(self, grid, sigma_coord, config):
        """si_substeps must be >= 1."""
        bad = config._replace(semi_implicit=True, si_substeps=0)
        with pytest.raises(ValueError):
            SpectralPrimitiveEquationModel(
                grid, sigma_coord, bad,
                allow_unsupported_backend=True,
            )

    def test_si_substeps_stabilize_large_dt_step(self, grid, sigma_coord, config):
        """SI substeps should keep large-dt updates finite."""
        si_cfg = config._replace(
            semi_implicit=True,
            si_substeps=2,
            si_hyperdiff_boost=8.0,
        )
        model = SpectralPrimitiveEquationModel(
            grid, sigma_coord, si_cfg,
            allow_unsupported_backend=True,
        )
        state = isothermal_rest_state_spectral(grid, sigma_coord)

        dt = 600.0
        for _ in range(10):
            state = model.step(state, dt)

        for field_name in ['vor_hat', 'div_hat', 'T_hat', 'lnps_hat']:
            data = getattr(state, field_name).data
            assert jnp.all(jnp.isfinite(data)), (
                f"With SI substeps, {field_name} has non-finite values"
            )


# =============================================================================
# Diagnostic Tests
# =============================================================================

class TestSpectralPEDiagnostics:
    """Tests for diagnostic utilities."""

    def test_spectral_pe_to_grid(self, rest_state, grid, sigma_coord):
        """spectral_pe_to_grid should return correct grid-point fields."""
        diag = spectral_pe_to_grid(rest_state, grid, sigma_coord)

        assert 'u' in diag
        assert 'v' in diag
        assert 'T' in diag
        assert 'p_s' in diag
        assert diag['T'].shape == (grid.n_lat, grid.n_lon, sigma_coord.n_levels)

        # Rest state: zero winds
        np.testing.assert_allclose(np.array(diag['u']), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.array(diag['v']), 0.0, atol=1e-10)

        # Rest state: uniform temperature
        np.testing.assert_allclose(np.array(diag['T']), 300.0, atol=1e-10)


# =============================================================================
# Differentiability Tests
# =============================================================================

class TestSpectralPEDifferentiability:
    """Tests for JAX differentiability of spectral PE."""

    def test_tendency_differentiable(self, grid, sigma_coord, config):
        """jax.grad should work through spectral_pe_tendencies."""
        state = isothermal_rest_state_spectral(grid, sigma_coord)

        def loss(T_hat_data):
            s = state._replace(
                T_hat=state.T_hat.replace(data=T_hat_data),
            )
            tend = spectral_pe_tendencies(s, grid, sigma_coord, config)
            return jnp.sum(jnp.abs(tend.T_hat.data) ** 2).real

        grad = jax.grad(loss)(state.T_hat.data)
        assert grad.shape == state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(grad)), "Gradient has non-finite values"


# =============================================================================
# Solver Axis Tests
# =============================================================================

class TestSpectralPESolverAxis:
    """Tests for solver axis resolution with spectral PE."""

    def test_resolve_hydrostatic_spectral(self):
        """dynamics=hydrostatic + discretization=spectral -> spectral_primitive_equations."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="hydrostatic", discretization="spectral")
        assert name == "spectral_primitive_equations"

    def test_create_model_from_name(self, grid, sigma_coord):
        """create_model with flat name produces SpectralPrimitiveEquationModel."""
        from legoesm.atmosphere.dynamics import create_model
        model = create_model(
            "spectral_primitive_equations",
            grid=grid,
            sigma_coord=sigma_coord,
            allow_unsupported_backend=True,
        )
        assert isinstance(model, SpectralPrimitiveEquationModel)

    def test_create_model_from_config(self, grid, sigma_coord):
        """create_model from Config resolves spectral PE correctly."""
        from legoesm.atmosphere.dynamics import create_model
        from legoesm.config import Config
        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "hydrostatic",
                "discretization": "spectral",
                "spectral": {"allow_unsupported": True},
            }
        })
        model = create_model(
            legoesm_config=cfg,
            grid=grid,
            sigma_coord=sigma_coord,
        )
        assert isinstance(model, SpectralPrimitiveEquationModel)

    def test_solver_axes_roundtrip(self):
        """solver_axes returns (hydrostatic, spectral) for spectral_primitive_equations."""
        from legoesm.atmosphere.dynamics import solver_axes
        axes = solver_axes("spectral_primitive_equations")
        assert axes == ("hydrostatic", "spectral")
