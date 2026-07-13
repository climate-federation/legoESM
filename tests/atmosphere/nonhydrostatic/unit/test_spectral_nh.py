"""Unit tests for the spectral non-hydrostatic compressible Euler model.

Tests cover:
- SpectralNHState structure and JIT compatibility
- Slow tendency computation (rest state, shapes, finite values)
- Acoustic substeps in grid space
- SpectralCompressibleEulerModel single/multi-step integration
- Exner perturbation with Gaussian grid shapes
- JAX differentiability
- Solver axis resolution
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis_3d,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.atmosphere.dynamics.spectral_nh import (
    SpectralNHState,
    SpectralNHConfig,
    SpectralCompressibleEulerModel,
    spectral_nh_slow_tendencies,
    _acoustic_substeps_grid,
    nh_rest_state_spectral,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
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
def height_coord():
    """10-level height coordinate with 30km top."""
    return create_height_coordinate(10, 30000.0)


@pytest.fixture(scope="module")
def terrain_metric(grid, height_coord):
    """Flat terrain metric for Gaussian grid."""
    z_s = jnp.zeros((grid.n_lat, grid.n_lon))
    return compute_terrain_metric(z_s, height_coord)


@pytest.fixture(scope="module")
def rest_state(grid, height_coord):
    """Rest-state initial condition."""
    return nh_rest_state_spectral(grid, height_coord, n_tracers=0)


@pytest.fixture(scope="module")
def config(grid):
    """Config with weak hyperdiffusion for stability."""
    return SpectralNHConfig(
        hyperdiff_coeff=_proper_hyperdiff(grid),
        hyperdiff_order=2,
        sponge_coeff=0.0,  # disable sponge for simpler tests
        n_acoustic_substeps=4,
    )


# =============================================================================
# State Tests
# =============================================================================

class TestSpectralNHState:
    """Tests for SpectralNHState."""

    def test_state_structure(self, rest_state, grid, height_coord):
        """State should have correct field names and shapes."""
        nlev = height_coord.n_levels
        n_sh = grid.n_sh

        assert rest_state.vor_hat.data.shape == (n_sh, nlev)
        assert rest_state.div_hat.data.shape == (n_sh, nlev)
        assert rest_state.w_hat.data.shape == (n_sh, nlev + 1)
        assert rest_state.theta_prime_hat.data.shape == (n_sh, nlev)
        assert rest_state.rho_prime_hat.data.shape == (n_sh, nlev)
        assert rest_state.phis_hat.data.shape == (n_sh,)
        assert rest_state.tracers_hat.data.ndim >= 2

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

    def test_rest_state_zero_perturbations(self, rest_state):
        """Rest state should have zero perturbations and winds."""
        assert float(jnp.max(jnp.abs(rest_state.vor_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(rest_state.div_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(rest_state.w_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(rest_state.theta_prime_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(rest_state.rho_prime_hat.data))) == 0.0


# =============================================================================
# Tendency Tests
# =============================================================================

class TestSpectralNHTendencies:
    """Tests for spectral NH slow tendency computation."""

    def test_rest_state_tendencies_small(
        self, rest_state, grid, height_coord, terrain_metric, config,
    ):
        """Rest state should produce near-zero slow tendencies."""
        tend = spectral_nh_slow_tendencies(
            rest_state, grid, height_coord, terrain_metric, config,
        )

        vor_max = float(jnp.max(jnp.abs(tend.vor_hat.data)))
        div_max = float(jnp.max(jnp.abs(tend.div_hat.data)))
        theta_max = float(jnp.max(jnp.abs(tend.theta_prime_hat.data)))
        rho_max = float(jnp.max(jnp.abs(tend.rho_prime_hat.data)))

        assert vor_max < 1e-10, f"Rest state vor tendency too large: {vor_max}"
        assert div_max < 1e-10, f"Rest state div tendency too large: {div_max}"
        assert theta_max < 1e-8, f"Rest state theta tendency too large: {theta_max}"
        assert rho_max < 1e-8, f"Rest state rho tendency too large: {rho_max}"

    def test_tendencies_shapes_match_state(
        self, rest_state, grid, height_coord, terrain_metric, config,
    ):
        """Tendencies should have the same pytree structure as state."""
        tend = spectral_nh_slow_tendencies(
            rest_state, grid, height_coord, terrain_metric, config,
        )

        assert tend.vor_hat.data.shape == rest_state.vor_hat.data.shape
        assert tend.div_hat.data.shape == rest_state.div_hat.data.shape
        assert tend.w_hat.data.shape == rest_state.w_hat.data.shape
        assert tend.theta_prime_hat.data.shape == rest_state.theta_prime_hat.data.shape
        assert tend.rho_prime_hat.data.shape == rest_state.rho_prime_hat.data.shape
        assert tend.tracers_hat.data.shape == rest_state.tracers_hat.data.shape

    def test_tendencies_finite(
        self, rest_state, grid, height_coord, terrain_metric, config,
    ):
        """All tendency values should be finite."""
        tend = spectral_nh_slow_tendencies(
            rest_state, grid, height_coord, terrain_metric, config,
        )

        for field_name in [
            'vor_hat', 'div_hat', 'w_hat',
            'theta_prime_hat', 'rho_prime_hat', 'tracers_hat',
        ]:
            data = getattr(tend, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"Tendency {field_name} has non-finite values"

    def test_phis_tendency_zero(
        self, rest_state, grid, height_coord, terrain_metric, config,
    ):
        """Surface geopotential tendency should always be zero (static)."""
        tend = spectral_nh_slow_tendencies(
            rest_state, grid, height_coord, terrain_metric, config,
        )
        assert float(jnp.max(jnp.abs(tend.phis_hat.data))) == 0.0


# =============================================================================
# Acoustic Substep Tests
# =============================================================================

class TestSpectralNHAcoustic:
    """Tests for acoustic substeps in grid space."""

    def test_acoustic_shapes(self, grid, height_coord, terrain_metric, config):
        """Acoustic substeps should preserve array shapes."""
        nlev = height_coord.n_levels
        shape_w = (grid.n_lat, grid.n_lon, nlev + 1)
        shape_3d = (grid.n_lat, grid.n_lon, nlev)

        w = jnp.zeros(shape_w, dtype=jnp.float64)
        theta_p = jnp.zeros(shape_3d, dtype=jnp.float64)
        rho_p = jnp.zeros(shape_3d, dtype=jnp.float64)

        w_new, theta_new, rho_new = _acoustic_substeps_grid(
            w, theta_p, rho_p, 1.0, 4,
            height_coord, terrain_metric, config,
        )

        assert w_new.shape == shape_w
        assert theta_new.shape == shape_3d
        assert rho_new.shape == shape_3d

    def test_acoustic_rest_state_stable(
        self, grid, height_coord, terrain_metric, config,
    ):
        """Acoustic substeps on rest state should not introduce perturbations."""
        nlev = height_coord.n_levels
        w = jnp.zeros((grid.n_lat, grid.n_lon, nlev + 1), dtype=jnp.float64)
        theta_p = jnp.zeros((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        rho_p = jnp.zeros((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)

        w_new, theta_new, rho_new = _acoustic_substeps_grid(
            w, theta_p, rho_p, 1.0, 10,
            height_coord, terrain_metric, config,
        )

        np.testing.assert_allclose(np.array(w_new), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.array(theta_new), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.array(rho_new), 0.0, atol=1e-15)

    def test_acoustic_spectral_roundtrip(self, grid, height_coord, terrain_metric, config):
        """Spectral->grid->acoustic->grid->spectral roundtrip should be finite."""
        nlev = height_coord.n_levels
        n_sh = grid.n_sh

        # Small perturbation in spectral space
        w_hat = jnp.zeros((n_sh, nlev + 1), dtype=jnp.complex128)
        theta_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
        rho_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

        # Small perturbation
        theta_p_hat = theta_p_hat.at[1, nlev // 2].set(0.01 + 0j)

        # To grid
        w_grid = sh_synthesis_3d(grid, w_hat)
        theta_p_grid = sh_synthesis_3d(grid, theta_p_hat)
        rho_p_grid = sh_synthesis_3d(grid, rho_p_hat)

        # Run acoustic substeps
        w_new, theta_new, rho_new = _acoustic_substeps_grid(
            w_grid, theta_p_grid, rho_p_grid, 0.5, 4,
            height_coord, terrain_metric, config,
        )

        # Back to spectral
        w_hat_new = sh_analysis_3d(grid, w_new)
        theta_hat_new = sh_analysis_3d(grid, theta_new)
        rho_hat_new = sh_analysis_3d(grid, rho_new)

        assert jnp.all(jnp.isfinite(w_hat_new)), "w_hat has non-finite after roundtrip"
        assert jnp.all(jnp.isfinite(theta_hat_new)), "theta_hat non-finite after roundtrip"
        assert jnp.all(jnp.isfinite(rho_hat_new)), "rho_hat non-finite after roundtrip"


# =============================================================================
# Exner Perturbation Tests
# =============================================================================

class TestSpectralNHExner:
    """Tests for Exner perturbation with Gaussian grid shapes."""

    def test_exner_zero_perturbation(self, height_coord):
        """Zero perturbations should give zero Exner perturbation."""
        n_lat, n_lon = 10, 20
        nlev = height_coord.n_levels
        rho_p = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
        theta_p = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        np.testing.assert_allclose(
            np.array(pi_p), 0.0, atol=1e-15,
            err_msg="Zero perturbations should give zero Exner perturbation",
        )

    def test_exner_shape(self, height_coord):
        """Exner perturbation should have same shape as input."""
        n_lat, n_lon = 8, 16
        nlev = height_coord.n_levels
        rho_p = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
        theta_p = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        assert pi_p.shape == (n_lat, n_lon, nlev)

    def test_exner_positive_perturbation(self, height_coord):
        """Positive rho and theta perturbations should give positive Exner perturbation."""
        n_lat, n_lon = 5, 10
        nlev = height_coord.n_levels
        # Small positive perturbations
        rho_p = jnp.full((n_lat, n_lon, nlev), 0.01, dtype=jnp.float64)
        theta_p = jnp.full((n_lat, n_lon, nlev), 0.1, dtype=jnp.float64)

        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        assert jnp.all(pi_p > 0), "Positive perturbations should give positive Exner"
        assert jnp.all(jnp.isfinite(pi_p)), "Exner perturbation has non-finite values"


# =============================================================================
# Model Tests
# =============================================================================

class TestSpectralNHModel:
    """Tests for SpectralCompressibleEulerModel."""

    def test_model_creation(self, grid, height_coord, terrain_metric, config):
        """Model should be creatable."""
        model = SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
            allow_unsupported_backend=True,
        )
        assert model.config == config

    def test_single_step_finite(self, grid, height_coord, terrain_metric, config):
        """Single time step should produce finite state."""
        model = SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
            allow_unsupported_backend=True,
        )
        state = nh_rest_state_spectral(grid, height_coord)
        dt = 10.0  # small dt for NH
        new_state = model.step(state, dt)

        for field_name in [
            'vor_hat', 'div_hat', 'w_hat',
            'theta_prime_hat', 'rho_prime_hat',
        ]:
            data = getattr(new_state, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"After 1 step, {field_name} has non-finite values"

    def test_multi_step_stability(self, grid, height_coord, terrain_metric, config):
        """Rest state should remain stable over 20 steps."""
        model = SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
            allow_unsupported_backend=True,
        )
        state = nh_rest_state_spectral(grid, height_coord)
        dt = 5.0

        for _ in range(20):
            state = model.step(state, dt)

        for field_name in [
            'vor_hat', 'div_hat', 'w_hat',
            'theta_prime_hat', 'rho_prime_hat',
        ]:
            data = getattr(state, field_name).data
            assert jnp.all(jnp.isfinite(data)), \
                f"After 20 steps, {field_name} has non-finite values"

        # Perturbations should remain small
        theta_max = float(jnp.max(jnp.abs(state.theta_prime_hat.data)))
        assert theta_max < 1.0, \
            f"Theta perturbation grew too large: {theta_max}"

    def test_integrate(self, grid, height_coord, terrain_metric, config):
        """integrate() should produce a trajectory."""
        model = SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
            allow_unsupported_backend=True,
        )
        state = nh_rest_state_spectral(grid, height_coord)
        dt = 10.0
        duration = 60.0  # 1 minute
        final, trajectory = model.integrate(state, duration, dt, save_every=3)

        assert len(trajectory) == 3  # initial + 6/3=2 saves
        assert jnp.all(jnp.isfinite(final.vor_hat.data))


# =============================================================================
# Differentiability Tests
# =============================================================================

class TestSpectralNHDifferentiability:
    """Tests for JAX differentiability of spectral NH."""

    def test_tendency_differentiable(
        self, grid, height_coord, terrain_metric, config,
    ):
        """jax.grad should work through spectral_nh_slow_tendencies."""
        state = nh_rest_state_spectral(grid, height_coord)

        def loss(theta_p_data):
            s = state._replace(
                theta_prime_hat=state.theta_prime_hat.replace(data=theta_p_data),
            )
            tend = spectral_nh_slow_tendencies(
                s, grid, height_coord, terrain_metric, config,
            )
            return jnp.sum(jnp.abs(tend.theta_prime_hat.data) ** 2).real

        grad = jax.grad(loss)(state.theta_prime_hat.data)
        assert grad.shape == state.theta_prime_hat.data.shape
        assert jnp.all(jnp.isfinite(grad)), "Gradient has non-finite values"


# =============================================================================
# Solver Axis Tests
# =============================================================================

class TestSpectralNHSolverAxis:
    """Tests for solver axis resolution with spectral NH."""

    def test_resolve_nonhydrostatic_spectral(self):
        """dynamics=nonhydrostatic + discretization=spectral -> spectral_compressible_euler."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(
            dynamics="nonhydrostatic", discretization="spectral",
        )
        assert name == "spectral_compressible_euler"

    def test_create_model_from_name(self, grid, height_coord, terrain_metric):
        """create_model with flat name produces SpectralCompressibleEulerModel."""
        from legoesm.atmosphere.dynamics import create_model
        model = create_model(
            "spectral_compressible_euler",
            grid=grid,
            height_coord=height_coord,
            terrain_metric=terrain_metric,
            allow_unsupported_backend=True,
        )
        assert isinstance(model, SpectralCompressibleEulerModel)

    def test_create_model_from_config(self, grid, height_coord, terrain_metric):
        """create_model from Config resolves spectral NH correctly."""
        from legoesm.atmosphere.dynamics import create_model
        from legoesm.config import Config
        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "nonhydrostatic",
                "discretization": "spectral",
                "spectral": {"allow_unsupported": True},
            }
        })
        model = create_model(
            legoesm_config=cfg,
            grid=grid,
            height_coord=height_coord,
            terrain_metric=terrain_metric,
        )
        assert isinstance(model, SpectralCompressibleEulerModel)

    def test_solver_axes_roundtrip(self):
        """solver_axes returns (nonhydrostatic, spectral) for spectral_compressible_euler."""
        from legoesm.atmosphere.dynamics import solver_axes
        axes = solver_axes("spectral_compressible_euler")
        assert axes == ("nonhydrostatic", "spectral")

    def test_all_six_combinations_resolve(self):
        """All 6 dynamics x discretization combinations should now resolve."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        expected = {
            ("shallow_water", "centered"): "cdgrid_shallow_water",
            ("shallow_water", "spectral"): "spectral_shallow_water",
            ("hydrostatic", "centered"): "cdgrid_primitive_equations",
            ("hydrostatic", "spectral"): "spectral_primitive_equations",
            ("nonhydrostatic", "centered"): "cdgrid_compressible_euler",
            ("nonhydrostatic", "spectral"): "spectral_compressible_euler",
        }
        for (dyn, disc), expected_name in expected.items():
            name = resolve_solver_name(dynamics=dyn, discretization=disc)
            assert name == expected_name, \
                f"({dyn}, {disc}) -> {name}, expected {expected_name}"


def test_batched_cpu_step_honors_threaded_target_mass():
    """codex 2026-07-12 round 2: _step_on_cpu must consume the THREADED
    target mass, not a closure-captured self._target_mass (which froze
    the first target into the compiled step)."""
    import jax.numpy as jnp
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    grid = create_gaussian_grid(21)
    state, hcoord, tmetric = dcmip25_tc1_init_spectral(grid, n_levels=8)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=4, semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = SpectralCompressibleEulerModel(
        grid, hcoord, tmetric, cfg, allow_unsupported_backend=True,
    )
    m0 = float(model.compute_dry_mass(state))

    out1 = model._step_on_cpu(state, 2.0, jnp.float64(m0))
    m1 = float(model.compute_dry_mass(out1))
    assert abs(m1 - m0) / m0 < 1e-12

    # SAME compiled step, new threaded target — must be honored.
    out2 = model._step_on_cpu(state, 2.0, jnp.float64(1.01 * m0))
    m2 = float(model.compute_dry_mass(out2))
    assert abs(m2 - 1.01 * m0) / m0 < 1e-9, (
        f"threaded target ignored: {m2} vs {1.01 * m0}"
    )
