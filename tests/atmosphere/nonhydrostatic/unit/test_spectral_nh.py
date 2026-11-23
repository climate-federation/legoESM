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
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
    SpectralNHState,
    SpectralNHConfig,
    SpectralCompressibleEulerModel,
    spectral_nh_slow_tendencies,
    _acoustic_substeps_grid,
    nh_rest_state_spectral,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
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
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
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


def test_semi_implicit_acoustic_matches_shared_kernel_on_stretched_grid():
    """On a stretched vertical grid the spectral semi-implicit w solve and
    continuity update must equal the shared column kernel's (stretched-grid
    metric; both couplings kept on the diagonal next to the rigid lid).  The
    θ' update differs only on the top/bottom levels by design (docstring)."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        semi_implicit_acoustic_column_kernel,
    )
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        _acoustic_substeps_grid_semi_implicit,
    )
    from legoesm.grids.vertical import create_height_coordinate_from_z_half

    dz = 50.0 * 1.4 ** np.arange(12)                       # 50 m ... ~2 km
    z_half = jnp.asarray(np.concatenate([[0.0], np.cumsum(dz)])[::-1])
    hc = create_height_coordinate_from_z_half(z_half)
    tm = compute_terrain_metric(jnp.zeros((3, 4)), hc)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    nlev = hc.n_levels
    theta_p = 0.5 * jax.random.normal(k1, (3, 4, nlev))
    rho_p = 1e-3 * jax.random.normal(k2, (3, 4, nlev))
    w = jnp.zeros((3, 4, nlev + 1))
    cfg = SpectralNHConfig()
    dt_s = 2.0

    w_s, th_s, rho_s = _acoustic_substeps_grid_semi_implicit(
        w, theta_p, rho_p, dt_s, 1, hc, tm, cfg)
    w_k, th_k, rho_k = semi_implicit_acoustic_column_kernel(
        w, theta_p, rho_p, hc, tm.jacobian, dt_s, 0.0, cfg.g)
    np.testing.assert_allclose(np.asarray(w_s), np.asarray(w_k), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(rho_s), np.asarray(rho_k), rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(np.asarray(th_s[..., 1:-1]), np.asarray(th_k[..., 1:-1]),
                               rtol=1e-12, atol=1e-14)


# T29: native spectral projection of continuity-paired tracer mass.
def t29_spectral_fixture(semi_implicit, *, nonflat=False):
    """Existing rough seed1 state on the census grid, with positive packed q."""
    from tests.unit.test_spectral_dealiasing import TestAliasingStabilityNH
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import dcmip25_tc2_init_spectral

    grid = create_gaussian_grid(10)
    hc = create_height_coordinate(10, 30000.0)
    state = TestAliasingStabilityNH()._rough_nh_state(grid, hc, n_tracers=1, seed=1)
    q = sh_synthesis_3d(grid, state.tracers_hat.data[..., 0]) + 0.01
    assert np.all(np.asarray(q) > 0)
    q_hat = sh_analysis_3d(grid, q)
    w = jnp.broadcast_to(
        jnp.sin(jnp.linspace(0, jnp.pi, 11)).at[-1].set(0.0),
        (grid.n_lat, grid.n_lon, 11),
    )
    state = state._replace(
        tracers_hat=state.tracers_hat.replace(data=jnp.stack([q_hat, 2 * q_hat], axis=-1)),
        w_hat=state.w_hat.replace(data=sh_analysis_3d(grid, w)),
    )
    metric = (dcmip25_tc2_init_spectral(grid, n_levels=10, subcase="a")[2]
              if nonflat else compute_terrain_metric(jnp.zeros_like(grid.grid_area), hc))
    jac = np.asarray(metric.jacobian)
    assert np.isfinite(jac).all() and np.all(jac > 0)
    if nonflat:
        assert np.max(np.abs(jac - 1)) > 1e-12 and np.ptp(jac) > 1e-12
    cfg = SpectralNHConfig(
        g=constants.g, hyperdiff_coeff=0.0, hyperdiff_order=2,
        sponge_coeff=0.0, sponge_width=10000.0, n_acoustic_substeps=6,
        small_earth_factor=1.0, semi_implicit_acoustic=semi_implicit,
        dealiasing_fraction=0.667, fix_mass=False, anchor_mass_to_initial=False,
        rotating=True,
    )
    model = SpectralCompressibleEulerModel(grid, hc, metric, cfg)
    rho, _ = t29_spectral_grid_state(model, state)
    assert np.all(np.asarray(rho) > 0) and np.ptp(np.asarray(rho), axis=(0, 1)).max() > 0
    assert np.any(np.asarray(state.vor_hat.data) != 0)
    assert np.any(np.asarray(state.div_hat.data) != 0) and np.any(np.asarray(w) != 0)
    return model, state


def t29_spectral_grid_state(model, state):
    """Actual represented density and packed mixing ratios, also used by census."""
    nsh, nz, nt = state.tracers_hat.data.shape
    q = sh_synthesis_3d(model.grid, state.tracers_hat.data.reshape(nsh, nz * nt))
    q = q.reshape(model.grid.n_lat, model.grid.n_lon, nz, nt)
    rho = model.height_coord.rho_ref + sh_synthesis_3d(model.grid, state.rho_prime_hat.data)
    return rho, q


class TestT29SpectralMass:
    @pytest.fixture(autouse=True)
    def fp64_policy(self):
        from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
        previous = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            yield
        finally:
            set_policy(previous)

    @pytest.mark.parametrize("semi_implicit", [False, True])
    @pytest.mark.parametrize("metric_kind", ["flat", "TC2a", "one_ulp"])
    def test_static_geometry_dispatch(self, semi_implicit, metric_kind, monkeypatch):
        """Select the actual integrator alias, eager and traced; plant misrouting."""
        from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
        from legoesm.timestepping.split_explicit import SplitExplicitConfig

        model, state = t29_spectral_fixture(semi_implicit, nonflat=metric_kind == "TC2a")
        if metric_kind == "one_ulp":
            jac = np.array(model.terrain_metric.jacobian, copy=True)
            assert jac.dtype == np.dtype("float64")
            assert np.all(jac == 1.0)
            jac.flat[0] = np.nextafter(jac.flat[0], np.float64(2.0))
            assert jac.flat[0] != 1.0
            metric = model.terrain_metric._replace(jacobian=jnp.asarray(jac))
            # This metric tests selection only: no numerical integration.
            model = SpectralCompressibleEulerModel(
                model.grid, model.height_coord, metric, model.config)
        expected = metric_kind == "flat"
        assert type(model._flat_tracer_metric) is bool
        assert model._flat_tracer_metric is expected
        monkeypatch.setattr(sn, "split_explicit_step", lambda *a, **kw: jnp.asarray(0))
        monkeypatch.setattr(sn, "split_explicit_tracer_mass_step", lambda *a, **kw: jnp.asarray(1))
        cfg = SplitExplicitConfig(n_substeps=model.config.n_acoustic_substeps)

        for compiled in (False, True):
            def check_dispatch():
                step = lambda s: model._split_step(s, 1.0, cfg)
                if compiled:
                    # Fresh closure for each check, including the planted
                    # violation: changing a captured bool cannot retrace a cache.
                    step = jax.jit(step)
                assert int(step(state)) == int(expected)

            check_dispatch()
            with monkeypatch.context() as control:
                control.setattr(model, "_flat_tracer_metric", not expected)
                with pytest.raises(AssertionError):
                    check_dispatch()

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_full_step_inventory_and_constant(self, semi_implicit):
        # Reversion selects this node only: no new production-helper import.
        from scripts.validate.diag_hybrid_tracer_smoothing import global_inventory_budget
        model, state = t29_spectral_fixture(semi_implicit)
        grid, hc, metric = model.grid, model.height_coord, model.terrain_metric
        volume = np.asarray(grid.grid_area[..., None] * metric.jacobian[..., None] * hc.dz)
        rho0, q0 = t29_spectral_grid_state(model, state)
        final = model.step(state, 1.0)
        rho1, q1 = t29_spectral_grid_state(model, final)
        row = global_inventory_budget(volume, rho0, q0, rho1, q1, np.zeros(2))
        assert row["verdicts"] == ["CONSERVES", "CONSERVES"], row
        q_const = sh_analysis_3d(grid, jnp.full(q0.shape[:-1], 0.01))
        constant = state._replace(tracers_hat=state.tracers_hat.replace(
            data=jnp.stack([q_const, q_const], axis=-1)))
        rc0, qc0 = t29_spectral_grid_state(model, constant)
        rc1, qc1 = t29_spectral_grid_state(model, model.step(constant, 1.0))
        np.testing.assert_allclose(qc1, 0.01, rtol=0, atol=1e-12)
        row = global_inventory_budget(volume, rc0, qc0, rc1, qc1, np.zeros(2))
        assert row["verdicts"] == ["CONSERVES", "CONSERVES"], row

    def test_native_projection_orthogonality_and_planted_violation(self):
        model, state = t29_spectral_fixture(False)
        grid = model.grid
        rho, q = t29_spectral_grid_state(model, state)
        mass = model.terrain_metric.jacobian[..., None, None] * rho[..., None] * q
        area = np.asarray(grid.grid_area)[..., None]

        def project(a):
            return np.asarray(sh_synthesis_3d(grid, sh_analysis_3d(grid, a)))

        for f, outside_span in ((np.asarray(rho), False), (np.asarray(mass[..., 0]), True)):
            pf = project(f)
            if outside_span:
                assert np.max(np.abs(f - pf)) > 1e-12
            for tracer in range(2):
                pg = project(mass[..., tracer])
                lhs = np.sum(area * f * pg, axis=(0, 1))
                rhs = np.sum(area * pf * pg, axis=(0, 1))
                norm = np.sum(area * np.abs(f * pg), axis=(0, 1))
                assert np.all(np.isfinite(norm)) and np.all(norm > 0)
                assert np.all(np.abs(lhs - rhs) / norm <= 1e-12)
                offset = 0.01 * np.max(np.abs(pg), axis=(0, 1))
                assert np.all(offset > 0)
                planted_lhs = np.sum(area * f * (pg + offset), axis=(0, 1))
                # SAME norm and predicate, changed only the left operand.
                assert np.all(np.abs(planted_lhs - rhs) / norm > 1e-12)

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_acoustic_export_pairs_density(self, semi_implicit):
        from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
        model, state = t29_spectral_fixture(semi_implicit)
        grid, hc, metric = model.grid, model.height_coord, model.terrain_metric
        w = sh_synthesis_3d(grid, state.w_hat.data)
        theta = sh_synthesis_3d(grid, state.theta_prime_hat.data)
        rho = sh_synthesis_3d(grid, state.rho_prime_hat.data)
        acoustic = (sn._acoustic_substeps_grid_semi_implicit if semi_implicit
                    else sn._acoustic_substeps_grid)
        (_, _, final_rho), flux = acoustic(
            w, theta, rho, 1.0 / 6, 6, hc, metric, model.config, return_mass_flux=True)
        rhs = (flux[..., 1:] - flux[..., :-1]) / (hc.dz * metric.jacobian[..., None])
        np.testing.assert_allclose(final_rho - rho, rhs, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(flux[..., (0, -1)], 0.0)
        assert np.max(np.abs(np.asarray(flux[..., 1:-1]))) > 0

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_stage_blend_and_recovery_budgets(self, semi_implicit, monkeypatch):
        from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
        from legoesm.timestepping import split_explicit as se
        model, state = t29_spectral_fixture(semi_implicit)
        weight = np.asarray(model.grid.grid_area[..., None, None] * model.height_coord.dz[:, None])
        jac = np.asarray(model.terrain_metric.jacobian)[..., None, None]
        observations = {"stage": [], "blend": [], "recovery": []}

        def mass_grid(mhat):
            nsh, nz, nt = mhat.shape
            return np.asarray(sh_synthesis_3d(model.grid, mhat.reshape(nsh, nz * nt))).reshape(
                model.grid.n_lat, model.grid.n_lon, nz, nt)

        def integral(mhat):
            return np.sum(weight * mass_grid(mhat), axis=(0, 1, 2))

        rho, q = t29_spectral_grid_state(model, state)
        norm = np.sum(weight * np.abs(jac * np.asarray(rho)[..., None] * np.asarray(q)), axis=(0, 1, 2))
        assert np.all(norm > 0)

        def close(a, b):
            assert np.all(np.abs(a - b) / norm <= 1e-12), (a - b) / norm

        def reset(carry):
            assert all(np.all(np.asarray(a) == 0) for a in jax.tree.leaves(carry[2:]))

        original_step = se.split_explicit_step
        original_blend = se._pytree_linear_combination
        original_helper = sn.split_explicit_tracer_mass_step

        def step(carry, slow, acoustic, dt, cfg):
            assert cfg.outer_integrator == "ssp_rk3"

            def observe(current, tendency, dt_s, n_sub, config):
                out = acoustic(current, tendency, dt_s, n_sub, config)
                before, after = integral(current[1]), integral(out[1])
                close(before, after)
                reset(out)
                observations["stage"].append((before, after, np.asarray(out[1])))
                return out

            return original_step(carry, slow, observe, dt, cfg)

        def blend(a, b, wa, wb):
            out = original_blend(a, b, wa, wb)
            before, after = wa * integral(a[1]) + wb * integral(b[1]), integral(out[1])
            close(before, after)
            reset(out)
            observations["blend"].append((before, after, np.asarray(out[1])))
            return out

        def helper(*args, **kwargs):
            recover = kwargs["recover_tracer_fn"]

            def observed(mhat, rho):
                qhat = recover(mhat, rho)
                before = integral(mhat)
                after = np.sum(weight * jac * np.asarray(rho)[..., None] * mass_grid(qhat), axis=(0, 1, 2))
                close(before, after)
                observations["recovery"].append((before, after))
                return qhat

            kwargs["recover_tracer_fn"] = observed
            return original_helper(*args, **kwargs)

        monkeypatch.setattr(se, "split_explicit_step", step)
        monkeypatch.setattr(se, "_pytree_linear_combination", blend)
        monkeypatch.setattr(sn, "split_explicit_tracer_mass_step", helper)
        with jax.disable_jit():
            model.step(state, 1.0)
        assert len(observations["stage"]) == 3
        assert len(observations["blend"]) == 2
        assert len(observations["recovery"]) == 7
        for kind, rows in observations.items():
            print("T29 spectral", semi_implicit, kind, [(a.tolist(), b.tolist()) for a, b, *_ in rows])

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_jit_eager_and_cpu_entry_parity(self, semi_implicit):
        model, state = t29_spectral_fixture(semi_implicit)
        compiled = model.step(state, 1.0)
        cpu_entry = model._step_on_cpu(state, 1.0)
        with jax.disable_jit():
            eager = model.step(state, 1.0)
        for other in (cpu_entry, eager):
            for a, b in zip(t29_spectral_grid_state(model, compiled), t29_spectral_grid_state(model, other)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_fp32_input_gradients_q_and_w(self, semi_implicit):
        """FP32 input gradients; spectral transforms retain native fp64."""
        model, state = t29_spectral_fixture(semi_implicit)

        def loss(q_scale, w_scale):
            initial = state._replace(
                tracers_hat=state.tracers_hat.replace(data=q_scale * state.tracers_hat.data),
                w_hat=state.w_hat.replace(data=w_scale * state.w_hat.data))
            _, q = t29_spectral_grid_state(model, model.step(initial, 1.0))
            return jnp.sum(q ** 2)

        one = jnp.asarray(1.0, dtype=jnp.float32)
        for grad in jax.grad(loss, argnums=(0, 1))(one, one):
            assert grad.dtype == jnp.float32
            assert np.isfinite(grad).all() and np.any(np.asarray(grad) != 0)

    @pytest.mark.parametrize("semi_implicit", [False, True])
    def test_fp64_check_grads_q_and_w(self, semi_implicit):
        from jax.test_util import check_grads
        model, state = t29_spectral_fixture(semi_implicit)

        def loss(q_scale, w_scale):
            initial = state._replace(
                tracers_hat=state.tracers_hat.replace(data=q_scale * state.tracers_hat.data),
                w_hat=state.w_hat.replace(data=w_scale * state.w_hat.data))
            _, q = t29_spectral_grid_state(model, model.step(initial, 1.0))
            return jnp.sum(q ** 2)

        one = jnp.asarray(1.0, dtype=jnp.float64)
        check_grads(loss, (one, one), order=1, modes=["rev"], rtol=1e-3, atol=1e-6)
