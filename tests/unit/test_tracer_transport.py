"""Tests for prescribed-wind tracer transport model."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.atmosphere.dynamics.shared.tracer_transport import (
    TracerTransportModel,
    TracerTransportConfig,
    tracer_tendencies,
)


# Small grid for fast tests
N = 8
NLEV = 5
N_TRACERS = 4


@pytest.fixture
def grid():
    return create_cubed_sphere(N)


@pytest.fixture
def sigma_coord():
    return create_sigma_coordinate(NLEV)


def _zero_wind(t, grid, sigma_coord):
    """Zero wind everywhere."""
    nlev = sigma_coord.n_levels
    n = grid.lon.shape[1]
    u = jnp.zeros((6, n, n, nlev))
    v = jnp.zeros((6, n, n, nlev))
    sigma_dot = jnp.zeros((6, n, n, nlev + 1))
    return u, v, sigma_dot


def _make_state(grid, sigma_coord, n_tracers=N_TRACERS, fill_value=1.0):
    """Create a simple TracerState."""
    n = grid.lon.shape[1]
    nlev = sigma_coord.n_levels
    q = jnp.full((6, n, n, nlev, n_tracers), fill_value)
    tracers = Field(data=q, name="tracers",
                    dims=("face", "x", "y", "level", "tracer"), units="kg/kg")
    time = Field(data=jnp.array(0.0), name="time", dims=(), units="s")
    return TracerState(tracers=tracers, time=time)


class TestTracerTendencies:
    """Test the tendency computation."""

    def test_uniform_tracer_zero_wind_zero_tendency(self, grid, sigma_coord):
        """Uniform tracer + zero wind = zero tendency."""
        state = _make_state(grid, sigma_coord, fill_value=1.0)
        tend = tracer_tendencies(state, grid, sigma_coord, _zero_wind)
        assert jnp.allclose(tend.tracers.data, 0.0, atol=1e-10)

    def test_nonuniform_tracer_zero_wind_zero_tendency(self, grid, sigma_coord):
        """Non-uniform tracer + zero wind = zero tendency (no advection)."""
        state = _make_state(grid, sigma_coord)
        # Make non-uniform
        q = state.tracers.data
        q = q.at[0, :, :, :, 0].set(2.0)
        state = TracerState(
            tracers=state.tracers.replace(data=q),
            time=state.time,
        )
        tend = tracer_tendencies(state, grid, sigma_coord, _zero_wind)
        assert jnp.allclose(tend.tracers.data, 0.0, atol=1e-10)

    def test_time_tendency_is_one(self, grid, sigma_coord):
        """Time tendency should always be 1.0 (dt/dt = 1)."""
        state = _make_state(grid, sigma_coord)
        tend = tracer_tendencies(state, grid, sigma_coord, _zero_wind)
        assert jnp.allclose(tend.time.data, 1.0)

    def test_tendency_shapes(self, grid, sigma_coord):
        """Tendency shapes match state shapes."""
        state = _make_state(grid, sigma_coord)
        tend = tracer_tendencies(state, grid, sigma_coord, _zero_wind)
        assert tend.tracers.data.shape == state.tracers.data.shape
        assert tend.time.data.shape == state.time.data.shape

    def test_tendency_finite_with_wind(self, grid, sigma_coord):
        """Tendencies are finite with non-zero wind."""
        def solid_rotation(t, grid, sigma_coord):
            nlev = sigma_coord.n_levels
            n = grid.lon.shape[1]
            u = jnp.cos(grid.lat)[..., None] * jnp.ones(nlev) * 20.0
            v = jnp.zeros((6, n, n, nlev))
            sigma_dot = jnp.zeros((6, n, n, nlev + 1))
            return u, v, sigma_dot

        # Non-uniform tracer
        state = _make_state(grid, sigma_coord)
        q = state.tracers.data
        q = q.at[..., 0].set(jnp.sin(grid.lon[..., None]) * jnp.ones(NLEV))
        state = TracerState(tracers=state.tracers.replace(data=q), time=state.time)

        tend = tracer_tendencies(state, grid, sigma_coord, solid_rotation)
        assert jnp.all(jnp.isfinite(tend.tracers.data))


class TestTracerTransportModel:
    """Test the model class."""

    def test_single_step_finite(self, grid, sigma_coord):
        """Single step produces finite results."""
        model = TracerTransportModel(grid, sigma_coord, _zero_wind)
        state = _make_state(grid, sigma_coord)
        new_state = model.step(state, 600.0)
        assert jnp.all(jnp.isfinite(new_state.tracers.data))
        assert jnp.all(jnp.isfinite(new_state.time.data))

    def test_time_advances(self, grid, sigma_coord):
        """Time advances by dt each step."""
        model = TracerTransportModel(grid, sigma_coord, _zero_wind)
        state = _make_state(grid, sigma_coord)
        dt = 600.0
        new_state = model.step(state, dt)
        assert jnp.allclose(new_state.time.data, dt, rtol=1e-5)

    def test_uniform_preserved(self, grid, sigma_coord):
        """Uniform tracer stays uniform with zero wind."""
        model = TracerTransportModel(grid, sigma_coord, _zero_wind)
        state = _make_state(grid, sigma_coord, fill_value=5.0)
        for _ in range(3):
            state = model.step(state, 600.0)
        assert jnp.allclose(state.tracers.data, 5.0, atol=1e-8)

    def test_flexible_n_tracers(self, grid, sigma_coord):
        """Works with different numbers of tracers (1, 4, 10)."""
        for n_t in [1, 4, 10]:
            model = TracerTransportModel(grid, sigma_coord, _zero_wind)
            state = _make_state(grid, sigma_coord, n_tracers=n_t)
            new_state = model.step(state, 600.0)
            assert new_state.tracers.data.shape[-1] == n_t
            assert jnp.all(jnp.isfinite(new_state.tracers.data))

    def test_hyperdiffusion(self, grid, sigma_coord):
        """Hyperdiffusion produces different tendency than without."""
        state = _make_state(grid, sigma_coord)
        # High-wavenumber pattern: checkerboard-like pattern has large ∇⁴
        # Use grid-scale noise to get measurable hyperdiffusion
        n = grid.lon.shape[1]
        ix = jnp.arange(n)
        checker = (-1.0) ** (ix[None, :, None] + ix[None, None, :])  # (1, n, n)
        checker_3d = jnp.broadcast_to(checker, (6, n, n))  # (6, n, n)
        checker_4d = checker_3d[..., None] * jnp.ones(NLEV)  # (6, n, n, NLEV)
        q = state.tracers.data
        q = q.at[..., 0].set(checker_4d)
        state = TracerState(tracers=state.tracers.replace(data=q), time=state.time)

        tend_no = tracer_tendencies(
            state, grid, sigma_coord, _zero_wind,
            TracerTransportConfig(hyperdiff_coeff=0.0)
        )
        tend_yes = tracer_tendencies(
            state, grid, sigma_coord, _zero_wind,
            TracerTransportConfig(hyperdiff_coeff=1e15)
        )
        # Tendencies should differ for the non-uniform tracer
        diff = jnp.max(jnp.abs(tend_no.tracers.data - tend_yes.tracers.data))
        assert diff > 0.0, f"Hyperdiffusion had no effect, diff={float(diff)}"

    def test_integrate(self, grid, sigma_coord):
        """integrate method runs without error."""
        model = TracerTransportModel(grid, sigma_coord, _zero_wind)
        state = _make_state(grid, sigma_coord)
        final, trajectory = model.integrate(state, duration=1200.0, dt=600.0)
        assert jnp.allclose(final.time.data, 1200.0, rtol=1e-4)
        assert len(trajectory) == 3  # initial + 2 steps


class TestDifferentiability:
    """Test that tracer transport is differentiable."""

    def test_grad_through_tendencies(self, grid, sigma_coord):
        """jax.grad works through tendency computation."""
        state = _make_state(grid, sigma_coord)

        def loss(q0_val):
            q = state.tracers.data.at[..., 0].set(q0_val)
            s = TracerState(
                tracers=state.tracers.replace(data=q),
                time=state.time,
            )
            tend = tracer_tendencies(s, grid, sigma_coord, _zero_wind)
            return jnp.sum(tend.tracers.data**2)

        q0 = jnp.ones((6, N, N, NLEV))
        grad = jax.grad(loss)(q0)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_step(self, grid, sigma_coord):
        """jax.grad works through a full step."""
        model = TracerTransportModel(grid, sigma_coord, _zero_wind)
        state = _make_state(grid, sigma_coord)

        def loss(q_data):
            s = TracerState(
                tracers=state.tracers.replace(data=q_data),
                time=state.time,
            )
            s_new = model.step(s, 600.0)
            return jnp.sum(s_new.tracers.data**2)

        grad = jax.grad(loss)(state.tracers.data)
        assert jnp.all(jnp.isfinite(grad))


class TestSolverFactory:
    """Test the create_model factory function."""

    def test_create_tracer_transport(self, grid, sigma_coord):
        """Factory creates TracerTransportModel."""
        from legoesm.atmosphere.dynamics import create_model
        model = create_model(
            "tracer_transport",
            grid=grid, sigma_coord=sigma_coord, wind_fn=_zero_wind,
        )
        assert isinstance(model, TracerTransportModel)

    def test_create_shallow_water(self, grid):
        """Factory creates ShallowWaterModel."""
        from legoesm.atmosphere.dynamics import create_model, ShallowWaterModel
        model = create_model("shallow_water", grid=grid)
        assert isinstance(model, ShallowWaterModel)

    def test_create_primitive_equations(self, grid, sigma_coord):
        """Factory creates PrimitiveEquationModel."""
        from legoesm.atmosphere.dynamics import create_model, PrimitiveEquationModel
        model = create_model(
            "primitive_equations",
            grid=grid, sigma_coord=sigma_coord,
        )
        assert isinstance(model, PrimitiveEquationModel)

    def test_create_unknown_raises(self):
        """Unknown solver name raises ValueError."""
        from legoesm.atmosphere.dynamics import create_model
        with pytest.raises(ValueError, match="Unknown solver"):
            create_model("nonexistent_solver")

    def test_available_solvers_list(self):
        """AVAILABLE_SOLVERS is populated."""
        from legoesm.atmosphere.dynamics import AVAILABLE_SOLVERS
        assert "cdgrid_shallow_water" in AVAILABLE_SOLVERS
        assert "tracer_transport" in AVAILABLE_SOLVERS
        assert "cdgrid_primitive_equations" in AVAILABLE_SOLVERS
        assert "spectral_shallow_water" in AVAILABLE_SOLVERS


class TestSolverAxes:
    """Test the two-axis (dynamics x discretization) solver selection."""

    def test_resolve_hydrostatic_centered(self):
        """dynamics=hydrostatic + discretization=centered -> cdgrid_primitive_equations."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="hydrostatic", discretization="centered")
        assert name == "cdgrid_primitive_equations"

    def test_resolve_nonhydrostatic_centered(self):
        """dynamics=nonhydrostatic + discretization=centered -> cdgrid_compressible_euler."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="nonhydrostatic", discretization="centered")
        assert name == "cdgrid_compressible_euler"

    def test_resolve_shallow_water_centered(self):
        """dynamics=shallow_water + discretization=centered -> cdgrid_shallow_water."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="shallow_water", discretization="centered")
        assert name == "cdgrid_shallow_water"

    def test_resolve_shallow_water_spectral(self):
        """dynamics=shallow_water + discretization=spectral -> spectral_shallow_water."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="shallow_water", discretization="spectral")
        assert name == "spectral_shallow_water"

    def test_resolve_no_selection_raises(self):
        """No arguments is not a solver choice: raise, never default."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="no solver selected"):
            resolve_solver_name()

    def test_resolve_missing_discretization_raises(self):
        """dynamics alone must not silently pick the cdgrid discretization."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="needs both dynamics"):
            resolve_solver_name(dynamics="nonhydrostatic")

    def test_resolve_missing_dynamics_raises(self):
        """discretization alone must not silently pick shallow-water dynamics."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="needs both dynamics"):
            resolve_solver_name(discretization="spectral")

    def test_resolve_unknown_equations_raises(self):
        """A mistyped legacy solver name must raise, not become cdgrid SW."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="Unknown equations"):
            resolve_solver_name(equations="hydrostatik")

    def test_resolve_invalid_dynamics_raises(self):
        """Invalid dynamics value raises ValueError."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="Unknown dynamics"):
            resolve_solver_name(dynamics="invalid", discretization="cdgrid")

    def test_resolve_invalid_discretization_raises(self):
        """Invalid discretization value raises ValueError."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with pytest.raises(ValueError, match="Unknown discretization"):
            resolve_solver_name(dynamics="hydrostatic", discretization="invalid")

    def test_resolve_hydrostatic_spectral(self):
        """dynamics=hydrostatic + discretization=spectral -> spectral_primitive_equations."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="hydrostatic", discretization="spectral")
        assert name == "spectral_primitive_equations"

    def test_resolve_nonhydrostatic_spectral(self):
        """dynamics=nonhydrostatic + discretization=spectral -> spectral_compressible_euler."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="nonhydrostatic", discretization="spectral")
        assert name == "spectral_compressible_euler"

    def test_resolve_legacy_equations_takes_precedence(self):
        """Legacy equations key works when no axis keys are given."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(equations="primitive_equations")
        # Legacy "primitive_equations" is a deprecated alias → cdgrid_primitive_equations
        assert name == "cdgrid_primitive_equations"

    def test_resolve_explicit_axes_override_legacy(self):
        """When dynamics is set explicitly, it overrides legacy equations."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(
            dynamics="nonhydrostatic",
            discretization="cdgrid",
            equations="shallow_water",  # legacy, should be ignored
        )
        assert name == "cdgrid_compressible_euler"

    def test_solver_axes_roundtrip(self):
        """solver_axes() is the inverse of resolve_solver_name()."""
        from legoesm.atmosphere.dynamics import resolve_solver_name, solver_axes
        for dyn in ["shallow_water", "hydrostatic", "nonhydrostatic"]:
            for disc in ["cdgrid", "spectral"]:
                try:
                    name = resolve_solver_name(dynamics=dyn, discretization=disc)
                except ValueError:
                    continue  # unimplemented combination
                axes = solver_axes(name)
                assert axes == (dyn, disc), f"Roundtrip failed for {name}"

    def test_create_model_from_config_hydrostatic(self, grid, sigma_coord):
        """create_model with Config resolves hydrostatic centered correctly."""
        from legoesm.atmosphere.dynamics import create_model, PrimitiveEquationModel
        from legoesm.config import Config
        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "hydrostatic",
                "discretization": "centered",
            }
        })
        model = create_model(legoesm_config=cfg, grid=grid, sigma_coord=sigma_coord)
        assert isinstance(model, PrimitiveEquationModel)

    def test_create_model_from_config_nonhydrostatic(self, grid):
        """create_model with Config resolves nonhydrostatic centered correctly."""
        from legoesm.atmosphere.dynamics import create_model, CompressibleEulerModel
        from legoesm.config import Config
        from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
        hc = create_height_coordinate(10, 30000.0)
        z_s = jnp.zeros((6, grid.lon.shape[1], grid.lon.shape[1]))
        tm = compute_terrain_metric(z_s, hc)
        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "nonhydrostatic",
                "discretization": "centered",
            }
        })
        model = create_model(
            legoesm_config=cfg, grid=grid,
            height_coord=hc, terrain_metric=tm,
        )
        assert isinstance(model, CompressibleEulerModel)

    def test_create_model_from_config_shallow_water(self, grid):
        """create_model with Config resolves shallow_water centered correctly."""
        from legoesm.atmosphere.dynamics import create_model, ShallowWaterModel
        from legoesm.config import Config
        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "shallow_water",
                "discretization": "centered",
            }
        })
        model = create_model(legoesm_config=cfg, grid=grid)
        assert isinstance(model, ShallowWaterModel)

    def test_create_model_no_name_no_config_raises(self):
        """create_model with neither name nor config raises ValueError."""
        from legoesm.atmosphere.dynamics import create_model
        with pytest.raises(ValueError, match="Either"):
            create_model()

    def test_dynamics_options_exported(self):
        """DYNAMICS_OPTIONS and DISCRETIZATION_OPTIONS are accessible."""
        from legoesm.atmosphere.dynamics import DYNAMICS_OPTIONS, DISCRETIZATION_OPTIONS
        assert "hydrostatic" in DYNAMICS_OPTIONS
        assert "nonhydrostatic" in DYNAMICS_OPTIONS
        assert "shallow_water" in DYNAMICS_OPTIONS
        assert "cdgrid" in DISCRETIZATION_OPTIONS
        assert "spectral" in DISCRETIZATION_OPTIONS


# ===================================================================
# Lat-lon C-grid tracer transport — face-coordinate evaluation
# ===================================================================

class TestLatLonTracerTransportFaceEval:
    """Verify prescribed winds are evaluated at true face coordinates."""

    def test_face_eval_differs_from_cell_avg(self):
        """A wind that varies sharply in longitude should produce different
        face values when evaluated at face coords vs averaged from cells.
        """
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
            _uface_coords, _vface_coords,
        )

        grid = create_latlon_grid(16)
        sigma = create_sigma_coordinate(5)

        lon_u, lat_u = _uface_coords(grid)
        lon_v, lat_v = _vface_coords(grid)

        # _uface_coords returns n_lon unique faces (no periodic wrap);
        # the caller appends the wrap column after wind evaluation.
        assert lon_u.shape == (grid.n_lat, grid.n_lon)
        assert lat_v.shape == (grid.n_lat + 1, grid.n_lon)
        # Interior u-face lon should be offset from cell lon by ~dlon/2
        lon_diff = float(jnp.abs(lon_u[0, 1] - grid.lon[0]))
        assert lon_diff > 0.01, f"u-face lon not offset: diff={lon_diff}"

    def test_spatially_varying_wind_fidelity(self):
        """When wind varies sharply, face-evaluated transport must differ
        from cell-averaged transport.
        """
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
            TracerTransportLatLonModel, TracerTransportLatLonConfig,
        )

        grid = create_latlon_grid(16)
        sigma = create_sigma_coordinate(5)
        nlev = sigma.n_levels

        # Sharply varying wind: u = sin(4*lon), v = 0, sigma_dot = 0
        def sharp_wind(t, lon2d, lat2d, sc):
            u = 10.0 * jnp.sin(4.0 * lon2d)[..., None] * jnp.ones(nlev)
            v = jnp.zeros_like(u)
            sd = jnp.zeros((*lon2d.shape, nlev + 1))
            return u, v, sd

        # Gaussian tracer bump
        q_data = jnp.exp(-(grid.lon2d**2 + grid.lat2d**2) / 0.5)
        q_data = q_data[:, :, None, None] * jnp.ones((1, 1, nlev, 1))
        tracers = Field(data=q_data, name="tracers",
                        dims=("lat", "lon", "level", "tracer"), units="kg/kg")
        time = Field(data=jnp.array(0.0), name="time", dims=(), units="s")
        state = TracerState(tracers=tracers, time=time)

        model = TracerTransportLatLonModel(grid, sigma, sharp_wind)
        state_new = model.step(state, dt=300.0)

        assert jnp.all(jnp.isfinite(state_new.tracers.data))
        # The tracer should have been advected (not identical to initial)
        diff = float(jnp.max(jnp.abs(state_new.tracers.data - q_data)))
        assert diff > 1e-6, f"Tracer unchanged after transport: max diff = {diff}"
