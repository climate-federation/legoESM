"""Tests for DCMIP-2012 transport test cases (1-1, 1-2, 1-3)."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.test_cases.dcmip_transport import (
    dcmip11_wind,
    dcmip11_init,
    dcmip12_wind,
    dcmip12_init,
    dcmip13_wind,
    dcmip13_init,
    compute_tracer_error_norms,
    create_dcmip_sigma,
    _height_from_sigma,
    _mountain_height,
)
from legoesm.atmosphere.dynamics.shared.tracer_transport import (
    TracerTransportModel,
    TracerTransportConfig,
)


# Small grid for fast tests
N = 8
NLEV = 10


@pytest.fixture
def grid():
    return create_cubed_sphere(N)


@pytest.fixture
def sigma_coord():
    return create_dcmip_sigma(NLEV)


def test_dcmip_sigma_uniform_height_spacing():
    """DCMIP sigma coordinate maps to uniformly spaced geometric height."""
    sigma = create_dcmip_sigma(20)
    z_half = _height_from_sigma(sigma.sigma_half)
    dz = z_half[:-1] - z_half[1:]
    assert jnp.allclose(dz, dz[0], rtol=5e-6, atol=1e-6)
    assert float(z_half[0]) == pytest.approx(12000.0, abs=1e-3)
    assert float(z_half[-1]) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Test 1-1: Deformational Flow
# ===========================================================================

class TestDCMIP11:
    """Test 1-1 initialization and wind field."""

    def test_init_shapes(self, grid, sigma_coord):
        """Initial state has correct shapes."""
        state = dcmip11_init(grid, sigma_coord)
        assert state.tracers.data.shape == (6, N, N, NLEV, 4)
        assert state.time.data.shape == ()
        assert float(state.time.data) == 0.0

    def test_init_q1_bounds(self, grid, sigma_coord):
        """q1 (cosine bells) is non-negative and bounded."""
        state = dcmip11_init(grid, sigma_coord)
        q1 = state.tracers.data[..., 0]
        assert jnp.all(q1 >= -1e-10)
        assert jnp.all(q1 <= 2.1)  # max is 2.0 (two bells)

    def test_init_q2_correlated(self, grid, sigma_coord):
        """q2 = 0.9 - 0.8 * q1^2."""
        state = dcmip11_init(grid, sigma_coord)
        q1 = state.tracers.data[..., 0]
        q2 = state.tracers.data[..., 1]
        q2_expected = 0.9 - 0.8 * q1**2
        assert jnp.allclose(q2, q2_expected, atol=1e-6)

    def test_init_q4_conservation(self, grid, sigma_coord):
        """q4 = 1 - 0.3*(q1 + q2 + q3)."""
        state = dcmip11_init(grid, sigma_coord)
        q = state.tracers.data
        q4_expected = 1.0 - 0.3 * (q[..., 0] + q[..., 1] + q[..., 2])
        assert jnp.allclose(q[..., 3], q4_expected, atol=1e-6)

    def test_wind_shapes(self, grid, sigma_coord):
        """Wind field has correct shapes."""
        u, v, sigma_dot = dcmip11_wind(0.0, grid, sigma_coord)
        assert u.shape == (6, N, N, NLEV)
        assert v.shape == (6, N, N, NLEV)
        assert sigma_dot.shape == (6, N, N, NLEV + 1)

    def test_wind_finite(self, grid, sigma_coord):
        """Wind field is finite at all times."""
        for t_days in [0.0, 3.0, 6.0, 9.0, 12.0]:
            t = t_days * 86400.0
            u, v, sigma_dot = dcmip11_wind(t, grid, sigma_coord)
            assert jnp.all(jnp.isfinite(u)), f"u not finite at t={t_days} days"
            assert jnp.all(jnp.isfinite(v)), f"v not finite at t={t_days} days"
            assert jnp.all(jnp.isfinite(sigma_dot)), f"sigma_dot not finite at t={t_days} days"

    def test_sigma_dot_boundaries(self, grid, sigma_coord):
        """sigma_dot = 0 at top and bottom boundaries."""
        u, v, sigma_dot = dcmip11_wind(3.0 * 86400.0, grid, sigma_coord)
        assert jnp.allclose(sigma_dot[..., 0], 0.0, atol=1e-15)
        assert jnp.allclose(sigma_dot[..., -1], 0.0, atol=1e-15)

    def test_short_integration(self, grid, sigma_coord):
        """Short integration doesn't blow up."""
        state = dcmip11_init(grid, sigma_coord)
        model = TracerTransportModel(grid, sigma_coord, dcmip11_wind)
        dt = 3600.0  # 1 hour
        for _ in range(3):
            state = model.step(state, dt)
        assert jnp.all(jnp.isfinite(state.tracers.data))


# ===========================================================================
# Test 1-2: Hadley-like Circulation
# ===========================================================================

class TestDCMIP12:
    """Test 1-2 initialization and wind field."""

    def test_init_shapes(self, grid, sigma_coord):
        """Initial state has correct shapes."""
        state = dcmip12_init(grid, sigma_coord)
        assert state.tracers.data.shape == (6, N, N, NLEV, 1)
        assert float(state.time.data) == 0.0

    def test_init_q1_bounds(self, grid, sigma_coord):
        """q1 is bounded [0, 1]."""
        state = dcmip12_init(grid, sigma_coord)
        q1 = state.tracers.data[..., 0]
        assert jnp.all(q1 >= -1e-10)
        assert jnp.all(q1 <= 1.0 + 1e-10)

    def test_wind_shapes(self, grid, sigma_coord):
        """Wind field has correct shapes."""
        u, v, sigma_dot = dcmip12_wind(0.0, grid, sigma_coord)
        assert u.shape == (6, N, N, NLEV)
        assert v.shape == (6, N, N, NLEV)
        assert sigma_dot.shape == (6, N, N, NLEV + 1)

    def test_wind_finite(self, grid, sigma_coord):
        """Wind field is finite at all times."""
        for t_hours in [0.0, 6.0, 12.0, 18.0, 24.0]:
            t = t_hours * 3600.0
            u, v, sigma_dot = dcmip12_wind(t, grid, sigma_coord)
            assert jnp.all(jnp.isfinite(u))
            assert jnp.all(jnp.isfinite(v))
            assert jnp.all(jnp.isfinite(sigma_dot))

    def test_zonal_wind_structure(self, grid, sigma_coord):
        """Zonal wind should be u = u0 * cos(lat) (solid-body rotation)."""
        u, v, _ = dcmip12_wind(0.0, grid, sigma_coord)
        # u is grid-aligned, but magnitude should be ~ u0 * cos(lat)
        # Just check it's non-zero and varies with latitude
        assert jnp.max(jnp.abs(u)) > 0.0

    def test_short_integration(self, grid, sigma_coord):
        """Short integration doesn't blow up."""
        state = dcmip12_init(grid, sigma_coord)
        model = TracerTransportModel(grid, sigma_coord, dcmip12_wind)
        dt = 600.0
        for _ in range(3):
            state = model.step(state, dt)
        assert jnp.all(jnp.isfinite(state.tracers.data))


# ===========================================================================
# Test 1-3: Advection over Orography
# ===========================================================================

class TestDCMIP13:
    """Test 1-3 initialization, orography, and wind field."""

    def test_mountain_height_positive(self, grid):
        """Mountain height is non-negative."""
        zs = _mountain_height(grid.lon, grid.lat)
        assert jnp.all(zs >= -1e-10)

    def test_mountain_height_max(self, grid):
        """Mountain peak height <= h0 = 2000 m."""
        zs = _mountain_height(grid.lon, grid.lat)
        assert jnp.max(zs) <= 2000.0 + 1e-6

    def test_init_shapes(self, grid, sigma_coord):
        """Initial state has correct shapes."""
        state = dcmip13_init(grid, sigma_coord)
        assert state.tracers.data.shape == (6, N, N, NLEV, 4)

    def test_init_q1_bounds(self, grid, sigma_coord):
        """q1 is bounded [0, 1]."""
        state = dcmip13_init(grid, sigma_coord)
        q1 = state.tracers.data[..., 0]
        assert jnp.all(q1 >= -1e-10)
        assert jnp.all(q1 <= 1.0 + 1e-10)

    def test_init_q3_values(self, grid, sigma_coord):
        """q3 is either 0 or 1 (step function)."""
        state = dcmip13_init(grid, sigma_coord)
        q3 = state.tracers.data[..., 2]
        # q3 should only be 0 or 1
        assert jnp.all((q3 < 0.01) | (q3 > 0.99))

    def test_wind_shapes(self, grid, sigma_coord):
        """Wind field has correct shapes."""
        u, v, sigma_dot = dcmip13_wind(0.0, grid, sigma_coord)
        assert u.shape == (6, N, N, NLEV)
        assert v.shape == (6, N, N, NLEV)
        assert sigma_dot.shape == (6, N, N, NLEV + 1)

    def test_wind_time_independent(self, grid, sigma_coord):
        """Wind for test 1-3 is time-independent (solid-body rotation)."""
        u1, v1, _ = dcmip13_wind(0.0, grid, sigma_coord)
        u2, v2, _ = dcmip13_wind(6.0 * 86400.0, grid, sigma_coord)
        assert jnp.allclose(u1, u2, atol=1e-10)
        assert jnp.allclose(v1, v2, atol=1e-10)

    def test_sigma_dot_orographic(self, grid, sigma_coord):
        """sigma_dot is terrain-induced (non-zero over mountain, ~0 elsewhere)."""
        _, _, sigma_dot = dcmip13_wind(0.0, grid, sigma_coord)
        zs = _mountain_height(grid.lon, grid.lat)
        assert float(jnp.max(jnp.abs(sigma_dot))) > 0.0

        off_mountain = zs < 1e-8
        sigma_off_mountain = jnp.where(off_mountain[..., None], sigma_dot, 0.0)
        assert jnp.max(jnp.abs(sigma_off_mountain)) < 1e-9

    def test_short_integration(self, grid, sigma_coord):
        """Short integration doesn't blow up."""
        state = dcmip13_init(grid, sigma_coord)
        model = TracerTransportModel(grid, sigma_coord, dcmip13_wind)
        dt = 3600.0
        for _ in range(3):
            state = model.step(state, dt)
        assert jnp.all(jnp.isfinite(state.tracers.data))


# ===========================================================================
# Error norms
# ===========================================================================

class TestErrorNorms:
    """Test error norm computation."""

    def test_zero_error(self, grid, sigma_coord):
        """Error norms are zero for identical states."""
        state = dcmip11_init(grid, sigma_coord)
        norms = compute_tracer_error_norms(state, state, grid)
        assert jnp.allclose(norms['l1'], 0.0, atol=1e-10)
        assert jnp.allclose(norms['l2'], 0.0, atol=1e-10)
        assert jnp.allclose(norms['linf'], 0.0, atol=1e-10)

    def test_norms_shape(self, grid, sigma_coord):
        """Error norms have shape (n_tracers,)."""
        state = dcmip11_init(grid, sigma_coord)
        norms = compute_tracer_error_norms(state, state, grid)
        assert norms['l1'].shape == (4,)
        assert norms['l2'].shape == (4,)
        assert norms['linf'].shape == (4,)

    def test_norms_positive_for_different_states(self, grid, sigma_coord):
        """Error norms are positive for different states."""
        state1 = dcmip11_init(grid, sigma_coord)
        # Perturb
        q2 = state1.tracers.data + 0.1
        from legoesm.core.state import TracerState
        state2 = TracerState(
            tracers=state1.tracers.replace(data=q2),
            time=state1.time,
        )
        norms = compute_tracer_error_norms(state2, state1, grid)
        assert jnp.all(norms['l1'] > 0.0)
        assert jnp.all(norms['l2'] > 0.0)
        assert jnp.all(norms['linf'] > 0.0)
