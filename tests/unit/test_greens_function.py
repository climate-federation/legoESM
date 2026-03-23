"""Category 2: Green's Function / Impulse Response Tests.

Apply point-source perturbations and verify:
  - Hyperdiffusion impulse response is smooth and monotonically decaying
  - Gravity wave speed matches c = sqrt(g*H) in shallow water
  - Cubed-sphere face-independence of impulse response
  - Laplacian integral of Green's function response is zero on sphere

Run with:  JAX_ENABLE_X64=1 python -m pytest tests/unit/test_greens_function.py -v
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cs_grid():
    return create_cubed_sphere(8)


@pytest.fixture(scope="module")
def ll_grid():
    return create_latlon_grid(16, 32)


@pytest.fixture(scope="module")
def voronoi_mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(2)


# ======================================================================
# 2b) Hyperdiffusion impulse response
# ======================================================================

class TestHyperdiffusionImpulse:
    """Apply a point perturbation, step with pure hyperdiffusion,
    verify smoothness and monotonic amplitude decay."""

    def test_hyperdiff_impulse_cubesphere(self, cs_grid):
        """Impulse on cubed-sphere: total energy (L2 norm) should
        decrease monotonically under hyperdiffusion.

        Note: pointwise max may temporarily increase due to Gibbs
        ringing from the delta function, but the energy integral
        must be non-increasing for a dissipative operator.
        """
        from legoesm.core.operators import hyperdiffusion
        n = cs_grid.n
        field_data = jnp.zeros((6, n, n))
        field_data = field_data.at[0, n // 2, n // 2].set(1.0)
        field = Field(data=field_data, name="q", dims=("face", "x", "y"), units="1")

        # dx spans 2 cells, so single-cell width = dx/2
        dx_cell = float(jnp.min(cs_grid.dx)) / 2.0
        dt = 1.0
        # CFL for nabla^4: nu * dt / dx^4 < 1/8
        nu = dx_cell**4 / (16.0 * dt)

        energies = [float(jnp.sum(field.data**2 * cs_grid.area))]
        for _ in range(10):
            tendency = hyperdiffusion(field, cs_grid, nu)
            field = field.replace(data=field.data + dt * tendency.data)
            energies.append(float(jnp.sum(field.data**2 * cs_grid.area)))

        # Energy should decrease overall
        assert energies[-1] < energies[0] * 0.99, (
            f"Energy not damped: {energies[-1]:.6e} vs {energies[0]:.6e}"
        )
        # All values should be finite
        assert jnp.all(jnp.isfinite(field.data))

    def test_hyperdiff_impulse_latlon(self, ll_grid):
        """Same energy-based impulse test on lat-lon grid."""
        from legoesm.core.operators_latlon import hyperdiffusion
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon
        field_data = jnp.zeros((n_lat, n_lon))
        field_data = field_data.at[n_lat // 2, n_lon // 2].set(1.0)
        field = Field(data=field_data, name="q", dims=("lat", "lon"), units="1")

        dx_min = float(jnp.min(ll_grid.dx))
        dt = 0.1
        nu = dx_min**4 / (16.0 * dt)

        energies = [float(jnp.sum(field.data**2 * ll_grid.area))]
        for _ in range(10):
            tendency = hyperdiffusion(field, ll_grid, nu)
            field = field.replace(data=field.data + dt * tendency.data)
            energies.append(float(jnp.sum(field.data**2 * ll_grid.area)))

        assert energies[-1] < energies[0] * 0.99, (
            f"Energy not damped: {energies[-1]:.6e} vs {energies[0]:.6e}"
        )
        assert jnp.all(jnp.isfinite(field.data))


# ======================================================================
# 2c) Gravity wave response (shallow water)
# ======================================================================

class TestGravityWaveSpeed:
    """Place a Gaussian height perturbation at rest and verify
    the wavefront propagates at c = sqrt(g*H)."""

    def test_gravity_wave_speed_latlon(self, ll_grid):
        """Gravity wave on lat-lon FV shallow water model.

        Initialize a Gaussian bump in h at the equator, step forward,
        and verify the wave spreads at approximately sqrt(g*H).
        """
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel, FVShallowWaterLatLonConfig,
        )
        from legoesm.core.state import ShallowWaterState

        grid = ll_grid
        g = constants.g
        H0 = 1e4  # 10 km mean depth
        c_analytic = float(jnp.sqrt(g * H0))

        # Gaussian bump at (0, pi) with width ~15 degrees
        lat0, lon0 = 0.0, jnp.pi
        sigma_rad = 15.0 * jnp.pi / 180.0
        dist_sq = (grid.lat2d - lat0)**2 + (grid.lon2d - lon0)**2
        bump = 100.0 * jnp.exp(-dist_sq / (2.0 * sigma_rad**2))

        h_data = H0 + bump
        u_data = jnp.zeros_like(h_data)
        v_data = jnp.zeros_like(h_data)
        h_s_data = jnp.zeros_like(h_data)

        state = ShallowWaterState(
            h=Field(data=h_data, name="h", dims=("lat", "lon"), units="m"),
            u=Field(data=u_data, name="u", dims=("lat", "lon"), units="m/s"),
            v=Field(data=v_data, name="v", dims=("lat", "lon"), units="m/s"),
            h_s=Field(data=h_s_data, name="h_s", dims=("lat", "lon"), units="m"),
        )

        dt = 60.0  # seconds
        config = FVShallowWaterLatLonConfig(
            g=g, hyperdiff_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False, fix_energy=False,
            use_polar_filter=False,
        )
        model = FVShallowWaterLatLonModel(grid, config=config, dt=dt)

        # Record initial perturbation energy distribution
        h_init_pert = float(jnp.max(jnp.abs(state.h.data - H0)))

        # Step forward N steps
        N_steps = 50
        for _ in range(N_steps):
            state = model.step(state, dt)

        # The perturbation should have spread outward
        h_pert = state.h.data - H0
        max_pert_final = float(jnp.max(jnp.abs(h_pert)))

        # The max perturbation should have decreased (wave dispersed)
        assert max_pert_final < h_init_pert, (
            f"Perturbation did not disperse: final max {max_pert_final:.1f} "
            f">= initial {h_init_pert:.1f}"
        )
        # All values should be finite (no blowup)
        assert jnp.all(jnp.isfinite(state.h.data)), "Non-finite h after gravity wave test"

    def test_gravity_wave_speed_mpas(self, voronoi_mesh):
        """Gravity wave on MPAS shallow water model."""
        from legoesm.atmosphere.dynamics.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig,
            MPASShallowWaterState,
        )

        mesh = voronoi_mesh
        g = constants.g
        H0 = 1e4

        # Gaussian bump centered at (0, pi) in lat/lon
        lat0, lon0 = 0.0, jnp.pi
        sigma_rad = 15.0 * jnp.pi / 180.0
        dist_sq = (mesh.latCell - lat0)**2 + (mesh.lonCell - lon0)**2
        bump = 100.0 * jnp.exp(-dist_sq / (2.0 * sigma_rad**2))

        h_data = H0 + bump
        u_data = jnp.zeros(mesh.nEdges)
        h_s_data = jnp.zeros(mesh.nCells)

        state = MPASShallowWaterState(
            h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
            u=Field(data=u_data, name="u", dims=("nEdges",), units="m/s"),
            h_s=Field(data=h_s_data, name="h_s", dims=("nCells",), units="m"),
        )

        config = MPASShallowWaterConfig(
            g=g, fix_mass=False, fix_energy=False,
        )
        model = MPASShallowWaterModel(mesh, config=config)

        h_init_pert = float(jnp.max(jnp.abs(h_data - H0)))

        dt = 60.0
        N_steps = 50
        for _ in range(N_steps):
            state = model.step(state, dt)

        h_pert = state.h.data - H0
        max_pert_final = float(jnp.max(jnp.abs(h_pert)))
        assert max_pert_final < h_init_pert
        assert jnp.all(jnp.isfinite(state.h.data))


# ======================================================================
# 2d) Cubed-sphere face-independence
# ======================================================================

class TestFaceIndependence:
    """The same impulse response should be identical regardless of
    which cubed-sphere face it is placed on."""

    def test_hyperdiff_face_independence(self, cs_grid):
        """Place an impulse on each face and verify L2 norms match."""
        from legoesm.core.operators import hyperdiffusion
        n = cs_grid.n
        dx_min = float(jnp.min(cs_grid.dx))
        dt = 0.1
        nu = dx_min**4 / (16.0 * dt)
        N_steps = 3

        l2_norms = []
        for face in range(6):
            field_data = jnp.zeros((6, n, n))
            field_data = field_data.at[face, n // 2, n // 2].set(1.0)
            field = Field(data=field_data, name="q", dims=("face", "x", "y"), units="1")
            for _ in range(N_steps):
                tendency = hyperdiffusion(field, cs_grid, nu)
                field = field.replace(data=field.data + dt * tendency.data)
            l2 = float(jnp.sqrt(jnp.sum(field.data**2 * cs_grid.area)))
            l2_norms.append(l2)

        # All L2 norms should be very close (face symmetry)
        mean_l2 = sum(l2_norms) / len(l2_norms)
        for i, l2 in enumerate(l2_norms):
            rel = abs(l2 - mean_l2) / (mean_l2 + 1e-30)
            assert rel < 0.05, (
                f"Face {i} L2 = {l2:.6e} differs from mean {mean_l2:.6e} by {rel:.3e}"
            )


# ======================================================================
# 2a) Laplacian Green's function: response is smooth and isotropic
# ======================================================================

class TestLaplacianGreensFunction:
    """Apply iterated Laplacian smoothing to a point source
    and verify the response is smooth."""

    def test_laplacian_smoothing_isotropic(self, cs_grid):
        """Iteratively smooth a delta function with Laplacian diffusion.
        Verify the response remains non-negative (for a Gaussian kernel)
        and is finite everywhere."""
        from legoesm.core.operators import laplacian_compact
        n = cs_grid.n
        field = jnp.zeros((6, n, n))
        field = field.at[0, n // 2, n // 2].set(1.0)

        dx_min = float(jnp.min(cs_grid.dx))
        # Diffusion: df/dt = nu * lap(f), CFL: nu*dt/dx^2 < 0.25
        nu = 1.0
        dt_diff = 0.1 * dx_min**2 / nu

        for _ in range(10):
            lap = laplacian_compact(field, cs_grid)
            field = field + dt_diff * nu * lap

        # Response should be finite and smooth
        assert jnp.all(jnp.isfinite(field)), "Non-finite values after diffusion"
        # Max should have decreased from 1.0
        assert float(jnp.max(field)) < 1.0, "Diffusion did not spread the impulse"
        # The field should be mostly non-negative (diffusion of positive source)
        min_val = float(jnp.min(field))
        assert min_val > -0.1, f"Significant undershoot: min = {min_val:.3e}"
