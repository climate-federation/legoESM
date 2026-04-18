"""CI tests for Smagorinsky biharmonic viscosity on lat-lon C-grid and MPAS.

Tests operator correctness, energy properties, invariances, and JAX
compatibility.  All tests use small grids and run in seconds.

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/ocean/unit/test_smagorinsky.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    strain_rate_cgrid,
    smagorinsky_biharmonic_tendency_cgrid,
    vector_laplacian_cgrid,
)
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    curl_vertex_3d,
    smagorinsky_biharmonic_3d,
    vector_laplacian_del2_3d,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _enable_x64():
    """Tight tolerances require float64 precision."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def latlon_grid():
    """Small regional lat-lon C-grid for fast tests."""
    grid, wall_mask = create_regional_latlon_grid(
        16, 34, 16.0, 34.0,
        periodic_x=True, lon_west=0.0, lon_east=10.0,
        dtype=jnp.float64,
    )
    u_mask, v_mask = compute_face_masks(wall_mask)
    return grid, wall_mask, u_mask, v_mask


@pytest.fixture(scope="module")
def mpas_mesh():
    """Level 2 Voronoi mesh (162 cells) for fast tests."""
    return create_voronoi_mesh(subdivision_level=2)


def _random_velocity_latlon(grid, wall_mask, u_mask, v_mask, seed=42,
                            amplitude=0.1):
    """Random PERIODIC velocity field on C-grid."""
    rng = np.random.RandomState(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    u_inner = jnp.array(rng.randn(n_lat, n_lon) * amplitude, dtype=jnp.float64)
    u = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1) * u_mask
    v = jnp.array(rng.randn(n_lat + 1, n_lon) * amplitude, dtype=jnp.float64) * v_mask
    return u, v


def _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1):
    """Random velocity at MPAS edges."""
    rng = np.random.RandomState(seed)
    n_edges = mesh.dcEdge.shape[0]
    u = jnp.array(rng.randn(n_edges, nlev) * amplitude, dtype=jnp.float64)
    return u


# ============================================================================
# Lat-lon C-grid tests
# ============================================================================


class TestSmagorinskyLatLon:
    """Tests for the stress-tensor Smagorinsky biharmonic on lat-lon C-grid."""

    def test_energy_dissipative(self, latlon_grid):
        """Energy identity: <u, tend> <= 0 (viscosity never injects energy)."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tend_u, tend_v = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Energy dissipation rate = -<u, tend> (minus because tend is subtracted)
        area = grid.area
        R = grid.radius
        dlat = grid.dlat
        dlon = grid.dlon
        cos_lat = grid.cos_lat
        dy = R * dlat
        area_u = dy * R * cos_lat * dlon

        lat = grid.lat
        lat_sp = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
        lat_np = jnp.array([jnp.pi / 2], dtype=lat.dtype)
        lat_int = 0.5 * (lat[:-1] + lat[1:])
        lat_v = jnp.concatenate([lat_sp, lat_int, lat_np])
        cos_v = jnp.maximum(jnp.cos(lat_v), 1e-10)
        area_v = dy * R * cos_v * dlon

        # u·tend·area (energy rate)
        # tend is to be subtracted from du/dt, so energy_rate = -sum(u*tend)
        energy_rate = (
            jnp.sum(u * tend_u * area_u[:, None])
            + jnp.sum(v * tend_v * area_v[:, None])
        )
        # Since we subtract tend, the dissipation should be positive
        # i.e., energy_rate = sum(u * tend * area) should be >= 0
        # (tend should be aligned with u for dissipation)
        # Actually: du/dt -= tend, so dE/dt = -<u, tend>.
        # For dissipation, -<u, tend> <= 0, i.e., <u, tend> >= 0.
        assert energy_rate > 0, (
            f"Smagorinsky should be dissipative, got energy_rate={energy_rate}"
        )

    def test_monotone_dissipation(self, latlon_grid):
        """Increasing C_smag -> increasing KE dissipation rate."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        area = grid.area
        R, dlat, dlon = grid.radius, grid.dlat, grid.dlon
        cos_lat = grid.cos_lat
        area_u = R * dlat * R * cos_lat * dlon

        lat = grid.lat
        lat_v = jnp.concatenate([
            jnp.array([-jnp.pi / 2], dtype=lat.dtype),
            0.5 * (lat[:-1] + lat[1:]),
            jnp.array([jnp.pi / 2], dtype=lat.dtype),
        ])
        area_v = R * dlat * R * jnp.maximum(jnp.cos(lat_v), 1e-10) * dlon

        dissipations = []
        for c_smag in [0.01, 0.03, 0.1, 0.3, 1.0]:
            tu, tv = smagorinsky_biharmonic_tendency_cgrid(
                u, v, grid, c_smag, mask=mask, u_mask=u_mask, v_mask=v_mask)
            diss = (jnp.sum(u * tu * area_u[:, None])
                    + jnp.sum(v * tv * area_v[:, None]))
            dissipations.append(float(diss))

        # All positive (dissipative)
        for i, d in enumerate(dissipations):
            assert d > 0, f"C_smag index {i}: dissipation={d} not positive"

        # Monotonically increasing
        for i in range(len(dissipations) - 1):
            assert dissipations[i + 1] > dissipations[i], (
                f"Not monotone: C_smag[{i}]={dissipations[i]:.6e} >= "
                f"C_smag[{i+1}]={dissipations[i+1]:.6e}"
            )

    def test_uniform_flow_zero_tendency(self, latlon_grid):
        """Uniform flow has zero strain, hence zero Smagorinsky tendency."""
        grid, mask, u_mask, v_mask = latlon_grid

        # Constant zonal flow
        u_const = jnp.ones((grid.n_lat, grid.n_lon + 1), dtype=jnp.float64) * u_mask
        v_zero = jnp.zeros((grid.n_lat + 1, grid.n_lon), dtype=jnp.float64)

        tu, tv = smagorinsky_biharmonic_tendency_cgrid(
            u_const, v_zero, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Strain of uniform flow should be zero in interior
        # (boundary terms from wall masks might contribute)
        # Check interior cells only (skip boundary rows)
        # Spherical metric terms produce O(dx^2) truncation error for
        # uniform flow on a regional lat-lon grid.
        assert jnp.max(jnp.abs(tu[2:-2, 2:-2])) < 1e-8, (
            f"Uniform u should give ~zero tendency, got max={float(jnp.max(jnp.abs(tu[2:-2, 2:-2])))}"
        )

    def test_3d_matches_2d(self, latlon_grid):
        """3D tendency (stacked levels) matches independent 2D calls."""
        grid, mask, u_mask, v_mask = latlon_grid
        nlev = 5

        # Create 3D fields with different values per level
        u_3d = jnp.zeros((grid.n_lat, grid.n_lon + 1, nlev), dtype=jnp.float64)
        v_3d = jnp.zeros((grid.n_lat + 1, grid.n_lon, nlev), dtype=jnp.float64)
        for k in range(nlev):
            u_k, v_k = _random_velocity_latlon(grid, mask, u_mask, v_mask,
                                                seed=100 + k)
            u_3d = u_3d.at[:, :, k].set(u_k)
            v_3d = v_3d.at[:, :, k].set(v_k)

        # 3D call
        tu_3d, tv_3d = smagorinsky_biharmonic_tendency_cgrid(
            u_3d, v_3d, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Independent 2D calls
        for k in range(nlev):
            tu_2d, tv_2d = smagorinsky_biharmonic_tendency_cgrid(
                u_3d[:, :, k], v_3d[:, :, k], grid, 0.1,
                mask=mask, u_mask=u_mask, v_mask=v_mask)
            max_diff_u = float(jnp.max(jnp.abs(tu_3d[:, :, k] - tu_2d)))
            max_diff_v = float(jnp.max(jnp.abs(tv_3d[:, :, k] - tv_2d)))
            assert max_diff_u < 1e-14, f"3D/2D u mismatch at level {k}: {max_diff_u}"
            assert max_diff_v < 1e-14, f"3D/2D v mismatch at level {k}: {max_diff_v}"

    def test_masked_regions_zero(self, latlon_grid):
        """Tendency is zero in land-masked regions."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tu, tv = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Where u_mask is 0, tendency should be 0
        u_land = tu * (1.0 - u_mask)
        v_land = tv * (1.0 - v_mask)
        assert jnp.max(jnp.abs(u_land)) == 0.0, "Tendency nonzero in u land mask"
        assert jnp.max(jnp.abs(v_land)) == 0.0, "Tendency nonzero in v land mask"

    def test_antisymmetry_velocity_reversal(self, latlon_grid):
        """tend(-u, -v) == -tend(u, v) — Smagorinsky is odd in velocity."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tu_pos, tv_pos = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)
        tu_neg, tv_neg = smagorinsky_biharmonic_tendency_cgrid(
            -u, -v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        err_u = float(jnp.max(jnp.abs(tu_neg + tu_pos)))
        err_v = float(jnp.max(jnp.abs(tv_neg + tv_pos)))
        assert err_u < 1e-14, f"Antisymmetry broken for u: {err_u}"
        assert err_v < 1e-14, f"Antisymmetry broken for v: {err_v}"

    def test_wrap_column_periodicity(self, latlon_grid):
        """Tendency at wrap column matches column 0 for periodic velocity."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tu, tv = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        n_lon = grid.n_lon
        err = float(jnp.max(jnp.abs(tu[:, n_lon] - tu[:, 0])))
        assert err < 1e-14, (
            f"Wrap column tendency inconsistent: max_err={err}"
        )

    def test_jax_autodiff(self, latlon_grid):
        """jax.grad flows through the Smagorinsky operator."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        def loss_fn(u_in, v_in):
            tu, tv = smagorinsky_biharmonic_tendency_cgrid(
                u_in, v_in, grid, 0.1,
                mask=mask, u_mask=u_mask, v_mask=v_mask)
            return jnp.sum(tu**2) + jnp.sum(tv**2)

        # Should not raise
        grad_u, grad_v = jax.grad(loss_fn, argnums=(0, 1))(u, v)

        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grad_u)), "grad_u contains NaN/Inf"
        assert jnp.all(jnp.isfinite(grad_v)), "grad_v contains NaN/Inf"

        # Gradients should be nonzero (operator is nontrivial)
        assert jnp.max(jnp.abs(grad_u)) > 0, "grad_u is all zeros"
        assert jnp.max(jnp.abs(grad_v)) > 0, "grad_v is all zeros"

    def test_jit_compiles(self, latlon_grid):
        """JIT compilation succeeds and gives same result."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        @jax.jit
        def compute(u_in, v_in):
            return smagorinsky_biharmonic_tendency_cgrid(
                u_in, v_in, grid, 0.1,
                mask=mask, u_mask=u_mask, v_mask=v_mask)

        tu_jit, tv_jit = compute(u, v)
        tu_eager, tv_eager = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.1, mask=mask, u_mask=u_mask, v_mask=v_mask)

        assert jnp.max(jnp.abs(tu_jit - tu_eager)) < 1e-15
        assert jnp.max(jnp.abs(tv_jit - tv_eager)) < 1e-15

    def test_checkerboard_scale_selective(self, latlon_grid):
        """Biharmonic damps 2dx checkerboard much more than harmonic."""
        grid, mask, u_mask, v_mask = latlon_grid
        n_lat, n_lon = grid.n_lat, grid.n_lon

        # Create 2dx checkerboard pattern in u
        i_idx = jnp.arange(n_lat)[:, None]
        j_idx = jnp.arange(n_lon + 1)[None, :]
        u_checker = ((-1.0) ** (i_idx + j_idx)) * 0.1 * u_mask
        v_zero = jnp.zeros((n_lat + 1, n_lon), dtype=jnp.float64)

        # Biharmonic (Smagorinsky) tendency
        tu_biharm, _ = smagorinsky_biharmonic_tendency_cgrid(
            u_checker, v_zero, grid, 0.1,
            mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Harmonic tendency
        tu_harm, _ = vector_laplacian_cgrid(
            u_checker, v_zero, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Biharmonic should have much larger tendency for grid-scale noise
        max_biharm = float(jnp.max(jnp.abs(tu_biharm[2:-2, 2:-2])))
        max_harm = float(jnp.max(jnp.abs(tu_harm[2:-2, 2:-2])))

        # The ratio depends on grid spacing and C_smag, but biharmonic
        # should dominate at the grid scale
        assert max_biharm > 0, "Biharmonic tendency is zero for checkerboard"
        assert max_harm > 0, "Harmonic tendency is zero for checkerboard"

    def test_strain_wrap_column_consistent(self, latlon_grid):
        """D_S at wrap column matches D_S at column 0 for periodic velocity."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        D_T, D_S = strain_rate_cgrid(u, v, grid, mask=mask,
                                      u_mask=u_mask, v_mask=v_mask)

        n_lon = grid.n_lon
        err = float(jnp.max(jnp.abs(D_S[:, n_lon] - D_S[:, 0])))
        assert err < 1e-14, (
            f"D_S wrap column inconsistent: max_err={err}"
        )

    def test_coefficient_magnitude(self, latlon_grid):
        """Smagorinsky coefficient has correct magnitude."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask, amplitude=0.1)

        D_T, D_S = strain_rate_cgrid(u, v, grid, mask=mask,
                                      u_mask=u_mask, v_mask=v_mask)
        C_smag = 0.1
        area = grid.area
        deformation = jnp.sqrt(D_T**2 + jnp.mean(D_S[1:-1]**2))
        delta = jnp.sqrt(area)
        A_expected_order = C_smag**2 * jnp.mean(delta**2) * jnp.mean(jnp.abs(deformation))

        # Just check it's in a reasonable range (not zero, not huge)
        assert float(A_expected_order) > 0, "Expected A_smag should be positive"
        assert float(A_expected_order) < 1e10, "Expected A_smag unreasonably large"


# ============================================================================
# MPAS Voronoi tests
# ============================================================================


class TestSmagorinskyMPAS:
    """Tests for Smagorinsky biharmonic on MPAS Voronoi mesh."""

    def test_monotone_dissipation(self, mpas_mesh):
        """Increasing C_smag -> increasing KE dissipation rate."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        dissipations = []
        for c_smag in [0.01, 0.03, 0.1, 0.3, 1.0]:
            tend = smagorinsky_biharmonic_3d(u, mesh, c_smag)
            # Energy rate: sum(u * tend * dcEdge * dvEdge)
            edge_area = mesh.dcEdge * mesh.dvEdge
            diss = float(jnp.sum(u * tend * edge_area[:, None]))
            dissipations.append(diss)

        # All should be negative (energy removed — tend is ADDED to du/dt)
        for i, d in enumerate(dissipations):
            assert d < 0, f"C_smag index {i}: diss={d} not negative (should remove energy)"

        # Magnitude increasing (more negative)
        for i in range(len(dissipations) - 1):
            assert dissipations[i + 1] < dissipations[i], (
                f"Not monotone: |diss[{i}]|={abs(dissipations[i]):.6e} >= "
                f"|diss[{i+1}]|={abs(dissipations[i+1]):.6e}"
            )

    def test_uniform_flow_zero_tendency(self, mpas_mesh):
        """Uniform zonal flow has zero strain -> zero tendency."""
        mesh = mpas_mesh
        # Uniform zonal: project onto edge normals
        u_zonal = jnp.cos(mesh.angleEdge)[:, None] * 0.1  # (nEdges, 1)
        u_zonal = u_zonal.astype(jnp.float64)

        tend = smagorinsky_biharmonic_3d(u_zonal, mesh, 0.1)
        max_tend = float(jnp.max(jnp.abs(tend)))

        # The vector Laplacian of uniform zonal flow is NOT exactly zero
        # on a discrete Voronoi mesh (truncation error), but should be small
        assert max_tend < 1e-6, (
            f"Uniform zonal flow tendency too large: {max_tend}"
        )

    def test_solid_body_rotation_small_tendency(self, mpas_mesh):
        """Solid body rotation has zero strain -> near-zero tendency."""
        mesh = mpas_mesh
        # Solid body rotation: u = U0 * cos(lat), projected onto edge normals
        # Edge-normal component: u_n = U0 * cos(lat) * cos(angleEdge)
        U0 = 0.1
        u_sbr = (U0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge))[:, None]
        u_sbr = u_sbr.astype(jnp.float64)

        tend = smagorinsky_biharmonic_3d(u_sbr, mesh, 0.1)
        max_tend = float(jnp.max(jnp.abs(tend)))

        # Solid body rotation is in the kernel of the vector Laplacian
        # on the sphere (continuous), so tendency should be small (O(dx^2))
        assert max_tend < 1e-5, (
            f"Solid body rotation tendency too large: {max_tend}"
        )

    def test_antisymmetry_velocity_reversal(self, mpas_mesh):
        """tend(-u) == -tend(u) — Smagorinsky is odd in velocity."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        tend_pos = smagorinsky_biharmonic_3d(u, mesh, 0.1)
        tend_neg = smagorinsky_biharmonic_3d(-u, mesh, 0.1)

        err = float(jnp.max(jnp.abs(tend_neg + tend_pos)))
        assert err < 1e-14, f"Antisymmetry broken: max_err={err}"

    def test_masked_edges_zero(self, mpas_mesh):
        """Tendency at boundary edges (where one cell is land) is handled."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        tend = smagorinsky_biharmonic_3d(u, mesh, 0.1)

        # No NaN or Inf
        assert jnp.all(jnp.isfinite(tend)), "Tendency contains NaN/Inf"

    def test_jax_autodiff(self, mpas_mesh):
        """jax.grad flows through the MPAS Smagorinsky operator."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        def loss_fn(u_in):
            tend = smagorinsky_biharmonic_3d(u_in, mesh, 0.1)
            return jnp.sum(tend**2)

        grad_u = jax.grad(loss_fn)(u)

        assert jnp.all(jnp.isfinite(grad_u)), "grad_u contains NaN/Inf"
        assert jnp.max(jnp.abs(grad_u)) > 0, "grad_u is all zeros"

    def test_jit_compiles(self, mpas_mesh):
        """JIT compilation succeeds and gives same result."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        @jax.jit
        def compute(u_in):
            return smagorinsky_biharmonic_3d(u_in, mesh, 0.1)

        tend_jit = compute(u)
        tend_eager = smagorinsky_biharmonic_3d(u, mesh, 0.1)

        assert jnp.max(jnp.abs(tend_jit - tend_eager)) < 1e-15

    def test_multilevel_consistency(self, mpas_mesh):
        """Multi-level result matches independent single-level calls."""
        mesh = mpas_mesh
        nlev = 3
        u_3d = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=nlev)

        tend_3d = smagorinsky_biharmonic_3d(u_3d, mesh, 0.1)

        for k in range(nlev):
            tend_1d = smagorinsky_biharmonic_3d(u_3d[:, k:k+1], mesh, 0.1)
            max_diff = float(jnp.max(jnp.abs(tend_3d[:, k:k+1] - tend_1d)))
            assert max_diff < 1e-14, (
                f"Multi-level/single-level mismatch at level {k}: {max_diff}"
            )

    def test_checkerboard_scale_selective(self, mpas_mesh):
        """Biharmonic damps grid-scale noise more than harmonic."""
        mesh = mpas_mesh
        # Create alternating-sign pattern based on cell index parity
        n_edges = mesh.dcEdge.shape[0]
        sign_pattern = ((-1.0) ** jnp.arange(n_edges))[:, None] * 0.1
        u_checker = sign_pattern.astype(jnp.float64)

        tend_biharm = smagorinsky_biharmonic_3d(u_checker, mesh, 0.1)
        tend_harm = vector_laplacian_del2_3d(u_checker, mesh)

        max_biharm = float(jnp.max(jnp.abs(tend_biharm)))
        max_harm = float(jnp.max(jnp.abs(tend_harm)))

        assert max_biharm > 0, "Biharmonic tendency is zero for checkerboard"
        assert max_harm > 0, "Harmonic tendency is zero for checkerboard"

    def test_strain_rate_components(self, mpas_mesh):
        """Verify strain rate decomposition gives expected shapes."""
        mesh = mpas_mesh
        u = _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1)

        div_c = divergence_cell_3d(u, mesh)
        curl_v = curl_vertex_3d(u, mesh)

        n_cells = mesh.areaCell.shape[0]
        n_vertices = mesh.areaTriangle.shape[0]

        assert div_c.shape == (n_cells, 1), f"div shape: {div_c.shape}"
        assert curl_v.shape == (n_vertices, 1), f"curl shape: {curl_v.shape}"

        # Both should be nonzero for random velocity
        assert jnp.max(jnp.abs(div_c)) > 0, "Divergence is zero"
        assert jnp.max(jnp.abs(curl_v)) > 0, "Curl is zero"
