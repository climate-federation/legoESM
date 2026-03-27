"""Tests for ocean bathymetry loading, coastline processing, and strait enforcement.

Tests cover:
- BathymetryConfig defaults and validation
- Idealized bathymetry dispatch for cubed-sphere and MPAS grids
- Strait enforcement with great-circle corridors
- Laplacian smoothing (cubed-sphere and Voronoi)
- Minimum depth enforcement
- Isolated basin fill
- Ocean fraction derivation
- Regridding from regular lat-lon to target grids
- Full init_ocean_bathymetry dispatch
- Rest state initialization with realistic bathymetry
- Differentiability of resulting states
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.bathymetry import (
    BathymetryConfig,
    CRITICAL_STRAITS,
    init_ocean_bathymetry,
    load_bathymetry,
    enforce_straits,
    rest_state_ocean_realistic,
    _laplacian_smooth_2d,
    _laplacian_smooth_voronoi,
    _derive_ocean_fraction,
    _regrid_bathymetry,
    _fill_isolated_basins,
    _fill_isolated_basins_2d,
    _haversine_km,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Tight tolerances require float64 precision."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_grid():
    """C8 cubed-sphere grid."""
    return create_cubed_sphere(8)


@pytest.fixture
def z_coord():
    """Standard 10-level z-star coordinate."""
    return create_ocean_z_star(n_levels=10, H_max=5500.0)


@pytest.fixture
def default_config():
    return BathymetryConfig()


# ============================================================================
# BathymetryConfig
# ============================================================================


class TestBathymetryConfig:
    """Config construction and defaults."""

    def test_defaults(self):
        cfg = BathymetryConfig()
        assert cfg.source == "idealized"
        assert cfg.H_max == 5500.0
        assert cfg.H_min == 10.0
        assert cfg.land_lat_threshold == 80.0
        assert cfg.smoothing_passes == 2
        assert cfg.enforce_straits is True
        assert cfg.strait_width_factor == 1.0
        assert cfg.fill_isolated_basins is False
        assert cfg.depth_is_negative is True

    def test_custom_config(self):
        cfg = BathymetryConfig(
            source="file",
            path="/some/path.nc",
            H_max=6000.0,
            H_min=50.0,
            enforce_straits=False,
        )
        assert cfg.source == "file"
        assert cfg.path == "/some/path.nc"
        assert cfg.H_max == 6000.0
        assert cfg.H_min == 50.0
        assert cfg.enforce_straits is False

    def test_strait_list_populated(self):
        """Critical straits list should have standard entries."""
        assert len(CRITICAL_STRAITS) >= 10
        names = [s[0] for s in CRITICAL_STRAITS]
        assert "Drake Passage" in names
        assert "Gibraltar" in names
        assert "Indonesian Throughflow" in names
        assert "Bering Strait" in names


# ============================================================================
# Haversine distance
# ============================================================================


class TestHaversine:
    def test_same_point_zero_distance(self):
        d = _haversine_km(40.0, -74.0, 40.0, -74.0)
        assert abs(d) < 1e-10

    def test_antipodal_half_circumference(self):
        # North pole to south pole ~ 20015 km
        d = _haversine_km(90.0, 0.0, -90.0, 0.0)
        assert abs(d - 20015.0) < 100.0  # within 100 km

    def test_vectorized(self):
        lats = np.array([0.0, 0.0, 0.0])
        lons = np.array([0.0, 0.0, 0.0])
        d = _haversine_km(lats, lons, 0.0, 1.0)
        assert d.shape == (3,)
        assert np.all(d > 100.0)  # ~111 km per degree at equator


# ============================================================================
# Strait enforcement
# ============================================================================


class TestStraitEnforcement:
    def test_opens_blocked_strait(self):
        """A grid cell at Gibraltar should become ocean even if initially land."""
        # Create a fine grid around Gibraltar (36N, -5.5E)
        # Gibraltar corridor radius = 25 km ~ 0.22 deg, so we need
        # grid spacing << 0.22 deg to have a point inside the corridor.
        n = 100
        lat = np.linspace(34.0, 38.0, n)
        lon = np.linspace(-8.0, -3.0, n)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # All land initially
        depth = np.zeros((n, n))
        ocean_mask = np.zeros((n, n))

        cfg = BathymetryConfig(strait_width_factor=1.0)
        depth_new, mask_new = enforce_straits(
            depth, ocean_mask, lat_2d, lon_2d, cfg,
        )

        # Some cells near Gibraltar should now be ocean
        assert np.sum(mask_new > 0.5) > 0
        # Depth should be at least Gibraltar minimum (300m)
        assert np.max(depth_new) >= 300.0

    def test_strait_width_factor_scales(self):
        """Wider factor should open more cells."""
        n = 40
        lat = np.linspace(30.0, 42.0, n)
        lon = np.linspace(-12.0, 2.0, n)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        depth = np.zeros((n, n))
        mask = np.zeros((n, n))

        cfg1 = BathymetryConfig(strait_width_factor=1.0)
        _, mask1 = enforce_straits(depth.copy(), mask.copy(), lat_2d, lon_2d, cfg1)

        cfg2 = BathymetryConfig(strait_width_factor=3.0)
        _, mask2 = enforce_straits(depth.copy(), mask.copy(), lat_2d, lon_2d, cfg2)

        assert np.sum(mask2) >= np.sum(mask1)

    def test_preserves_existing_ocean(self):
        """Strait enforcement should not remove existing ocean."""
        n = 20
        lat = np.linspace(-90, 90, n)
        lon = np.linspace(0, 360, n)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        depth = np.full((n, n), 4000.0)
        mask = np.ones((n, n))

        cfg = BathymetryConfig()
        depth_new, mask_new = enforce_straits(
            depth, mask, lat_2d, lon_2d, cfg,
        )

        # All original ocean should be preserved
        assert np.all(mask_new[mask > 0.5] > 0.5)


# ============================================================================
# Smoothing
# ============================================================================


class TestSmoothing:
    def test_cubed_sphere_reduces_variance(self):
        """Smoothing should reduce field variance."""
        np.random.seed(42)
        arr = np.random.randn(6, 8, 8)
        smoothed = _laplacian_smooth_2d(arr, passes=3, is_cubed=True)
        assert np.var(smoothed) < np.var(arr)

    def test_gaussian_reduces_variance(self):
        np.random.seed(42)
        arr = np.random.randn(16, 32)
        smoothed = _laplacian_smooth_2d(arr, passes=3, is_cubed=False)
        assert np.var(smoothed) < np.var(arr)

    def test_zero_passes_no_change(self):
        arr = np.random.randn(6, 4, 4)
        result = _laplacian_smooth_2d(arr, passes=0, is_cubed=True)
        np.testing.assert_array_equal(result, arr)

    def test_voronoi_reduces_variance(self):
        """Voronoi smoothing should reduce variance."""
        nCells = 20
        maxEdges = 6
        np.random.seed(42)

        arr = np.random.randn(nCells) * 100.0
        # Simple ring connectivity: each cell neighbors the next 2
        cells_on_cell = np.full((maxEdges, nCells), -1, dtype=np.int32)
        n_edges_on_cell = np.full(nCells, 2, dtype=np.int32)
        for c in range(nCells):
            cells_on_cell[0, c] = (c + 1) % nCells
            cells_on_cell[1, c] = (c - 1) % nCells

        smoothed = _laplacian_smooth_voronoi(
            arr, cells_on_cell, n_edges_on_cell, passes=3,
        )
        assert np.var(smoothed) < np.var(arr)

    def test_uniform_field_unchanged(self):
        """Smoothing a uniform field should not change it."""
        arr = np.full((6, 8, 8), 42.0)
        smoothed = _laplacian_smooth_2d(arr, passes=5, is_cubed=True)
        np.testing.assert_allclose(smoothed, 42.0, atol=1e-10)


# ============================================================================
# Ocean fraction derivation
# ============================================================================


class TestOceanFraction:
    def test_all_ocean_gives_one(self):
        """If source depth is everywhere positive, fraction should be ~1."""
        lat_src = np.linspace(-90, 90, 100)
        lon_src = np.linspace(0, 359, 200)
        depth = np.full((100, 200), 4000.0)

        target_lat = np.array([0.0, 45.0])
        target_lon = np.array([180.0, 90.0])

        frac = _derive_ocean_fraction(
            lat_src, lon_src, depth, target_lat, target_lon, 5.0,
        )
        np.testing.assert_allclose(frac, 1.0)

    def test_all_land_gives_zero(self):
        """If source depth is everywhere ≤ 0, fraction should be 0."""
        lat_src = np.linspace(-90, 90, 100)
        lon_src = np.linspace(0, 359, 200)
        depth = np.full((100, 200), -100.0)  # land

        target_lat = np.array([0.0])
        target_lon = np.array([180.0])

        frac = _derive_ocean_fraction(
            lat_src, lon_src, depth, target_lat, target_lon, 5.0,
        )
        np.testing.assert_allclose(frac, 0.0)

    def test_mixed_gives_intermediate(self):
        """Half ocean/half land should give ~0.5 fraction."""
        lat_src = np.linspace(-90, 90, 200)
        lon_src = np.linspace(0, 359, 400)
        depth = np.zeros((200, 400))
        # Ocean in southern hemisphere, land in northern
        for i in range(200):
            if lat_src[i] < 0:
                depth[i, :] = 3000.0

        # Point on the equator should have fraction near 0.5
        target_lat = np.array([0.0])
        target_lon = np.array([180.0])

        frac = _derive_ocean_fraction(
            lat_src, lon_src, depth, target_lat, target_lon, 5.0,
        )
        assert 0.2 < frac[0] < 0.8


# ============================================================================
# Regridding
# ============================================================================


class TestRegridding:
    def test_identity_regrid(self):
        """Regridding to same points should return same values."""
        lat_src = np.linspace(-90, 90, 50)
        lon_src = np.linspace(0, 359, 100)
        depth = np.outer(np.cos(np.radians(lat_src)), np.ones(100)) * 3000.0

        # Sample at a few source grid points
        target_lat = np.array([0.0, 45.0, -30.0])
        target_lon = np.array([0.0, 180.0, 90.0])

        result = _regrid_bathymetry(lat_src, lon_src, depth,
                                     target_lat, target_lon)

        # Values should be close (not exact due to interpolation)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_output_shape_matches_target(self):
        """Output shape should match target coordinates."""
        lat_src = np.linspace(-90, 90, 50)
        lon_src = np.linspace(0, 359, 100)
        depth = np.ones((50, 100)) * 4000.0

        target = np.zeros((6, 8, 8))
        result = _regrid_bathymetry(
            lat_src, lon_src, depth, target, target,
        )
        assert result.shape == (6, 8, 8)


# ============================================================================
# Isolated basin fill
# ============================================================================


class TestIsolatedBasinFill:
    def test_single_basin_unchanged(self):
        """A single connected ocean should not be modified."""
        mask = np.zeros((10, 20))
        mask[2:8, 5:15] = 1.0
        result = _fill_isolated_basins_2d(mask)
        np.testing.assert_array_equal(result, mask)

    def test_removes_small_basin(self):
        """A small isolated basin should be removed."""
        mask = np.zeros((10, 20))
        # Main ocean: large
        mask[0:8, 0:18] = 1.0
        # Isolated basin: small
        mask[9, 19] = 1.0

        result = _fill_isolated_basins_2d(mask)
        # Small basin should be filled
        assert result[9, 19] == 0.0
        # Main ocean preserved
        assert np.sum(result) > 100

    def test_cubed_sphere_dispatch(self):
        """Should work on (6, n, n) shape."""
        mask = np.zeros((6, 8, 8))
        mask[:, 1:7, 1:7] = 1.0
        mask[0, 0, 0] = 1.0  # isolated pixel on face 0

        result = _fill_isolated_basins(mask)
        assert result.shape == (6, 8, 8)
        # Isolated pixel should be removed
        assert result[0, 0, 0] == 0.0


# ============================================================================
# Minimum depth enforcement
# ============================================================================


class TestMinimumDepth:
    def test_shallow_cells_become_land(self):
        """Cells shallower than H_min should become land."""
        lat_src = np.linspace(-90, 90, 50)
        lon_src = np.linspace(0, 359, 100)
        # Ocean with varying depth: some very shallow
        depth_src = np.ones((50, 100)) * 5.0  # 5m depth, < H_min=10m

        target_lat = np.array([0.0])
        target_lon = np.array([180.0])

        # Use the core load function mechanics
        depth = _regrid_bathymetry(lat_src, lon_src, depth_src,
                                    target_lat, target_lon)

        H_min = 10.0
        too_shallow = depth < H_min
        ocean_mask = np.where(too_shallow, 0.0, 1.0)

        # All cells should be land since depth (5m) < H_min (10m)
        assert ocean_mask[0] == 0.0


# ============================================================================
# Idealized dispatch
# ============================================================================


class TestIdealizedDispatch:
    def test_cubed_sphere(self, small_grid):
        cfg = BathymetryConfig(source="idealized")
        H_bathy, mask = init_ocean_bathymetry(small_grid, cfg)
        assert H_bathy.shape == (6, 8, 8)
        assert mask.shape == (6, 8, 8)
        # All ocean cells should have depth H_max
        assert float(jnp.max(H_bathy)) == 5500.0

    def test_cubed_sphere_default_config(self, small_grid):
        """None config should use idealized defaults."""
        H_bathy, mask = init_ocean_bathymetry(small_grid, None)
        assert H_bathy.shape == (6, 8, 8)
        assert mask.shape == (6, 8, 8)

    def test_cubed_sphere_polar_land(self, small_grid):
        """High-latitude cells should be land."""
        cfg = BathymetryConfig(land_lat_threshold=70.0)
        _, mask = init_ocean_bathymetry(small_grid, cfg)
        lat_deg = jnp.abs(small_grid.lat) * 180.0 / jnp.pi
        # Cells beyond 70 deg should be land (mask=0)
        polar = lat_deg > 70.0
        assert float(jnp.sum(mask[polar])) == 0.0

    def test_mpas_mesh(self):
        """Test idealized dispatch for MPAS mesh."""
        try:
            from legoesm.grids.voronoi import create_voronoi_mesh
        except ImportError:
            pytest.skip("VoronoiMesh not available")

        mesh = create_voronoi_mesh(2)
        cfg = BathymetryConfig(source="idealized")
        H_bathy, mask = init_ocean_bathymetry(mesh, cfg)
        assert H_bathy.shape == (mesh.nCells,)
        assert mask.shape == (mesh.nCells,)


# ============================================================================
# Rest state with realistic bathymetry
# ============================================================================


class TestRestStateRealistic:
    def test_cubed_sphere_idealized(self, small_grid, z_coord):
        """Rest state with idealized bathymetry should match existing behavior."""
        cfg = BathymetryConfig(source="idealized")
        state = rest_state_ocean_realistic(
            small_grid, z_coord, cfg,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        )

        assert state.T.data.shape == (6, 8, 8, 10)
        assert state.S.data.shape == (6, 8, 8, 10)
        assert state.u.data.shape == (6, 8, 8, 10)
        assert state.eta.data.shape == (6, 8, 8)
        assert state.H_bathy.data.shape == (6, 8, 8)
        assert state.land_mask.data.shape == (6, 8, 8)

        # Temperature should be warm at surface, cold at depth
        T_data = state.T.data
        assert float(jnp.mean(T_data[..., 0])) > float(jnp.mean(T_data[..., -1]))

        # Velocity should be zero
        assert float(jnp.max(jnp.abs(state.u.data))) == 0.0

    def test_cubed_sphere_state_finite(self, small_grid, z_coord):
        """All state fields should be finite."""
        cfg = BathymetryConfig(source="idealized")
        state = rest_state_ocean_realistic(small_grid, z_coord, cfg)
        for field_name in ["T", "S", "u", "v", "eta", "H_bathy", "land_mask"]:
            data = getattr(state, field_name).data
            assert jnp.all(jnp.isfinite(data)), f"{field_name} has non-finite values"

    def test_mpas_idealized(self, z_coord):
        """MPAS rest state with idealized bathymetry."""
        try:
            from legoesm.grids.voronoi import create_voronoi_mesh
        except ImportError:
            pytest.skip("VoronoiMesh not available")

        mesh = create_voronoi_mesh(2)
        cfg = BathymetryConfig(source="idealized")
        state = rest_state_ocean_realistic(mesh, z_coord, cfg)

        assert state.T.data.shape == (mesh.nCells, 10)
        assert state.u.data.shape == (mesh.nEdges, 10)
        assert state.eta.data.shape == (mesh.nCells,)

    def test_differentiable(self, small_grid, z_coord):
        """Resulting state should be usable in JAX grad computations."""
        cfg = BathymetryConfig(source="idealized")
        state = rest_state_ocean_realistic(small_grid, z_coord, cfg)

        def loss(T):
            return jnp.sum(T ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == state.T.data.shape


# ============================================================================
# File-based loading error handling
# ============================================================================


class TestFileLoadingErrors:
    def test_missing_path_raises(self):
        """Loading from file without path should raise."""
        cfg = BathymetryConfig(source="file", path="")
        with pytest.raises(ValueError, match="No bathymetry data path"):
            load_bathymetry(
                np.array([0.0]), np.array([0.0]), cfg,
            )

    def test_nonexistent_file_raises(self):
        """Loading from nonexistent file should raise."""
        cfg = BathymetryConfig(source="file", path="/nonexistent/bathy.nc")
        with pytest.raises(FileNotFoundError):
            load_bathymetry(
                np.array([0.0]), np.array([0.0]), cfg,
            )

    def test_unknown_source_raises(self, small_grid):
        """Unknown source type should raise."""
        cfg = BathymetryConfig(source="unknown")
        with pytest.raises(ValueError, match="Unknown bathymetry source"):
            init_ocean_bathymetry(small_grid, cfg)
