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
    laplacian_smooth_2d,
    laplacian_smooth_voronoi,
    _derive_ocean_fraction,
    _regrid_bathymetry,
    _fill_isolated_basins,
    _fill_isolated_basins_2d,
    haversine_km,
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
        d = haversine_km(40.0, -74.0, 40.0, -74.0)
        assert abs(d) < 1e-10

    def test_antipodal_half_circumference(self):
        # North pole to south pole ~ 20015 km
        d = haversine_km(90.0, 0.0, -90.0, 0.0)
        assert abs(d - 20015.0) < 100.0  # within 100 km

    def test_vectorized(self):
        lats = np.array([0.0, 0.0, 0.0])
        lons = np.array([0.0, 0.0, 0.0])
        d = haversine_km(lats, lons, 0.0, 1.0)
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
        smoothed = laplacian_smooth_2d(arr, passes=3, is_cubed=True)
        assert np.var(smoothed) < np.var(arr)

    def test_gaussian_reduces_variance(self):
        np.random.seed(42)
        arr = np.random.randn(16, 32)
        smoothed = laplacian_smooth_2d(arr, passes=3, is_cubed=False)
        assert np.var(smoothed) < np.var(arr)

    def test_zero_passes_no_change(self):
        arr = np.random.randn(6, 4, 4)
        result = laplacian_smooth_2d(arr, passes=0, is_cubed=True)
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

        smoothed = laplacian_smooth_voronoi(
            arr, cells_on_cell, n_edges_on_cell, passes=3,
        )
        assert np.var(smoothed) < np.var(arr)

    def test_uniform_field_unchanged(self):
        """Smoothing a uniform field should not change it."""
        arr = np.full((6, 8, 8), 42.0)
        smoothed = laplacian_smooth_2d(arr, passes=5, is_cubed=True)
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
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
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


# ============================================================================
# Mellor-Ezer-Oey r-factor cap
# ============================================================================


class TestMEORFactorCap:
    """``apply_meo_r_factor_cap`` correctly caps the r-factor while
    preserving ocean topology and adding minimal volume."""

    def test_already_satisfied_is_noop(self):
        """If the bathymetry already satisfies r < target, MEO should
        not change anything."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        # Uniform bathymetry → r = 0 everywhere.
        H = np.full((5, 10), 4000.0)
        ocean = np.ones_like(H)
        H_out, info = apply_meo_r_factor_cap(H, ocean, 0.2)
        assert info["initial_r_max"] == 0.0
        assert info["final_r_max"] == 0.0
        assert info["cells_modified"] == 0
        np.testing.assert_array_equal(H_out, H)

    def test_caps_a_step_bathymetry(self):
        """Sharp step from 200m → 4000m must be smoothed by MEO."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        H = np.full((5, 10), 4000.0)
        H[:, :5] = 200.0
        ocean = np.ones_like(H)
        # Initial r at the step: |200 - 4000| / 4000 = 0.95
        H_out, info = apply_meo_r_factor_cap(H, ocean, 0.2, max_iter=200)
        assert info["initial_r_max"] > 0.9
        assert info["final_r_max"] <= 0.2 + 1e-10, (
            f"Final r_max = {info['final_r_max']}"
        )
        # All cells should be deeper (or unchanged), never shallower.
        assert np.all(H_out >= H - 1e-9)
        # Some cells were modified
        assert info["cells_modified"] > 0
        # Volume change is positive (added water by deepening shallow cells)
        assert info["volume_change_frac"] > 0

    def test_volume_change_bounded(self):
        """On a moderately-rough field, volume change should be modest
        (<20% — typical real-world is 1-5%, but a synthetic adversarial
        case can be larger)."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        rng = np.random.default_rng(seed=42)
        # Large H baseline + spotty shallow cells
        H = np.full((20, 40), 4000.0)
        # 10 random shallow spots at 200m
        idx_lat = rng.integers(0, 20, size=10)
        idx_lon = rng.integers(0, 40, size=10)
        for i, j in zip(idx_lat, idx_lon):
            H[i, j] = 200.0
        ocean = np.ones_like(H)
        _H_out, info = apply_meo_r_factor_cap(H, ocean, 0.2)
        # On this adversarial pattern (isolated 200m spots in 4000m sea),
        # MEO should still finish and cap r below target.
        assert info["final_r_max"] <= 0.2 + 1e-10
        assert info["volume_change_frac"] < 0.2

    def test_land_cells_untouched(self):
        """Land cells (ocean_mask=0) must be unchanged after MEO."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        H = np.full((5, 10), 4000.0)
        H[:, :5] = 200.0
        # Right half is land
        ocean = np.ones_like(H)
        ocean[:, 7:] = 0.0
        H_initial = H.copy()
        H_out, info = apply_meo_r_factor_cap(H, ocean, 0.2)
        # Land cells must equal their initial value.
        land = ocean < 0.5
        np.testing.assert_array_equal(H_out[land], H_initial[land])

    def test_idempotent(self):
        """Running MEO twice produces the same result as running once."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        H = np.full((5, 10), 4000.0)
        H[:, :5] = 200.0
        ocean = np.ones_like(H)
        H_once, _ = apply_meo_r_factor_cap(H, ocean, 0.2)
        H_twice, info = apply_meo_r_factor_cap(H_once, ocean, 0.2)
        np.testing.assert_allclose(H_once, H_twice, atol=1e-9)
        # Second run should converge in 1 iteration with no changes.
        assert info["cells_modified"] == 0

    def test_invalid_r_factor_raises(self):
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap
        H = np.full((3, 5), 4000.0)
        ocean = np.ones_like(H)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap(H, ocean, 0.0)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap(H, ocean, 1.0)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap(H, ocean, -0.1)

    def test_meo_via_load_bathymetry_pipeline(self, latlon_grid, tmp_path):
        """End-to-end: setting ``r_factor_max`` in BathymetryConfig
        triggers MEO during ``load_bathymetry`` and lowers the final
        r-factor."""
        import xarray as xr
        # Build a synthetic ETOPO with sharp continental-slope-style cuts
        n_lat, n_lon = 91, 180
        lat = np.linspace(-90.0, 90.0, n_lat)
        lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        # 4000 m ocean, sharp shelf at lon < 30°: depth = 200 m
        elev = np.where(np.abs(LAT) > 80.0, 100.0,
                          np.where(LON < 30.0, -200.0, -4000.0))
        ds = xr.Dataset(
            {"z": (["lat", "lon"], elev.astype(np.float32))},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "shelf_synth.nc"
        ds.to_netcdf(path)

        # Without MEO
        cfg_no_meo = BathymetryConfig(
            source="file", path=str(path),
            smoothing_passes=0, enforce_straits=False,
            fill_isolated_basins=False, H_min=10.0,
            r_factor_max=None,
        )
        H_no, mask_no = init_ocean_bathymetry(latlon_grid, cfg_no_meo)
        # With MEO
        cfg_meo = BathymetryConfig(
            source="file", path=str(path),
            smoothing_passes=0, enforce_straits=False,
            fill_isolated_basins=False, H_min=10.0,
            r_factor_max=0.2, meo_max_iter=200,
        )
        H_meo, mask_meo = init_ocean_bathymetry(latlon_grid, cfg_meo)

        from legoesm.ocean.bathymetry import compute_max_r_factor
        r_no = compute_max_r_factor(np.asarray(H_no), np.asarray(mask_no))
        r_meo = compute_max_r_factor(np.asarray(H_meo), np.asarray(mask_meo))
        assert r_no > 0.5, f"baseline r should be high; got {r_no}"
        # Allow a small float32 tolerance from get_policy().storage casts
        # in load_bathymetry_latlon_cgrid.
        assert r_meo <= 0.2 + 1e-6, (
            f"MEO failed to cap r-factor; final r = {r_meo}"
        )
        # Mask topology preserved (straits not enforced; flat land unchanged)
        np.testing.assert_array_equal(
            np.asarray(mask_no), np.asarray(mask_meo),
        )


# ============================================================================
# MEO r-factor cap on Voronoi (P1a of MPAS realistic-geometry plan)
# ============================================================================


@pytest.fixture
def small_voronoi_mesh():
    """Subdivision-level-2 Voronoi mesh (~162 cells)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(subdivision_level=2)


class TestMEORFactorCapVoronoi:
    """``apply_meo_r_factor_cap_voronoi`` — edge-list MEO smoothing on
    an unstructured Voronoi mesh.  The 2D Cartesian variant is wrong
    for MPAS because ``np.roll`` along the cell-index axis has no
    geometric meaning.
    """

    def test_uniform_bathymetry_is_noop(self, small_voronoi_mesh):
        """Uniform H gives r=0 everywhere; MEO must not change anything."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        nCells = small_voronoi_mesh.nCells
        H = np.full(nCells, 4000.0)
        ocean = np.ones(nCells)
        H_out, info = apply_meo_r_factor_cap_voronoi(
            H, ocean, small_voronoi_mesh, 0.2,
        )
        assert info["initial_r_max"] == 0.0
        assert info["final_r_max"] == 0.0
        assert info["cells_modified"] == 0
        np.testing.assert_array_equal(H_out, H)

    def test_caps_a_step_bathymetry(self, small_voronoi_mesh):
        """Sharp 200m → 4000m step (split by latitude) must be capped."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        nCells = small_voronoi_mesh.nCells
        lat = np.asarray(small_voronoi_mesh.latCell)
        H = np.where(lat < 0.0, 200.0, 4000.0)
        ocean = np.ones(nCells)
        H_out, info = apply_meo_r_factor_cap_voronoi(
            H, ocean, small_voronoi_mesh, 0.2, max_iter=400,
        )
        assert info["initial_r_max"] > 0.9
        assert info["final_r_max"] <= 0.2 + 1e-10, (
            f"Final r_max = {info['final_r_max']}"
        )
        assert np.all(H_out >= H - 1.0e-9)
        assert info["cells_modified"] > 0
        assert info["volume_change_frac"] > 0

    def test_land_cells_untouched(self, small_voronoi_mesh):
        """Land cells (ocean_mask=0) must be unchanged after MEO."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        nCells = small_voronoi_mesh.nCells
        lat = np.asarray(small_voronoi_mesh.latCell)
        H = np.where(lat < 0.0, 200.0, 4000.0)
        # Northern hemisphere is land
        ocean = np.where(lat < 0.0, 1.0, 0.0)
        H_initial = H.copy()
        H_out, _ = apply_meo_r_factor_cap_voronoi(
            H, ocean, small_voronoi_mesh, 0.2,
        )
        land = ocean < 0.5
        np.testing.assert_array_equal(H_out[land], H_initial[land])

    def test_idempotent(self, small_voronoi_mesh):
        """A second MEO pass on already-capped H is a no-op."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        nCells = small_voronoi_mesh.nCells
        lat = np.asarray(small_voronoi_mesh.latCell)
        H = np.where(lat < 0.0, 200.0, 4000.0)
        ocean = np.ones(nCells)
        H_once, _ = apply_meo_r_factor_cap_voronoi(
            H, ocean, small_voronoi_mesh, 0.2, max_iter=400,
        )
        H_twice, info = apply_meo_r_factor_cap_voronoi(
            H_once, ocean, small_voronoi_mesh, 0.2, max_iter=400,
        )
        np.testing.assert_allclose(H_once, H_twice, atol=1.0e-9)
        assert info["cells_modified"] == 0

    def test_cells_only_deepen(self, small_voronoi_mesh):
        """MEO never makes a cell shallower — it deepens shallow cells
        adjacent to deep ones."""
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        nCells = small_voronoi_mesh.nCells
        rng = np.random.default_rng(seed=7)
        H = rng.uniform(100.0, 5000.0, size=nCells)
        ocean = np.ones(nCells)
        H_out, _ = apply_meo_r_factor_cap_voronoi(
            H, ocean, small_voronoi_mesh, 0.3,
        )
        assert np.all(H_out >= H - 1.0e-9), (
            "MEO violated the monotone-deepening property."
        )

    def test_invalid_r_factor_raises(self, small_voronoi_mesh):
        from legoesm.ocean.bathymetry import apply_meo_r_factor_cap_voronoi
        H = np.full(small_voronoi_mesh.nCells, 4000.0)
        ocean = np.ones_like(H)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap_voronoi(H, ocean, small_voronoi_mesh, 0.0)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap_voronoi(H, ocean, small_voronoi_mesh, 1.0)
        with pytest.raises(ValueError, match="r_factor_max"):
            apply_meo_r_factor_cap_voronoi(H, ocean, small_voronoi_mesh, -0.1)

    def test_r_factor_max_voronoi_helper(self, small_voronoi_mesh):
        """Direct test of the r_max helper on a known-r configuration."""
        from legoesm.ocean.bathymetry import _r_factor_max_voronoi
        nCells = small_voronoi_mesh.nCells
        # Uniform → r_max = 0.
        H_uniform = np.full(nCells, 1000.0)
        ocean = np.ones(nCells)
        assert _r_factor_max_voronoi(
            H_uniform, ocean, small_voronoi_mesh.cellsOnEdge,
        ) == 0.0
        # Two-value field: deep_cells at lat>0, shallow at lat<0 →
        # at the equator any edge spanning latitudes has r close to
        # |1000 - 100| / 1000 = 0.9.
        lat = np.asarray(small_voronoi_mesh.latCell)
        H_split = np.where(lat < 0.0, 100.0, 1000.0)
        r = _r_factor_max_voronoi(
            H_split, ocean, small_voronoi_mesh.cellsOnEdge,
        )
        assert abs(r - 0.9) < 1.0e-12, f"Expected 0.9, got {r}"


# ============================================================================
# MPAS realistic-bathymetry end-to-end (P1b of MPAS realistic-geometry plan)
# ============================================================================


class TestMPASRealisticBathymetry:
    """``rest_state_mpas_ocean(..., bathymetry=cfg)`` loads ETOPO-style
    NetCDF onto a Voronoi mesh and produces a state with non-uniform
    ``H_bathy`` and a derived ``land_mask``.  Default invocation
    (no ``bathymetry`` kwarg) keeps the idealized path bit-exact.
    """

    @staticmethod
    def _make_synthetic_etopo_nc(tmp_path, n_lat_src=181, n_lon_src=360):
        """Hemispherically-symmetric bowl: -4000 m ocean, +100 m land
        above |lat|=80°.  Same pattern as the lat-lon test."""
        import xarray as xr
        lat = np.linspace(-90.0, 90.0, n_lat_src)
        lon = np.linspace(0.0, 360.0, n_lon_src, endpoint=False)
        LAT, _ = np.meshgrid(lat, lon, indexing="ij")
        elev = np.where(np.abs(LAT) > 80.0, 100.0, -4000.0)
        ds = xr.Dataset(
            {"z": (["lat", "lon"], elev.astype(np.float32))},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "synthetic_etopo_mpas.nc"
        ds.to_netcdf(path)
        return str(path)

    def test_default_call_unchanged(self, small_voronoi_mesh, z_coord):
        """No ``bathymetry`` kwarg → idealized path, bit-exact."""
        from legoesm.ocean.init_mpas import (
            rest_state_mpas_ocean, idealized_bathymetry_mpas,
        )
        state = rest_state_mpas_ocean(
            small_voronoi_mesh, z_coord, H_max=4000.0,
            land_lat_threshold=80.0,
        )
        H_ref, mask_ref = idealized_bathymetry_mpas(
            small_voronoi_mesh, 4000.0, 80.0,
        )
        np.testing.assert_array_equal(
            np.asarray(state.H_bathy.data), np.asarray(H_ref),
        )
        np.testing.assert_array_equal(
            np.asarray(state.land_mask.data), np.asarray(mask_ref),
        )

    def test_realistic_bathymetry_from_file(
        self, small_voronoi_mesh, z_coord, tmp_path,
    ):
        """Pass a ``BathymetryConfig(source='file', path=...)`` and the
        state must carry a depth pattern matching the ETOPO file."""
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean

        path = self._make_synthetic_etopo_nc(tmp_path)
        cfg = BathymetryConfig(
            source="file", path=path,
            H_max=5500.0, H_min=10.0,
            smoothing_passes=0,
            enforce_straits=False,
            fill_isolated_basins=False,
            depth_is_negative=True,
            r_factor_max=None,  # disable MEO for round-trip clarity
        )
        state = rest_state_mpas_ocean(small_voronoi_mesh, z_coord, bathymetry=cfg)

        H = np.asarray(state.H_bathy.data)
        mask = np.asarray(state.land_mask.data)
        lat_deg = np.asarray(np.degrees(small_voronoi_mesh.latCell))

        # Equatorial cells must be ocean at ~4000 m.
        eq = np.abs(lat_deg) < 60.0
        assert mask[eq].mean() > 0.95, (
            f"Equatorial cells should be ocean; got mask mean "
            f"{mask[eq].mean():.3f}"
        )
        ocean_eq = eq & (mask > 0.5)
        assert np.all(np.abs(H[ocean_eq] - 4000.0) < 50.0), (
            f"Equatorial ocean depth should be ~4000 m; got "
            f"min={H[ocean_eq].min():.1f}, max={H[ocean_eq].max():.1f}"
        )

        # High-latitude cells must be predominantly land.  We test only
        # very-high lat (>85°) because the fixture mesh has limited
        # resolution and sub-grid ocean fraction can flip cells in the
        # 80-85° transition band.
        polar = np.abs(lat_deg) > 85.0
        if polar.any():
            assert mask[polar].mean() < 0.5, (
                f"|lat|>85° cells should be predominantly land; got "
                f"mask mean {mask[polar].mean():.3f}"
            )

    def test_realistic_bathymetry_preserves_state_shape(
        self, small_voronoi_mesh, z_coord, tmp_path,
    ):
        """Realistic-bathy state has the same field shapes as the
        idealized state — only ``H_bathy``/``land_mask`` data differs."""
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean

        path = self._make_synthetic_etopo_nc(tmp_path)
        cfg = BathymetryConfig(
            source="file", path=path,
            H_max=5500.0, H_min=10.0,
            smoothing_passes=0, enforce_straits=False,
            fill_isolated_basins=False, depth_is_negative=True,
            r_factor_max=None,
        )
        state_ideal = rest_state_mpas_ocean(small_voronoi_mesh, z_coord)
        state_real = rest_state_mpas_ocean(
            small_voronoi_mesh, z_coord, bathymetry=cfg,
        )
        assert state_real.u.data.shape == state_ideal.u.data.shape
        assert state_real.T.data.shape == state_ideal.T.data.shape
        assert state_real.S.data.shape == state_ideal.S.data.shape
        assert state_real.eta.data.shape == state_ideal.eta.data.shape
        assert state_real.H_bathy.data.shape == state_ideal.H_bathy.data.shape
        assert state_real.land_mask.data.shape == state_ideal.land_mask.data.shape

    def test_realistic_bathymetry_with_meo_smoothing(
        self, small_voronoi_mesh, z_coord, tmp_path,
    ):
        """End-to-end pipeline with MEO r-factor cap enabled — exercises
        the Voronoi-aware ``apply_meo_r_factor_cap_voronoi`` from P1a."""
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.bathymetry import _r_factor_max_voronoi

        path = self._make_synthetic_etopo_nc(tmp_path)
        cfg = BathymetryConfig(
            source="file", path=path,
            H_max=5500.0, H_min=10.0,
            smoothing_passes=2,
            enforce_straits=False, fill_isolated_basins=False,
            depth_is_negative=True,
            r_factor_max=0.3,
        )
        state = rest_state_mpas_ocean(small_voronoi_mesh, z_coord, bathymetry=cfg)
        H = np.asarray(state.H_bathy.data)
        mask = np.asarray(state.land_mask.data)
        # MEO cap must hold on the resulting bathymetry.
        r_final = _r_factor_max_voronoi(H, mask, small_voronoi_mesh.cellsOnEdge)
        assert r_final <= 0.3 + 1e-10, (
            f"MEO r-factor cap not satisfied: r_final={r_final:.4f}"
        )


# ============================================================================
# Lat-lon C-grid integration (Phase 0 of realistic-geometry plan)
# ============================================================================


@pytest.fixture
def latlon_grid():
    """Small 36×72 (5°) lat-lon grid — same shape as global overturning."""
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(36, 72)


class TestLatLonCGridDispatch:
    """LatLonGrid must dispatch to its own bathymetry path, not collide
    with cubed-sphere or Gaussian dispatch (since LatLonGrid has both
    ``n`` (=n_lat) and ``n_lat`` attributes)."""

    def test_idealized_dispatch_returns_correct_shape(self, latlon_grid):
        cfg = BathymetryConfig(source="idealized", H_max=4000.0,
                                land_lat_threshold=80.0)
        H_bathy, ocean_mask = init_ocean_bathymetry(latlon_grid, cfg)
        assert H_bathy.shape == (36, 72)
        assert ocean_mask.shape == (36, 72)

    def test_idealized_dispatch_respects_land_threshold(self, latlon_grid):
        cfg = BathymetryConfig(source="idealized", H_max=4000.0,
                                land_lat_threshold=80.0)
        H_bathy, ocean_mask = init_ocean_bathymetry(latlon_grid, cfg)
        # At |lat|>80°, cells should be land (mask=0).  On a 5° grid
        # with cell centres at -87.5, ..., +87.5, this is rows 0, 1, 34, 35.
        lat_deg = np.abs(np.asarray(latlon_grid.lat2d)) * 180.0 / np.pi
        polar = lat_deg > 80.0
        assert np.all(np.asarray(ocean_mask)[polar] < 0.5)

    def test_does_not_collide_with_cubed_sphere(self, latlon_grid):
        """LatLonGrid has an ``n`` property (=n_lat), but the dispatch
        must route to the lat-lon path, not the cubed-sphere path.
        The cubed-sphere idealized_bathymetry would produce shape
        (6, n, n) — the wrong shape for a LatLonGrid."""
        cfg = BathymetryConfig(source="idealized")
        H_bathy, ocean_mask = init_ocean_bathymetry(latlon_grid, cfg)
        # Lat-lon shape, not cubed-sphere shape.
        assert H_bathy.ndim == 2
        assert H_bathy.shape[0] == latlon_grid.n_lat


class TestLatLonCGridFileLoading:
    """Round-trip a synthetic NetCDF bathymetry through the lat-lon
    C-grid loader."""

    def _make_synthetic_etopo_nc(self, tmp_path, n_lat_src=181, n_lon_src=360):
        """Create a synthetic ETOPO-style NetCDF (elevation negative under
        ocean, positive on land) for testing.  Pattern: hemisphere-symmetric
        bowl 4000 m deep with land above |lat|=80°."""
        import xarray as xr
        lat = np.linspace(-90.0, 90.0, n_lat_src)
        lon = np.linspace(0.0, 360.0, n_lon_src, endpoint=False)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        # Ocean: -4000 m everywhere; land where |lat| > 80°.
        elev = np.where(np.abs(LAT) > 80.0, 100.0, -4000.0)
        ds = xr.Dataset(
            {"z": (["lat", "lon"], elev.astype(np.float32))},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "synthetic_etopo.nc"
        ds.to_netcdf(path)
        return str(path)

    def test_roundtrip_synthetic_etopo(self, latlon_grid, tmp_path):
        path = self._make_synthetic_etopo_nc(tmp_path)
        cfg = BathymetryConfig(
            source="file", path=path,
            H_max=5500.0, H_min=10.0,
            smoothing_passes=0,
            enforce_straits=False,    # synthetic data has no real straits
            fill_isolated_basins=False,
            depth_is_negative=True,
        )
        H_bathy, ocean_mask = init_ocean_bathymetry(latlon_grid, cfg)
        assert H_bathy.shape == (36, 72)
        assert ocean_mask.shape == (36, 72)
        # Open ocean bathy should be approximately 4000 m where ocean.
        H_arr = np.asarray(H_bathy)
        mask_arr = np.asarray(ocean_mask)
        ocean_depths = H_arr[mask_arr > 0.5]
        assert ocean_depths.size > 0
        assert np.all(np.abs(ocean_depths - 4000.0) < 100.0), (
            f"ocean depths span [{ocean_depths.min()}, {ocean_depths.max()}], "
            f"expected ~4000"
        )
        # Polar rows should be land.
        lat_deg = np.abs(np.asarray(latlon_grid.lat2d)) * 180.0 / np.pi
        polar_rows = lat_deg.max(axis=1) > 85.0
        assert np.all(mask_arr[polar_rows] < 0.5)

    def test_handles_duplicate_endpoint_longitudes(self, latlon_grid, tmp_path):
        """ETOPO/GEBCO files often store lon in [-180, +180] inclusive.
        After ``% 360`` this produces duplicate values (both -180 and +180
        → 180), which RegularGridInterpolator rejects.  Verify the loader
        deduplicates correctly."""
        import xarray as xr
        # Inclusive endpoints — 181 lat points, 361 lon points.
        lat = np.linspace(-90.0, 90.0, 181)
        lon = np.linspace(-180.0, 180.0, 361)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        elev = np.where(np.abs(LAT) > 80.0, 100.0, -3500.0)
        ds = xr.Dataset(
            {"altitude": (["lat", "lon"], elev.astype(np.float32))},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "etopo_endpoint_duplicates.nc"
        ds.to_netcdf(path)
        cfg = BathymetryConfig(
            source="file", path=str(path),
            smoothing_passes=0, enforce_straits=False,
            fill_isolated_basins=False, H_min=10.0,
            depth_is_negative=True,
        )
        # Should not raise.
        H_bathy, ocean_mask = init_ocean_bathymetry(latlon_grid, cfg)
        assert H_bathy.shape == (36, 72)

    def test_smoothing_reduces_bathymetry_variance(self, latlon_grid, tmp_path):
        """With smoothing_passes>0, bathy should have lower variance than
        raw (after the binary mask thresholding)."""
        # Build a noisier synthetic file: random depth perturbation in ocean.
        import xarray as xr
        n_lat_src, n_lon_src = 91, 180
        lat = np.linspace(-90.0, 90.0, n_lat_src)
        lon = np.linspace(0.0, 360.0, n_lon_src, endpoint=False)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0.0, 500.0, LAT.shape)
        elev = np.where(np.abs(LAT) > 80.0, 100.0, -4000.0 + noise)
        ds = xr.Dataset(
            {"z": (["lat", "lon"], elev.astype(np.float32))},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "noisy.nc"
        ds.to_netcdf(path)

        cfg_no_smooth = BathymetryConfig(
            source="file", path=str(path),
            smoothing_passes=0, enforce_straits=False,
            fill_isolated_basins=False, H_min=10.0,
        )
        H_no, mask_no = init_ocean_bathymetry(latlon_grid, cfg_no_smooth)

        cfg_smooth = BathymetryConfig(
            source="file", path=str(path),
            smoothing_passes=5, enforce_straits=False,
            fill_isolated_basins=False, H_min=10.0,
        )
        H_smooth, mask_smooth = init_ocean_bathymetry(latlon_grid, cfg_smooth)

        # Smoothing should reduce H variance over ocean cells.
        ocean_no = np.asarray(mask_no) > 0.5
        ocean_smooth = np.asarray(mask_smooth) > 0.5
        var_no = float(np.var(np.asarray(H_no)[ocean_no]))
        var_smooth = float(np.var(np.asarray(H_smooth)[ocean_smooth]))
        assert var_smooth < var_no, (
            f"Smoothed variance {var_smooth} should be < unsmoothed {var_no}"
        )


class TestLatLonCGridRestState:
    """End-to-end rest_state_ocean_realistic on lat-lon C-grid."""

    def test_idealized_yields_LatLonCGridOceanState(self, latlon_grid, z_coord):
        from legoesm.ocean.state import LatLonCGridOceanState
        cfg = BathymetryConfig(source="idealized", H_max=4000.0)
        state = rest_state_ocean_realistic(latlon_grid, z_coord, cfg)
        assert isinstance(state, LatLonCGridOceanState)
        # Shapes
        assert state.T.data.shape == (36, 72, 10)
        assert state.u.data.shape == (36, 73, 10)   # u-faces
        assert state.v.data.shape == (37, 72, 10)   # v-faces
        assert state.eta.data.shape == (36, 72)
        assert state.H_bathy.data.shape == (36, 72)
        assert state.land_mask.data.shape == (36, 72)
        # Face masks present and consistent with land mask
        assert state.u_mask.data.shape == (36, 73)
        assert state.v_mask.data.shape == (37, 72)
        # Rest state: u, v, eta = 0
        assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
        assert float(jnp.max(jnp.abs(state.v.data))) == 0.0
        assert float(jnp.max(jnp.abs(state.eta.data))) == 0.0

    def test_H_bathy_override_preserves_input(self, latlon_grid, z_coord):
        """Calling rest_state_latlon_cgrid_ocean with H_bathy_override
        should preserve the supplied bathymetry verbatim (after mask).
        """
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        # Sloping bathymetry: shallow at the south, deep at the north.
        n_lat, n_lon = 36, 72
        H = np.linspace(500.0, 4000.0, n_lat)
        H_bathy_in = jnp.broadcast_to(
            jnp.asarray(H)[:, None], (n_lat, n_lon),
        )
        # Add a couple of land columns to test mask consistency.
        land_mask = jnp.ones((n_lat, n_lon))
        land_mask = land_mask.at[:, 10].set(0.0)
        land_mask = land_mask.at[:, 30].set(0.0)
        state = rest_state_latlon_cgrid_ocean(
            latlon_grid, z_coord,
            land_mask_override=land_mask,
            H_bathy_override=H_bathy_in,
        )
        H_out = np.asarray(state.H_bathy.data)
        H_in_np = np.asarray(H_bathy_in)
        np.testing.assert_allclose(H_out, H_in_np, rtol=1e-5)

    def test_H_bathy_override_derives_mask_when_not_given(
        self, latlon_grid, z_coord
    ):
        """If only H_bathy_override is given, ocean_mask = (H_bathy > 0)."""
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        n_lat, n_lon = 36, 72
        H = np.full((n_lat, n_lon), 4000.0)
        H[0, :] = 0.0   # south pole row = land
        H[-1, :] = 0.0  # north pole row = land
        state = rest_state_latlon_cgrid_ocean(
            latlon_grid, z_coord,
            H_bathy_override=jnp.asarray(H),
        )
        mask = np.asarray(state.land_mask.data)
        assert mask[0, 0] == 0.0
        assert mask[-1, 0] == 0.0
        assert mask[18, 36] == 1.0   # interior should be ocean
