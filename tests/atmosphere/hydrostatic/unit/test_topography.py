"""Tests for real topography loading and land-sea mask derivation (Task 6)."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.topography import (
    TopographyConfig,
    load_real_topography,
    _regrid_to_target,
    _derive_land_fraction,
    _laplacian_smooth_cubed_sphere,
    _laplacian_smooth_gaussian,
    gaussian_mountain,
    phis_from_topography,
    land_mask_from_topography,
)


def _make_synthetic_topo_netcdf(
    path: str,
    n_lat: int = 90,
    n_lon: int = 180,
    mountains: bool = True,
) -> None:
    """Create a simple synthetic topography NetCDF for testing."""
    import xarray as xr

    lat = np.linspace(-89.5, 89.5, n_lat)
    lon = np.linspace(0.5, 359.5, n_lon)

    if mountains:
        # Create synthetic continents and mountains
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        z = np.zeros((n_lat, n_lon), dtype=np.float64)

        # "Eurasia" - broad elevated region
        z += 500.0 * np.exp(
            -((lat2d - 45.0) ** 2 / (20.0**2) + (lon2d - 60.0) ** 2 / (40.0**2))
        )
        # "Himalayas" - sharp peak
        z += 5000.0 * np.exp(
            -((lat2d - 30.0) ** 2 / (5.0**2) + (lon2d - 85.0) ** 2 / (8.0**2))
        )
        # "Andes" - narrow ridge
        z += 3000.0 * np.exp(
            -((lon2d - 290.0) ** 2 / (3.0**2))
        ) * np.exp(-((lat2d + 15.0) ** 2 / (20.0**2)))
        # "Africa" - broad low continent
        z += 300.0 * np.exp(
            -((lat2d - 5.0) ** 2 / (25.0**2) + (lon2d - 25.0) ** 2 / (15.0**2))
        )
        # Ocean floor (negative)
        ocean_mask = z < 10.0
        z = np.where(ocean_mask, -3000.0, z)
    else:
        z = np.zeros((n_lat, n_lon), dtype=np.float64)

    ds = xr.Dataset(
        {"z": (("lat", "lon"), z)},
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


class TestRegridding(unittest.TestCase):
    """Test bilinear regridding of elevation data."""

    def test_uniform_field(self):
        """Uniform elevation should regrid exactly."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0, 355, 72)
        elev = np.full((37, 72), 500.0)

        target_lat = np.array([0.0, 30.0, -45.0])
        target_lon = np.array([90.0, 180.0, 270.0])

        result = _regrid_to_target(lat_src, lon_src, elev, target_lat, target_lon)
        npt.assert_allclose(result, 500.0, atol=1e-10)

    def test_latitude_gradient(self):
        """Linear latitude gradient should interpolate correctly."""
        lat_src = np.linspace(-90, 90, 181)
        lon_src = np.linspace(0, 359, 360)
        lat2d, _ = np.meshgrid(lat_src, lon_src, indexing="ij")
        elev = lat2d * 10.0  # elevation = 10 * latitude_degrees

        target_lat = np.array([0.0, 45.0, -60.0])
        target_lon = np.array([100.0, 200.0, 300.0])

        result = _regrid_to_target(lat_src, lon_src, elev, target_lat, target_lon)
        expected = target_lat * 10.0
        npt.assert_allclose(result, expected, atol=0.5)

    def test_longitude_wrapping(self):
        """Interpolation near 0/360 boundary should be smooth."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0.5, 359.5, 360)
        _, lon2d = np.meshgrid(lat_src, lon_src, indexing="ij")
        elev = np.cos(np.radians(lon2d)) * 1000.0

        # Test at lon=0 (near 360 wrap)
        target_lat = np.array([0.0])
        target_lon = np.array([0.0])

        result = _regrid_to_target(lat_src, lon_src, elev, target_lat, target_lon)
        expected = 1000.0  # cos(0) = 1
        npt.assert_allclose(result, expected, atol=5.0)

    def test_output_shape(self):
        """Result should match target shape."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0, 355, 72)
        elev = np.zeros((37, 72))

        target_lat = np.ones((6, 8, 8))
        target_lon = np.ones((6, 8, 8))

        result = _regrid_to_target(lat_src, lon_src, elev, target_lat, target_lon)
        self.assertEqual(result.shape, (6, 8, 8))


class TestLandFraction(unittest.TestCase):
    """Test sub-grid land fraction derivation."""

    def test_all_ocean(self):
        """Negative elevation everywhere should give f_land=0."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0, 355, 72)
        elev = np.full((37, 72), -3000.0)

        target_lat = np.array([0.0, 45.0])
        target_lon = np.array([90.0, 180.0])

        f_land = _derive_land_fraction(
            lat_src, lon_src, elev, target_lat, target_lon, 5.0
        )
        npt.assert_allclose(f_land, 0.0)

    def test_all_land(self):
        """Positive elevation everywhere should give f_land=1."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0, 355, 72)
        elev = np.full((37, 72), 500.0)

        target_lat = np.array([0.0, 45.0])
        target_lon = np.array([90.0, 180.0])

        f_land = _derive_land_fraction(
            lat_src, lon_src, elev, target_lat, target_lon, 5.0
        )
        npt.assert_allclose(f_land, 1.0)

    def test_mixed_gives_fractional(self):
        """Coast regions should have fractional land."""
        lat_src = np.linspace(-90, 90, 181)
        lon_src = np.linspace(0, 359, 360)
        lat2d, _ = np.meshgrid(lat_src, lon_src, indexing="ij")
        # Land above 30N, ocean below
        elev = np.where(lat2d > 30.0, 500.0, -3000.0)

        # Target point right at boundary
        target_lat = np.array([30.0])
        target_lon = np.array([180.0])

        f_land = _derive_land_fraction(
            lat_src, lon_src, elev, target_lat, target_lon, 5.0
        )
        # Should be around 0.5 at the boundary
        self.assertGreater(float(f_land[0]), 0.2)
        self.assertLess(float(f_land[0]), 0.8)


class TestSmoothing(unittest.TestCase):
    """Test Laplacian smoothing."""

    def test_cubed_sphere_reduces_noise(self):
        """Smoothing should reduce field variance."""
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 100, (6, 16, 16))
        smoothed = _laplacian_smooth_cubed_sphere(arr, passes=4)
        self.assertLess(np.var(smoothed), np.var(arr))

    def test_cubed_sphere_preserves_mean(self):
        """Smoothing should roughly preserve the global mean."""
        rng = np.random.default_rng(42)
        arr = rng.normal(500, 100, (6, 16, 16))
        smoothed = _laplacian_smooth_cubed_sphere(arr, passes=4)
        npt.assert_allclose(np.mean(smoothed), np.mean(arr), rtol=0.1)

    def test_cubed_sphere_zero_passes(self):
        """Zero passes should return unchanged array."""
        arr = np.random.default_rng(42).normal(0, 100, (6, 8, 8))
        result = _laplacian_smooth_cubed_sphere(arr, passes=0)
        npt.assert_array_equal(result, arr)

    def test_gaussian_reduces_noise(self):
        """Smoothing should reduce field variance on Gaussian grid."""
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 100, (32, 64))
        smoothed = _laplacian_smooth_gaussian(arr, passes=4)
        self.assertLess(np.var(smoothed), np.var(arr))

    def test_gaussian_preserves_mean(self):
        """Smoothing should roughly preserve the mean on Gaussian grid."""
        rng = np.random.default_rng(42)
        arr = rng.normal(500, 100, (32, 64))
        smoothed = _laplacian_smooth_gaussian(arr, passes=4)
        npt.assert_allclose(np.mean(smoothed), np.mean(arr), rtol=0.1)


class TestLoadRealTopography(unittest.TestCase):
    """Test loading and regridding from NetCDF."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.grid = create_cubed_sphere(8)

    def test_load_cubed_sphere(self):
        """Load synthetic topo onto cubed-sphere grid."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(
            source="file", path=path,
            smoothing_passes=2, edge_blend_strength=0.1,
        )
        phis, f_land = load_real_topography(self.grid, config=config)

        self.assertEqual(phis.shape, (6, 8, 8))
        self.assertEqual(f_land.shape, (6, 8, 8))

    def test_phis_non_negative(self):
        """With clip_negative=True (default), phis should be >= 0."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path=path)
        phis, _ = load_real_topography(self.grid, config=config)

        self.assertTrue(float(jnp.min(phis)) >= 0.0)

    def test_load_latlon_with_smoothing(self):
        """Load real topography onto the lat-lon grid WITH smoothing passes.

        The grid was previously mis-classified as cubed-sphere (it exposes
        ``n``), so its 2-D field hit the (6, n, n) smoother and crashed.  Now
        classified by coordinate rank; the 2-D Gaussian smoother applies.
        """
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(n_lat=24, radius=constants.R_earth,
                                  omega=constants.Omega)
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(
            source="file", path=path,
            smoothing_passes=4, edge_blend_strength=0.1,
        )
        phis, f_land = load_real_topography(grid, config=config)

        self.assertEqual(phis.shape, (grid.n_lat, grid.n_lon))
        self.assertEqual(f_land.shape, (grid.n_lat, grid.n_lon))
        self.assertTrue(bool(jnp.all(jnp.isfinite(phis))))
        self.assertGreaterEqual(float(jnp.min(phis)), 0.0)
        self.assertTrue(0.0 <= float(jnp.min(f_land))
                        and float(jnp.max(f_land)) <= 1.0)

    def test_phis_has_mountains(self):
        """Should have nonzero phis where mountains exist."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path, mountains=True)

        config = TopographyConfig(
            source="file", path=path, smoothing_passes=1,
        )
        phis, _ = load_real_topography(self.grid, config=config)

        z_max = float(jnp.max(phis)) / constants.g
        self.assertGreater(z_max, 100.0)  # some mountains survived regridding

    def test_flat_has_no_phis(self):
        """Flat NetCDF should give near-zero phis."""
        path = str(Path(self.tmpdir) / "flat.nc")
        _make_synthetic_topo_netcdf(path, mountains=False)

        config = TopographyConfig(source="file", path=path)
        phis, _ = load_real_topography(self.grid, config=config)

        npt.assert_allclose(phis, 0.0, atol=1e-10)

    def test_land_fraction_bounded(self):
        """Land fraction should be in [0, 1]."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path=path)
        _, f_land = load_real_topography(self.grid, config=config)

        self.assertTrue(float(jnp.min(f_land)) >= 0.0)
        self.assertTrue(float(jnp.max(f_land)) <= 1.0)

    def test_land_fraction_mixed(self):
        """With continents, should have both land and ocean."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path, mountains=True)

        config = TopographyConfig(source="file", path=path)
        _, f_land = load_real_topography(self.grid, config=config)

        mean_land = float(jnp.mean(f_land))
        self.assertGreater(mean_land, 0.01)  # some land
        self.assertLess(mean_land, 0.99)     # some ocean

    def test_data_path_override(self):
        """data_path parameter should override config.path."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path="/nonexistent.nc")
        phis, f_land = load_real_topography(
            self.grid, config=config, data_path=path,
        )
        self.assertEqual(phis.shape, (6, 8, 8))

    def test_no_edge_blend(self):
        """edge_blend_strength=0 should disable blending."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(
            source="file", path=path,
            edge_blend_strength=0.0, smoothing_passes=0,
        )
        phis, _ = load_real_topography(self.grid, config=config)
        self.assertEqual(phis.shape, (6, 8, 8))


class TestLoadGaussianGrid(unittest.TestCase):
    """Test loading topography onto Gaussian grid."""

    def setUp(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        self.grid = create_gaussian_grid(21)  # T21
        self.tmpdir = tempfile.mkdtemp()

    def test_gaussian_grid_shape(self):
        """Topography should have Gaussian grid shape."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path=path)
        phis, f_land = load_real_topography(self.grid, config=config)

        self.assertEqual(phis.shape, (self.grid.n_lat, self.grid.n_lon))
        self.assertEqual(f_land.shape, (self.grid.n_lat, self.grid.n_lon))

    def test_gaussian_phis_non_negative(self):
        """phis should be >= 0 on Gaussian grid."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path=path)
        phis, _ = load_real_topography(self.grid, config=config)

        self.assertTrue(float(jnp.min(phis)) >= 0.0)


class TestTopographyConfig(unittest.TestCase):
    """Test TopographyConfig defaults and creation."""

    def test_defaults(self):
        config = TopographyConfig()
        self.assertEqual(config.source, "flat")
        self.assertEqual(config.smoothing_passes, 4)
        self.assertEqual(config.edge_blend_strength, 0.3)
        self.assertTrue(config.clip_negative)

    def test_custom_config(self):
        config = TopographyConfig(
            source="file", path="/data/etopo1.nc",
            smoothing_passes=8, edge_blend_strength=0.5,
        )
        self.assertEqual(config.path, "/data/etopo1.nc")
        self.assertEqual(config.smoothing_passes, 8)


class TestEdgeBlending(unittest.TestCase):
    """Test that edge blending reduces discontinuities at face boundaries."""

    def test_blending_reduces_edge_jumps(self):
        """Edge blending should reduce jumps at face boundaries."""
        grid = create_cubed_sphere(16)
        tmpdir = tempfile.mkdtemp()
        path = str(Path(tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        # Without blending
        config_no_blend = TopographyConfig(
            source="file", path=path,
            smoothing_passes=0, edge_blend_strength=0.0,
        )
        phis_no_blend, _ = load_real_topography(grid, config=config_no_blend)

        # With blending
        config_blend = TopographyConfig(
            source="file", path=path,
            smoothing_passes=0, edge_blend_strength=0.5,
            edge_blend_width=2,
        )
        phis_blend, _ = load_real_topography(grid, config=config_blend)

        # Compute edge differences for face 0-1 boundary
        # Face 0 east edge vs face 1 west edge
        edge_diff_no_blend = float(jnp.mean(jnp.abs(
            phis_no_blend[0, -1, :] - phis_no_blend[1, 0, :]
        )))
        edge_diff_blend = float(jnp.mean(jnp.abs(
            phis_blend[0, -1, :] - phis_blend[1, 0, :]
        )))

        # Blended version should have smaller edge jumps
        self.assertLessEqual(edge_diff_blend, edge_diff_no_blend + 1e-10)


class TestInitialization(unittest.TestCase):
    """Test that topography flows through to the dynamical state."""

    def test_held_suarez_with_phis(self):
        """held_suarez_init should accept phis and adjust p_s."""
        from legoesm.atmosphere.held_suarez import held_suarez_init
        from legoesm.grids.vertical import create_sigma_coordinate

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)

        # Create a simple mountain
        z_s = gaussian_mountain(grid, h0=2000.0)
        phis = phis_from_topography(z_s)

        state = held_suarez_init(grid, sigma, T_init=280.0, phis=phis)

        # phis should be stored
        npt.assert_allclose(state.phis.data, phis, atol=1e-10)

        # p_s should be reduced over mountains
        p_s_flat = 1e5
        p_s_mountain = float(jnp.min(state.p_s.data))
        self.assertLess(p_s_mountain, p_s_flat)

    def test_spectral_with_phis(self):
        """isothermal_rest_state_spectral should accept phis."""
        from legoesm.atmosphere.dynamics.spectral_pe import (
            isothermal_rest_state_spectral,
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
        from legoesm.grids.vertical import create_sigma_coordinate

        grid = create_gaussian_grid(21)
        sigma = create_sigma_coordinate(10)

        # Create phis on Gaussian grid (simple zonal mountain)
        lat2d = np.asarray(grid.lat2d)
        lon2d = np.asarray(grid.lon2d)
        z_s = 2000.0 * np.exp(
            -((lat2d - np.radians(30)) ** 2 / np.radians(10) ** 2)
            - ((lon2d - np.radians(90)) ** 2 / np.radians(15) ** 2)
        )
        phis = jnp.array(constants.g * z_s)

        state = isothermal_rest_state_spectral(
            grid, sigma, T_init=280.0, phis=phis,
        )

        # phis_hat should be nonzero
        phis_hat_max = float(jnp.max(jnp.abs(state.phis_hat.data)))
        self.assertGreater(phis_hat_max, 0.0)

        # Recover phis on grid and check
        phis_recovered = sh_synthesis(grid, state.phis_hat.data)
        # Spectral truncation means not exact, but close
        npt.assert_allclose(phis_recovered, phis, rtol=0.3, atol=500.0)

        # p_s should be non-uniform (reduced over mountains)
        lnps_grid = sh_synthesis(grid, state.lnps_hat.data)
        p_s_grid = jnp.exp(lnps_grid)
        p_s_range = float(jnp.max(p_s_grid) - jnp.min(p_s_grid))
        self.assertGreater(p_s_range, 100.0)


class TestLatLonAnalyticTopography(unittest.TestCase):
    """Analytic mountain generators must work on the lat-lon grid, whose
    ``lat``/``lon`` are 1-D axes (n_lat,) / (n_lon,) — previously they broadcast
    ``grid.lat - grid.lon`` as (n_lat,)+(n_lon,) and crashed."""

    def setUp(self):
        from legoesm.grids.latlon import create_latlon_grid
        self.grid = create_latlon_grid(
            n_lat=24, radius=constants.R_earth, omega=constants.Omega)
        self.shape = (self.grid.n_lat, self.grid.n_lon)

    def test_gaussian_mountain_latlon_shape_and_nonflat(self):
        z = gaussian_mountain(self.grid, h0=2500.0)
        self.assertEqual(z.shape, self.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(z))))
        self.assertGreater(float(jnp.max(z)), 1000.0)
        self.assertAlmostEqual(float(jnp.min(z)), 0.0, delta=10.0)

    def test_zonal_ridge_latlon_shape_and_zonal(self):
        from legoesm.grids.topography import zonal_ridge
        z = zonal_ridge(self.grid, h0=2500.0)
        self.assertEqual(z.shape, self.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(z))))
        # Zonally symmetric: each latitude row is constant in longitude
        # (to float32 precision).
        z = np.asarray(z)
        self.assertLess(float(z.std(axis=1).max()), 1e-4 * float(z.max()))

    def test_schaer_mountain_latlon_shape(self):
        from legoesm.grids.topography import schaer_mountain
        z = schaer_mountain(self.grid)
        self.assertEqual(z.shape, self.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(z))))

    def test_cubed_sphere_unchanged(self):
        # Regression: the 2-D-coordinate helper must not change cubed-sphere
        # output (lat/lon already 2-D there).
        grid = create_cubed_sphere(8)
        z = gaussian_mountain(grid, h0=2500.0)
        self.assertEqual(z.shape, (6, 8, 8))
        self.assertGreater(float(jnp.max(z)), 1000.0)


class TestLaplacianSmoothCrossFace(unittest.TestCase):
    """Cube-imprint guard: the topography Laplacian smoothing must use the
    cross-face halo, NOT per-face boundary clamping (which smooths each face in
    isolation and leaves a cube-edge seam — the cube imprint). The authoritative
    check is the nightly cube-SW visual-regression gate; this asserts the
    necessary cross-face-leakage property deterministically."""

    def test_smoothing_leaks_across_face_boundaries(self):
        from legoesm.grids.topography import _laplacian_smooth_cubed_sphere

        n = 6
        arr = np.zeros((6, n, n))
        arr[0] = 1.0  # face 0 hot, all other faces zero
        out = _laplacian_smooth_cubed_sphere(arr, passes=1)

        # With a real cross-face halo the faces bordering face 0 receive a
        # positive contribution. One-sided boundary clamping (the cube-imprint
        # bug) leaves every non-face-0 cell exactly 0.
        self.assertTrue(
            np.any(out[1:] > 1e-6),
            "smoothing did not cross cube face boundaries — per-face clamping "
            "would leave a cube-edge seam (cube imprint)",
        )
        # Face 0 mixes toward its (zero) neighbours, so its min drops below 1.
        self.assertLess(float(out[0].min()), 1.0)

    def test_constant_field_is_preserved(self):
        """Smoothing a constant field must return it unchanged (no spurious
        edge artifact from the halo stencil)."""
        from legoesm.grids.topography import _laplacian_smooth_cubed_sphere

        arr = np.full((6, 6, 6), 3.0)
        out = _laplacian_smooth_cubed_sphere(arr, passes=3)
        npt.assert_allclose(out, 3.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
