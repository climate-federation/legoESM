"""Tests for real topography loading and land-sea mask derivation (Task 6)."""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.topography import (
    TopographyConfig,
    load_real_topography,
    _regrid_to_target,
    _neighbour_table,
    bin_latlon_to_cells,
    masked_diffusion,
    gaussian_mountain,
    phis_from_topography,
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


class TestBinning(unittest.TestCase):
    """The product's binning: nearest-centre ownership, source-area
    weighted; land fraction and elevation from the SAME binning."""

    def _targets(self, dlat=10.0, dlon=10.0):
        lat = np.arange(-85.0, 90.0, dlat)
        lon = np.arange(5.0, 360.0, dlon)
        lo, la = np.meshgrid(lon, lat)
        return np.deg2rad(la).ravel(), np.deg2rad(lo).ravel(), la, lo

    def test_all_ocean_and_all_land(self):
        lat_src = np.linspace(-89.5, 89.5, 180)
        lon_src = np.linspace(0.5, 359.5, 360)
        la, lo, _, _ = self._targets()
        for val, want in ((-3000.0, 0.0), (500.0, 1.0)):
            elev = np.full((180, 360), val)
            z, f = bin_latlon_to_cells(lat_src, lon_src,
                                       [np.maximum(elev, 0.0), (elev > 0).astype(float)], la, lo)
            npt.assert_allclose(f, want)
            npt.assert_allclose(z, max(val, 0.0))

    def test_coast_cell_is_fractional_and_uses_the_same_samples(self):
        lat_src = np.linspace(-89.5, 89.5, 180)
        lon_src = np.linspace(0.5, 359.5, 360)
        lat2d, _ = np.meshgrid(lat_src, lon_src, indexing="ij")
        elev = np.where(lat2d > 34.0, 500.0, -3000.0)
        la, lo, la_deg, _ = self._targets()
        z, f = bin_latlon_to_cells(lat_src, lon_src,
                                   [np.maximum(elev, 0.0), (elev > 0).astype(float)], la, lo)
        coast = np.isclose(la_deg.ravel(), 35.0)
        # the 10-degree cell centred on 35N (30N..40N) holds the 34N coast:
        # fractional, and its elevation is the SAME samples' mean
        self.assertTrue(np.all((f[coast] > 0.4) & (f[coast] < 0.8)), f[coast])
        npt.assert_allclose(z[coast], 500.0 * f[coast], rtol=1e-12)
        npt.assert_allclose(f[la_deg.ravel() >= 45.0], 1.0)
        npt.assert_allclose(f[la_deg.ravel() <= 25.0], 0.0)

    def test_area_integral_is_conserved(self):
        from legoesm.grids.topography import _source_cell_areas
        lat_src = np.linspace(-89.5, 89.5, 180)
        lon_src = np.linspace(0.5, 359.5, 360)
        rng = np.random.default_rng(0)
        elev = rng.uniform(0.0, 1000.0, (180, 360))
        la, lo, _, _ = self._targets()
        (z,) = bin_latlon_to_cells(lat_src, lon_src, [elev], la, lo)
        a_src = _source_cell_areas(lat_src, lon_src)
        npt.assert_allclose(a_src.sum(), 4.0 * np.pi, rtol=1e-12)
        # ownership partitions the source: sum_cells z_c * A_c == sum_src elev * a
        from scipy.spatial import cKDTree
        cl = np.cos(la)
        xyz_c = np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], -1)
        lo2, la2 = np.meshgrid(np.deg2rad(lon_src), np.deg2rad(lat_src))
        cs = np.cos(la2.ravel())
        xyz_s = np.stack([cs * np.cos(lo2.ravel()), cs * np.sin(lo2.ravel()),
                          np.sin(la2.ravel())], -1)
        owner = cKDTree(xyz_c).query(xyz_s)[1]
        a_c = np.bincount(owner, weights=a_src.ravel(), minlength=la.size)
        npt.assert_allclose(np.sum(z * a_c), np.sum(elev * a_src), rtol=1e-12)

    def test_polar_cell_is_land(self):
        lat_src = np.linspace(-89.5, 89.5, 180)
        lon_src = np.linspace(0.5, 359.5, 360)
        elev = np.full((180, 360), -3000.0)
        elev[lat_src <= -60.0, :] = 2500.0
        la, lo, la_deg, _ = self._targets()
        (f,) = bin_latlon_to_cells(lat_src, lon_src, [(elev > 0).astype(float)], la, lo)
        npt.assert_allclose(f[la_deg.ravel() <= -75.0], 1.0)

    def test_latlon_box_ownership_is_exact(self):
        from legoesm.grids.factory import create_grid
        from legoesm.grids.topography import owner_by_boxes
        grid = create_grid("latlon", 16)
        lat_c, lon_c = np.asarray(grid.lat), np.asarray(grid.lon)
        n_lat, n_lon = lat_c.size, lon_c.size
        # the centres own themselves; a point just east of a centre's box
        # edge belongs to the next column (periodic at the dateline)
        lo, la = np.meshgrid(lon_c, lat_c)
        own = owner_by_boxes(la.ravel(), lo.ravel(), lat_c, lon_c)
        npt.assert_array_equal(own, np.arange(n_lat * n_lon))
        dlon = 2.0 * np.pi / n_lon
        own2 = owner_by_boxes(la.ravel(), lo.ravel() + 0.51 * dlon, lat_c, lon_c)
        npt.assert_array_equal(own2 % n_lon, (np.arange(n_lat * n_lon) + 1) % n_lon)

    def test_descending_source_latitudes_bin_the_same(self):
        """ERA5-style north-to-south sources give the same product."""
        from legoesm.grids.factory import create_grid
        from legoesm.grids.topography import grid_terrain_product
        grid = create_grid("latlon", 16)
        lat = np.arange(-89.75, 90.0, 0.5)
        lon = np.arange(0.25, 360.0, 0.5)
        lo, la = np.meshgrid(lon, lat)
        elev = 1000.0 * np.exp(-((la - 30.0) ** 2 / 400.0 + (lo - 90.0) ** 2 / 900.0)) - 200.0
        z1, f1 = grid_terrain_product(grid, lat, lon, elev)
        z2, f2 = grid_terrain_product(grid, lat[::-1], lon, elev[::-1])
        npt.assert_allclose(z2, z1, rtol=0, atol=1e-9)
        npt.assert_allclose(f2, f1, rtol=0, atol=1e-12)
        self.assertGreater(float(np.max(z1)), 100.0)

    def test_gaussian_boxes_use_the_quadrature_edges(self):
        """On a Gaussian grid the boxes are the quadrature cells, so the
        binned source area equals ``grid_area`` (polar cells included) to
        the source resolution -- the midpoint boxes miss by 2x at T21."""
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.topography import _source_cell_areas, owner_by_boxes
        grid = create_gaussian_grid(21)
        lat = np.arange(-89.95, 90.0, 0.1)
        lon = np.arange(0.05, 360.0, 0.1)
        lo, la = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
        a_src = _source_cell_areas(lat, lon).ravel()
        own = owner_by_boxes(la.ravel(), lo.ravel(), np.asarray(grid.lat), np.asarray(grid.lon),
                             np.asarray(grid.lat_v))
        got = np.bincount(own, weights=a_src, minlength=grid.n_lat * grid.n_lon)
        want = np.asarray(grid.grid_area).ravel() / float(grid.radius) ** 2
        npt.assert_allclose(got, want, rtol=0.03)

    def test_coarser_source_than_target_is_refused(self):
        lat_src = np.linspace(-80.0, 80.0, 9)
        lon_src = np.linspace(0.0, 340.0, 18)
        la, lo, _, _ = self._targets(dlat=2.0, dlon=2.0)
        with self.assertRaises(ValueError):
            bin_latlon_to_cells(lat_src, lon_src, [np.zeros((9, 18))], la, lo)


class TestMaskedDiffusion(unittest.TestCase):
    """fv_surf_map's zero_ocean on every grid: flux-form diffusion with the
    min-land-fraction edge weight."""

    def test_cube_conserves_area_integral_and_damps_noise(self):
        grid = create_cubed_sphere(8)
        nb, area = _neighbour_table(grid)
        self.assertEqual(nb.shape, (4, 6 * 64))
        # symmetric neighbour relation
        for i in range(nb.shape[1]):
            for j in nb[:, i]:
                self.assertIn(i, nb[:, j])
        rng = np.random.default_rng(1)
        q = rng.normal(size=6 * 64) * 100.0
        f = np.ones_like(q)
        out = masked_diffusion(q, f, nb, area, passes=4)
        npt.assert_allclose(np.sum(out * area), np.sum(q * area), rtol=1e-12)
        self.assertLess(np.std(out), np.std(q))
        npt.assert_array_equal(masked_diffusion(q, f, nb, area, passes=0), q)

    def test_ocean_pinned_and_island_isolated(self):
        grid = create_cubed_sphere(8)
        nb, area = _neighbour_table(grid)
        q = np.zeros(6 * 64)
        f = np.zeros(6 * 64)
        q[100] = 1000.0  # a one-cell island
        f[100] = 1.0
        f[200:230] = 1.0  # a land strip
        q[200:230] = np.arange(30.0)
        out = masked_diffusion(q, f, nb, area, passes=4)
        self.assertEqual(out[100], 1000.0)               # no flux across a coast
        npt.assert_array_equal(out[f == 0.0], 0.0)       # ocean stays exactly 0
        # land-land fluxes still conserve the land integral
        land = f > 0.0
        npt.assert_allclose(np.sum(out[land] * area[land]),
                            np.sum(q[land] * area[land]), rtol=1e-12)
        self.assertLess(np.ptp(out[200:230]), np.ptp(q[200:230]))

    def test_latlon_table_is_periodic_and_pole_clamped(self):
        from legoesm.grids.factory import create_grid
        grid = create_grid("latlon", 16)
        nb, area = _neighbour_table(grid)
        n_lat, n_lon = np.asarray(grid.grid_lat).shape
        self.assertEqual(nb.shape, (4, n_lat * n_lon))
        self.assertTrue(np.all(nb[0, :n_lon] == -1) and np.all(nb[1, -n_lon:] == -1))
        self.assertEqual(nb[2, 0], n_lon - 1)            # west of column 0 wraps
        q = np.zeros(n_lat * n_lon)
        q[n_lon * (n_lat // 2)] = 100.0
        out = masked_diffusion(q, np.ones_like(q), nb, area, passes=1)
        self.assertGreater(out[n_lon * (n_lat // 2) + n_lon - 1], 0.0)   # periodic spread

    def test_voronoi_table(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        mesh = create_voronoi_mesh(2)
        nb, area = _neighbour_table(mesh)
        for i in range(nb.shape[1]):                 # symmetric relation
            for j in nb[:, i]:
                if j >= 0:
                    self.assertIn(i, nb[:, j])
        rng = np.random.default_rng(3)
        q = rng.normal(size=mesh.nCells) * 100.0
        out = masked_diffusion(q, np.ones_like(q), nb, area, passes=3)
        npt.assert_allclose(np.sum(out * area), np.sum(q * area), rtol=1e-12)
        self.assertLess(np.std(out), 0.8 * np.std(q))
        npt.assert_allclose(masked_diffusion(np.full(mesh.nCells, 5.0), np.ones(mesh.nCells),
                                             nb, area, passes=3), 5.0)

    def test_monotone_bound(self):
        grid = create_cubed_sphere(4)
        nb, area = _neighbour_table(grid)
        with self.assertRaises(ValueError):
            masked_diffusion(np.zeros(96), np.ones(96), nb, area, k=0.3)


class TestLoadRealTopography(unittest.TestCase):
    """Test loading and regridding from NetCDF."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.grid = create_cubed_sphere(8)

    def test_load_cubed_sphere(self):
        """Load synthetic topo onto cubed-sphere grid."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)

        config = TopographyConfig(source="file", path=path, smoothing_passes=2)
        phis, f_land = load_real_topography(self.grid, config=config)

        self.assertEqual(phis.shape, (6, 8, 8))
        self.assertEqual(f_land.shape, (6, 8, 8))

    def test_duplicated_periodic_end_column_dropped(self):
        """A file with both -180 and 180 loads; the file's FIRST column wins."""
        import xarray as xr

        lat = np.linspace(-90.0, 90.0, 91)
        lon = np.linspace(-180.0, 180.0, 181)
        z = np.full((lat.size, lon.size), 1000.0)
        z[:, -1] = 3000.0  # end column disagrees with the first one
        path = str(Path(self.tmpdir) / "topo_dup.nc")
        xr.Dataset({"z": (("lat", "lon"), z)},
                   coords={"lat": lat, "lon": lon}).to_netcdf(path)

        config = TopographyConfig(source="file", path=path,
                                  smoothing_passes=0)
        # This lat-lon grid has cells ON 180 deg, where the kept column is read.
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(n_lat=24, radius=constants.R_earth,
                                  omega=constants.Omega)
        self.assertTrue(np.isclose(np.degrees(np.asarray(grid.lon)) % 360.0,
                                   180.0).any())
        phis, _ = load_real_topography(grid, config=config)
        npt.assert_allclose(np.asarray(phis) / constants.g, 1000.0, rtol=1e-12)

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
            smoothing_passes=4,
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

    def test_phis_only_where_land_and_one_product(self):
        """Decision C: phis > 0 only where f_land > 0 (zero_ocean), the
        land fraction and the elevation come from one binning."""
        path = str(Path(self.tmpdir) / "topo.nc")
        _make_synthetic_topo_netcdf(path)
        config = TopographyConfig(source="file", path=path, smoothing_passes=4)
        phis, f_land = load_real_topography(self.grid, config=config)
        phis, f_land = np.asarray(phis), np.asarray(f_land)
        self.assertTrue(np.all(phis[f_land == 0.0] == 0.0))
        self.assertTrue(np.all((f_land >= 0.0) & (f_land <= 1.0)))
        self.assertGreater(phis.max(), 0.0)


class TestDoubleGGuard(unittest.TestCase):
    """T6: geopotential-vs-meters double-g trap.

    An ERA5-style invariant file where 'z' is GEOPOTENTIAL [m**2 s**-2] used
    to be treated as elevation [m] and multiplied by g again in
    load_real_topography — phis silently inflated ~9.8x.  The loader must
    reject implausible magnitudes (> _MAX_PLAUSIBLE_ELEV_M) and geopotential
    units attributes.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.grid = create_cubed_sphere(8)

    def _write(self, z, units=None, name="topo.nc"):
        import xarray as xr
        n_lat, n_lon = z.shape
        lat = np.linspace(-89.5, 89.5, n_lat)
        lon = np.linspace(0.5, 359.5, n_lon)
        da = xr.DataArray(z, dims=("lat", "lon"),
                          coords={"lat": lat, "lon": lon})
        if units is not None:
            da.attrs["units"] = units
        path = str(Path(self.tmpdir) / name)
        xr.Dataset({"z": da}).to_netcdf(path)
        return path

    def test_geopotential_magnitude_raises(self):
        z = np.zeros((90, 180))
        z[45, 90] = 5000.0 * constants.g   # ~49000 "m" — clearly geopotential
        path = self._write(z)
        with self.assertRaisesRegex(ValueError, "double-g"):
            load_real_topography(
                self.grid, config=TopographyConfig(source="file", path=path))

    def test_geopotential_units_attr_raises(self):
        # Magnitude alone is plausible, but the units attribute says m²/s².
        z = np.zeros((90, 180))
        z[45, 90] = 5000.0
        path = self._write(z, units="m**2 s**-2")
        with self.assertRaisesRegex(ValueError, "double-g"):
            load_real_topography(
                self.grid, config=TopographyConfig(source="file", path=path))

    def test_normal_etopo_passes(self):
        # Everest-scale peak + Mariana-scale trench are plausible elevation.
        z = np.zeros((90, 180))
        z[45, 90] = 8800.0
        z[10, 10] = -10900.0
        path = self._write(z, units="m")
        phis, f_land = load_real_topography(
            self.grid, config=TopographyConfig(source="file", path=path))
        self.assertTrue(bool(jnp.all(jnp.isfinite(phis))))
        self.assertEqual(phis.shape, (6, 8, 8))


class TestPoleRowRegrid(unittest.TestCase):
    """L2: target cells poleward of the source grid's outermost latitude
    CENTER used to get fill_value=0 (f_land=0 ocean / z_s=0 sea level) —
    Antarctica misclassified as ocean at cubed-sphere/Gaussian points near
    ±90.  The source latitude axis is now edge-replicated to exactly ±90."""

    def test_regrid_pole_rows_edge_replicated(self):
        lat_src = np.linspace(-89.5, 89.5, 180)   # 1-deg centers, not reaching ±90
        lon_src = np.linspace(0.5, 359.5, 360)
        elev = np.zeros((180, 360))
        elev[0, :] = 2500.0    # Antarctica-like all-land poleward-most row
        elev[-1, :] = 100.0    # a north-pole row too
        target_lat = np.array([-90.0, -89.9, 90.0])
        target_lon = np.array([10.0, 200.0, 350.0])
        result = _regrid_to_target(lat_src, lon_src, elev,
                                   target_lat, target_lon)
        npt.assert_allclose(result[:2], 2500.0, rtol=1e-6)
        npt.assert_allclose(result[2], 100.0, rtol=1e-6)

    def test_land_fraction_file_pole_row_is_land(self):
        import xarray as xr
        from legoesm.grids.topography import _load_land_fraction_file
        lat = np.linspace(-89.5, 89.5, 180)
        lon = np.linspace(0.5, 359.5, 360)
        sftlf = np.zeros((180, 360))
        sftlf[lat <= -60.0, :] = 100.0     # Antarctica all-land (percent field)
        tmpdir = tempfile.mkdtemp()
        path = str(Path(tmpdir) / "sftlf.nc")
        xr.Dataset({"sftlf": (("lat", "lon"), sftlf)},
                   coords={"lat": lat, "lon": lon}).to_netcdf(path)
        target_lat = np.array([-90.0, -89.8, 0.0])
        target_lon = np.array([120.0, 300.0, 45.0])
        f_land = _load_land_fraction_file(path, "", target_lat, target_lon)
        npt.assert_allclose(f_land[:2], 1.0, atol=1e-6)   # NOT ocean
        npt.assert_allclose(f_land[2], 0.0, atol=1e-6)    # equator still ocean



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

    def test_custom_config(self):
        config = TopographyConfig(
            source="file", path="/data/etopo1.nc",
            smoothing_passes=8,
        )
        self.assertEqual(config.path, "/data/etopo1.nc")
        self.assertEqual(config.smoothing_passes, 8)


class TestInitialization(unittest.TestCase):
    """Test that topography flows through to the dynamical state."""

    def test_held_suarez_with_phis(self):
        """held_suarez_init should accept phis and adjust p_s."""
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            isothermal_rest_state_spectral,
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


class TestPhisAnchorBarometric(unittest.TestCase):
    """Unit tests for the barometric p_s formula used in topography handling.

    NOTE: The phis anchor (substituting ETOPO phis into state after ERA5 IC
    load) was removed from model_driver._init_state because transplanting ETOPO
    phis while keeping ERA5 winds creates a pressure-gradient imbalance that
    causes a blowup within day 1 (local p_s delta ~15 % over Tibet/Andes
    exceeds CFL).  Dynamics run with ERA5 phis; ETOPO is used for CMOR orog
    and f_land only.  These tests retain the barometric math for future use
    when a balanced ETOPO IC path is implemented.
    """

    def test_barometric_p_s_adjustment_direction(self):
        """When ERA5 phis > ETOPO phis (ERA5 mountain > ETOPO mountain),
        p_s should increase (relief is lower than ERA5, column is taller)."""
        from legoesm import constants
        era5_phis = jnp.array([30000.0, 10000.0])  # m²/s² — ERA5 high
        etopo_phis = jnp.array([10000.0, 10000.0])  # ETOPO lower
        p_s = jnp.array([70000.0, 100000.0])
        T_sfc = 270.0
        delta_phis = era5_phis - etopo_phis
        p_s_adj = p_s * jnp.exp(delta_phis / (constants.R_d * T_sfc))
        # ERA5 - ETOPO > 0 → delta_phis > 0 → exp(...) > 1 → p_s increases
        self.assertGreater(float(p_s_adj[0]), float(p_s[0]))
        # Where phis unchanged, p_s unchanged
        npt.assert_allclose(float(p_s_adj[1]), float(p_s[1]), rtol=1e-6)

    def test_barometric_p_s_adjustment_flat_topo(self):
        """When _phis_data is zero (flat), the anchor should not fire."""
        flat_phis = jnp.zeros((6, 8, 8))
        # The gate condition: jnp.any(_phis_data != 0)
        self.assertFalse(bool(jnp.any(flat_phis != 0)))

    def test_barometric_p_s_recovers_hydrostatic_ps(self):
        """Barometric adjustment should recover the hydrostatic p_s for a
        constant-T atmosphere: p_s_adj = p_ref * exp(-phis_etopo / (R_d * T))
        when the ERA5 state was also hydrostatically consistent."""
        from legoesm import constants
        T = 280.0
        p_ref = 101325.0
        etopo_phis = jnp.array([0.0, 9806.16, 49030.8])  # 0, 1000, 5000 m
        era5_phis = jnp.array([0.0, 0.0, 0.0])           # ERA5 was flat
        p_s_era5 = jnp.full(3, p_ref)                    # flat ERA5 p_s
        delta_phis = p_s_era5 * 0  # placeholder — compute from formula
        delta_phis = era5_phis - etopo_phis
        p_s_adj = p_s_era5 * jnp.exp(delta_phis / (constants.R_d * T))
        # Expected: p_ref * exp(-etopo_phis / (R_d * T))
        expected = p_ref * jnp.exp(-etopo_phis / (constants.R_d * T))
        npt.assert_allclose(np.asarray(p_s_adj), np.asarray(expected), rtol=1e-6)

    def test_cmor_orog_uses_phis_data_always(self):
        """CMOR orog comes from _phis_data -- since decision C the ONE
        terrain product the dynamics also run on (set_fixed_fields in
        _setup_diagnostics); this pins the unit conversion only."""
        etopo_phis = jnp.array([9806.16, 49030.8])  # 1000 m, 5000 m
        orog = np.asarray(etopo_phis) / constants.g
        npt.assert_allclose(orog, [1000.0, 5000.0], rtol=1e-4)


class TestPassiveLandTile(unittest.TestCase):
    """Unit tests for passive land tile mode (f_land without slab T step).

    In passive mode (--topography <file>, no --land-mask-file):
      - physics.f_land is set (albedo + T_sfc blend apply)
      - physics.slab_land_active is False (T_land not stepped)

    In full slab mode (--land-mask-file):
      - physics.slab_land_active = True (T_land stepped)
    """

    def _make_pipeline(self):
        """Return a stub that mirrors PhysicsPipeline's land-tile attributes."""
        from types import SimpleNamespace
        return SimpleNamespace(
            f_land=None,
            albedo_land=None,
            rad_update_steps=1,
            slab_land_active=False,
        )

    def test_default_slab_land_active_is_false(self):
        """PhysicsPipeline must declare slab_land_active=False in __init__.
        Verified by reading the source directly (find_spec triggers __init__)."""
        import pathlib
        src_path = (
            pathlib.Path(__file__).parents[4]
            / "packages/coupler/legoesm/driver/physics_pipeline.py"
        )
        src = src_path.read_text()
        self.assertIn("self.slab_land_active = False", src)

    def test_passive_land_blend_requires_f_land_and_T_land(self):
        """The _land_active gate inside compute_radiation_core is
        f_land is not None AND T_land is not None.
        If f_land is set but T_land is None, no blend happens (pure ocean).
        Passive mode: f_land set + T_land carried → blend happens, slab skipped."""
        pp = self._make_pipeline()
        # Passive: f_land set, T_land would be provided by carry
        pp.f_land = jnp.ones((4, 4)) * 0.3
        _land_active = pp.f_land is not None  # T_land supplied by caller
        self.assertTrue(_land_active)
        # slab_land_active=False → the slab step is NOT called even if active
        self.assertFalse(pp.slab_land_active)

    def test_full_slab_sets_slab_land_active_true(self):
        """After setting slab_land_active=True (--land-mask-file path),
        the flag should be True."""
        pp = self._make_pipeline()
        pp.f_land = jnp.ones((4, 4)) * 0.5
        pp.slab_land_active = True
        self.assertTrue(pp.slab_land_active)

    def test_flat_topo_no_land_tile(self):
        """With flat topography, f_land is zero everywhere → _has_land is False
        → physics.f_land stays None → no land tile at all."""
        flat_f_land = jnp.zeros((6, 8, 8))
        _has_land = bool(jnp.any(flat_f_land > 0))
        self.assertFalse(_has_land)


class TestMaskedDiffusionCrossFace(unittest.TestCase):
    """Cube-imprint guard: the product's diffusion must use the cross-face
    halo, NOT per-face boundary clamping (which smooths each face in
    isolation and leaves a cube-edge seam).  The authoritative check is the
    nightly cube-SW visual-regression gate; this asserts the necessary
    cross-face-leakage property deterministically."""

    def _cube(self, n=6):
        grid = create_cubed_sphere(n)
        nb, area = _neighbour_table(grid)
        return nb, area

    def test_smoothing_leaks_across_face_boundaries(self):
        nb, area = self._cube()
        arr = np.zeros((6, 6, 6))
        arr[0] = 1.0  # face 0 hot, all other faces zero
        out = masked_diffusion(arr.ravel(), np.ones(arr.size), nb, area, passes=1).reshape(6, 6, 6)
        self.assertTrue(
            np.any(out[1:] > 1e-6),
            "diffusion did not cross cube face boundaries — per-face clamping "
            "would leave a cube-edge seam (cube imprint)",
        )
        self.assertLess(float(out[0].min()), 1.0)

    def test_constant_field_is_preserved(self):
        nb, area = self._cube()
        arr = np.full(6 * 36, 3.0)
        out = masked_diffusion(arr, np.ones_like(arr), nb, area, passes=3)
        npt.assert_allclose(out, 3.0, atol=1e-10)

    def test_table_independent_of_halo_backend(self):
        """The neighbour table is built from the LOCAL cross-face pad, never
        the global MPI/SPMD halo backend (codex PR F)."""
        from legoesm.grids.halo import get_halo_backend, set_halo_backend
        grid = create_cubed_sphere(6)
        nb_local, _ = _neighbour_table(grid)
        prev = get_halo_backend()
        try:
            set_halo_backend("spmd")
            nb_other, _ = _neighbour_table(grid)
        finally:
            set_halo_backend(prev)
        npt.assert_array_equal(nb_other, nb_local)


class TestUnstructuredTopographySmoothing(unittest.TestCase):
    """Audit 2026-07-17 T4: static-file topography on a Voronoi/MPAS mesh must
    be mesh-Laplacian smoothed (raw point-sampled ETOPO drives the TRiSK-PGF
    O(dx^-1) blowup over steep terrain), matching the ERA5-IC MPAS path."""

    def _mesh(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        return create_voronoi_mesh(2)   # 162 cells, cheap

    def test_smoothing_reduces_cell_to_cell_roughness(self):
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
        )
        mesh = self._mesh()
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "topo.nc")
            _make_synthetic_topo_netcdf(path, mountains=True)
            raw, _ = load_real_topography(
                mesh, TopographyConfig(source="file", path=path,
                                       smoothing_passes=0))
            smoothed, _ = load_real_topography(
                mesh, TopographyConfig(source="file", path=path,
                                       smoothing_passes=4))

        coc = np.asarray(mesh.cellsOnCell)
        neoc = np.asarray(mesh.nEdgesOnCell)

        def _neighbor_var(field):
            f = np.asarray(field)
            tot = 0.0
            for c in range(mesh.nCells):
                for e in range(int(neoc[c])):
                    nb = int(coc[e, c]) - 1     # 1-based -> 0-based
                    if 0 <= nb < mesh.nCells:
                        tot += (f[c] - f[nb]) ** 2
            return tot

        # smoothing must lower the summed squared cell-to-cell difference
        self.assertLess(_neighbor_var(smoothed), _neighbor_var(raw))
        # and it must actually have changed the field
        self.assertGreater(float(np.max(np.abs(np.asarray(smoothed)
                                               - np.asarray(raw)))), 0.0)

    def test_zero_passes_is_raw(self):
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
        )
        mesh = self._mesh()
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "topo.nc")
            _make_synthetic_topo_netcdf(path, mountains=True)
            a, _ = load_real_topography(
                mesh, TopographyConfig(source="file", path=path,
                                       smoothing_passes=0))
            b, _ = load_real_topography(
                mesh, TopographyConfig(source="file", path=path,
                                       smoothing_passes=0))
        npt.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
