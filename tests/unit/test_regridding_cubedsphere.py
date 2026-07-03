"""Tests for face-aware cubed-sphere → lat-lon regridding.

Verifies that the face-aware bilinear interpolator in
``legoesm.grids.regridding`` produces smooth lat-lon fields
without cube-edge artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.regridding import (
    CubedSphereToLatLonWeights,
    apply_cubedsphere_to_latlon,
    apply_cubedsphere_to_latlon_3d,
    compute_cubedsphere_to_latlon_weights,
    compute_latlon_to_cs_weights,
    regrid_scalar,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TILE = 16  # Small grid for fast tests


def _make_grid_and_weights(n: int = N_TILE, n_lon: int = 360, n_lat: int = 181):
    grid = create_cubed_sphere(n)
    weights = compute_cubedsphere_to_latlon_weights(n, n_lon=n_lon, n_lat=n_lat)
    return grid, weights


def _edge_distance(weights: CubedSphereToLatLonWeights) -> np.ndarray:
    """Distance (in fractional cells) from each target point to the nearest
    face boundary.  Small values indicate points near a cube edge.

    Indices in weights are in padded coordinates (interior at 1..n),
    so the face boundary is at padded index 0.5 and n+0.5.
    """
    n = weights.n
    # Fractional padded index
    fi = weights.i0.astype(np.float64) + weights.wi
    fj = weights.j0.astype(np.float64) + weights.wj
    # Interior spans [1, n] in padded coords; boundary is at 0.5 and n+0.5.
    di = np.minimum(fi - 0.5, (n + 0.5) - fi)
    dj = np.minimum(fj - 0.5, (n + 0.5) - fj)
    return np.minimum(di, dj)


# ---------------------------------------------------------------------------
# Test: zonally symmetric field stays longitude-invariant
# ---------------------------------------------------------------------------

class TestZonalSymmetry:
    """Regrid u = cos(lat) from cubed-sphere to lat-lon and verify
    near-perfect zonal symmetry on the output grid.
    """

    @pytest.fixture(scope="class")
    def regridded_cos_lat(self):
        grid, weights = _make_grid_and_weights()
        lat = np.asarray(grid.lat, dtype=np.float64)  # (6, n, n) radians
        field = np.cos(lat)
        ll = apply_cubedsphere_to_latlon(field, weights)
        return ll, weights

    def test_max_zonal_std_small(self, regridded_cos_lat):
        """Standard deviation across longitudes at each latitude should
        be tiny compared to the field amplitude.
        """
        ll, _w = regridded_cos_lat
        # Exclude polar rows (|lat| > 85°) where cos(lat) ~ 0
        lat_cent = _w.lat_cent
        mask = np.abs(lat_cent) < 85.0
        zonal_std = np.std(ll[mask, :], axis=1)
        max_std = np.max(zonal_std)
        assert max_std < 0.02, (
            f"Zonal std of cos(lat) after remap = {max_std:.4f} "
            f"(expected < 0.02)"
        )

    def test_mean_matches_analytic(self, regridded_cos_lat):
        """The zonal mean should closely track the analytic cos(lat)."""
        ll, _w = regridded_cos_lat
        lat_cent = _w.lat_cent
        zonal_mean = np.mean(ll, axis=1)
        analytic = np.cos(np.deg2rad(lat_cent))
        err = np.max(np.abs(zonal_mean - analytic))
        # C16 has large cells (~11 deg); bilinear interpolation error
        # is O(dalpha^2) ~ 0.01 at face centers but larger near corners.
        assert err < 0.10, (
            f"Max error in zonal mean of cos(lat) = {err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: cube-edge artifact ratio
# ---------------------------------------------------------------------------

class TestCubeEdgeArtifact:
    """Verify that the remapped error near cube edges is comparable
    to the interior — the amplification ratio should stay near 1
    and clearly below 2.

    The metric compares RMS interpolation error (vs. analytic cos(lat))
    for target points near face boundaries against those in the face
    interior.  A face-mixing remap (old KD-tree) has a ratio >> 1
    because face-boundary mixing inflates edge errors; the face-aware
    bilinear remap should give a ratio near 1.
    """

    @pytest.fixture(scope="class")
    def edge_interior_ratio(self):
        grid, weights = _make_grid_and_weights()
        lat = np.asarray(grid.lat, dtype=np.float64)
        field = np.cos(lat)
        ll = apply_cubedsphere_to_latlon(field, weights)

        # Analytic reference on the lat-lon grid
        lat_rad = np.deg2rad(weights.lat_cent)
        analytic = np.cos(lat_rad)[:, None] * np.ones((1, weights.n_lon))

        error = np.abs(ll - analytic)
        dist = _edge_distance(weights).reshape(weights.n_lat, weights.n_lon)

        # "Edge" = within 1.5 cells of face boundary
        edge_mask = dist < 1.5
        interior_mask = dist >= 3.0

        # Exclude polar regions
        lat_abs = np.abs(weights.lat_cent)
        row_ok = lat_abs < 80.0
        edge_mask = edge_mask & row_ok[:, None]
        interior_mask = interior_mask & row_ok[:, None]

        rms_edge = np.sqrt(np.mean(error[edge_mask] ** 2)) if edge_mask.any() else 0.0
        rms_interior = np.sqrt(np.mean(error[interior_mask] ** 2)) if interior_mask.any() else 1e-15

        ratio = rms_edge / max(rms_interior, 1e-15)
        return ratio

    def test_ratio_below_2(self, edge_interior_ratio):
        assert edge_interior_ratio < 2.0, (
            f"Edge/interior error ratio = {edge_interior_ratio:.2f} (want < 2)"
        )

    def test_ratio_near_1(self, edge_interior_ratio):
        """Soft check: ratio should be in a reasonable neighbourhood of 1."""
        assert edge_interior_ratio < 1.5, (
            f"Edge/interior error ratio = {edge_interior_ratio:.2f} (ideal ~ 1)"
        )


# ---------------------------------------------------------------------------
# Test: 3D remap path
# ---------------------------------------------------------------------------

class TestRegrid3D:
    """Cover the 3-D remap path (6, n, n, nlev) → (n_lat, n_lon, nlev)."""

    def test_3d_zonal_field(self):
        grid, weights = _make_grid_and_weights()
        lat = np.asarray(grid.lat, dtype=np.float64)
        nlev = 3
        # Create a field that varies smoothly with latitude and level
        field_3d = np.stack([np.cos(lat) * (k + 1) for k in range(nlev)], axis=-1)
        ll3d = apply_cubedsphere_to_latlon_3d(field_3d, weights)

        assert ll3d.shape == (weights.n_lat, weights.n_lon, nlev)

        for k in range(nlev):
            # Each level should be nearly zonal
            lat_ok = np.abs(weights.lat_cent) < 85.0
            zonal_std = np.std(ll3d[lat_ok, :, k], axis=1)
            assert np.max(zonal_std) < 0.02 * (k + 1), (
                f"Level {k}: max zonal std = {np.max(zonal_std):.4f}"
            )

    def test_3d_flat_input(self):
        """Accept flattened (6*n*n, nlev) input."""
        grid, weights = _make_grid_and_weights()
        lat = np.asarray(grid.lat, dtype=np.float64)
        nlev = 2
        field_3d = np.stack([np.cos(lat) * (k + 1) for k in range(nlev)], axis=-1)
        flat = field_3d.reshape(-1, nlev)
        ll3d = apply_cubedsphere_to_latlon_3d(flat, weights)
        assert ll3d.shape == (weights.n_lat, weights.n_lon, nlev)


# ---------------------------------------------------------------------------
# Test: uniform field is preserved exactly
# ---------------------------------------------------------------------------

def test_uniform_field_exact():
    """A uniform field should remap to exactly the same constant."""
    _grid, weights = _make_grid_and_weights()
    field = np.full((6, N_TILE, N_TILE), 42.0)
    ll = apply_cubedsphere_to_latlon(field, weights)
    np.testing.assert_allclose(ll, 42.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: weights structure is valid
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test: REGULAR lat-lon -> cubed-sphere weights (compute_latlon_to_cs_weights)
# ---------------------------------------------------------------------------

class TestLatLonToCubedSphere:
    """Regression for the ERA5 -> cubed-sphere regrid bug (iter 109).

    ``era5_to_cubedsphere_carry`` previously built the KD-tree weights from a
    GAUSSIAN PROXY of the (uniform lat-lon) ERA5 grid.  The proxy's quadrature
    latitudes do not coincide with a uniform lat-lon grid and its latitude COUNT
    generally differs, so the flat ``src_indices`` gathered the WRONG ERA5 cells
    (pulling near-antipodal latitudes — a ``T = lat`` field showed max ~154 deg
    error).  ``compute_latlon_to_cs_weights`` builds weights from the ACTUAL
    source nodes; the same field now reproduces each CS cell's latitude to within
    the grid-resolution + IDW-smoothing residual (a couple of degrees).
    """

    @staticmethod
    def _uniform_latlon_deg(n_lat: int = 73, n_lon: int = 144):
        # ERA5-like uniform grid: lat 90 -> -90 (descending), lon 0..360 (excl).
        lat_deg = np.linspace(90.0, -90.0, n_lat)
        lon_deg = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        return lat_deg, lon_deg

    def test_temperature_equals_lat_reproduces_cell_latitude(self):
        """A field equal to source latitude regrids to each CS cell's own
        latitude — the exact check that exposed the proxy bug (was ~154 deg)."""
        grid = create_cubed_sphere(8)
        lat_deg, lon_deg = self._uniform_latlon_deg()
        weights = compute_latlon_to_cs_weights(
            np.deg2rad(lat_deg), np.deg2rad(lon_deg), grid
        )
        # Source field T[i, j] = lat_deg[i] (C-order flatten: lat slowest).
        field = np.broadcast_to(
            lat_deg[:, None], (lat_deg.size, lon_deg.size)
        ).astype(np.float32)
        out = np.asarray(regrid_scalar(field, weights))   # (6, n, n)
        cell_lat_deg = np.rad2deg(np.asarray(grid.lat))
        err = np.abs(out - cell_lat_deg)
        # Resolution (C8 cells ~11 deg) + IDW smoothing residual only; the proxy
        # bug produced max ~154 deg / mean ~69 deg here.
        assert err.max() < 6.0, f"max |T_cs - cell_lat| = {err.max():.2f} deg"
        assert err.mean() < 3.0, f"mean = {err.mean():.2f} deg"

    def test_uniform_field_preserved(self):
        """A constant lat-lon field maps to the same constant on the CS grid
        (IDW weights sum to 1 per target)."""
        grid = create_cubed_sphere(8)
        lat_deg, lon_deg = self._uniform_latlon_deg()
        weights = compute_latlon_to_cs_weights(
            np.deg2rad(lat_deg), np.deg2rad(lon_deg), grid
        )
        field = np.full((lat_deg.size, lon_deg.size), 42.0, dtype=np.float32)
        out = np.asarray(regrid_scalar(field, weights))
        np.testing.assert_allclose(out, 42.0, rtol=1e-5)

    def test_weights_structure(self):
        """Indices reference the SOURCE grid (not a proxy of different size),
        weights are convex, target matches the CS shape."""
        grid = create_cubed_sphere(8)
        lat_deg, lon_deg = self._uniform_latlon_deg(n_lat=37, n_lon=72)
        weights = compute_latlon_to_cs_weights(
            np.deg2rad(lat_deg), np.deg2rad(lon_deg), grid
        )
        src_size = lat_deg.size * lon_deg.size
        assert weights.src_flat_size == src_size
        assert int(np.asarray(weights.src_indices).max()) < src_size
        assert int(np.asarray(weights.src_indices).min()) >= 0
        assert weights.target_shape == tuple(int(s) for s in grid.lat.shape)
        w = np.asarray(weights.weights)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-6)
        assert (w >= 0.0).all()


def test_weights_valid():
    """Sanity-check weight indices and values."""
    _grid, weights = _make_grid_and_weights()
    assert weights.face.min() >= 0
    assert weights.face.max() <= 5
    # Indices are in padded coordinates: 0..n for i0/j0, 1..n+1 for i1/j1.
    assert weights.i0.min() >= 0
    assert weights.i0.max() <= N_TILE
    assert weights.j0.min() >= 0
    assert weights.j0.max() <= N_TILE
    assert np.all(weights.wi >= 0.0)
    assert np.all(weights.wi <= 1.0 + 1e-12)
    assert np.all(weights.wj >= 0.0)
    assert np.all(weights.wj <= 1.0 + 1e-12)
    assert len(weights.lon_cent) == weights.n_lon
    assert len(weights.lat_cent) == weights.n_lat
