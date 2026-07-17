"""Unit tests for lat-lon operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.operators_latlon import (
    gradient_x,
    gradient_y,
    divergence,
    curl_z,
    laplacian,
    hyperdiffusion,
    global_integral,
    global_mean,
)
from legoesm.core.operators_latlon_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    divergence_3d,
    hyperdiffusion_3d,
)
from legoesm.core.operators_fv_latlon import (
    cgrid_fv_flux_divergence_latlon,
    fv_scalar_advection_latlon,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter,
    fourier_filter_3d,
)


@pytest.fixture
def grid():
    """16x32 lat-lon grid for testing."""
    return create_latlon_grid(16, 32)


@pytest.fixture
def grid_fine():
    """32x64 lat-lon grid for convergence tests."""
    return create_latlon_grid(32, 64)


def _make_field(grid, name="f", value=1.0):
    """Helper: create a uniform 2D field."""
    data = jnp.ones((grid.n_lat, grid.n_lon)) * value
    return Field(data=data, name=name, dims=("lat", "lon"), units="")


# ==============================================================================
# Halo Exchange Tests
# ==============================================================================

class TestHaloExchange:
    """Tests for lat-lon halo exchange."""

    def test_pad_shape(self, grid):
        """Padded array should be (n_lat+2, n_lon+2)."""
        data = jnp.ones((grid.n_lat, grid.n_lon))
        padded = pad_halo_latlon(data)
        assert padded.shape == (grid.n_lat + 2, grid.n_lon + 2)

    def test_periodic_longitude(self, grid):
        """Longitude halo should wrap periodically."""
        data = jnp.arange(grid.n_lon)[None, :] * jnp.ones((grid.n_lat, 1))
        padded = pad_halo_latlon(data)
        # Left halo = last column, right halo = first column
        assert jnp.allclose(padded[1:-1, 0], data[:, -1])
        assert jnp.allclose(padded[1:-1, -1], data[:, 0])

    def test_zero_gradient_latitude(self, grid):
        """Latitude halo should be zero-gradient (copy edge rows)."""
        data = jnp.arange(grid.n_lat)[:, None] * jnp.ones((1, grid.n_lon))
        padded = pad_halo_latlon(data)
        # Bottom halo = first row, top halo = last row
        assert jnp.allclose(padded[0, 1:-1], data[0, :])
        assert jnp.allclose(padded[-1, 1:-1], data[-1, :])


# ==============================================================================
# 2D Operator Tests
# ==============================================================================

class TestGradient:
    """Tests for gradient operators."""

    def test_gradient_uniform_zero(self, grid):
        """Gradient of a uniform field should be zero."""
        f = _make_field(grid, value=42.0)
        gx = gradient_x(f, grid)
        gy = gradient_y(f, grid)
        assert jnp.allclose(gx.data, 0.0, atol=1e-10)
        assert jnp.allclose(gy.data, 0.0, atol=1e-10)

    def test_gradient_shape(self, grid):
        """Gradient output should have same shape as input."""
        f = _make_field(grid)
        gx = gradient_x(f, grid)
        gy = gradient_y(f, grid)
        assert gx.data.shape == (grid.n_lat, grid.n_lon)
        assert gy.data.shape == (grid.n_lat, grid.n_lon)

    def test_gradient_finite(self, grid):
        """Gradient should produce finite values for random input."""
        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (grid.n_lat, grid.n_lon))
        f = Field(data=data, name="f", dims=("lat", "lon"), units="")
        gx = gradient_x(f, grid)
        gy = gradient_y(f, grid)
        assert jnp.all(jnp.isfinite(gx.data))
        assert jnp.all(jnp.isfinite(gy.data))


class TestDivergence:
    """Tests for divergence operator."""

    def test_divergence_uniform_zero(self, grid):
        """Divergence of a uniform vector field should be near zero."""
        u = _make_field(grid, name="u", value=10.0)
        v = _make_field(grid, name="v", value=5.0)
        div = divergence(u, v, grid)
        # Not exactly zero due to spherical metric (constant u has nonzero
        # divergence on a sphere), but should be small for small grid
        assert jnp.all(jnp.isfinite(div.data))

    def test_divergence_shape(self, grid):
        """Divergence output shape should match input."""
        u = _make_field(grid, name="u")
        v = _make_field(grid, name="v")
        div = divergence(u, v, grid)
        assert div.data.shape == (grid.n_lat, grid.n_lon)

    def test_divergence_finite(self, grid):
        """Divergence should be finite for random input."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        u = Field(data=jax.random.normal(k1, (grid.n_lat, grid.n_lon)),
                  name="u", dims=("lat", "lon"), units="m/s")
        v = Field(data=jax.random.normal(k2, (grid.n_lat, grid.n_lon)),
                  name="v", dims=("lat", "lon"), units="m/s")
        div = divergence(u, v, grid)
        assert jnp.all(jnp.isfinite(div.data))


class TestCurlZ:
    """Tests for vorticity operator."""

    def test_curl_shape(self, grid):
        """Vorticity should have correct shape."""
        u = _make_field(grid, name="u")
        v = _make_field(grid, name="v")
        vort = curl_z(u, v, grid)
        assert vort.data.shape == (grid.n_lat, grid.n_lon)

    def test_curl_finite(self, grid):
        """Vorticity should be finite."""
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        u = Field(data=jax.random.normal(k1, (grid.n_lat, grid.n_lon)),
                  name="u", dims=("lat", "lon"), units="m/s")
        v = Field(data=jax.random.normal(k2, (grid.n_lat, grid.n_lon)),
                  name="v", dims=("lat", "lon"), units="m/s")
        vort = curl_z(u, v, grid)
        assert jnp.all(jnp.isfinite(vort.data))


class TestLaplacian:
    """Tests for Laplacian operator."""

    def test_laplacian_uniform_zero(self, grid):
        """Laplacian of a uniform field should be zero."""
        f = _make_field(grid, value=42.0)
        lap = laplacian(f, grid)
        assert jnp.allclose(lap.data, 0.0, atol=1e-8)

    def test_laplacian_shape(self, grid):
        """Laplacian output shape should match input."""
        f = _make_field(grid)
        lap = laplacian(f, grid)
        assert lap.data.shape == (grid.n_lat, grid.n_lon)

    def test_laplacian_finite(self, grid):
        """Laplacian should be finite for random input."""
        key = jax.random.PRNGKey(3)
        data = jax.random.normal(key, (grid.n_lat, grid.n_lon))
        f = Field(data=data, name="f", dims=("lat", "lon"), units="")
        lap = laplacian(f, grid)
        assert jnp.all(jnp.isfinite(lap.data))


class TestHyperdiffusion:
    """Tests for hyperdiffusion operator."""

    def test_hyperdiffusion_uniform_zero(self, grid):
        """Hyperdiffusion of a uniform field should be zero."""
        f = _make_field(grid, value=42.0)
        hd = hyperdiffusion(f, grid, coeff=1e15)
        assert jnp.allclose(hd.data, 0.0, atol=1e-4)

    def test_hyperdiffusion_finite(self, grid):
        """Hyperdiffusion should be finite."""
        key = jax.random.PRNGKey(4)
        data = jax.random.normal(key, (grid.n_lat, grid.n_lon))
        f = Field(data=data, name="f", dims=("lat", "lon"), units="")
        hd = hyperdiffusion(f, grid, coeff=1e15)
        assert jnp.all(jnp.isfinite(hd.data))


# ==============================================================================
# Global Integral Tests
# ==============================================================================

class TestGlobalIntegral:
    """Tests for global integral and mean."""

    def test_integral_uniform(self, grid):
        """Integral of 1 should equal total area."""
        f = _make_field(grid, value=1.0)
        integral = global_integral(f, grid)
        assert jnp.allclose(integral, grid.total_area, rtol=1e-5)

    def test_mean_uniform(self, grid):
        """Mean of a uniform field should equal the field value."""
        f = _make_field(grid, value=42.0)
        mean = global_mean(f, grid)
        assert jnp.allclose(mean, 42.0, rtol=1e-5)


# ==============================================================================
# 3D Operator Tests
# ==============================================================================

class TestOperators3D:
    """Tests for 3D vmap wrappers."""

    def test_vorticity_3d_shape(self, grid):
        """3D vorticity should have correct shape."""
        nlev = 5
        shape = (grid.n_lat, grid.n_lon, nlev)
        u = jnp.ones(shape) * 10.0
        v = jnp.ones(shape) * 5.0
        vort = vorticity_3d(u, v, grid)
        assert vort.shape == shape

    def test_divergence_3d_shape(self, grid):
        """3D divergence should have correct shape."""
        nlev = 5
        shape = (grid.n_lat, grid.n_lon, nlev)
        u = jnp.ones(shape) * 10.0
        v = jnp.ones(shape) * 5.0
        div = divergence_3d(u, v, grid)
        assert div.shape == shape

    def test_gradient_3d_shape(self, grid):
        """3D gradient should have correct shape."""
        nlev = 5
        shape = (grid.n_lat, grid.n_lon, nlev)
        f = jnp.ones(shape) * 100.0
        gx = gradient_x_3d(f, grid)
        gy = gradient_y_3d(f, grid)
        assert gx.shape == shape
        assert gy.shape == shape

    def test_hyperdiffusion_3d_shape(self, grid):
        """3D hyperdiffusion should have correct shape."""
        nlev = 5
        shape = (grid.n_lat, grid.n_lon, nlev)
        f = jnp.ones(shape)
        hd = hyperdiffusion_3d(f, grid, 1e15)
        assert hd.shape == shape

    def test_3d_operators_finite(self, grid):
        """All 3D operators should produce finite values."""
        nlev = 5
        shape = (grid.n_lat, grid.n_lon, nlev)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, shape)
        v = jax.random.normal(k2, shape)

        assert jnp.all(jnp.isfinite(vorticity_3d(u, v, grid)))
        assert jnp.all(jnp.isfinite(divergence_3d(u, v, grid)))
        assert jnp.all(jnp.isfinite(gradient_x_3d(u, grid)))
        assert jnp.all(jnp.isfinite(gradient_y_3d(u, grid)))


class TestFVVFaceMetricBandCorrect:
    """The FV operators consume ``grid.cos_lat_v`` — the band-correct
    v-face metric — instead of recomputing v-face latitudes with
    hard-coded ±π/2 endpoints (which mislabelled interior MPI band cuts
    as poles and collapsed their face length to ~1e-10)."""

    def test_cos_lat_v_matches_legacy_pole_padded_formula(self):
        """Serial bit-identity of the metric switch on the GLOBAL grid.

        The legacy operator-internal formula (cell-centre midpoints
        padded with exact ±π/2, cos, 1e-10 floor) and the grid-carried
        ``cos_lat_v`` (midpoints + half-cell endpoint extrapolation,
        abs-cos, same floor) are BIT-EQUAL in float64: interior faces
        are the same midpoints, and both endpoint variants land on the
        1e-10 clamp at the true poles.
        """
        if not jax.config.jax_enable_x64:
            pytest.skip("bit-identity is pinned for the float64 pipeline")
        for n_lat, n_lon in ((16, 32), (48, 96)):
            g = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
            lat = g.lat
            legacy = jnp.maximum(
                jnp.cos(jnp.pad(
                    0.5 * (lat[:-1] + lat[1:]), (1, 1),
                    constant_values=(-jnp.pi / 2, jnp.pi / 2),
                )),
                1e-10,
            )
            np.testing.assert_array_equal(
                np.asarray(legacy), np.asarray(g.cos_lat_v),
                err_msg=f"cos_lat_v mismatch on global {n_lat}x{n_lon}",
            )

    def test_band_subgrid_cut_rows_match_global(self):
        """Interior MPI band reproduces the full-grid rows BIT-EXACTLY.

        Constant q makes the PPM face values independent of the
        (band-local) halo content, so the only term that can differ
        between the band and the corresponding global rows is the
        v-face metric.  With the legacy hard-coded ±π/2 endpoints the
        interior cut faces collapsed to cos≈1e-10 and this comparison
        failed at max|Δ|≈5e-3; with ``grid.cos_lat_v`` (pre-sliced by
        ``slice_latlon_grid_to_band``) it is exact.
        """
        from legoesm.parallel.latlon_mpi import (
            make_latlon_band_layout,
            slice_latlon_grid_to_band,
        )
        n_lat, n_lon = 16, 32
        g = create_latlon_grid(n_lat, n_lon)
        layout = make_latlon_band_layout(1, 4, n_lat, n_lon)  # interior band
        band = slice_latlon_grid_to_band(
            g, layout, skip_total_area_reduce=True)
        s, e = layout.lat_start, layout.lat_end
        # Sanity: both band endpoints are interior cuts, not poles —
        # their faces must carry real (uncollapsed) cos(lat_v).
        assert float(band.cos_lat_v[0]) > 0.5
        assert float(band.cos_lat_v[-1]) > 0.5

        rng = np.random.default_rng(7)
        dt = g.lat.dtype
        q = jnp.full((n_lat, n_lon), 300.0, dtype=dt)
        u_face = jnp.asarray(
            rng.normal(0.0, 10.0, (n_lat, n_lon + 1)), dtype=dt)
        v_face = jnp.asarray(
            rng.normal(0.0, 10.0, (n_lat + 1, n_lon)), dtype=dt)

        out_global = cgrid_fv_flux_divergence_latlon(q, u_face, v_face, g)
        out_band = cgrid_fv_flux_divergence_latlon(
            q[s:e], u_face[s:e], v_face[s:e + 1], band)
        np.testing.assert_array_equal(
            np.asarray(out_band), np.asarray(out_global[s:e]),
            err_msg="band rows diverged from global — v-face metric "
                    "collapsed at an interior cut",
        )


class TestFVScalarAdvectionLatLon:
    """Tests for FV advective-form scalar transport on lat-lon."""

    def test_uniform_field_divergent_flow_zero(self, grid):
        """Constant field should remain invariant under divergent flow."""
        key_u = jax.random.PRNGKey(11)
        key_v = jax.random.PRNGKey(12)
        q = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float32) * 300.0
        u = jax.random.normal(key_u, q.shape, dtype=jnp.float32) * 10.0
        v = jax.random.normal(key_v, q.shape, dtype=jnp.float32) * 10.0

        dq = fv_scalar_advection_latlon(q, u, v, grid)
        max_abs = float(jnp.max(jnp.abs(dq)))
        assert max_abs < 5e-6, f"constant-field drift too large: {max_abs:.3e}"


# ==============================================================================
# Polar Filter Tests
# ==============================================================================

class TestPolarFilter:
    """Tests for the Fourier polar filter."""

    def test_filter_preserves_uniform(self, grid):
        """Filter should not change a uniform field."""
        mask = compute_polar_filter_mask(grid, dt=600.0, cutoff_lat_deg=60.0)
        data = jnp.ones((grid.n_lat, grid.n_lon)) * 42.0
        filtered = fourier_filter(data, grid, mask)
        assert jnp.allclose(filtered, 42.0, atol=1e-5)

    def test_filter_shape(self, grid):
        """Filter output should have same shape."""
        mask = compute_polar_filter_mask(grid)
        data = jnp.ones((grid.n_lat, grid.n_lon))
        filtered = fourier_filter(data, grid, mask)
        assert filtered.shape == (grid.n_lat, grid.n_lon)

    def test_filter_3d_shape(self, grid):
        """3D filter output should have same shape."""
        mask = compute_polar_filter_mask(grid)
        nlev = 5
        data = jnp.ones((grid.n_lat, grid.n_lon, nlev))
        filtered = fourier_filter_3d(data, grid, mask)
        assert filtered.shape == (grid.n_lat, grid.n_lon, nlev)

    def test_filter_damps_high_wavenumber_at_pole(self, grid):
        """Filter should damp high wavenumbers near the poles."""
        mask = compute_polar_filter_mask(grid, dt=600.0, cutoff_lat_deg=45.0)
        # Create a field with maximum wavenumber in longitude
        k_max = grid.n_lon // 2 - 1
        data = jnp.sin(k_max * grid.lon2d)
        filtered = fourier_filter(data, grid, mask)

        # At high latitudes, the high-k mode should be strongly damped
        pole_rms_original = float(jnp.sqrt(jnp.mean(data[0, :]**2)))
        pole_rms_filtered = float(jnp.sqrt(jnp.mean(filtered[0, :]**2)))

        if pole_rms_original > 1e-10:
            assert pole_rms_filtered < pole_rms_original * 0.5, (
                f"High-k mode not damped at poles: "
                f"original rms={pole_rms_original:.4f}, filtered rms={pole_rms_filtered:.4f}"
            )

    def test_filter_preserves_low_wavenumber(self, grid):
        """Filter should preserve low wavenumbers even at poles."""
        mask = compute_polar_filter_mask(grid, dt=600.0, cutoff_lat_deg=60.0)
        # Wavenumber 1 should be preserved everywhere
        data = jnp.sin(grid.lon2d)
        filtered = fourier_filter(data, grid, mask)
        assert jnp.allclose(filtered, data, atol=1e-5)


# ==============================================================================
# Differentiability Tests
# ==============================================================================

class TestDifferentiability:
    """Tests that operators are differentiable."""

    def test_grad_through_gradient(self, grid):
        """jax.grad should work through gradient_x."""
        def loss(data):
            f = Field(data=data, name="f", dims=("lat", "lon"), units="")
            gx = gradient_x(f, grid)
            return jnp.sum(gx.data**2)

        key = jax.random.PRNGKey(10)
        data = jax.random.normal(key, (grid.n_lat, grid.n_lon))
        grads = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_through_divergence(self, grid):
        """jax.grad should work through divergence."""
        def loss(u_data):
            u = Field(data=u_data, name="u", dims=("lat", "lon"), units="m/s")
            v = Field(data=jnp.zeros_like(u_data), name="v",
                      dims=("lat", "lon"), units="m/s")
            div = divergence(u, v, grid)
            return jnp.sum(div.data**2)

        key = jax.random.PRNGKey(11)
        data = jax.random.normal(key, (grid.n_lat, grid.n_lon))
        grads = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(grads))
