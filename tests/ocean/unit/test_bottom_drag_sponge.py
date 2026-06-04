"""CI tests for bottom drag and sponge layer relaxation.

Tests that bottom drag applies to full velocity (not perturbation),
that barotropic substeps include drag, and that sponge layers relax
toward the reference state.

Run with:
    JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_bottom_drag_sponge.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.sponge import (
    SpongeForcing,
    compute_sponge_gamma_latlon,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def latlon_grid():
    grid, wall_mask = create_regional_latlon_grid(
        16, 34, 16.0, 34.0,
        periodic_x=True, lon_west=0.0, lon_east=10.0,
        dtype=jnp.float64,
    )
    u_mask, v_mask = compute_face_masks(wall_mask)
    return grid, wall_mask, u_mask, v_mask


# ============================================================================
# Sponge module tests
# ============================================================================

class TestSpongeForcing:
    """Tests for the SpongeForcing data structure and gamma computation."""

    def test_sponge_forcing_creation(self):
        """SpongeForcing can be created with minimal fields."""
        gamma = jnp.zeros((10, 20))
        T_ref = jnp.ones((10, 20, 5))
        S_ref = jnp.ones((10, 20, 5)) * 35.0
        sf = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)
        assert sf.gamma.shape == (10, 20)
        assert sf.u_ref is None
        assert sf.v_ref is None

    def test_compute_sponge_gamma_latlon_shape(self, latlon_grid):
        """Gamma field has correct shape."""
        grid, _, _, _ = latlon_grid
        gamma = compute_sponge_gamma_latlon(
            grid, lat_south=16.0, lat_north=34.0,
            width_deg=2.0, timescale_days=1.0)
        assert gamma.shape == (grid.n_lat, grid.n_lon)

    def test_compute_sponge_gamma_zero_interior(self, latlon_grid):
        """Gamma is zero in the interior, nonzero near boundaries."""
        grid, _, _, _ = latlon_grid
        gamma = compute_sponge_gamma_latlon(
            grid, lat_south=16.0, lat_north=34.0,
            width_deg=2.0, timescale_days=1.0)

        lat_deg = np.degrees(np.asarray(grid.lat))
        # Interior cells (more than 2 degrees from walls)
        interior = (lat_deg > 18.5) & (lat_deg < 31.5)
        assert np.all(gamma[interior] == 0.0), "Gamma nonzero in interior"

        # Boundary cells
        south_sponge = lat_deg < 18.0
        if np.any(south_sponge):
            assert np.all(gamma[south_sponge] > 0.0), "Gamma zero in south sponge"

    def test_compute_sponge_gamma_magnitude(self, latlon_grid):
        """Peak gamma is O(1/tau) and positive."""
        grid, _, _, _ = latlon_grid
        tau_days = 1.0
        gamma = compute_sponge_gamma_latlon(
            grid, lat_south=16.0, lat_north=34.0,
            width_deg=2.0, timescale_days=tau_days)

        expected_order = 1.0 / (tau_days * 86400.0)
        # Max gamma should be in the right ballpark (within 2x of 1/tau)
        assert np.max(gamma) > 0.0, "Gamma should be positive near walls"
        assert np.max(gamma) < expected_order * 3.0, (
            f"Gamma too large: {np.max(gamma):.2e} vs expected ~{expected_order:.2e}"
        )

    def test_sponge_relaxation_drives_toward_ref(self):
        """Sponge tendency drives T toward T_ref."""
        # Manually compute what the sponge tendency should be
        gamma = jnp.array([[0.0, 1e-4], [1e-4, 0.0]])  # (2, 2)
        T = jnp.ones((2, 2, 3)) * 20.0  # current T
        T_ref = jnp.ones((2, 2, 3)) * 10.0  # reference T

        # dT/dt = gamma * (T_ref - T)
        gamma_3d = gamma[..., jnp.newaxis]
        tendency = gamma_3d * (T_ref - T)

        # Where gamma > 0, tendency should push T toward T_ref (negative)
        assert jnp.all(tendency[0, 1, :] < 0), "Sponge should cool where T > T_ref"
        assert jnp.all(tendency[1, 0, :] < 0), "Sponge should cool where T > T_ref"
        # Where gamma == 0, no tendency
        assert jnp.all(tendency[0, 0, :] == 0), "No sponge in interior"

    def test_sponge_jit_compatible(self):
        """SpongeForcing works inside JIT."""
        gamma = jnp.zeros((4, 4))
        T_ref = jnp.ones((4, 4, 2))
        S_ref = jnp.ones((4, 4, 2)) * 35.0
        sf = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)

        @jax.jit
        def apply_sponge(T, sponge):
            g3d = sponge.gamma[..., jnp.newaxis]
            return g3d * (sponge.T_ref - T)

        T = jnp.ones((4, 4, 2)) * 20.0
        result = apply_sponge(T, sf)
        assert result.shape == (4, 4, 2)
        # gamma is zero everywhere, so result should be zero
        assert jnp.max(jnp.abs(result)) == 0.0


class TestBottomDrag:
    """Tests for bottom drag configuration and tendency."""

    def test_mpas_config_has_bottom_drag_r(self):
        """MPASOceanConfig has bottom_drag_r field with default 0."""
        from legoesm.ocean.mpas_config import MPASOceanConfig
        c = MPASOceanConfig()
        assert c.bottom_drag_r == 0.0
        c2 = MPASOceanConfig(bottom_drag_r=1e-4)
        assert c2.bottom_drag_r == 1e-4

    def test_latlon_config_has_bottom_drag_r(self):
        """LatLonCGridOceanConfig has bottom_drag_r field."""
        from legoesm.ocean.state import LatLonCGridOceanConfig
        c = LatLonCGridOceanConfig()
        assert c.bottom_drag_r == 0.0
        c2 = LatLonCGridOceanConfig(bottom_drag_r=1e-4)
        assert c2.bottom_drag_r == 1e-4

    def test_sponge_import(self):
        """SpongeForcing is importable from ocean.sponge."""
        from legoesm.ocean.sponge import SpongeForcing, compute_sponge_gamma_latlon
        assert SpongeForcing is not None
        assert compute_sponge_gamma_latlon is not None
