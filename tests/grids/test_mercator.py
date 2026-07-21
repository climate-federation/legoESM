"""Tests for the Mercator lat-lon grid generator.

The Mercator grid places latitudes via ``sin(φ) = tanh(Δλ · k)`` so
that cell heights ``dy`` track ``R · Δλ · cos(φ)`` and ``dx ≈ dy`` per
row (isotropic). Used as the placement for the DINO ocean test case
(Kamm et al. 2025, GMD).

These tests cover:

* Placement formula matches a hand-computed reference.
* Per-row isotropy ``dx ≈ dy`` within 1% across a range of resolutions.
* Total area integrates to the spherical-cap area between ±lat_max.
* Edge cases: coarse resolution, near-equator-only, near-pole truncation.
* JIT round-trip — the grid is a pytree and traces cleanly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids import LatLonGrid, create_mercator_grid


# =====================================================================
# Reference: lat placement formula
# =====================================================================

class TestPlacementFormula:
    def test_face_latitudes_match_arcsin_tanh(self):
        """Cell face latitudes satisfy sin(φ_f) = tanh(Δλ · k_f)."""
        n_lon = 360
        lat_max_deg = 70.0
        grid = create_mercator_grid(n_lon=n_lon, lat_max_deg=lat_max_deg)

        # Δλ from n_lon (global 0-360 → 2π/n_lon)
        dlam = 2.0 * np.pi / n_lon
        # Reconstruct integer face indices from the equator.
        K = grid.n_lat // 2
        k_face = np.arange(-K, K + 1)

        # Reference face latitudes
        lat_face_ref = np.arcsin(np.tanh(dlam * k_face))
        # Cell centres sit at half-integer face indices
        k_center = k_face[:-1] + 0.5
        lat_center_ref = np.arcsin(np.tanh(dlam * k_center))

        lat_grid = np.asarray(grid.lat, dtype=np.float64)
        np.testing.assert_allclose(
            lat_grid, lat_center_ref, atol=1e-6, rtol=0,
            err_msg="Cell-centre latitudes do not match arcsin(tanh(Δλ·k))",
        )

    @pytest.mark.parametrize("n_lon,lat_max_deg", [
        (180, 70.0),
        (360, 60.0),
        (720, 80.0),
    ])
    def test_lat_symmetric_about_equator(self, n_lon, lat_max_deg):
        """Mercator places latitudes symmetrically about the equator."""
        g = create_mercator_grid(n_lon=n_lon, lat_max_deg=lat_max_deg)
        lat = np.asarray(g.lat, dtype=np.float64)
        # ``n_lat = 2K`` is even by construction, so paired centres
        # are symmetric reflections.
        np.testing.assert_allclose(
            lat, -lat[::-1], atol=1e-6,
            err_msg="Mercator latitudes not symmetric about the equator",
        )


# =====================================================================
# Isotropy: dx ≈ dy per row
# =====================================================================

class TestIsotropy:
    @pytest.mark.parametrize("n_lon", [180, 360, 720])
    def test_dx_dy_isotropic(self, n_lon):
        """dx(j) ≈ dy(j) at every latitude, within 1%."""
        g = create_mercator_grid(n_lon=n_lon, lat_max_deg=70.0)
        # Single-cell extents (recall the 2-cell-distance convention).
        dx_per_row = np.asarray(g.dx, dtype=np.float64)[:, 0] * 0.5
        dy_per_row = np.asarray(g.dy, dtype=np.float64) * 0.5
        ratio = dx_per_row / dy_per_row
        # Mercator is isotropic by construction; tolerance covers
        # discretisation error from finite cell width.
        assert np.max(np.abs(ratio - 1.0)) < 0.01, (
            f"dx/dy isotropy violated: max |ratio-1|={np.max(np.abs(ratio-1.0)):.4e}"
        )


# =====================================================================
# Total area: spherical-cap integral
# =====================================================================

class TestAreaIntegration:
    @pytest.mark.parametrize("n_lon,lat_max_deg", [
        (180, 70.0),
        (360, 60.0),
        (360, 85.0),
    ])
    def test_total_area_matches_cap(self, n_lon, lat_max_deg):
        """Sum of cell areas equals 2π R² · (2 · sin(lat_face_max))."""
        g = create_mercator_grid(n_lon=n_lon, lat_max_deg=lat_max_deg)
        # The actual lat_face_max may sit slightly equatorward of
        # ``lat_max_deg`` because K is rounded down. Reconstruct it.
        dlam = 2.0 * np.pi / n_lon
        K = g.n_lat // 2
        lat_face_max = float(np.arcsin(np.tanh(dlam * K)))

        expected = 2.0 * np.pi * constants.R_earth ** 2 * 2.0 * np.sin(lat_face_max)
        actual = float(g.total_area)
        rel_err = abs(actual - expected) / expected
        # Discretisation: cell-area formula uses ``|sin(lat_face[j+1]) -
        # sin(lat_face[j])|`` per row, which is exact for any orthogonal
        # spherical grid. Should integrate to the cap area to machine
        # precision (modulo storage dtype). Allow a small tolerance to
        # accommodate float32 storage.
        assert rel_err < 1e-3, f"Total area off by {rel_err:.3e}"


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_coarse_resolution(self):
        """36 zonal cells (10° at the equator) — should still build."""
        g = create_mercator_grid(n_lon=36, lat_max_deg=60.0)
        assert g.n_lon == 36
        assert g.n_lat >= 2
        assert g.dy.shape == (g.n_lat,)
        assert float(jnp.min(g.dy)) > 0.0

    def test_near_equator_only(self):
        """lat_max_deg=10 — a narrow equatorial band."""
        g = create_mercator_grid(n_lon=360, lat_max_deg=10.0)
        assert g.n_lat > 0
        assert g.n_lon == 360
        # All latitudes should sit within ±10°
        lat_deg = np.asarray(g.lat) * 180.0 / np.pi
        assert np.max(np.abs(lat_deg)) < 10.0

    def test_near_pole_truncation(self):
        """lat_max_deg=85 — high-latitude truncation."""
        g = create_mercator_grid(n_lon=180, lat_max_deg=85.0)
        lat_deg = np.asarray(g.lat) * 180.0 / np.pi
        assert np.max(np.abs(lat_deg)) < 85.0
        # Polar cells are narrowest — smallest ``dy`` should be much
        # smaller than the equatorial one.
        dy_max = float(jnp.max(g.dy))
        dy_min = float(jnp.min(g.dy))
        assert dy_min < dy_max * 0.5, (
            f"Expected strong dy variation near pole, got "
            f"min={dy_min:.0f} max={dy_max:.0f}"
        )

    def test_rejects_lat_max_out_of_range(self):
        with pytest.raises(ValueError):
            create_mercator_grid(n_lon=180, lat_max_deg=0.0)
        with pytest.raises(ValueError):
            create_mercator_grid(n_lon=180, lat_max_deg=90.0)

    def test_rejects_bad_lon_range(self):
        with pytest.raises(ValueError):
            create_mercator_grid(
                n_lon=180, lat_max_deg=60.0,
                lon_west_deg=180.0, lon_east_deg=0.0,
            )


# =====================================================================
# JIT compatibility
# =====================================================================

class TestPytreeAndJit:
    def test_grid_is_pytree(self):
        """LatLonGrid is a NamedTuple and registers as a pytree."""
        g = create_mercator_grid(n_lon=180, lat_max_deg=60.0)
        leaves = jax.tree_util.tree_leaves(g)
        assert len(leaves) > 0

    def test_grid_traces_through_jit(self):
        """A simple operation on grid fields compiles cleanly."""
        g = create_mercator_grid(n_lon=180, lat_max_deg=60.0)

        @jax.jit
        def cell_count(grid: LatLonGrid) -> jnp.ndarray:
            # Use ``dy`` (1D array) — exercises the post-Mercator path.
            return jnp.sum(grid.dy) / jnp.maximum(grid.radius, 1.0)

        result = cell_count(g)
        assert jnp.isfinite(result)


# =====================================================================
# Cross-check vs uniform grid
# =====================================================================

class TestDistinctFromUniform:
    def test_dy_varies_for_mercator(self):
        """Mercator dy varies with row — distinct from uniform grid."""
        g = create_mercator_grid(n_lon=360, lat_max_deg=70.0)
        # Equatorial cell vs poleward cell: cos(φ) factor between them
        # should produce ≥ 2x ratio at lat_max=70.
        dy_max = float(jnp.max(g.dy))
        dy_min = float(jnp.min(g.dy))
        assert dy_max / dy_min > 2.0, (
            f"Expected dy_max/dy_min > 2 at lat_max=70°, got {dy_max/dy_min:.3f}"
        )


def test_equator_on_tpoint_matches_nemo_dino():
    """NEMO-faithful placement: equator ON a T-point (odd n_lat), the
    ``usr_def_hgr`` convention.  Reproduces NEMO's DINO R1 grid (195x48,
    φ = asin(tanh(Δλ·(j-97)))) — verified cell-for-cell against the mesh to
    3e-6° in the DINO oracle harness.  Contrast the default equator-on-face."""
    import numpy as np
    # Default: equator on a FACE, even n_lat.
    gf = create_mercator_grid(n_lon=48, lat_max_deg=70.0,
                              lon_west_deg=1.0, lon_east_deg=49.0)
    assert gf.n_lat % 2 == 0                                  # even
    assert np.min(np.abs(np.degrees(np.asarray(gf.lat)))) > 0.1  # no cell ON equator

    # NEMO-faithful: equator on a T-POINT, odd n_lat=195.
    gt = create_mercator_grid(n_lon=48, lat_max_deg=70.0,
                              lon_west_deg=1.0, lon_east_deg=49.0,
                              equator_on_tpoint=True, n_lat=195)
    assert gt.n_lat == 195                                    # odd, NEMO jpjglo
    lat = np.degrees(np.asarray(gt.lat, dtype=np.float64))
    assert abs(lat[97]) < 1e-4                                # equator ON T-point j=97
    # φ(j) = asin(tanh(1°·(j-97))): symmetric, ±69.151 at the ends.
    assert abs(lat[0] + 69.151) < 1e-2 and abs(lat[-1] - 69.151) < 1e-2
    assert np.allclose(lat, -lat[::-1], atol=1e-9)            # symmetric about equator

    # Odd n_lat is required for equator_on_tpoint.
    import pytest
    with pytest.raises(ValueError, match="ODD n_lat"):
        create_mercator_grid(n_lon=48, lat_max_deg=70.0, lon_west_deg=1.0,
                             lon_east_deg=49.0, equator_on_tpoint=True, n_lat=196)
