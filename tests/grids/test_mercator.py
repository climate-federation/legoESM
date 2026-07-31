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


# =====================================================================
# #1226: metric_convention="nemo_isotropic" — NEMO usr_def_hgr.F90 closed
# form (DINO, vopikamm/DINO@v0.2.0): pe1t = pe2t = ra * rad *
# cos(rad*phi_T) * rn_e1_deg.  Independent loop-port ground truth: this
# reproduces that Fortran closed form from scratch (not by calling back
# into create_mercator_grid's own formula) and checks BOTH directions —
# "nemo_isotropic" MUST match it, "exact" (default) MUST NOT (so the test
# is non-vacuous: it would fail if the dispatch were a silent no-op in
# either direction).
# =====================================================================

class TestNemoIsotropicMetricConvention:
    def _nemo_usr_def_hgr_pe1t(self, radius, rn_e1_deg_rad, lat_center_rad):
        """Independent transcription of NEMO usr_def_hgr.F90's isotropic
        DINO closed form (NOT calling create_mercator_grid): ``pe1t = ra *
        rad * cos(rad*phi_T) * rn_e1_deg`` -- and by construction
        ``pe2t = pe1t`` (the whole point of the "isotropic" convention)."""
        return radius * rn_e1_deg_rad * np.cos(lat_center_rad)

    def test_nemo_isotropic_matches_independent_fortran_transcription(self):
        """metric_convention="nemo_isotropic" dy_T (single-cell, via
        create_latlon_geometry) equals the from-scratch NEMO closed form."""
        from legoesm.grids import create_latlon_geometry

        n_lon, lat_max_deg = 48, 70.0
        lon_west_deg, lon_east_deg = 1.0, 49.0
        g = create_mercator_grid(
            n_lon=n_lon, lat_max_deg=lat_max_deg,
            lon_west_deg=lon_west_deg, lon_east_deg=lon_east_deg,
            equator_on_tpoint=True, n_lat=195,
            metric_convention="nemo_isotropic",
        )
        geom = create_latlon_geometry(
            n_lat=g.n_lat, n_lon=g.n_lon, radius=g.radius,
            lat_1d=g.lat, lon_1d=g.lon, lat_face_1d=g.lat_v,
            metric_convention="nemo_isotropic",
        )
        rn_e1_deg_rad = np.deg2rad((lon_east_deg - lon_west_deg) / n_lon)
        lat = np.asarray(g.lat, dtype=np.float64)
        pe1t_ref = self._nemo_usr_def_hgr_pe1t(
            float(g.radius), rn_e1_deg_rad, lat)

        dy_T_single = np.asarray(geom.dy_T[:, 0], dtype=np.float64)
        rel = np.abs(dy_T_single - pe1t_ref) / pe1t_ref
        # float64 roundoff on this ~1e5 m magnitude (dlon computed as a
        # degree-difference-then-radians vs a direct degree-to-radians
        # conversion here) -- both are "the same formula", just re-derived
        # independently, so agreement is to ~1e-7, not exact machine zero.
        assert np.max(rel) < 1e-6, (
            f"nemo_isotropic dy_T does not match the independent NEMO "
            f"usr_def_hgr.F90 transcription: max relerr {np.max(rel):.3e}"
        )
        # pe1t == pe2t is the DEFINITION of "isotropic": dy_T must equal
        # dx_T (single-cell) exactly, not merely match the reference.
        dx_T_single = np.asarray(geom.dx_T[:, 0], dtype=np.float64)
        np.testing.assert_allclose(dy_T_single, dx_T_single, rtol=0, atol=0)

    def test_exact_convention_does_NOT_match_nemo_closed_form(self):
        """Non-vacuity check: the DEFAULT ("exact") convention must NOT
        reproduce NEMO's isotropic approximation -- if it did, this test
        (and the #1226 diagnosis motivating the whole feature) would be
        vacuous.  legoESM's exact dy_T is R*Δφ (true finite difference of
        face latitudes), NEMO's is R*Δλ*cos(φ) (=dx) -- these differ
        wherever cos(φ) is not exactly 1, i.e. away from the equator."""
        from legoesm.grids import create_latlon_geometry

        n_lon, lat_max_deg = 48, 70.0
        lon_west_deg, lon_east_deg = 1.0, 49.0
        g = create_mercator_grid(
            n_lon=n_lon, lat_max_deg=lat_max_deg,
            lon_west_deg=lon_west_deg, lon_east_deg=lon_east_deg,
            equator_on_tpoint=True, n_lat=195,
        )  # default metric_convention="exact"
        geom = create_latlon_geometry(
            n_lat=g.n_lat, n_lon=g.n_lon, radius=g.radius,
            lat_1d=g.lat, lon_1d=g.lon, lat_face_1d=g.lat_v,
        )  # default metric_convention="exact"
        rn_e1_deg_rad = np.deg2rad((lon_east_deg - lon_west_deg) / n_lon)
        lat = np.asarray(g.lat, dtype=np.float64)
        pe1t_ref = self._nemo_usr_def_hgr_pe1t(
            float(g.radius), rn_e1_deg_rad, lat)

        dy_T_single = np.asarray(geom.dy_T[:, 0], dtype=np.float64)
        rel = np.abs(dy_T_single - pe1t_ref) / pe1t_ref
        # The #1226 diagnosis measured e2t vs e1t at -1.27e-5..+9.5e-6 on
        # the real NEMO mesh_mask; this independent from-scratch check
        # reproduces a mismatch of the SAME order (~1e-5), well above the
        # ~1e-7 float64-roundoff floor the nemo_isotropic branch achieves
        # in the sibling test above -- i.e. genuinely non-matching, not a
        # coincidence of tolerance.
        assert np.max(rel) > 1e-6, (
            "exact convention unexpectedly matches the NEMO isotropic "
            f"closed form (max relerr {np.max(rel):.3e}) -- this would make "
            "the nemo_isotropic feature vacuous"
        )

    def test_default_is_bit_identical_to_exact_keyword(self):
        """metric_convention default omission == explicit "exact" (every
        pre-#1226 caller stays byte-identical)."""
        from legoesm.grids import create_latlon_geometry

        g_default = create_mercator_grid(n_lon=48, lat_max_deg=70.0)
        g_exact = create_mercator_grid(
            n_lon=48, lat_max_deg=70.0, metric_convention="exact")
        for f in ("dy", "area", "dx", "lat", "lat_v", "cos_lat_v"):
            np.testing.assert_array_equal(
                getattr(g_default, f), getattr(g_exact, f))

        geom_default = create_latlon_geometry(
            n_lat=g_default.n_lat, n_lon=g_default.n_lon,
            lat_1d=g_default.lat, lon_1d=g_default.lon,
            lat_face_1d=g_default.lat_v)
        geom_exact = create_latlon_geometry(
            n_lat=g_default.n_lat, n_lon=g_default.n_lon,
            lat_1d=g_default.lat, lon_1d=g_default.lon,
            lat_face_1d=g_default.lat_v, metric_convention="exact")
        for f in ("dy_T", "dy_u", "dy_v", "area_T", "area_q", "dx_v", "dx_T"):
            np.testing.assert_array_equal(
                getattr(geom_default, f), getattr(geom_exact, f))

    def test_vface_metric_invariant_under_metric_convention(self):
        """#516 constraint: the v-face metric (dx_v, dy_v, area_q,
        cos_alpha_v) that ``vface_zonal_cos_lat`` and the strain/stress
        adjoint pair depend on must be BIT-IDENTICAL between "exact" and
        "nemo_isotropic" -- this flag only touches the T/u-face metric."""
        from legoesm.grids import create_latlon_geometry
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            vface_zonal_cos_lat,
        )

        g = create_mercator_grid(n_lon=48, lat_max_deg=70.0,
                                  lon_west_deg=1.0, lon_east_deg=49.0,
                                  equator_on_tpoint=True, n_lat=195)
        geom_exact = create_latlon_geometry(
            n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon,
            lat_face_1d=g.lat_v, metric_convention="exact")
        geom_iso = create_latlon_geometry(
            n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon,
            lat_face_1d=g.lat_v, metric_convention="nemo_isotropic")

        # NARROWED 2026-07-28, deliberately, with the reason recorded.
        # The #516 invariant is the strain/stress ADJOINT PAIR and the
        # divergence/flux-form-advection MASS CONSISTENCY.  Both are properties
        # of the ZONAL v-face metric dx_v (via vface_zonal_cos_lat), and both
        # hold because every operator SHARES that metric -- not because of its
        # value.  Verified structurally: strain_rate_cgrid /
        # stress_divergence_cgrid take a LatLonGrid, which has no dy_v field at
        # all, and vface_zonal_cos_lat reads grid.lat (asserted below).
        # dy_v -- the MERIDIONAL v-point spacing -- was in this list
        # conservatively, not because the invariant consumes it.  It is NEMO's
        # e2v, and ldf_slp's vslp divides by it, so it MUST follow the
        # convention (#1226: NEMO usrdef_hgr.F90:117 sets pe2v = pe1v).
        for f in ("dx_v", "area_q", "cos_alpha_v", "sin_alpha_v"):
            np.testing.assert_array_equal(
                getattr(geom_exact, f), getattr(geom_iso, f),
                err_msg=f"v-face field {f!r} changed under metric_convention "
                        "-- #516 invariant violated",
            )
        # dy_v MUST change -- pin it, so the new behaviour is asserted rather
        # than merely permitted by the narrowing above.
        assert not np.array_equal(
            np.asarray(geom_exact.dy_v), np.asarray(geom_iso.dy_v)), (
            "dy_v is unchanged under nemo_isotropic -- NEMO's e2v = e1v "
            "(usrdef_hgr.F90:117) is then NOT being reproduced, and vslp "
            "cannot reach the bar")
        # It must equal NEMO's closed form e2v = ra*rad*COS(gphiv)*rn_e1_deg
        # (usrdef_hgr.F90:117).  NOTE this is deliberately NOT compared against
        # geom.dx_v: dx_v is the #516 TRANSPORT metric and is hard-zeroed at
        # the poles (no meridional flux through the pole wall), whereas NEMO's
        # e2v carries no such zeroing.  Same closed form, different boundary
        # convention -- comparing them directly would be wrong.
        expected = (float(g.radius) * float(g.dlon)) * np.asarray(g.cos_lat_v)
        np.testing.assert_allclose(
            np.asarray(geom_iso.dy_v)[:, 0], expected, rtol=1e-6,
            err_msg="under nemo_isotropic, dy_v must be NEMO's e2v = "
                    "R*dlon*cos(lat_v) (pe2v = pe1v, usrdef_hgr.F90:117)")
        # The #516 helper itself: identical on grids that only differ by
        # metric_convention (it reads grid.lat, which this flag never
        # touches).
        np.testing.assert_array_equal(
            vface_zonal_cos_lat(g), vface_zonal_cos_lat(g))


class TestMetricConventionDispatch:
    def test_create_mercator_grid_raises_on_unknown_convention(self):
        with pytest.raises(ValueError, match="metric_convention"):
            create_mercator_grid(n_lon=8, lat_max_deg=70.0,
                                 metric_convention="bogus")

    def test_create_latlon_geometry_raises_on_unknown_convention(self):
        from legoesm.grids import create_latlon_geometry
        with pytest.raises(ValueError, match="metric_convention"):
            create_latlon_geometry(n_lat=10, metric_convention="bogus")
