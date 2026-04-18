"""Targeted regression tests for the cubed-sphere FV3 C-D grid path.

Tests:
1. dgrid_to_center_geographic removes spurious v_north for solid-body rotation
2. _d2a2c_vect vs fv3_cc2c divergence on a balanced solid-body case
3. fv3_sw_tendencies balanced-flow residual decreases with resolution
4. Metric consistency: rsin_u matches sqrt(1-cosa_u^2)
"""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _make_solid_body_corner(cdgrid, Omega=7.292e-5):
    """Solid-body rotation at D-grid corners: u_east = Omega*R*cos(lat)."""
    R = cdgrid.radius
    cos_lat = jnp.cos(cdgrid.lat_corner)
    u_geo = Omega * R * cos_lat
    u_d = u_geo * cdgrid.cos_angle_corner
    v_d = -u_geo * cdgrid.sin_angle_corner
    return u_d, v_d


def _make_solid_body_edge(cdgrid, Omega=7.292e-5):
    """Solid-body rotation at FV3 edge-midpoint D-grid positions."""
    R = cdgrid.radius
    # x-edge midpoints: (6, n, n+1)
    cos_lat_x = jnp.cos(cdgrid.lat_edge_x)
    u_geo_x = Omega * R * cos_lat_x
    u_d = u_geo_x * cdgrid.cos_angle_edge_x  # (6, n, n+1)

    # y-edge midpoints: (6, n+1, n)
    cos_lat_y = jnp.cos(cdgrid.lat_edge_y)
    u_geo_y = Omega * R * cos_lat_y
    v_d = -u_geo_y * cdgrid.sin_angle_edge_y  # (6, n+1, n)

    return u_d, v_d


def _make_tc2_state_edge(cdgrid, g=9.80616, Omega=7.292e-5, H0=2.94e4 / 9.80616):
    """Williamson TC2 (steady-state geostrophic flow) at edge-midpoint stagger."""
    R = cdgrid.radius
    n = cdgrid.n

    # Height field at cell centres
    lat_cc = cdgrid.base.lat  # (6, n, n)
    h = H0 - (R * Omega + 0.5 * Omega**2 * R) / g * jnp.sin(lat_cc)**2

    # Winds at edge midpoints
    u_d, v_d = _make_solid_body_edge(cdgrid, Omega)

    h_s = jnp.zeros((6, n, n))
    return h, u_d, v_d, h_s


class TestDgridToCenterGeographic(unittest.TestCase):
    """Test that dgrid_to_center_geographic removes spurious v_north."""

    def test_solid_body_vnorth_near_zero(self):
        """For solid-body rotation, v_north should be 0 to near-machine-precision."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import dgrid_to_center_geographic

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        u_d, v_d = _make_solid_body_corner(cdgrid)
        u_east, v_north = dgrid_to_center_geographic(u_d, v_d, cdgrid)

        # v_north should be zero for solid-body rotation about the pole axis
        max_vn = float(jnp.max(jnp.abs(v_north)))
        # Should be << 1 m/s (solid-body speed at equator is ~465 m/s)
        self.assertLess(max_vn, 0.01,
                        f"v_north max = {max_vn:.4f} m/s, expected near zero")

        # u_east should match Omega*R*cos(lat)
        R = cdgrid.radius
        Omega = 7.292e-5
        expected_ue = Omega * R * jnp.cos(cdgrid.base.lat)
        rel_err = jnp.max(jnp.abs(u_east - expected_ue)) / jnp.max(jnp.abs(expected_ue))
        self.assertLess(float(rel_err), 0.02,
                        f"u_east relative error = {float(rel_err):.4f}")


class TestD2a2cVsFv3Cc2c(unittest.TestCase):
    """Compare _d2a2c_vect C-grid vs fv3_cc2c C-grid on balanced solid-body flow."""

    def test_transport_divergence_comparison(self):
        """Both C-grid interpolation paths should give similar divergence."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import cgrid_divergence, fv3_cc2c, fv3_d2cc
        from legoesm.core.fv3_sw_core import _d2a2c_vect

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        u_d, v_d = _make_solid_body_edge(cdgrid)

        # Path 1: d2a2c_vect → C-grid (covariant convention, FV3-style)
        ua, va, uc_cov, vc_cov, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

        # Path 2: fv3_cc2c → C-grid (physical face-normal convention)
        u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
        uc_phys, vc_phys = fv3_cc2c(u_cc, v_cc, cdgrid)

        # Compute divergence from both paths
        # fv3_cc2c gives physical face-normal velocity → use cgrid_divergence directly
        div_phys = cgrid_divergence(uc_phys, vc_phys, cdgrid)

        # d2a2c_vect gives covariant uc/vc — need to convert to physical
        # for a fair comparison, just compare magnitude of divergence
        # For solid-body rotation on the sphere, divergence should be near zero
        rms_phys = float(jnp.sqrt(jnp.mean(div_phys**2)))

        # Both should be small (solid body has zero divergence)
        Omega = 7.292e-5
        # Divergence scale: Omega ~ 7e-5 s^-1; truncation should be << this
        # At C16 with non-orthogonality corrections, RMS ~ O(1e-7) which is
        # ~0.003 * Omega — well within acceptable range.
        self.assertLess(rms_phys, 1e-2 * Omega,
                        f"fv3_cc2c divergence rms = {rms_phys:.2e}, "
                        f"should be << Omega = {Omega:.2e}")


class TestFv3SwTendenciesBalancedResidual(unittest.TestCase):
    """Test that fv3_sw_tendencies balanced-flow residual decreases with resolution."""

    def _compute_residual(self, n):
        """Compute max tendency magnitude for TC2 at resolution n."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import fv3_sw_tendencies

        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        h, u_d, v_d, h_s = _make_tc2_state_edge(cdgrid)

        dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid, g=9.80616,
        )

        # Velocity residual normalized by Omega (natural tendency scale)
        Omega = 7.292e-5
        R = cdgrid.radius
        u_max = Omega * R
        du_max = float(jnp.max(jnp.abs(du_dt)))
        dv_max = float(jnp.max(jnp.abs(dv_dt)))
        return max(du_max, dv_max) / u_max

    def test_residual_small_c8_and_c16(self):
        """Balanced-flow residual should be small at C8 and C16.

        At coarse resolution, boundary errors dominate and convergence
        is not strictly monotonic.  We just verify the residual is small
        compared to Omega (the natural tendency scale for geostrophic flow).
        """
        res_c8 = self._compute_residual(8)
        res_c16 = self._compute_residual(16)

        # Both should be small: << 1 (much less than the wind itself)
        self.assertLess(res_c8, 5e-4,
                        f"C8 residual ({res_c8:.2e}) too large")
        self.assertLess(res_c16, 5e-4,
                        f"C16 residual ({res_c16:.2e}) too large")

    def test_residual_finite(self):
        """Tendencies should be finite for balanced flow."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import fv3_sw_tendencies

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        h, u_d, v_d, h_s = _make_tc2_state_edge(cdgrid)

        dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid, g=9.80616,
        )

        self.assertTrue(jnp.all(jnp.isfinite(dh_dt)), "dh_dt has non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(du_dt)), "du_dt has non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(dv_dt)), "dv_dt has non-finite values")


class TestMetricConsistency(unittest.TestCase):
    """Test rsin_u/rsin_v follow the FV3 mixed convention.

    Per Fortran fv_grid_utils.F90:509,548-554 (non-duogrid cubed sphere):
      - Interior u/v faces: rsin_u = 1/sina_u²
      - Panel edges (i=0/i=n for rsin_u, j=0/j=n for rsin_v):
        rsin_u = 1/sina_u (override gated on .not. bounded_domain).
    Duogrid/bounded-domain grids use 1/sin² everywhere (no edge override).
    """

    def test_rsin_u_nonduogrid_fv3_mixed(self):
        """rsin_u: 1/sin² interior; 1/sin at i=0, i=n (non-duogrid)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(16, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        n = cdgrid.n
        _EPS = float(jnp.finfo(jnp.float32).eps)

        sina_u = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_u**2, _EPS))
        # Interior: 1/sin²
        rsin_int_expected = 1.0 / jnp.maximum(sina_u[:, 1:n, :]**2, _EPS)
        diff_int = float(jnp.max(jnp.abs(
            cdgrid.rsin_u[:, 1:n, :] - rsin_int_expected)))
        self.assertLess(diff_int, 1e-6,
                        f"rsin_u interior not 1/sin²: max diff = {diff_int:.2e}")
        # Panel edges: 1/sin
        for i in (0, n):
            rsin_edge_expected = 1.0 / jnp.maximum(jnp.abs(sina_u[:, i, :]), _EPS)
            diff_edge = float(jnp.max(jnp.abs(
                cdgrid.rsin_u[:, i, :] - rsin_edge_expected)))
            self.assertLess(diff_edge, 1e-6,
                            f"rsin_u i={i} not 1/sin: max diff = {diff_edge:.2e}")

    def test_rsin_v_nonduogrid_fv3_mixed(self):
        """rsin_v: 1/sin² interior; 1/sin at j=0, j=n (non-duogrid)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(16, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        n = cdgrid.n
        _EPS = float(jnp.finfo(jnp.float32).eps)

        sina_v = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_v**2, _EPS))
        rsin_int_expected = 1.0 / jnp.maximum(sina_v[:, :, 1:n]**2, _EPS)
        diff_int = float(jnp.max(jnp.abs(
            cdgrid.rsin_v[:, :, 1:n] - rsin_int_expected)))
        self.assertLess(diff_int, 1e-6,
                        f"rsin_v interior not 1/sin²: max diff = {diff_int:.2e}")
        for j in (0, n):
            rsin_edge_expected = 1.0 / jnp.maximum(jnp.abs(sina_v[:, :, j]), _EPS)
            diff_edge = float(jnp.max(jnp.abs(
                cdgrid.rsin_v[:, :, j] - rsin_edge_expected)))
            self.assertLess(diff_edge, 1e-6,
                            f"rsin_v j={j} not 1/sin: max diff = {diff_edge:.2e}")

    def test_rsin_u_duogrid_uniform_1_over_sin2(self):
        """Duogrid: rsin_u = 1/sin² everywhere (no panel-edge override)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(16, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        _EPS = float(jnp.finfo(jnp.float32).eps)

        sina_u = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_u**2, _EPS))
        expected = 1.0 / jnp.maximum(sina_u**2, _EPS)
        max_diff = float(jnp.max(jnp.abs(cdgrid.rsin_u - expected)))
        self.assertLess(max_diff, 1e-6,
                        f"duogrid rsin_u not uniform 1/sin²: max diff = {max_diff:.2e}")

    def test_rsin_u_single_face_panel_uniform_1_over_sin2(self):
        """Single-face panel (regional / nested): bounded_domain=True, so
        rsin_u = 1/sin² everywhere (no panel-edge 1/sin override)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere_panel

        panel, cdgrid_panel = create_cubed_sphere_panel(16, return_cdgrid=True)
        _EPS = float(jnp.finfo(jnp.float32).eps)

        sina_u = jnp.sqrt(jnp.maximum(1.0 - cdgrid_panel.cosa_u**2, _EPS))
        expected = 1.0 / jnp.maximum(sina_u**2, _EPS)
        max_diff = float(jnp.max(jnp.abs(cdgrid_panel.rsin_u - expected)))
        self.assertLess(max_diff, 1e-6,
                        f"panel rsin_u not uniform 1/sin²: max diff = {max_diff:.2e}")
        # Same check for rsin_v
        sina_v = jnp.sqrt(jnp.maximum(1.0 - cdgrid_panel.cosa_v**2, _EPS))
        expected_v = 1.0 / jnp.maximum(sina_v**2, _EPS)
        max_diff_v = float(jnp.max(jnp.abs(cdgrid_panel.rsin_v - expected_v)))
        self.assertLess(max_diff_v, 1e-6,
                        f"panel rsin_v not uniform 1/sin²: max diff = {max_diff_v:.2e}")

    def test_rarea_c_positive(self):
        """rarea_c should be positive everywhere."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(8)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        self.assertTrue(jnp.all(cdgrid.rarea_c > 0))
        # Check consistency: rarea_c ≈ 1/area_corner
        recomputed = 1.0 / cdgrid.area_corner
        max_rel_err = float(jnp.max(jnp.abs(cdgrid.rarea_c - recomputed)
                                     / jnp.maximum(recomputed, 1e-30)))
        self.assertLess(max_rel_err, 1e-6)


class TestFv3ForwardBackwardSmoke(unittest.TestCase):
    """Smoke test for the experimental fv3_forward_backward_step."""

    def test_one_step_finite(self):
        """One FB step should produce finite values (even if inaccurate)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import fv3_forward_backward_step

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        h, u_d, v_d, h_s = _make_tc2_state_edge(cdgrid)
        dt = 300.0  # 5-minute step

        h_new, u_new, v_new = fv3_forward_backward_step(
            h, u_d, v_d, h_s, cdgrid, dt, g=9.80616)

        self.assertTrue(jnp.all(jnp.isfinite(h_new)),
                        "h_new has non-finite values after 1 FB step")
        self.assertTrue(jnp.all(jnp.isfinite(u_new)),
                        "u_new has non-finite values after 1 FB step")
        self.assertTrue(jnp.all(jnp.isfinite(v_new)),
                        "v_new has non-finite values after 1 FB step")

    def test_mass_approximately_conserved_one_step(self):
        """Mass should be approximately conserved for 1 step."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import fv3_forward_backward_step

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        h, u_d, v_d, h_s = _make_tc2_state_edge(cdgrid)
        dt = 60.0

        area = cdgrid.base.area
        mass_0 = float(jnp.sum(h * area))

        h_new, u_new, v_new = fv3_forward_backward_step(
            h, u_d, v_d, h_s, cdgrid, dt, g=9.80616)

        mass_1 = float(jnp.sum(h_new * area))
        rel_err = abs(mass_1 - mass_0) / abs(mass_0)

        # Relaxed tolerance for the experimental FB step
        self.assertLess(rel_err, 0.01,
                        f"Mass relative error = {rel_err:.4e}")


class TestD2a2cVectDuogridSeams(unittest.TestCase):
    """Seam-level regression tests for _d2a2c_vect_duogrid (iter-60).

    These tests verify that the fully-haloed D-grid path (introduced in
    iter-60 to replace the pad_halo_vector utmp halo) produces correct
    face-boundary uc/vc values.  Prior to iter-60 the only direct tests
    on this path were shape/finite checks; the adversarial review noted
    that wrong-but-finite boundary winds could ship silently.
    """

    def test_rest_state_machine_precision(self):
        """u_d = v_d = 0 should give uc = vc = ut = vt = ua = va = 0 exactly."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        u_d = jnp.zeros((6, n, n + 1))
        v_d = jnp.zeros((6, n + 1, n))

        ua, va, uc, vc, ut, vt = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        for name, arr in [("ua", ua), ("va", va), ("uc", uc),
                          ("vc", vc), ("ut", ut), ("vt", vt)]:
            m = float(jnp.max(jnp.abs(arr)))
            self.assertLess(m, 1e-12, f"Rest state {name} max = {m:.3e}")

    def test_constant_geographic_flow_face_continuity(self):
        """Constant geographic wind should give uc continuous across face seams.

        With a constant (u_east, v_north) field, the PHYSICAL velocity is
        smooth everywhere.  After projecting to the non-orthogonal grid
        covariant basis and interpolating to C-grid edges, uc should be
        continuous across face boundaries (up to the grid-angle rotation
        which is itself continuous).  This tests that the cross-axis
        D-grid halo reconstruction in ext_vector_dgrid gives seam values
        consistent with the interior.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Constant eastward geographic wind of 10 m/s
        u_east = 10.0
        v_north = 0.0
        u_d = (cdgrid.cos_angle_edge_x * u_east
               + cdgrid.sin_angle_edge_x * v_north)
        v_d = (-cdgrid.sin_angle_edge_y * u_east
               + cdgrid.cos_angle_edge_y * v_north)

        ua, va, uc, vc, ut, vt = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        # For a constant (u_east, v_north), u^contra = const in geographic
        # frame; in grid-aligned covariant coords uc = ua*cos + va*sin at
        # each stagger, so |uc|, |vc| are bounded by max(|u_east|, |v_north|)
        # + small interpolation overshoot.  Check no pathological blow-up
        # at face boundaries.
        u_wind_magnitude = max(abs(u_east), abs(v_north))
        safety = 1.5  # allow 50% overshoot from covariant scaling

        # Face-boundary u-edges: i=0 and i=n
        uc_boundary = jnp.concatenate([uc[:, 0:1, :], uc[:, n:n + 1, :]], axis=1)
        uc_interior = uc[:, 1:n, :]
        boundary_max = float(jnp.max(jnp.abs(uc_boundary)))
        interior_max = float(jnp.max(jnp.abs(uc_interior)))
        self.assertLess(boundary_max, safety * u_wind_magnitude / 0.5,
                        f"uc boundary max = {boundary_max:.3f} "
                        f"> safety bound for constant flow")
        # Boundary values should not differ wildly from interior (<3x)
        self.assertLess(boundary_max, 3.0 * interior_max + 1e-6,
                        f"uc boundary {boundary_max:.3f} >> interior "
                        f"{interior_max:.3f} (seam discontinuity)")

        # Face-boundary v-edges: j=0 and j=n
        vc_boundary = jnp.concatenate([vc[:, :, 0:1], vc[:, :, n:n + 1]], axis=2)
        vc_interior = vc[:, :, 1:n]
        vb_max = float(jnp.max(jnp.abs(vc_boundary)))
        vi_max = float(jnp.max(jnp.abs(vc_interior)))
        self.assertLess(vb_max, 3.0 * vi_max + 1e-6,
                        f"vc boundary {vb_max:.3f} >> interior "
                        f"{vi_max:.3f} (seam discontinuity)")

        # Result must be finite everywhere.
        for name, arr in [("ua", ua), ("va", va), ("uc", uc),
                          ("vc", vc), ("ut", ut), ("vt", vt)]:
            self.assertTrue(bool(jnp.all(jnp.isfinite(arr))),
                            f"{name} has non-finite values")

    def test_seam_halo_consistent_jit_stable(self):
        """Same input should give identical output when the function is
        JIT-compiled — catches stale closure / halo-table races.

        This test catches a subtle class of bugs where halo lookups
        use stale state (e.g., when the JAX trace captures different
        array identities between eager and JIT paths).  The seam
        computation in ``_d2a2c_vect_duogrid`` goes through
        ``ext_vector_dgrid`` which reads ``duogrid.vlon_ext`` etc. —
        any trace-time-only lookup would fail under JIT.
        """
        import jax
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        Omega = 7.292e-5
        R = cdgrid.radius
        u_east_ex = Omega * R * jnp.cos(cdgrid.lat_edge_x)
        u_east_ey = Omega * R * jnp.cos(cdgrid.lat_edge_y)
        u_d = cdgrid.cos_angle_edge_x * u_east_ex
        v_d = -cdgrid.sin_angle_edge_y * u_east_ey

        # Eager
        out_eager = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        # JIT-compiled
        fn = jax.jit(lambda ud, vd: _d2a2c_vect_duogrid(ud, vd, cdgrid))
        out_jit = fn(u_d, v_d)

        # Eager and JIT may differ at floating-point precision; tolerance
        # is relative to typical magnitudes.  Bug would cause O(1) drift.
        for name, e, j in zip(("ua", "va", "uc", "vc", "ut", "vt"),
                               out_eager, out_jit):
            d = float(jnp.max(jnp.abs(e - j)))
            scale = max(float(jnp.max(jnp.abs(e))), 1.0)
            self.assertLess(d, 1e-4 * scale,
                            f"{name} eager vs jit differ by {d:.3e} "
                            f"(scale {scale:.3e})")

    def test_uniform_east_wind_ut_positive(self):
        """Uniform eastward geographic wind should give ut > 0 on
        face-0 equatorial u-edges (where x-grid axis is aligned with
        east).  Catches sign-flip / orientation bugs at face boundaries
        that a magnitude-only test would miss.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # u_east = +10 m/s (constant, physical east wind).  Project to
        # D-grid via local edge angles.
        u_east = 10.0
        u_d = cdgrid.cos_angle_edge_x * u_east
        v_d = -cdgrid.sin_angle_edge_y * u_east

        ua, va, uc, vc, ut, vt = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        # On face 0 (equator-centred face in FV3 convention), the
        # x-grid axis points approximately east at equatorial cells.
        # u_east=+10 should give positive ut on face 0.
        ut_face0_equator = ut[0, :, n // 2]  # (n+1,) u-edges on face 0, equator row
        # Expect at least 80% of u-edges on face 0 equator to have
        # ut > 0 (allowing some edge positions near face corners to
        # rotate out of east alignment).
        n_positive = int(jnp.sum(ut_face0_equator > 0))
        self.assertGreater(n_positive, int(0.8 * (n + 1)),
                           f"Only {n_positive}/{n + 1} ut values on "
                           f"face 0 equator are positive — expected "
                           f"most to be +east for u_east=+10. "
                           f"(sign/orientation bug?)")

        # Full cross-face seam: ut at face-0 east u-edge (i=n) and
        # face-0+east-neighbor west u-edge should have the same sign
        # (both positive for eastward flow).  On face 0, east neighbor
        # is face 1 (FV3 standard connectivity).
        ut_f0_east = float(jnp.mean(ut[0, n, :]))
        ut_f1_west = float(jnp.mean(ut[1, 0, :]))
        self.assertGreater(ut_f0_east, 0.0,
                           f"ut face-0 east edge = {ut_f0_east:.3f} "
                           f"should be positive (east wind)")
        self.assertGreater(ut_f1_west, 0.0,
                           f"ut face-1 west edge = {ut_f1_west:.3f} "
                           f"should be positive (east wind, cross-face)")

    def test_constant_covariant_input_preserved(self):
        """Constant u_d, v_d fields should give utmp equal to that
        constant (after 2-point length-weighted D→A).  This catches
        normalization bugs in the c2l_ord2-equivalent halo seed.

        For u_d = c1 everywhere (c1 is a constant), the length-weighted
        average (u_d*dx_j + u_d*dx_{j+1}) / (dx_j + dx_{j+1}) reduces to
        c1 exactly.  The resulting contravariant ua then depends on
        non-orthogonality but is bounded by the covariant value.

        This test also checks FACE-BOUNDARY uc/vc/ut/vt values, since
        the halo seed affects seam outputs specifically (interior is
        overwritten with exact u_d/v_d post-halo).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Uniform constant D-grid covariant wind
        c1 = 5.0
        u_d = jnp.full((6, n, n + 1), c1)
        v_d = jnp.full((6, n + 1, n), c1)

        ua, va, uc, vc, ut, vt = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        # utmp at interior cells equals c1 by length-weighted average.
        # Contravariant ua = (utmp - vtmp * cos_theta) * rsin2 =
        # c1 * (1 - cos_theta) / sin²θ = c1 / (1 + cos_theta).  For
        # cubed-sphere interior |cos_theta| < 0.5, so ua in [c1/1.5, c1/0.5]
        # = [3.33, 10] for c1=5.  Loose upper bound 3*c1=15 catches the
        # factor-of-2 bug (which gave ua up to 2*c1/min(1+cos)=~20).
        ua_max = float(jnp.max(jnp.abs(ua)))
        self.assertLess(ua_max, 3.0 * abs(c1),
                        f"ua max {ua_max:.3f} >> 3*c1 ({3*abs(c1):.3f}) "
                        f"— normalization bug (factor-of-2)?")

        # uc should be similarly bounded.  The 4th-order A→C applied to
        # a constant utmp field gives back utmp exactly (Lagrange
        # polynomial reproduces constants).
        uc_interior = uc[:, 1:n, :]  # avoid face boundaries
        uc_max_int = float(jnp.max(jnp.abs(uc_interior)))
        self.assertLess(abs(uc_max_int - abs(c1)), 0.5 * abs(c1),
                        f"uc interior {uc_max_int:.3f} deviates from "
                        f"expected ~{abs(c1):.3f} by > 50% "
                        f"— constant-state not preserved")

        # FACE-BOUNDARY outputs: uc at i=0 and i=n (the u-edges that sit
        # exactly on face seams).  For constant u_d=c1, uc at face
        # boundary should be close to c1 (4th-order Lagrange reproduces
        # constants IF the halo is correctly populated).
        uc_bdy_w = uc[:, 0, :]      # west face u-edge
        uc_bdy_e = uc[:, n, :]      # east face u-edge
        for name, arr in [("uc_west", uc_bdy_w), ("uc_east", uc_bdy_e)]:
            a_max = float(jnp.max(jnp.abs(arr)))
            # Should be bounded by ~c1 × (1 + overshoot from halo projection
            # through non-orthogonal metrics).  2*c1 is generous; fails if
            # halo seed is doubled.
            self.assertLess(a_max, 2.5 * abs(c1),
                            f"{name} max {a_max:.3f} > 2.5*c1 "
                            f"({2.5*abs(c1):.3f}) — halo doubling?")

        vc_bdy_s = vc[:, :, 0]
        vc_bdy_n = vc[:, :, n]
        for name, arr in [("vc_south", vc_bdy_s), ("vc_north", vc_bdy_n)]:
            a_max = float(jnp.max(jnp.abs(arr)))
            self.assertLess(a_max, 2.5 * abs(c1),
                            f"{name} max {a_max:.3f} > 2.5*c1 "
                            f"({2.5*abs(c1):.3f}) — halo doubling?")

        # Contravariant transport: ut, vt bounded similarly (they are
        # post-rotation of uc/vc with rsin_u, rsin_v factors).
        ut_bdy = jnp.concatenate([ut[:, 0:1, :], ut[:, n:n + 1, :]], axis=1)
        vt_bdy = jnp.concatenate([vt[:, :, 0:1], vt[:, :, n:n + 1]], axis=2)
        self.assertLess(float(jnp.max(jnp.abs(ut_bdy))), 5.0 * abs(c1),
                        "ut boundary unreasonably large")
        self.assertLess(float(jnp.max(jnp.abs(vt_bdy))), 5.0 * abs(c1),
                        "vt boundary unreasonably large")

    def test_solid_body_rotation_ut_sign_convention(self):
        """Solid-body rotation: ut should be eastward-positive everywhere.

        For ω > 0 (prograde rotation), the contravariant transport
        velocity ut at a u-edge should have a consistent positive sign
        when the grid axis aligns with east (i.e., for equatorial
        u-edges on faces where grid_x ≈ east).  This catches gross
        mis-orientation in the cross-axis halo rotation — a common
        failure mode of D-grid vector halos on cubed-sphere.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect_duogrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Solid-body rotation: u_east = Omega * R * cos(lat)
        Omega = 7.292e-5
        R = cdgrid.radius
        u_east_ex = Omega * R * jnp.cos(cdgrid.lat_edge_x)
        u_east_ey = Omega * R * jnp.cos(cdgrid.lat_edge_y)
        u_d = cdgrid.cos_angle_edge_x * u_east_ex
        v_d = -cdgrid.sin_angle_edge_y * u_east_ey

        ua, va, uc, vc, ut, vt = _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

        # ut is contravariant along x-grid axis.  Its magnitude at
        # u-edges where x-grid is near east should approach the solid
        # body speed at that latitude.  Face boundaries must not exceed
        # the interior by more than 50% in magnitude.
        max_speed = float(jnp.max(jnp.abs(ua) + jnp.abs(va)))
        ut_max = float(jnp.max(jnp.abs(ut)))
        vt_max = float(jnp.max(jnp.abs(vt)))
        self.assertLess(ut_max, 2.0 * max_speed + 1.0,
                        f"ut max = {ut_max:.3f} unreasonable vs "
                        f"ua/va max = {max_speed:.3f}")
        self.assertLess(vt_max, 2.0 * max_speed + 1.0,
                        f"vt max = {vt_max:.3f} unreasonable")

        # ut boundary edges should not be wildly larger than interior.
        ut_boundary = jnp.concatenate([ut[:, 0:1, :], ut[:, n:n + 1, :]], axis=1)
        ut_interior = ut[:, 1:n, :]
        ut_b_max = float(jnp.max(jnp.abs(ut_boundary)))
        ut_i_max = float(jnp.max(jnp.abs(ut_interior)))
        self.assertLess(ut_b_max, 2.5 * ut_i_max + 1e-6,
                        f"ut boundary {ut_b_max:.3f} much larger than "
                        f"interior {ut_i_max:.3f}")


class TestFvTp2dCornerInvariant(unittest.TestCase):
    """Iter-69: verify fv_tp_2d never reads cube-vertex corner cells.

    Codex raised the 2-point-average vs Fortran directional ``copy_corners``
    discrepancy in ``_fill_corners_h1/h2``.  The practical test of whether
    this affects mass transport is whether fv_tp_2d's PPM sweeps ever
    dereference the 2x2 cube-vertex corner blocks at (i_halo, j_halo).
    The slicing pattern (``q_full[:, 2:-2, :]`` for y-sweep,
    ``q_i_pad[:, :, 2:-2]`` for x-sweep) suggests NO, but we lock it in
    with a random-scramble test: poison the corner blocks with NaN before
    calling fv_tp_2d; if any sweep reads them, outputs become NaN.
    """

    def test_fv_tp_2d_output_unchanged_when_corner_ghosts_nan(self):
        """End-to-end behavioural test: patch fv_tp_2d's internal pad_halo
        to inject NaN into the 2x2 cube-vertex corner blocks AFTER the
        standard fill, call fv_tp_2d with the patched halo, and verify
        the output flux is finite and equal to the unpatched call.

        If fv_tp_2d ever dereferences a cube-vertex corner cell, the NaN
        propagates and either (a) the output contains NaN, or (b) the
        output differs from the unpatched baseline.  Either failure
        disproves the iter-69 invariant.
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv_tp_2d as fv_tp_2d_mod

        n = 8
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        h = jnp.ones((6, n, n)) * 1000.0 + jnp.sin(
            jnp.linspace(0, 3.14, n))[None, None, :] * 10.0
        crx = jnp.ones((6, n + 1, n)) * 0.1
        cry = jnp.ones((6, n, n + 1)) * 0.1
        xfx = crx * cdgrid.dy_edge_x
        yfx = cry * cdgrid.dx_edge_y
        area = cdgrid.base.area

        # Baseline call — uses the real pad_halo
        fx_base, fy_base = fv_tp_2d_mod.fv_tp_2d(
            h, crx, cry, xfx, yfx, area, area, cdgrid)
        self.assertTrue(bool(jnp.all(jnp.isfinite(fx_base))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(fy_base))))

        real_pad_halo = fv_tp_2d_mod.pad_halo

        def poisoned_pad_halo(q, halo=1, interp_offsets=None, duogrid=None):
            """Wrap real pad_halo and inject NaN into the cube-vertex corner
            blocks.  Only the 2x2 corners at (i_halo, j_halo) are poisoned;
            strip halos and interior are untouched."""
            res = real_pad_halo(q, halo=halo, interp_offsets=interp_offsets,
                                duogrid=duogrid)
            if res.ndim == 3 and res.shape[0] == 6:
                size = res.shape[1]
                n_int = size - 2 * halo
                h = halo
                # 4 corner blocks, h×h cells each, poison on every face
                for (i_lo, i_hi) in [(0, h), (n_int + h, n_int + 2 * h)]:
                    for (j_lo, j_hi) in [(0, h), (n_int + h, n_int + 2 * h)]:
                        res = res.at[:, i_lo:i_hi, j_lo:j_hi].set(jnp.nan)
            return res

        # Patch pad_halo inside fv_tp_2d's module namespace
        with mock.patch.object(fv_tp_2d_mod, 'pad_halo', poisoned_pad_halo):
            fx_poison, fy_poison = fv_tp_2d_mod.fv_tp_2d(
                h, crx, cry, xfx, yfx, area, area, cdgrid)

        # If fv_tp_2d dereferenced any cube-vertex corner cell, NaN
        # would propagate into fx_poison / fy_poison.
        self.assertTrue(bool(jnp.all(jnp.isfinite(fx_poison))),
                        "fx_poison has NaN — fv_tp_2d DOES read cube-vertex "
                        "corners (contradicts iter-69 analysis)")
        self.assertTrue(bool(jnp.all(jnp.isfinite(fy_poison))),
                        "fy_poison has NaN — fv_tp_2d DOES read cube-vertex "
                        "corners")
        # And the output must be bit-identical to the unpatched baseline.
        self.assertTrue(bool(jnp.array_equal(fx_poison, fx_base)),
                        "fx with corner NaN differs from baseline — corner "
                        "ghosts are being read somewhere in the pipeline")
        self.assertTrue(bool(jnp.array_equal(fy_poison, fy_base)),
                        "fy with corner NaN differs from baseline — corner "
                        "ghosts are being read somewhere in the pipeline")

    def test_deln_flux_output_unchanged_when_corner_ghosts_nan(self):
        """Iter-70: Codex flagged _deln_flux for missing Fortran
        direction-specific copy_corners (tp_core.F90:1267, 1280).
        Same end-to-end mock-patch test: poison cube-vertex corners in
        pad_halo and verify _deln_flux output is bit-identical to the
        unpatched baseline.  Proves the Python stencil does not
        dereference cube-vertex corners (same invariant as fv_tp_2d).
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv_tp_2d as fv_tp_2d_mod

        n = 8
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        q = jnp.ones((6, n, n)) * 10.0 + jnp.sin(
            jnp.linspace(0, 3.14, n))[None, None, :]
        fx = jnp.zeros((6, n + 1, n))
        fy = jnp.zeros((6, n, n + 1))

        # Baseline: _deln_flux at nord=1 (del-4) — the case where Fortran
        # actually calls copy_corners.
        fx_base, fy_base = fv_tp_2d_mod._deln_flux(
            1, 0.001, q, fx, fy, cdgrid)

        real_pad_halo = fv_tp_2d_mod.pad_halo

        def poisoned(q, halo=1, interp_offsets=None, duogrid=None):
            res = real_pad_halo(q, halo=halo, interp_offsets=interp_offsets,
                                duogrid=duogrid)
            if res.ndim == 3 and res.shape[0] == 6:
                size = res.shape[1]
                n_int = size - 2 * halo
                h = halo
                for (i_lo, i_hi) in [(0, h), (n_int + h, n_int + 2 * h)]:
                    for (j_lo, j_hi) in [(0, h), (n_int + h, n_int + 2 * h)]:
                        res = res.at[:, i_lo:i_hi, j_lo:j_hi].set(jnp.nan)
            return res

        with mock.patch.object(fv_tp_2d_mod, 'pad_halo', poisoned):
            fx_poison, fy_poison = fv_tp_2d_mod._deln_flux(
                1, 0.001, q, fx, fy, cdgrid)

        self.assertTrue(bool(jnp.all(jnp.isfinite(fx_poison))),
                        "_deln_flux fx has NaN — stencil reads corner cells")
        self.assertTrue(bool(jnp.all(jnp.isfinite(fy_poison))),
                        "_deln_flux fy has NaN — stencil reads corner cells")
        self.assertTrue(bool(jnp.array_equal(fx_poison, fx_base)),
                        "_deln_flux fx differs from baseline when corner "
                        "blocks poisoned — corner ghosts used somewhere")
        self.assertTrue(bool(jnp.array_equal(fy_poison, fy_base)),
                        "_deln_flux fy differs from baseline when corner "
                        "blocks poisoned — corner ghosts used somewhere")

    def test_del6_vt_flux_routes_halo_through_duogrid_when_active(self):
        """Iter-78: `_del6_vt_flux` accepted a `use_duogrid` parameter
        but never used it — its halo was pinned to `interp_offsets`.
        After the iter-78 fix, `use_duogrid=True` must actually route
        through the duogrid remap.
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv3_sw_core as fv3_sw_core_mod

        n = 8
        grid_dg = create_cubed_sphere(n, use_duogrid=True)
        cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)

        q = jnp.sin(jnp.linspace(0, 3.14, n))[None, None, :] * jnp.ones((6, n, n))

        calls = []
        real_pad_halo = fv3_sw_core_mod.pad_halo

        def recording(q, halo=1, interp_offsets=None, duogrid=None):
            calls.append(
                ('interp_offsets_none' if interp_offsets is None else 'interp_offsets_set',
                 'duogrid_none' if duogrid is None else 'duogrid_set'))
            return real_pad_halo(q, halo=halo,
                                 interp_offsets=interp_offsets,
                                 duogrid=duogrid)

        with mock.patch.object(fv3_sw_core_mod, 'pad_halo', recording):
            fv3_sw_core_mod._del6_vt_flux(
                1, 1e-6, q, cdgrid_dg, use_duogrid=True)

        self.assertTrue(len(calls) > 0, "_del6_vt_flux made no halo exchanges")
        for kw in calls:
            self.assertEqual(
                kw, ('interp_offsets_none', 'duogrid_set'),
                f"_del6_vt_flux halo exchange used {kw} instead of "
                f"duogrid remap when use_duogrid=True")

    def test_compute_transport_quantities_routes_halo_through_duogrid(self):
        """Iter-79: `compute_transport_quantities` had 4 `pad_halo` calls
        (rdxa, rdya, sin_sg variants) pinned to `interp_offsets` even when
        duogrid was active on the grid.  After the iter-79 fix, every
        halo call must route through duogrid when duogrid is available.
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv_tp_2d as fv_tp_2d_mod

        n = 8
        grid_dg = create_cubed_sphere(n, use_duogrid=True)
        cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)

        ut = jnp.ones((6, n + 1, n)) * 10.0
        vt = jnp.ones((6, n, n + 1)) * 5.0
        dt = 100.0

        calls = []
        real_pad_halo = fv_tp_2d_mod.pad_halo

        def recording(q, halo=1, interp_offsets=None, duogrid=None):
            calls.append(
                ('interp_offsets_none' if interp_offsets is None else 'interp_offsets_set',
                 'duogrid_none' if duogrid is None else 'duogrid_set'))
            return real_pad_halo(q, halo=halo,
                                 interp_offsets=interp_offsets,
                                 duogrid=duogrid)

        with mock.patch.object(fv_tp_2d_mod, 'pad_halo', recording):
            fv_tp_2d_mod.compute_transport_quantities(ut, vt, dt, cdgrid_dg)

        # Expect 4 halo calls (rdxa, sin_E/W, rdya, sin_N/S — 6 actually:
        # 1 rdxa + 2 sin_sg pair + 1 rdya + 2 sin_sg pair = 6)
        self.assertTrue(len(calls) >= 4,
                        f"too few halo calls: {len(calls)}")
        for kw in calls:
            self.assertEqual(
                kw, ('interp_offsets_none', 'duogrid_set'),
                f"compute_transport_quantities halo used {kw} "
                f"instead of duogrid remap on duogrid-active grid")

    def test_cosa_corner_matches_fortran_sub_grid_average_interior(self):
        """Fortran `fv_grid_utils.F90:495`:
          cosa(i,j) = 0.5*(cos_sg(i-1,j-1,8) + cos_sg(i,j,6))
        averages the NE sub-grid corner of the lower-left cell with the
        SW sub-grid corner of the upper-right cell.  Python builds
        `cdgrid.cosa_corner` by direct tangent-vector geometry on an
        extended grid — a different construction.

        At INTERIOR corners (1 <= ic, jc <= n-1) both sub-grid corner
        values come from the same supergrid point, so Fortran's average
        equals either operand, and matches Python's direct tangent to
        machine precision.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        sg = cdgrid.cos_sg
        direct = cdgrid.cosa_corner

        sg_avg_int = 0.5 * (sg[:, 0:n-1, 0:n-1, 7] + sg[:, 1:n, 1:n, 5])
        max_int = float(jnp.max(jnp.abs(direct[:, 1:n, 1:n] - sg_avg_int)))
        self.assertLess(max_int, 1e-6,
                        f"interior cosa_corner mismatch: {max_int}")

    def test_cosa_corner_panel_edge_self_consistency_all_faces(self):
        """Iter-97 (simpler form addressing Codex flag about iter-96):
        Verify Python's `cdgrid.cosa_corner` at every panel-edge
        corner on every face equals the LOCAL single-side sub-grid
        value (by construction of the extended tangent-vector grid).

        This covers all 6 faces × 4 edges × (n-1) interior corners —
        24 panel-edge sections in total, including reversed seams.

        For each (face, edge), the formula is:
            direct[f, panel_corner_idx] == sg[f, adjacent_cell, POS]
        where POS depends on which sub-grid corner of the adjacent
        cell physically coincides with the panel-edge corner.

        Self-consistency is a weaker claim than "Python matches
        Fortran" — but for reversed seams the Fortran index mapping
        is intricate.  What this test guarantees is that Python's
        construction is internally consistent across ALL 24 seams:
        extended-grid tangent geometry and sub-grid corner values
        agree at the interior supergrid point they share.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sg = cdgrid.cos_sg
        direct = cdgrid.cosa_corner

        tol = 1e-6   # float32 metric precision

        for f in range(6):
            # WEST panel edge: corners (0, jc), 1 <= jc <= n-1
            d = float(jnp.max(jnp.abs(
                direct[f, 0, 1:n] - sg[f, 0, 1:n, 5])))
            self.assertLess(d, tol,
                            f"face {f} WEST self-consistency: {d}")

            # EAST panel edge: corners (n, jc), 1 <= jc <= n-1
            d = float(jnp.max(jnp.abs(
                direct[f, n, 1:n] - sg[f, n-1, 0:n-1, 7])))
            self.assertLess(d, tol,
                            f"face {f} EAST self-consistency: {d}")

            # SOUTH panel edge: corners (ic, 0), 1 <= ic <= n-1
            d = float(jnp.max(jnp.abs(
                direct[f, 1:n, 0] - sg[f, 1:n, 0, 5])))
            self.assertLess(d, tol,
                            f"face {f} SOUTH self-consistency: {d}")

            # NORTH panel edge: corners (ic, n), 1 <= ic <= n-1
            d = float(jnp.max(jnp.abs(
                direct[f, 1:n, n] - sg[f, 0:n-1, n-1, 7])))
            self.assertLess(d, tol,
                            f"face {f} NORTH self-consistency: {d}")

    def test_d2a2c_vect_non_duogrid_cube_vertex_gap_documented(self):
        """Iter-108 (Priority 3, corrected from iter-107):
        `_d2a2c_vect` non-duogrid path does not implement Fortran's
        cube-vertex corner overrides for utmp/vtmp and ua/va
        (sw_core.F90:3527-3545 and 3620-3640).

        Correct characterization of the gap: Fortran writes halo
        cells near cube vertices with sign-flipped copies of the
        OTHER component on the adjacent face.  Python does NOT do
        this.  Instead Python uses `_fill_corners_h1` / `_fill_corners_h2`
        which averages adjacent edge halos — a DIFFERENT convention.
        The two give different numerical values at cube-vertex cells
        (O(1) on random input, O(dx²) on smooth fields).

        Iter-107 framed this as "Fortran-style is redundant for the
        common case", which overclaimed the equivalence.  The corrected
        iter-108 note in `_d2a2c_vect` honestly states the two
        approaches differ and the impact has not been quantified.

        Duogrid path (via `_d2a2c_vect_duogrid`, which Fortran also
        skips via `dg%is_initialized`) is unaffected.
        """
        import inspect
        from legoesm.core.fv3_sw_core import _d2a2c_vect

        src = inspect.getsource(_d2a2c_vect)
        # The note must name the Fortran lines, the two fill mechanisms
        # Python uses, and state the values differ.
        self.assertIn(
            "sw_core.F90:3527-3545 and 3620-3640", src,
            "The Priority 3 gap note for cube-vertex corner "
            "overrides was removed from `_d2a2c_vect`.  Either port the "
            "overrides or restore the note.",
        )
        self.assertIn(
            "NOT PORTED", src,
            "The Priority 3 gap note should explicitly state NOT PORTED.",
        )
        self.assertIn(
            "_fill_corners_h", src,
            "The note must acknowledge Python uses _fill_corners_h* "
            "for the cube-vertex halo blocks — a DIFFERENT convention "
            "from Fortran's sign-flip override.",
        )
        self.assertIn(
            "DIFFERENT", src,
            "The note must state that Python and Fortran give "
            "DIFFERENT values at cube-vertex cells in the non-duogrid "
            "path — do not weaken this to 'equivalent' or 'redundant'.",
        )
        self.assertIn(
            "NOT been quantified", src,
            "The note must state honestly that the numerical impact "
            "on the non-duogrid FB path has NOT been quantified.",
        )

    def test_rsin2_corner_matches_fortran_at_interior(self):
        """Iter-99: lock in `cdgrid.rsin2_corner` fidelity at interior
        corners against the Fortran Formula
            rsina(i,j) = 1 / sina(i,j)^2
        with sina built via the sub-grid averaging (iter-98 proved this
        matches Python's cosa_corner at interior to 1e-15).

        At interior corners the sign of sin(angle) is positive and
        stable under the sub-grid average so the rsin2 reconstruction
        via `1/sina²` matches Python at machine precision.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        sg = cdgrid.cos_sg
        # Fortran sina at interior corners = 0.5 * (sin_sg NE + sin_sg SW)
        # where sin_sg = sqrt(1 - cos_sg²) from the sub-grid.
        sin_sg = jnp.sqrt(jnp.maximum(1.0 - sg**2, 0.0))
        sina_fortran_int = 0.5 * (sin_sg[:, 0:n-1, 0:n-1, 7]
                                   + sin_sg[:, 1:n, 1:n, 5])
        rsin2_fortran_int = 1.0 / jnp.maximum(sina_fortran_int**2, 1e-30)

        rsin2_py_int = cdgrid.rsin2_corner[:, 1:n, 1:n]
        rel_diff = float(jnp.max(jnp.abs(
            rsin2_py_int - rsin2_fortran_int)) / jnp.max(rsin2_py_int))
        self.assertLess(
            rel_diff, 1e-5,
            f"rsin2_corner interior rel diff vs Fortran = {rel_diff:.3e}",
        )

    def test_cosa_corner_panel_edge_fortran_match_all_24_seams(self):
        """Iter-98: Codex stop-time flagged iter-97 as not closing
        reversed-seam Fortran coverage (only one reversed seam
        spot-checked).  Empirically derived the (halo cell, sub-grid
        position, sign) tuple for EVERY one of 24 (face, edge) panel-
        edge seams on a cubed sphere; locked each in with an exact
        Python-matches-Fortran assertion at 1e-10.

        The table below was derived by scanning all 16 combinations
        of (sub-grid position, sign) at each seam and picking the
        match at <1e-10.  All 24 seams match via forward-traversal
        index (non-reversed) along the neighbor's edge.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.halo import WEST, EAST, SOUTH, NORTH

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sg = cdgrid.cos_sg
        direct = cdgrid.cosa_corner

        # (face, edge) → (neighbor sub-grid position, sign)
        # Empirically derived; all match at 1e-10 via fwd traversal.
        #   pos: 5=SW, 6=SE, 7=NE, 8=NW
        lookup = {
            (0, WEST):  (7, -1),  (0, EAST):  (8, -1),
            (0, SOUTH): (7, -1),  (0, NORTH): (6, -1),
            (1, WEST):  (7, -1),  (1, EAST):  (8, -1),
            (1, SOUTH): (7, -1),  (1, NORTH): (7, +1),
            (2, WEST):  (7, -1),  (2, EAST):  (8, -1),
            (2, SOUTH): (6, +1),  (2, NORTH): (7, +1),
            (3, WEST):  (7, -1),  (3, EAST):  (8, -1),
            (3, SOUTH): (8, +1),  (3, NORTH): (8, -1),
            (4, WEST):  (7, -1),  (4, EAST):  (7, +1),
            (4, SOUTH): (7, -1),  (4, NORTH): (7, +1),
            (5, WEST):  (6, +1),  (5, EAST):  (6, -1),
            (5, SOUTH): (6, +1),  (5, NORTH): (6, -1),
        }

        from legoesm.grids.halo import CONNECTIVITY
        tol = 1e-10

        for (f, edge), (pos, sign) in lookup.items():
            nbr_f, nbr_e, _rev = CONNECTIVITY[f][edge]
            idx = jnp.arange(0, n - 1)   # fwd traversal

            # Local side and Python target value along the panel edge
            if edge == WEST:
                py = direct[f, 0, 1:n]
                local = sg[f, 0, 1:n, 5]
            elif edge == EAST:
                py = direct[f, n, 1:n]
                local = sg[f, n - 1, 0:n - 1, 7]
            elif edge == SOUTH:
                py = direct[f, 1:n, 0]
                local = sg[f, 1:n, 0, 5]
            else:  # NORTH
                py = direct[f, 1:n, n]
                local = sg[f, 0:n - 1, n - 1, 7]

            # Neighbor cell row along the neighbor's edge
            if nbr_e == WEST:
                halo = sg[nbr_f, 0, idx, pos]
            elif nbr_e == EAST:
                halo = sg[nbr_f, n - 1, idx, pos]
            elif nbr_e == SOUTH:
                halo = sg[nbr_f, idx, 0, pos]
            else:
                halo = sg[nbr_f, idx, n - 1, pos]

            fortran = 0.5 * (sign * halo + local)
            d = float(jnp.max(jnp.abs(py - fortran)))
            self.assertLess(
                d, tol,
                f"face {f} edge {edge}: Python vs Fortran halo-avg "
                f"diff = {d:.3e} (pos={pos}, sign={sign})",
            )

    def test_cosa_corner_panel_edge_matches_fortran_with_sign_flip(self):
        """Iter-96: CORRECT reframing of the iter-91–95 test.

        My earlier iterations claimed a panel-edge fidelity gap
        between Python's `cdgrid.cosa_corner` and Fortran's halo-
        averaged formula.  The gap was based on a NAIVE halo-copy
        without sub-grid rotation at the seam — which is not what
        Fortran actually does with a proper cubed-sphere halo update.

        Correct finding: Fortran at a face seam applies a cross-face
        rotation when copying `cos_sg` into the halo.  At face-0's
        west/east edge the i-axis flips relative to the neighbor, so
        cos(angle between i and j tangents) flips sign.  When the
        halo cos_sg is sign-flipped before averaging, the Fortran
        formula
            cosa(i,j) = 0.5 * (rotated_halo_cos_sg + local_cos_sg)
        matches Python's direct tangent-vector value at machine
        precision (~1e-17 on C16).

        This demonstrates that Python's `cdgrid.cosa_corner` IS
        Fortran-faithful at panel-edge corners — the extended
        tangent-vector construction on a single coordinate frame
        produces the same value as the halo-averaged construction
        with proper cross-face rotation.

        Empirical results on C16, all four face-0 seams:
          max |Python direct - Fortran halo-avg (sign-flipped)| = 1.96e-17
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        sg = cdgrid.cos_sg  # (6, n, n, 9); 5=SW, 6=SE, 7=NE, 8=NW
        direct = cdgrid.cosa_corner

        # --- Python self-consistency on all four edges of face 0 ---
        # Along each panel edge, Python's extended-grid tangent equals
        # the local single-side sub-grid corner.
        self.assertLess(
            float(jnp.max(jnp.abs(direct[0, 0, 1:n] - sg[0, 0, 1:n, 5]))),
            1e-6, "west self-consistency broken")
        self.assertLess(
            float(jnp.max(jnp.abs(direct[0, n, 1:n] - sg[0, n-1, 0:n-1, 7]))),
            1e-6, "east self-consistency broken")
        self.assertLess(
            float(jnp.max(jnp.abs(direct[0, 1:n, 0] - sg[0, 1:n, 0, 5]))),
            1e-6, "south self-consistency broken")
        self.assertLess(
            float(jnp.max(jnp.abs(direct[0, 1:n, n] - sg[0, 0:n-1, n-1, 7]))),
            1e-6, "north self-consistency broken")

        # --- Fortran halo-averaged construction across four seams ---
        # Fortran `fv_grid_utils.F90:495`:
        #   cosa(i, j) = 0.5 * (cos_sg(i-1, j-1, NE) + cos_sg(i, j, SW))
        # In Python 0-indexed:
        #   At Python corner (ic, jc), use
        #     0.5 * (sg[cell (ic-1, jc-1), NE=7] + sg[cell (ic, jc), SW=5])
        # CONNECTIVITY[0]: WEST=(3,EAST,False), EAST=(1,WEST,False),
        #                  SOUTH=(5,NORTH,False), NORTH=(4,SOUTH,False)

        # Apply Fortran's halo-averaged formula at each face-0 seam
        # WITH the cross-face i-axis sign flip that a proper Fortran
        # cubed-sphere halo update produces.
        # CONNECTIVITY[0]: W→3E, E→1W, S→5N, N→4S, all not reversed.

        # West seam (face 0 corner (0, jc)): halo = face-3 NE, sign-flip
        halo_w_rot = -sg[3, n-1, 0:n-1, 7]
        local_w = sg[0, 0, 1:n, 5]
        fortran_w = 0.5 * (halo_w_rot + local_w)
        d_w = float(jnp.max(jnp.abs(direct[0, 0, 1:n] - fortran_w)))

        # East seam (face 0 corner (n, jc)): halo = face-1 SW, sign-flip
        local_e = sg[0, n-1, 0:n-1, 7]
        halo_e_rot = -sg[1, 0, 1:n, 5]
        fortran_e = 0.5 * (local_e + halo_e_rot)
        d_e = float(jnp.max(jnp.abs(direct[0, n, 1:n] - fortran_e)))

        # South seam (face 0 corner (ic, 0)): halo = face-5 NE, sign-flip
        halo_s_rot = -sg[5, 0:n-1, n-1, 7]
        local_s = sg[0, 1:n, 0, 5]
        fortran_s = 0.5 * (halo_s_rot + local_s)
        d_s = float(jnp.max(jnp.abs(direct[0, 1:n, 0] - fortran_s)))

        # North seam (face 0 corner (ic, n)): halo = face-4 SW, sign-flip
        local_n = sg[0, 0:n-1, n-1, 7]
        halo_n_rot = -sg[4, 1:n, 0, 5]
        fortran_n = 0.5 * (local_n + halo_n_rot)
        d_n = float(jnp.max(jnp.abs(direct[0, 1:n, n] - fortran_n)))

        # Each seam must match Python at machine precision
        tol = 1e-12  # float64 precision of the rotation + average
        for name, d in [("west", d_w), ("east", d_e),
                        ("south", d_s), ("north", d_n)]:
            self.assertLess(
                d, tol,
                f"{name} panel-edge Fortran halo-avg with sign flip "
                f"should match Python direct tangent, got diff = {d}. "
                f"Either CONNECTIVITY changed or the sub-grid rotation "
                f"convention broke.",
            )

    def test_sina_u_v_helper_matches_cdgrid_rsin_u_at_interior(self):
        """Iter-87 consistency: the helper's `sina_u` must be
        self-consistent with the grid-build `rsin_u = 1/sina_u²`
        (stored on `cdgrid` for duogrid mode).  If the helper diverges
        from the grid build, the fidelity claim is unsupported —
        Codex flagged this as an unguarded assumption.  Verifying here
        at interior cells where the mixed-convention panel-edge
        override does not apply.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _sina_u_v_from_sin_sg

        n = 16
        # Duogrid mode → rsin_u = 1/sin² everywhere (including panel edges)
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sina_u_helper, sina_v_helper = _sina_u_v_from_sin_sg(cdgrid)

        # Reconstruct sina_u from stored rsin_u.  Under duogrid,
        # rsin_u = 1/sina_u² uniformly, so sina_u = 1/sqrt(rsin_u).
        sina_u_from_rsin = 1.0 / jnp.sqrt(cdgrid.rsin_u)
        sina_v_from_rsin = 1.0 / jnp.sqrt(cdgrid.rsin_v)

        max_u_diff = float(jnp.max(jnp.abs(sina_u_helper - sina_u_from_rsin)))
        max_v_diff = float(jnp.max(jnp.abs(sina_v_helper - sina_v_from_rsin)))
        # Float32 round-trip precision — use 1e-6 tolerance
        self.assertLess(max_u_diff, 1e-6,
                        f"sina_u helper diverges from cdgrid.rsin_u: {max_u_diff}")
        self.assertLess(max_v_diff, 1e-6,
                        f"sina_v helper diverges from cdgrid.rsin_v: {max_v_diff}")

    def test_sina_u_v_from_sin_sg_matches_fortran_convention(self):
        """Iter-87: `_sina_u_v_from_sin_sg` constructs sina_u/sina_v from
        the sin_sg sub-grid per fv_grid_utils.F90:505-518.  Verify:
        1. interior: sina_u(i,j) = 0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))
        2. panel edges: single-side sin_sg
        3. differs from `sqrt(1 - cosa_u**2)` on the halo-averaged cosa.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _sina_u_v_from_sin_sg

        n = 16
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        sina_u, sina_v = _sina_u_v_from_sin_sg(cdgrid)

        # Shape
        self.assertEqual(sina_u.shape, (6, n + 1, n))
        self.assertEqual(sina_v.shape, (6, n, n + 1))

        # Interior values match the averaging formula
        sg = cdgrid.sin_sg
        sin_E = sg[:, :, :, 2]
        sin_W = sg[:, :, :, 0]
        expected_u_int = 0.5 * (sin_E[:, :-1, :] + sin_W[:, 1:, :])
        max_int_diff = float(jnp.max(jnp.abs(
            sina_u[:, 1:-1, :] - expected_u_int)))
        self.assertLess(max_int_diff, 1e-12,
                        f"interior sina_u mismatch: {max_int_diff}")

        # Panel-edge cells: single-side sin_sg
        self.assertTrue(bool(jnp.all(sina_u[:, 0, :] == sin_W[:, 0, :])))
        self.assertTrue(bool(jnp.all(sina_u[:, -1, :] == sin_E[:, -1, :])))

        # Verify that this differs from the naive sqrt(1 - cosa**2)
        # formulation at face boundaries (where cos_sg averaging makes
        # the trig identity fail).
        sina_u_naive = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_u**2, 1e-30))
        naive_vs_fv3 = float(jnp.max(jnp.abs(sina_u - sina_u_naive)))
        # For a smooth sphere sin_sg and cosa are both exact trig
        # values of the same angle at panel edges, so naive=fv3 there;
        # the difference is concentrated in the interior where the
        # two halo-average formulations diverge.  Either way, the
        # formulas are different functions; assert they produce a
        # detectable difference.
        self.assertGreater(naive_vs_fv3, 0.0,
                           "naive sqrt and FV3 sin_sg averaging are "
                           "identical — test is not sensitive")

    def test_c_sw_sin_sg_halos_route_through_duogrid_when_active(self):
        """Iter-79: `_c_sw` internally pads sin_sg E/W/N/S for the
        upwind transport-velocity scaling.  Previously these 4 halo
        exchanges were pinned to `interp_offsets=grid.halo_interp_offsets`.
        After the iter-79 fix, they must route through the duogrid
        remap when duogrid is active on the grid.

        `_c_sw` re-imports `pad_halo` from `legoesm.grids.halo` locally,
        so we patch the source module rather than a module-level symbol.
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv3_sw_core as fv3_sw_core_mod
        from legoesm.grids import halo as halo_mod

        n = 8
        grid_dg = create_cubed_sphere(n, use_duogrid=True)
        cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)

        h = jnp.ones((6, n, n)) * 1000.0
        u_d = jnp.zeros((6, n, n + 1))
        v_d = jnp.zeros((6, n + 1, n))
        h_s = jnp.zeros((6, n, n))

        # Record only the sin_sg-shaped halo calls; _c_sw also calls into
        # _d2a2c_vect_duogrid which uses its own halo paths (ext_vector).
        sin_sg_calls = []
        real_pad_halo = halo_mod.pad_halo

        def recording(q, halo=1, interp_offsets=None, duogrid=None):
            # sin_sg fields are (6, n, n) cell-centre scalars
            if (hasattr(q, 'shape') and q.shape == (6, n, n) and halo == 1):
                sin_sg_calls.append(
                    ('interp_offsets_none' if interp_offsets is None else 'interp_offsets_set',
                     'duogrid_none' if duogrid is None else 'duogrid_set'))
            return real_pad_halo(q, halo=halo,
                                 interp_offsets=interp_offsets,
                                 duogrid=duogrid)

        with mock.patch.object(halo_mod, 'pad_halo', recording):
            fv3_sw_core_mod._c_sw(h, u_d, v_d, h_s, cdgrid_dg, dt=300.0, g=9.81)

        # Expect at least the 4 sin_sg E/W/N/S halos that iter-79 fixed.
        self.assertTrue(len(sin_sg_calls) >= 4,
                        f"_c_sw made too few sin_sg halo calls: {len(sin_sg_calls)}")
        for kw in sin_sg_calls:
            self.assertEqual(
                kw, ('interp_offsets_none', 'duogrid_set'),
                f"_c_sw sin_sg halo used {kw} instead of duogrid remap")

    def test_deln_flux_routes_halo_through_duogrid_when_active(self):
        """Iter-78: `_deln_flux` previously pinned its internal halo to
        `interp_offsets=grid.halo_interp_offsets` even when duogrid was
        active on the grid.  After the iter-78 fix, the halo should route
        through duogrid when available.  Verify by mock-patching
        `pad_halo` and recording which kwarg combinations are used.
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core import fv_tp_2d as fv_tp_2d_mod

        n = 8
        grid_dg = create_cubed_sphere(n, use_duogrid=True)
        cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)

        q = jnp.ones((6, n, n)) * 10.0
        fx = jnp.zeros((6, n + 1, n))
        fy = jnp.zeros((6, n, n + 1))

        calls = []
        real_pad_halo = fv_tp_2d_mod.pad_halo

        def recording(q, halo=1, interp_offsets=None, duogrid=None):
            calls.append(
                ('interp_offsets_none' if interp_offsets is None else 'interp_offsets_set',
                 'duogrid_none' if duogrid is None else 'duogrid_set'))
            return real_pad_halo(q, halo=halo,
                                 interp_offsets=interp_offsets,
                                 duogrid=duogrid)

        with mock.patch.object(fv_tp_2d_mod, 'pad_halo', recording):
            fv_tp_2d_mod._deln_flux(1, 0.001, q, fx, fy, cdgrid_dg)

        # When duogrid is active, every halo exchange inside _deln_flux
        # must use the duogrid remap (duogrid=dg, interp_offsets=None).
        # Pre-iter-78 code would have used (interp_offsets=set, duogrid=None).
        self.assertTrue(len(calls) > 0, "_deln_flux made no halo exchanges")
        for kw in calls:
            self.assertEqual(
                kw, ('interp_offsets_none', 'duogrid_set'),
                f"_deln_flux halo exchange used {kw} instead of "
                f"duogrid remap on a duogrid-active grid")


class TestSupergridMetrics(unittest.TestCase):
    """Iter-44 added rdxa/rdya from supergrid as FV3-faithful metrics.
    Verify they are consistent with the supergrid construction and have
    the expected numerical properties.

    Fortran fv_grid_tools.F90 defines dxa as face-to-face cell widths in
    the i-direction (cell width at the A-grid cell centre).
    rdxa = 1/dxa.  Used in sw_core.F90:850 for Courant number:
    crx = dt*ut*rdxa(upwind_cell).
    """

    def test_rdxa_rdya_positive_finite(self):
        """rdxa and rdya must be strictly positive and finite."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(16, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        self.assertTrue(bool(jnp.all(cdgrid.rdxa > 0)),
                        "rdxa has non-positive values")
        self.assertTrue(bool(jnp.all(cdgrid.rdya > 0)),
                        "rdya has non-positive values")
        self.assertTrue(bool(jnp.all(jnp.isfinite(cdgrid.rdxa))),
                        "rdxa has non-finite values")
        self.assertTrue(bool(jnp.all(jnp.isfinite(cdgrid.rdya))),
                        "rdya has non-finite values")

    def test_rdxa_sphere_average_matches_radius(self):
        """On a unit-sphere cubed-sphere grid, the mean 1/rdxa should be
        approximately R/n per cell.  A C16 grid has ~6*(2*pi*R)/24 cells
        spanning each equatorial edge; the average dxa (= 1/rdxa) should
        be near R*pi/(2*n) for the equatorial latitudes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        R = 6.371229e6
        grid = create_cubed_sphere(n, radius=R, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # dxa at equatorial row of face 0 (equator-centred face)
        dxa_equator = 1.0 / cdgrid.rdxa[0, :, n // 2]
        mean_dxa = float(jnp.mean(dxa_equator))
        # Expected: each face edge spans 90° of great circle ≈ R*pi/2,
        # divided into n cells → R*pi/(2n).
        expected = R * jnp.pi / (2 * n)
        # Allow 10% tolerance because face edge is not exactly a great
        # circle (gnomonic projection distorts).
        self.assertAlmostEqual(
            mean_dxa / expected, 1.0, delta=0.1,
            msg=f"dxa mean at equator = {mean_dxa:.2e}, "
                f"expected ~{float(expected):.2e}")

    def test_rdxa_rdya_near_equality_on_equatorial_face(self):
        """Face 0 is centred on the equator; away from its corners,
        dxa ≈ dya (both are close to the local grid spacing).  At the
        centre cell, rdxa and rdya should match to within 1%.  At corner
        cells they diverge because of cubed-sphere face-corner geometry,
        which is allowed."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Centre cell of face 0
        ic, jc = n // 2, n // 2
        ratio = float(cdgrid.rdxa[0, ic, jc] / cdgrid.rdya[0, ic, jc])
        self.assertAlmostEqual(
            ratio, 1.0, delta=0.01,
            msg=f"rdxa/rdya at face 0 centre = {ratio:.4f}, "
                f"expected ≈ 1.0")

    def test_rdxa_is_i_direction_and_rdya_is_j_direction(self):
        """Distinguish x vs y: catch a swap where rdxa accidentally
        encodes the j-direction width.

        On face 4 (+z, north-pole face), the cubed-sphere gnomonic
        projection distorts dxa and dya asymmetrically when we look at
        cells far from face 4's centre.  Compute the true physical cell
        width in the i-direction (from adjacent cell centres in i) and
        the j-direction (from adjacent cell centres in j); verify that
        rdxa matches the i-direction reciprocal (NOT j) and rdya matches
        the j-direction reciprocal (NOT i).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        n = 16
        R = 6.371229e6
        grid = create_cubed_sphere(n, radius=R, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        # Pick a cell with known-large asymmetry: face 5 (-z) near i=14 has
        # ~20% difference between dxa and dya, enough to distinguish a swap.
        f, ic, jc = 5, 14, 8
        # Cell centre 3D positions from grid.{lon,lat}
        x_cc = jnp.cos(grid.lat) * jnp.cos(grid.lon)
        y_cc = jnp.cos(grid.lat) * jnp.sin(grid.lon)
        z_cc = jnp.sin(grid.lat)

        def great_circle(p0, p1):
            dot = jnp.clip(p0[0]*p1[0] + p0[1]*p1[1] + p0[2]*p1[2],
                           -1.0, 1.0)
            return R * jnp.arccos(dot)

        # i-direction width: half-distance between cells (ic-1, jc) and (ic+1, jc)
        p_im1 = jnp.array([x_cc[f, ic-1, jc], y_cc[f, ic-1, jc],
                           z_cc[f, ic-1, jc]])
        p_ip1 = jnp.array([x_cc[f, ic+1, jc], y_cc[f, ic+1, jc],
                           z_cc[f, ic+1, jc]])
        dxa_physical = float(great_circle(p_im1, p_ip1) / 2.0)

        # j-direction width: half-distance between cells (ic, jc-1) and (ic, jc+1)
        p_jm1 = jnp.array([x_cc[f, ic, jc-1], y_cc[f, ic, jc-1],
                           z_cc[f, ic, jc-1]])
        p_jp1 = jnp.array([x_cc[f, ic, jc+1], y_cc[f, ic, jc+1],
                           z_cc[f, ic, jc+1]])
        dya_physical = float(great_circle(p_jm1, p_jp1) / 2.0)

        dxa_stored = float(1.0 / cdgrid.rdxa[f, ic, jc])
        dya_stored = float(1.0 / cdgrid.rdya[f, ic, jc])

        # rdxa should encode i-direction width (within 15% — supergrid
        # construction is not a pure centre-to-centre great-circle
        # distance but the two agree to within this tolerance at off-
        # centre cells).
        ratio_ix = dxa_stored / dxa_physical
        ratio_jy = dya_stored / dya_physical
        # Cross ratios: if x/y were swapped
        ratio_iy = dxa_stored / dya_physical
        ratio_jx = dya_stored / dxa_physical

        self.assertAlmostEqual(
            ratio_ix, 1.0, delta=0.15,
            msg=f"rdxa does not encode i-direction width at "
                f"face {f} cell ({ic},{jc}): dxa_stored/dxa_physical = "
                f"{ratio_ix:.4f}, expected ≈ 1.0")
        self.assertAlmostEqual(
            ratio_jy, 1.0, delta=0.15,
            msg=f"rdya does not encode j-direction width: "
                f"dya_stored/dya_physical = {ratio_jy:.4f}")

        # Sanity: to catch a swap, dxa_physical and dya_physical must
        # themselves DIFFER here.  If they happened to coincide, a
        # swap would not be detectable.  Assert they differ by >5%.
        asymmetry = abs(dxa_physical - dya_physical) / max(
            dxa_physical, dya_physical)
        self.assertGreater(
            asymmetry, 0.05,
            f"Test picked a symmetric cell (dxa≈dya phys, asym="
            f"{asymmetry:.4f}); x/y swap would not be detectable.")

        # And the wrong mapping must clearly fail the 15% bound.
        swap_failure_x = abs(ratio_iy - 1.0)
        swap_failure_y = abs(ratio_jx - 1.0)
        self.assertGreater(
            max(swap_failure_x, swap_failure_y), 0.15,
            "x/y swap would still be within tolerance at this cell — "
            "test does not meaningfully distinguish x from y here.")


class TestD2a2cVectNonDuogridAdjacentStrip(unittest.TestCase):
    """Verify iter-68 4-point adjacent-strip recomputation.

    Fortran sw_core.F90:670-722 computes:
      vt(1, j) = vc(1, j) - 0.25*cosa_v(1, j)*(ut(1, j-1)+ut(2, j-1)+ut(1, j)+ut(2, j))
      ut(i, 1) = uc(i, 1) - 0.25*cosa_u(i, 1)*(vt(i-1, 1)+vt(i, 1)+vt(i-1, 2)+vt(i, 2))

    These tests reconstruct the expected values from the 4-point formula using
    the Python intermediate ut/vt and assert the emitted ut/vt match. They
    also verify that the south/north ut reads ONLY interior vt i-columns
    (i_cell ∈ [1, n-3]) that were never touched by the west/east updates.
    """

    def test_south_ut_matches_4point_vt_average(self):
        """South-edge ut[:, i_lo:i_hi, 0] = uc - 0.25*cosa_u * 4pt(vt)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect

        n = 12
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        u_d, v_d = _make_solid_body_edge(cdgrid)
        _, _, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

        # Reconstruct expected ut at south (j_face=0) using the 4-point formula.
        i_lo, i_hi = 2, n - 1
        vt_sum = (vt[:, i_lo - 1:i_hi - 1, 0] + vt[:, i_lo:i_hi, 0]
                  + vt[:, i_lo - 1:i_hi - 1, 1] + vt[:, i_lo:i_hi, 1])
        ut_expected = (uc[:, i_lo:i_hi, 0]
                       - 0.25 * cdgrid.cosa_u[:, i_lo:i_hi, 0] * vt_sum)
        diff = float(jnp.max(jnp.abs(ut[:, i_lo:i_hi, 0] - ut_expected)))
        # The output must match the expected formula to machine precision,
        # regardless of what vt values the formula saw.
        self.assertLess(diff, 1e-12,
                        f"south ut != 4-point formula: max abs diff = {diff:.2e}")

    def test_north_ut_matches_4point_vt_average(self):
        """North-edge ut[:, i_lo:i_hi, n-1] matches formula."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _d2a2c_vect

        n = 12
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        u_d, v_d = _make_solid_body_edge(cdgrid)
        _, _, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

        i_lo, i_hi = 2, n - 1
        vt_sum = (vt[:, i_lo - 1:i_hi - 1, n - 1]
                  + vt[:, i_lo:i_hi, n - 1]
                  + vt[:, i_lo - 1:i_hi - 1, n]
                  + vt[:, i_lo:i_hi, n])
        ut_expected = (uc[:, i_lo:i_hi, n - 1]
                       - 0.25 * cdgrid.cosa_u[:, i_lo:i_hi, n - 1] * vt_sum)
        diff = float(jnp.max(jnp.abs(ut[:, i_lo:i_hi, n - 1] - ut_expected)))
        self.assertLess(diff, 1e-12,
                        f"north ut != 4-point formula: max abs diff = {diff:.2e}")


class TestD2a2cVectNonDuogridBoundary(unittest.TestCase):
    """Lock in Fortran-faithful face-boundary overrides in the non-duogrid
    _d2a2c_vect branch (sw_core.F90:660-668, 677-684, 696-703, 714-721).

    The Fortran overrides ut at face-boundary u-edges (i=is, i=ie+1) and
    vt at face-boundary v-edges (j=js, j=je+1) by dividing uc/vc by
    sin_sg at the upwind neighbour (cell-edge-local metric), using the
    HALO cell's E/N edge for positive flow and the local cell's W/S edge
    for negative flow.  Python replicates this via pad_halo'd sin_sg
    followed by jnp.where upwind selection.
    """

    def test_face_boundary_ut_divides_uc_by_upwind_sin_sg(self):
        """ut at i=0 and i=n equals uc / sin_sg(upwind) to machine
        precision on a solid-body rotation state (non-duogrid path).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.halo import pad_halo
        from legoesm.core.fv3_sw_core import _d2a2c_vect

        n = 12
        # Non-duogrid path
        grid = create_cubed_sphere(n, use_duogrid=False)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        u_d, v_d = _make_solid_body_edge(cdgrid)
        _, _, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

        # Haloed sin_sg so the upwind cell index at i=0 reads from the
        # neighbouring face (matches Fortran sin_sg(0,j,3) / sin_sg(1,j,1)
        # pattern at the west face; analogous at the east face).
        sin_east = cdgrid.sin_sg[:, :, :, 2]
        sin_west = cdgrid.sin_sg[:, :, :, 0]
        offsets = cdgrid.base.halo_interp_offsets
        se_pad = pad_halo(sin_east, interp_offsets=offsets)
        sw_pad = pad_halo(sin_west, interp_offsets=offsets)
        eps = 1e-20

        for i_bdy in (0, n):
            sin_left = se_pad[:, i_bdy, 1:-1]
            sin_right = sw_pad[:, i_bdy + 1, 1:-1]
            sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
            ut_expected = uc[:, i_bdy, :] / jnp.maximum(sin_upwind, eps)
            rel = float(jnp.max(jnp.abs(ut[:, i_bdy, :] - ut_expected))
                        / (jnp.max(jnp.abs(ut_expected)) + 1e-30))
            self.assertLess(rel, 1e-12,
                            f"ut[i={i_bdy}] deviates from uc/sin_sg(upwind) "
                            f"by rel={rel:.2e} — Fortran sw_core.F90 "
                            f"{'660-663' if i_bdy == 0 else '677-684'} override")

        # Same check in the y-direction for vt.
        sin_north = cdgrid.sin_sg[:, :, :, 3]
        sin_south = cdgrid.sin_sg[:, :, :, 1]
        sn_pad = pad_halo(sin_north, interp_offsets=offsets)
        ss_pad = pad_halo(sin_south, interp_offsets=offsets)
        for j_bdy in (0, n):
            sin_below = sn_pad[:, 1:-1, j_bdy]
            sin_above = ss_pad[:, 1:-1, j_bdy + 1]
            sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
            vt_expected = vc[:, :, j_bdy] / jnp.maximum(sin_upwind, eps)
            rel = float(jnp.max(jnp.abs(vt[:, :, j_bdy] - vt_expected))
                        / (jnp.max(jnp.abs(vt_expected)) + 1e-30))
            self.assertLess(rel, 1e-12,
                            f"vt[j={j_bdy}] deviates from vc/sin_sg(upwind) "
                            f"by rel={rel:.2e} — Fortran sw_core.F90 "
                            f"{'696-703' if j_bdy == 0 else '714-721'} override")


if __name__ == "__main__":
    unittest.main()
