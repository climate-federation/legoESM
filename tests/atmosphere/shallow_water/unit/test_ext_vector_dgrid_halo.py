"""Acceptance test: FB-chain D-grid cross-face halo vs analytic W2 winds.

Root cause of the FB SW panel-edge instability (2026-07-10): this model's
D winds are ``u_d = V·x̂`` and ``v_d = V·x̂⊥`` (``angle_edge_y`` is the
i-tangent angle at y-edge midpoints), but ``ext_vector_dgrid`` ran the
FV3 covariant machinery on them (cosa_s inversion in step 1,
``ew_ext[...,1]`` y-line-tangent projection in step 3).  The resulting
O(cosa_s·|V|) convention error flips sign across panel seams and is a
~100% error in the small v_d projection — e.g. at C36, face4 west-end
j-halo carried ~35-40 m/s (u magnitude) where the physical v_d is
~13-17 m/s, and face0 halo values were sign-flipped.

The fix is ``basis="orthogonal"`` in ``ext_vector_dgrid`` +
``cubed_a2d_halo_orthogonal``, used by the FB-chain halo helpers
(``_pad_halo_dgrid_for_ppm``, ``_pad_halo_uc_vc_via_d2a2c``).

NOTE (probing lesson): a direction-smoothness probe on ew_ext/es_ext
shows ~+0.999 continuity — the convention error is invisible that way.
Only value-level checks against analytic winds (this test) expose it.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.duogrid import (
    cubed_a2d_halo_orthogonal,
    ext_vector_dgrid,
)
from legoesm.core.fv3_sw_core import (
    _pad_halo_dgrid_for_ppm,
    _sina_u_v_from_sin_sg,
    fb_v_d_to_covariant,
    fb_v_d_to_orthogonal,
    fv3_fb_sw_step,
)

N = 36
H = 2


@pytest.fixture(scope="module")
def w2_setup():
    """C36 duogrid + analytic Williamson-2 edge-staggered D winds."""
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    dg = grid.duogrid
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    # Matrix-runner W2 IC recipe: u_east = u0 cos(lat), v_north = 0.
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    # Analytic reference at the EXTENDED halo positions: exact geographic
    # wind at the duogrid ext A-points pushed through the same (model-
    # convention) A→D projection used by the pipeline.
    ng = dg.ng
    n_p = N + 2 * H
    sl = slice(ng - H, ng - H + n_p)
    ue_exact = u0 * jnp.cos(dg.ext_lat[:, sl, sl])
    ud_ref_f, vd_ref_f = cubed_a2d_halo_orthogonal(
        ue_exact, jnp.zeros_like(ue_exact), dg, H)
    ud_ref = ud_ref_f[:, :, H - 1:H + N]   # (6, n+2H, n+1)
    vd_ref = vd_ref_f[:, H - 1:H + N, :]   # (6, n+1, n+2H)
    return grid, cdgrid, u_d, v_d, ud_ref, vd_ref


class TestReferenceAnchoring:
    def test_reference_interior_matches_analytic_ic(self, w2_setup):
        """The analytic reference must reproduce the model-convention IC in
        the interior — anchors the reference itself (guards against a
        pipeline-vs-pipeline tautology)."""
        _, _, u_d, v_d, ud_ref, vd_ref = w2_setup
        int_u = float(jnp.max(jnp.abs(ud_ref[:, H:H + N, :] - u_d)))
        int_v = float(jnp.max(jnp.abs(vd_ref[:, :, H:H + N] - v_d)))
        assert int_u < 0.05, f"u reference interior err {int_u:.4f}"
        assert int_v < 0.05, f"v reference interior err {int_v:.4f}"


class TestPadHaloDgridForPPM:
    def test_halo_matches_analytic_extended_winds(self, w2_setup):
        """All cross-face halo strips of both D components must match the
        analytic winds at the extended positions to 0.2 m/s at C36."""
        _, cdgrid, u_d, v_d, ud_ref, vd_ref = w2_setup
        u_h, v_h = _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=H, basis="orthogonal")
        du = jnp.abs(u_h - ud_ref)   # u: i-halo on axis 1
        dv = jnp.abs(v_h - vd_ref)   # v: j-halo on axis 2
        u_halo_err = float(jnp.maximum(
            jnp.max(du[:, :H, :]), jnp.max(du[:, H + N:, :])))
        v_halo_err = float(jnp.maximum(
            jnp.max(dv[:, :, :H]), jnp.max(dv[:, :, H + N:])))
        assert u_halo_err < 0.2, f"u_d halo err {u_halo_err:.4f} m/s"
        assert v_halo_err < 0.2, f"v_d halo err {v_halo_err:.4f} m/s"

    def test_interior_preserved_exactly(self, w2_setup):
        _, cdgrid, u_d, v_d, _, _ = w2_setup
        u_h, v_h = _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=H, basis="orthogonal")
        np.testing.assert_array_equal(
            np.asarray(u_h[:, H:H + N, :]), np.asarray(u_d))
        np.testing.assert_array_equal(
            np.asarray(v_h[:, :, H:H + N]), np.asarray(v_d))

    def test_face0_halo_not_sign_flipped(self, w2_setup):
        """Broken behaviour: face0 ring-1 j-halo sign-flipped vs the smooth
        continuation.  Fixed: sign and value agree with analytic."""
        _, cdgrid, u_d, v_d, _, vd_ref = w2_setup
        _, v_h = _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=H, basis="orthogonal")
        ring = np.asarray(v_h[0, :, H - 1])
        ref = np.asarray(vd_ref[0, :, H - 1])
        big = np.abs(ref) > 0.5
        assert np.all(np.sign(ring[big]) == np.sign(ref[big])), (
            f"face0 ring-1 j-halo sign flip: got {ring[big][:6]} "
            f"vs ref {ref[big][:6]}")

    def test_face4_halo_end_magnitude_physical(self, w2_setup):
        """Broken behaviour: face4 halo carried ~35-40 m/s (u magnitude)
        where physical v_d is ~13-17 m/s.  Fixed: end-of-strip halo values
        stay in the physical band."""
        _, cdgrid, u_d, v_d, _, _ = w2_setup
        _, v_h = _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=H, basis="orthogonal")
        end_val = float(jnp.abs(v_h[4, 0, H - 1]))
        assert 10.0 < end_val < 20.0, (
            f"face4 ring-1 j-halo west-end |v| = {end_val:.2f}, expected "
            f"~13-17 (u-magnitude ~35-40 = broken covariant projection)")


class TestFBCovariantConversion:
    """2026-07-10 FB wind-convention fix: the FB chain converts the model's
    orthogonal v_d to true FV3 covariant at entry and back at exit."""

    def test_entry_matches_analytic_covariant(self, w2_setup):
        """v_cov = cosa_u*(V.x) + sina_u*(V.rot90(x)) must match the analytic
        covariant projection everywhere incl. panel seams (W2, C36)."""
        _, cdgrid, u_d, v_d, _, _ = w2_setup
        u0 = 2.0 * np.pi * float(cdgrid.radius) / (12.0 * 86400.0)
        ue = u0 * jnp.cos(cdgrid.lat_edge_y)
        u_o = ue * jnp.cos(cdgrid.angle_edge_y)
        v_o = -ue * jnp.sin(cdgrid.angle_edge_y)
        sina_u, _ = _sina_u_v_from_sin_sg(cdgrid)
        v_cov_ref = cdgrid.cosa_u * u_o + sina_u * v_o
        v_cov = fb_v_d_to_covariant(u_d, v_d, cdgrid)
        err = float(jnp.max(jnp.abs(v_cov - v_cov_ref)))
        assert err < 0.05, f"entry conversion err {err:.4f} m/s"
        # the conversion is O(cosa*u) large at seams — guard non-triviality
        assert float(jnp.max(jnp.abs(v_cov - v_d))) > 5.0

    def test_roundtrip_identity(self, w2_setup):
        """to_orthogonal(to_covariant(v_d)) == v_d (interior exact; seam
        residual contracts ≤~0.15 per fixed-point pass).  Seam rows asserted
        SPECIFICALLY: the residual is a dt-INDEPENDENT per-step kick there —
        the 2-pass inverse left ~7e-4 m/s at seams (would fail the 5e-4
        gate); the 3-pass inverse measures ~7e-5 (W2 C36, 2026-07-10)."""
        _, cdgrid, u_d, v_d, _, _ = w2_setup
        v_rt = fb_v_d_to_orthogonal(
            u_d, fb_v_d_to_covariant(u_d, v_d, cdgrid), cdgrid)
        err = jnp.abs(v_rt - v_d)   # v_d (6, n+1, n)
        seam_err = float(jnp.max(jnp.stack([
            jnp.max(err[:, :2, :]), jnp.max(err[:, -2:, :]),   # i seam rows
            jnp.max(err[:, :, :2]), jnp.max(err[:, :, -2:]),   # j seam rows
        ])))
        assert seam_err < 5e-4, f"seam-row roundtrip err {seam_err:.2e} m/s"
        assert float(jnp.max(err)) < 5e-4, (
            f"roundtrip err {float(jnp.max(err)):.2e} m/s")

    def test_fb_step_w2_near_steady(self, w2_setup):
        """One fv3_fb_sw_step on steady W2: the one-step wind increment must
        stay at the balanced-truncation level everywhere (the pre-fix vertex
        v_d kick was 0.71 m/s; post-fix ~0.03)."""
        grid, cdgrid, u_d, v_d, _, _ = w2_setup
        from tests.test_cases.williamson import williamson_test2
        sw = williamson_test2(grid)
        h1, u1, v1 = fv3_fb_sw_step(
            sw.h.data, u_d, v_d, sw.h_s.data, cdgrid, 300.0,
            d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1)
        dv = float(jnp.max(jnp.abs(v1 - v_d)))
        du = float(jnp.max(jnp.abs(u1 - u_d)))
        assert dv < 0.15, f"W2 one-step max|dv| {dv:.4f} m/s (seam kick back?)"
        assert du < 0.15, f"W2 one-step max|du| {du:.4f} m/s"


class TestExtVectorDgridDispatch:
    def test_unknown_basis_raises(self, w2_setup):
        grid, cdgrid, _, _, _, _ = w2_setup
        utmp = jnp.zeros((6, N, N))
        with pytest.raises(ValueError, match="basis"):
            ext_vector_dgrid(
                utmp, utmp, grid.duogrid,
                grid.cos_angle, grid.sin_angle,
                cdgrid.cos_sg[:, :, :, 4],
                halo=H, basis="typo")

    def test_covariant_default_unchanged(self, w2_setup):
        """Production `_d2a2c_vect_duogrid` path (basis='covariant') must be
        bit-identical to the pre-fix behaviour: the orthogonal fix is
        opt-in for the FB chain only."""
        grid, cdgrid, u_d, v_d, _, _ = w2_setup
        dg = grid.duogrid
        dx_u = cdgrid.dx_edge_y
        dy_v = cdgrid.dy_edge_x
        wu, wv = u_d * dx_u, v_d * dy_v
        utmp = (wu[:, :, :-1] + wu[:, :, 1:]) / (
            dx_u[:, :, :-1] + dx_u[:, :, 1:])
        vtmp = (wv[:, :-1, :] + wv[:, 1:, :]) / (
            dy_v[:, :-1, :] + dy_v[:, 1:, :])
        args = (utmp, vtmp, dg, grid.cos_angle, grid.sin_angle,
                cdgrid.cos_sg[:, :, :, 4])
        ud_default, vd_default = ext_vector_dgrid(*args, halo=H)
        ud_cov, vd_cov = ext_vector_dgrid(*args, halo=H, basis="covariant")
        np.testing.assert_array_equal(
            np.asarray(ud_default), np.asarray(ud_cov))
        np.testing.assert_array_equal(
            np.asarray(vd_default), np.asarray(vd_cov))
        # And the orthogonal basis genuinely differs at the seam halo.
        ud_orth, vd_orth = ext_vector_dgrid(
            *args, halo=H, basis="orthogonal")
        assert float(jnp.max(jnp.abs(vd_orth - vd_cov))) > 1.0
