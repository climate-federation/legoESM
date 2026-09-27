"""Direct unit tests for tripole internals.

Addresses PR #268 slopbuster REJECT #2: ``_detect_fold``,
``_compute_rotation_angles``, ``_read_nemo_mesh_mask`` were reachable
only via the global-overturning tripolar runner.  These tests
exercise each one against a synthetic NEMO-style mesh.
"""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest


# -------------------------------------------------------------------------
# _detect_fold
# -------------------------------------------------------------------------


class TestDetectFold:
    def test_ideal_symmetric_fold(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 16, 32
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))

        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
        # Force the fold row to be perfectly symmetric (constant in lon).
        gphit = gphit.at[-1].set(60.0)

        # A constant fold row is self-symmetric under BOTH index conventions, so
        # ``auto`` cannot disambiguate (see test_constant_fold_row_is_ambiguous);
        # pass the convention explicitly to exercise the n_lon-1-i descriptor.
        fold = _detect_fold(glamt, gphit, n_lat, n_lon,
                            fold_convention="n_lon-1-i")

        assert fold.is_active is True
        assert fold.fold_j == n_lat - 1
        # perm_T[i] = n_lon - 1 - i
        assert jnp.all(fold.perm_T == jnp.arange(n_lon - 1, -1, -1))
        assert jnp.all(fold.perm_v == fold.perm_T)
        assert fold.vector_sign_u == -1.0
        assert fold.vector_sign_v == -1.0

    def test_constant_fold_row_is_ambiguous_under_auto(self):
        """PR B #4: a constant (perfectly symmetric) fold row fits BOTH index
        conventions, so ``fold_convention='auto'`` must raise rather than
        silently guess n_lon-1-i (the wrong origin for a de-haloed mesh)."""
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 16, 32
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon)).at[-1].set(60.0)

        with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
            _detect_fold(glamt, gphit, n_lat, n_lon)  # auto

        # Both explicit conventions resolve it (no ambiguity once chosen).
        for conv in ("n_lon-1-i", "(n_lon-i)%n_lon"):
            fold = _detect_fold(glamt, gphit, n_lat, n_lon, fold_convention=conv)
            assert fold.is_active is True

    def test_asymmetric_fold_raises(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 16, 32
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
        # Break the fold-row symmetry: monotonically rising in i
        gphit = gphit.at[-1].set(jnp.linspace(50.0, 70.0, n_lon))

        with pytest.raises(ValueError, match="Fold symmetry check failed"):
            _detect_fold(glamt, gphit, n_lat, n_lon)

    def test_asymmetry_threshold_kwarg_loosens_check(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 8, 16
        glamt = jnp.zeros((n_lat, n_lon))
        gphit = jnp.broadcast_to(
            jnp.linspace(-70.0, 60.0, n_lat)[:, None], (n_lat, n_lon),
        )
        # Inject a 0.5 deg lat bump that the default 0.1 tolerance rejects.
        # Perturb column 1 (not 0): column 0 is a fixed point of the
        # (n_lon-i)%n_lon convention, so a col-0 bump would be "symmetric"
        # under that perm and slip past the auto-detected check. Column 1 is
        # asymmetric under BOTH supported conventions, so both reject it.
        gphit = gphit.at[-1].set(60.0)
        gphit = gphit.at[-1, 1].set(60.5)

        # Default 0.1 tolerance rejects the 0.5 deg asymmetry outright.
        with pytest.raises(ValueError, match="Fold symmetry check failed"):
            _detect_fold(glamt, gphit, n_lat, n_lon)
        # Loosening max_fold_asym_deg to 1.0 passes the symmetry check. But a
        # col-1 bump is asymmetric by the SAME amount under both conventions
        # (a tie), so 'auto' now correctly raises as ambiguous (PR B #4) —
        # naming the convention explicitly confirms the loosened threshold let
        # the 0.5 deg bump through.
        with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
            _detect_fold(glamt, gphit, n_lat, n_lon, max_fold_asym_deg=1.0)
        fold = _detect_fold(
            glamt, gphit, n_lat, n_lon, max_fold_asym_deg=1.0,
            fold_convention="n_lon-1-i",
        )
        assert fold.is_active is True

    def test_near_constant_fold_row_nonzero_tie_raises(self):
        """PR B #4 (codex): the tie test is on the DIFFERENCE of the two
        asymmetries, not their magnitude. A near-constant fold row whose two
        candidate asymmetries are both small-but-nonzero AND essentially equal
        is still ambiguous and must raise — the earlier 'both <= tol' form
        wrongly let such a row through."""
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 8, 16
        glamt = jnp.zeros((n_lat, n_lon))
        gphit = jnp.broadcast_to(
            jnp.linspace(-70.0, 60.0, n_lat)[:, None], (n_lat, n_lon),
        )
        # Tiny col-1 bump (1e-4 deg): both conventions see asym 1e-4 (>> the
        # 1e-6 default tie tol in magnitude) but their DIFFERENCE is ~0, so the
        # row is genuinely ambiguous.
        gphit = gphit.at[-1].set(60.0)
        gphit = gphit.at[-1, 1].set(60.0 + 1e-4)
        with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
            _detect_fold(glamt, gphit, n_lat, n_lon)

    def test_pure_periodic_fold_convention(self):
        """De-haloed NEMO meshes (e.g. eORCA025) self-permute the fold row
        under perm[i] = (n_lon - i) % n_lon, not n_lon-1-i. _detect_fold must
        auto-detect this convention rather than rejecting the grid."""
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 12, 24
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
        # cos(2*pi*i/n_lon) is even under i -> (n_lon - i) % n_lon (fixed points
        # at i=0 and i=n_lon/2) but NOT under i -> n_lon-1-i, so this fold row
        # is symmetric only in the pure-periodic convention.
        i = jnp.arange(n_lon)
        fold_row = 60.0 + 5.0 * jnp.cos(2.0 * jnp.pi * i / n_lon)
        gphit = gphit.at[-1].set(fold_row)

        fold = _detect_fold(glamt, gphit, n_lat, n_lon)
        assert fold.is_active is True
        expected = (n_lon - jnp.arange(n_lon)) % n_lon
        assert jnp.all(fold.perm_T == expected)
        assert jnp.all(fold.perm_v == fold.perm_T)
        # A constant fold row is a genuine tie (symmetric under BOTH origins):
        # auto now RAISES instead of silently guessing n_lon-1-i (PR B #4); the
        # eORCA1.2 origin is recovered only by naming it explicitly.
        gphit2 = gphit.at[-1].set(60.0)
        with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
            _detect_fold(glamt, gphit2, n_lat, n_lon)
        fold2 = _detect_fold(glamt, gphit2, n_lat, n_lon,
                             fold_convention="n_lon-1-i")
        assert jnp.all(fold2.perm_T == jnp.arange(n_lon - 1, -1, -1))

    def test_explicit_fold_convention_overrides_ambiguous_tie(self):
        """A constant fold-row latitude is a genuine tie: ``auto`` raises (PR B
        #4) rather than silently guessing. An explicit ``fold_convention``
        resolves it (still verified against the tolerance), giving a de-haloed
        mesh its correct pure-periodic origin."""
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 12, 24
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(
            jnp.linspace(-80.0, 60.0, n_lat)[:, None], (n_lat, n_lon))
        gphit = gphit.at[-1].set(60.0)   # constant fold row -> auto ties
        # auto -> raises (ambiguous, neither origin distinguishable)
        with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
            _detect_fold(glamt, gphit, n_lat, n_lon)
        # explicit pure-periodic -> the OTHER origin, verified (asym 0 <= tol)
        f = _detect_fold(glamt, gphit, n_lat, n_lon,
                         fold_convention="(n_lon-i)%n_lon")
        assert jnp.all(f.perm_T == (n_lon - jnp.arange(n_lon)) % n_lon)
        assert jnp.all(f.perm_v == f.perm_T)

    def test_explicit_wrong_convention_raises(self):
        """An explicit convention that does not actually fold the grid is
        rejected by the tolerance check (not silently accepted); a bogus name
        is rejected outright."""
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 12, 24
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(
            jnp.linspace(-80.0, 60.0, n_lat)[:, None], (n_lat, n_lon))
        # cos(2*pi*i/n) is symmetric ONLY under (n_lon-i)%n_lon, not n_lon-1-i.
        i = jnp.arange(n_lon)
        gphit = gphit.at[-1].set(60.0 + 5.0 * jnp.cos(2.0 * jnp.pi * i / n_lon))
        with pytest.raises(ValueError, match="Fold symmetry check failed"):
            _detect_fold(glamt, gphit, n_lat, n_lon,
                         fold_convention="n_lon-1-i")
        with pytest.raises(ValueError, match="fold_convention must be"):
            _detect_fold(glamt, gphit, n_lat, n_lon, fold_convention="bogus")


# -------------------------------------------------------------------------
# _compute_rotation_angles
# -------------------------------------------------------------------------


class TestComputeRotationAngles:
    def test_below_cap_is_identity(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        n_lat, n_lon = 12, 24
        # Regular lat-lon coordinates → no rotation expected
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 80.0, n_lat)
        glam = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphi = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))

        # cap_j past the end → all rows below cap
        cos_au, sin_au, cos_av, sin_av = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=n_lat,
        )
        assert jnp.allclose(cos_au, 1.0)
        assert jnp.allclose(sin_au, 0.0)
        assert jnp.allclose(cos_av, 1.0)
        assert jnp.allclose(sin_av, 0.0)

    def test_above_cap_uses_local_grid_orientation(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        n_lat, n_lon = 12, 24
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glam = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphi = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))

        cos_au, sin_au, _, _ = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=n_lat // 2,
        )
        # In the no-cap (regular lat-lon) limit the local i-axis is east,
        # so cos≈1, sin≈0 even above the cap row.
        assert jnp.allclose(cos_au, 1.0, atol=1e-5)
        assert jnp.allclose(sin_au, 0.0, atol=1e-5)


class TestRotationAngleSignAndSeam:
    """H5 (v-point sin sign) + H6 (branch-cut unwrap) gates for
    _compute_rotation_angles.  The pre-existing TestComputeRotationAngles cases
    use regular lat-lon caps (dlat=0 => sin==0) so they cannot catch either
    bug; these build a genuinely TILTED / SEAM-CROSSING cap and check the
    SIGNED angle."""

    @staticmethod
    def _tilted_cap(alpha_deg, lon0=0.0, lat0=60.0, d=0.05, n=6):
        # Rigid in-tangent-plane rotation of a lat-lon patch by alpha about
        # (lon0, lat0): i-axis -> (cos a east, sin a north); j-axis ->
        # (-sin a east, cos a north).  Both u- and v-faces sample the SAME cell
        # orientation, so cos/sin must agree at both after the fix.
        a = np.deg2rad(alpha_deg)
        # indexing='xy' (default): ii varies along axis1 (i), jj along axis0 (j)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n))
        cphi = np.cos(np.deg2rad(lat0))
        east = (ii * d * np.cos(a)) + (jj * (-d) * np.sin(a))
        north = (ii * d * np.sin(a)) + (jj * d * np.cos(a))
        glam = lon0 + east / cphi
        gphi = lat0 + north
        return jnp.asarray(glam), jnp.asarray(gphi)

    def test_h5_vpoint_sin_matches_uface_convention(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        alpha = 30.0
        glam, gphi = self._tilted_cap(alpha)
        cos_au, sin_au, cos_av, sin_av = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=0)
        j, i = 2, 2  # interior cell (avoid the wrap-padded last col/row)
        s = np.sin(np.deg2rad(alpha))
        c = np.cos(np.deg2rad(alpha))
        # u-face recovers +alpha (sanity; passes before and after)
        assert np.isclose(float(sin_au[j, i]), s, atol=5e-3)
        # H5: v-face MUST carry the SAME +sin(alpha), not -sin(alpha)
        assert np.isclose(float(sin_av[j, i]), s, atol=5e-3)
        assert np.isclose(float(cos_av[j, i]), c, atol=5e-3)
        assert np.isclose(float(sin_av[j, i]), float(sin_au[j, i]), atol=5e-3)

    def test_h5_consumer_matrix_is_orthonormal_roundtrip(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        glam, gphi = self._tilted_cap(30.0)
        cos_au, sin_au, cos_av, sin_av = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=0)
        j, i = 2, 2
        # Exact 2x2 the consumers assemble (i-row +sin_au, j-row -sin_av).
        M = np.array([[float(cos_au[j, i]), float(sin_au[j, i])],
                      [-float(sin_av[j, i]), float(cos_av[j, i])]])
        # Cross-row inner product = 2*sin*cos (~0.87) WITH the bug, ~0 after the
        # fix.  (A magnitude-only cos^2+sin^2==1 test would NOT catch this.)
        # atol 1.5e-2 accommodates the u-vs-v finite-difference floor: the u- and
        # v-faces sample the SAME cell angle at half-cell-offset points under a
        # varying cos(lat) metric, so sin_au and sin_av agree only to ~3e-3 on
        # this coarse (d=0.05) patch.  The bug leaves a 0.87 cross-term -- ~60x
        # this tol -- so the check stays decisively non-vacuous.
        assert np.allclose(M @ M.T, np.eye(2), atol=1.5e-2)
        v_geo = np.array([1.0, 0.0])          # pure eastward geographic vector
        v_back = M.T @ (M @ v_geo)             # geo->grid->geo round-trip
        assert np.allclose(v_back, v_geo, atol=1.5e-2)

    def test_h6_seam_cell_matches_interior_neighbor(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        # A due-east cap row (true i-axis = east everywhere) whose longitudes
        # are wrapped across +-180 at one interior column.  dlat=0 isolates H6.
        n_lat, n_lon = 4, 6
        true_lon = 176.0 + 2.0 * np.arange(n_lon)          # monotonic true lon
        wrapped = ((true_lon + 180.0) % 360.0) - 180.0     # ...,178,-180,-178,...
        glam = jnp.asarray(
            np.broadcast_to(wrapped[None, :], (n_lat, n_lon)).copy())
        gphi = jnp.asarray(np.full((n_lat, n_lon), 75.0))
        cos_au, sin_au, _, _ = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=0)
        # i-axis is due east at EVERY interior column, incl. the seam column
        # (index 1, the 178 -> -180 step): cos=1, sin=0 everywhere.
        assert np.allclose(np.asarray(cos_au[:, :n_lon - 1]), 1.0, atol=1e-6)
        assert np.allclose(np.asarray(sin_au[:, :n_lon - 1]), 0.0, atol=1e-6)



# -------------------------------------------------------------------------
# _read_nemo_mesh_mask  (skip without netCDF4)
# -------------------------------------------------------------------------


netcdf4 = pytest.importorskip("netCDF4")


def _write_synthetic_mesh_mask(
    path: str, n_lat: int = 8, n_lon: int = 16, curved_fold: bool = False,
) -> None:
    """Write a minimal NEMO-style mesh_mask.nc file.

    The default fold row (gphit[-1]) is constant — a deliberately AMBIGUOUS
    fold. ``curved_fold=True`` instead writes a realistic non-constant fold row
    that is self-symmetric under exactly ONE convention (like a real ORCA
    bipolar cap), so ``fold_convention='auto'`` resolves it WITHOUT raising.
    """
    ds = netcdf4.Dataset(path, "w")
    ds.createDimension("y", n_lat)
    ds.createDimension("x", n_lon)

    glamt = np.broadcast_to(
        np.linspace(0.0, 360.0, n_lon, endpoint=False)[None, :],
        (n_lat, n_lon),
    ).astype(np.float64)
    gphit = np.array(
        np.broadcast_to(
            np.linspace(-80.0, 80.0, n_lat)[:, None], (n_lat, n_lon),
        ),
        dtype=np.float64,
    )
    if curved_fold:
        # cos(2*pi*i/n_lon) is even under i -> (n_lon - i) % n_lon but NOT under
        # i -> n_lon-1-i, so this fold row fits exactly one convention -> auto
        # disambiguates it (no tie), mirroring a real (non-flat) ORCA fold.
        i = np.arange(n_lon)
        gphit[-1, :] = 80.0 + 5.0 * np.cos(2.0 * np.pi * i / n_lon)
    ones = np.ones((n_lat, n_lon), dtype=np.float64)

    for name, arr in (
        ("glamt", glamt), ("gphit", gphit),
        ("glamu", glamt), ("gphiu", gphit),
        ("glamv", glamt), ("gphiv", gphit),
        ("e1t", ones * 1.0e4), ("e2t", ones * 1.0e4),
        ("e1u", ones * 1.0e4), ("e2u", ones * 1.0e4),
        ("e1v", ones * 1.0e4), ("e2v", ones * 1.0e4),
        ("tmask", ones), ("umask", ones), ("vmask", ones),
    ):
        v = ds.createVariable(name, "f8", ("y", "x"))
        v[:] = arr
    ds.close()


class TestReadNemoMeshMask:
    def test_reads_2d_vars(self):
        from legoesm.grids.tripole import _read_nemo_mesh_mask

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=8, n_lon=16)
            raw = _read_nemo_mesh_mask(path)

        for name in ("glamt", "gphit", "e1t", "e2t",
                     "e1u", "e2u", "e1v", "e2v",
                     "tmask", "umask", "vmask"):
            assert name in raw, f"missing {name}"
            assert raw[name].shape == (8, 16)

    def test_squeezes_extra_singleton_axes(self):
        """NEMO mesh files often carry singleton time/depth axes; ensure
        they are squeezed to plain 2D."""
        from legoesm.grids.tripole import _read_nemo_mesh_mask

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            ds = netcdf4.Dataset(path, "w")
            ds.createDimension("t", 1)
            ds.createDimension("z", 1)
            ds.createDimension("y", 4)
            ds.createDimension("x", 8)
            v = ds.createVariable("glamt", "f8", ("t", "z", "y", "x"))
            v[:] = np.zeros((1, 1, 4, 8))
            v = ds.createVariable("gphit", "f8", ("t", "z", "y", "x"))
            v[:] = np.zeros((1, 1, 4, 8))
            for name in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v"):
                v = ds.createVariable(name, "f8", ("y", "x"))
                v[:] = np.ones((4, 8))
            ds.close()

            raw = _read_nemo_mesh_mask(path)
            assert raw["glamt"].shape == (4, 8)
            assert raw["gphit"].shape == (4, 8)


# -------------------------------------------------------------------------
# gradient_y_cgrid: zero-dy_v south pole row (NEMO min_dx_m=0.0) must not
# produce NaN/inf before the polar wall-BC overwrite (PR358 review).
# -------------------------------------------------------------------------


class TestPadTripoleGridSouth:
    """``pad_tripole_grid_south`` (the SPMD ``n_lat % N == 0`` enabler)."""

    def test_pad_preserves_wet_rows_and_keeps_fold_at_north(self):
        from legoesm.grids.tripole import (
            create_synthetic_tripole, pad_tripole_grid_south,
        )

        n_lat, n_lon, n_pad = 15, 24, 5    # 15 -> 20 (= 4 * 5), the eORCA025 case
        g = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
        assert g.fold.is_active
        gp = pad_tripole_grid_south(g, n_pad)

        # shape grew by n_pad on the latitude (T/u leading) axis
        assert int(gp.n_lat) == n_lat + n_pad
        assert int(gp.n_lon) == n_lon
        assert gp.lat_T.shape == (n_lat + n_pad, n_lon)
        assert gp.dx_u.shape == (n_lat + n_pad, n_lon + 1)
        assert gp.dx_v.shape == (n_lat + n_pad + 1, n_lon)      # v leads n_lat+1
        assert gp.area_q.shape == (n_lat + n_pad + 1, n_lon + 1)

        # the ORIGINAL rows are bit-exact, just shifted +n_pad (south padding)
        for name in ("lat_T", "lon_T", "dx_T", "dy_T", "area_T", "f_T",
                     "dx_u", "dy_u", "cos_alpha_u", "sin_alpha_u"):
            a = np.asarray(getattr(g, name))
            b = np.asarray(getattr(gp, name))
            np.testing.assert_array_equal(
                b[n_pad:], a, err_msg=f"{name} wet rows not preserved")
        # v/q rows (leading n_lat+1): original block preserved after the n_pad
        # prepended south-wall rows.
        for name in ("dx_v", "dy_v", "f_v", "area_q", "cos_alpha_v",
                     "sin_alpha_v"):
            a = np.asarray(getattr(g, name))
            b = np.asarray(getattr(gp, name))
            np.testing.assert_array_equal(
                b[n_pad:], a, err_msg=f"{name} wet rows not preserved")

        # fold seam stays at the NEW northernmost row; cap shifts +n_pad
        assert gp.fold.is_active is True
        assert int(gp.fold.fold_j) == int(g.fold.fold_j) + n_pad
        assert int(gp.fold.fold_j) == (n_lat + n_pad) - 1
        assert int(gp.fold.cap_j) == int(g.fold.cap_j) + n_pad
        np.testing.assert_array_equal(np.asarray(gp.fold.perm_T),
                                      np.asarray(g.fold.perm_T))

        # added south rows are FINITE + positive metrics + strictly-south latitude
        assert bool(np.all(np.isfinite(np.asarray(gp.lat_T))))
        assert bool(np.all(np.asarray(gp.dx_T) > 0.0))
        assert bool(np.all(np.asarray(gp.area_T) > 0.0))
        lat_col = np.asarray(gp.lat_T)[:, 0]
        assert bool(np.all(np.diff(lat_col) > 0.0)), "latitude not monotone north"

        # total_area is PRESERVED (the area-weighted-mean denominator must not
        # pick up the spurious land padding)
        np.testing.assert_array_equal(np.asarray(gp.total_area),
                                      np.asarray(g.total_area))

    def test_pad_zero_is_identity_and_inactive_fold_raises(self):
        from legoesm.grids.tripole import (
            create_synthetic_tripole, pad_tripole_grid_south,
        )
        from legoesm.grids.latlon import create_latlon_geometry

        g = create_synthetic_tripole(n_lat=12, n_lon=24)
        assert pad_tripole_grid_south(g, 0) is g       # n_pad=0 -> identity

        reg = create_latlon_geometry(n_lat=12, n_lon=24)
        assert not reg.fold.is_active
        with pytest.raises(ValueError, match="ACTIVE bipolar fold"):
            pad_tripole_grid_south(reg, 3)


class TestGradientYZeroPolarMetric:
    def test_zero_south_dy_v_is_finite_and_differentiable(self):
        import equinox as eqx
        import jax
        from legoesm.grids.tripole import create_synthetic_tripole
        from legoesm.grids.operators_latlon_cgrid import gradient_y_cgrid

        g = create_synthetic_tripole(n_lat=16, n_lon=24)
        # Emulate a NEMO tripole built with min_dx_m=0.0: exact-zero south
        # pole metric row.  The unified gradient_y path divides every row by
        # dy_v before zeroing the polar v-faces, so without the safe floor
        # the south row would be nonzero/0.
        g0 = eqx.tree_at(lambda t: t.dy_v, g, g.dy_v.at[0].set(0.0))
        f = (jnp.arange(16, dtype=jnp.float64)[:, None]
             * jnp.ones((16, 24)))

        out = gradient_y_cgrid(f, g0)
        assert bool(jnp.all(jnp.isfinite(out))), "gradient_y_cgrid not finite"
        # Polar wall BC: south v-face gradient is zero.
        assert float(jnp.max(jnp.abs(out[0]))) == 0.0

        grad = jax.grad(
            lambda x: jnp.sum(gradient_y_cgrid(x, g0) ** 2))(f)
        assert bool(jnp.all(jnp.isfinite(grad))), "AD grad not finite"


class TestCreateTripoleGridFoldDefault:
    """PR B #4 (codex rounds 2-3): the public loader on an ambiguous (constant)
    fold row must fail LOUD by default — a runtime warning is insufficient for
    batch/long runs where a wrong seam origin silently corrupts the northern
    halo. The synthetic mesh has a constant fold row (gphit[-1] is uniform). The
    legacy n_lon-1-i fallback is available only as an EXPLICIT opt-in."""

    def test_default_auto_loads_realistic_curved_fold(self):
        """A REALISTIC mesh has a curved (non-constant) fold row — like a real
        ORCA bipolar cap — so the default ``fold_convention='auto'`` resolves it
        WITHOUT raising. This is the production path (eORCA1.2 etc.): the
        fail-loud default only trips on a degenerate constant fold row, so it is
        not an operational break for real meshes."""
        import warnings as _warnings
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=12, n_lon=24, curved_fold=True)
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")  # no warning on a resolvable fold
                geom = create_tripole_grid(path)  # default auto, no convention
        assert geom is not None

    def test_default_auto_raises_on_ambiguous_fold(self):
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=8, n_lon=16)
            with pytest.raises(ValueError, match="(?i)ambiguous fold_convention"):
                create_tripole_grid(path)  # default: fail loud on a flat fold

    def test_legacy_fold_fallback_is_explicit_opt_in(self):
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=8, n_lon=16)
            with pytest.warns(RuntimeWarning, match="(?i)legacy"):
                geom = create_tripole_grid(path, allow_ambiguous_legacy_fold=True)
        assert geom is not None  # opt-in loads with the legacy origin

    def test_explicit_convention_silences_warning(self):
        import warnings as _warnings
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=8, n_lon=16)
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")  # any warning becomes an error
                geom = create_tripole_grid(path, fold_convention="n_lon-1-i")
        assert geom is not None

    def test_native_u_fields_start_at_redundant_east_face(self):
        """NEMO U(i) is T(i)'s east face; legoESM U[0] is the west image."""
        from legoesm.grids.tripole import (
            _compute_rotation_angles,
            create_tripole_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            n_lat, n_lon = 8, 16
            _write_synthetic_mesh_mask(
                path, n_lat=n_lat, n_lon=n_lon, curved_fold=True)
            with netcdf4.Dataset(path, "a") as ds:
                columns = np.arange(n_lon, dtype=np.float64)[None, :]
                rows = np.arange(n_lat, dtype=np.float64)[:, None]
                ds["e1u"][:] = 1000.0 + 10.0 * rows + columns
                ds["e2u"][:] = 2000.0 + 10.0 * rows + columns
                ds["e1v"][:] = 3000.0 + 10.0 * rows + columns
                ds["e2v"][:] = 4000.0 + 10.0 * rows + columns
                glamu = np.asarray(ds["glamu"][:])
                gphiu = np.asarray(ds["gphiu"][:])
                glamv = np.asarray(ds["glamv"][:])
                gphiv = np.asarray(ds["gphiv"][:])
                e1u = np.asarray(ds["e1u"][:])
                e2u = np.asarray(ds["e2u"][:])
                e1v = np.asarray(ds["e1v"][:])
                e2v = np.asarray(ds["e2v"][:])
                gphit = np.asarray(ds["gphit"][:])

            grid = create_tripole_grid(
                path, dtype=jnp.float64,
                fold_convention="(n_lon-i)%n_lon")

        np.testing.assert_array_equal(np.asarray(grid.dx_u)[:, 1:], e1u)
        np.testing.assert_array_equal(np.asarray(grid.dy_u)[:, 1:], e2u)
        np.testing.assert_array_equal(np.asarray(grid.dx_u)[:, 0], e1u[:, -1])
        np.testing.assert_array_equal(np.asarray(grid.dy_u)[:, 0], e2u[:, -1])
        np.testing.assert_array_equal(np.asarray(grid.dx_u)[:, -1], e1u[:, -1])
        np.testing.assert_array_equal(np.asarray(grid.dy_u)[:, -1], e2u[:, -1])

        # V metrics and generic V-face Coriolis retain their pre-change map.
        np.testing.assert_array_equal(np.asarray(grid.dx_v)[1:], e1v)
        np.testing.assert_array_equal(np.asarray(grid.dy_v)[1:], e2v)
        f_t = grid.f_T
        expected_f_v = jnp.concatenate(
            [f_t[0:1], 0.5 * (f_t[:-1] + f_t[1:]), f_t[-1:]], axis=0)
        np.testing.assert_array_equal(np.asarray(grid.f_v),
                                      np.asarray(expected_f_v))

        native_cos_u, native_sin_u, _, _ = _compute_rotation_angles(
            jnp.asarray(glamu), jnp.asarray(gphiu),
            jnp.asarray(glamv), jnp.asarray(gphiv), grid.fold.cap_j)
        np.testing.assert_array_equal(
            np.asarray(grid.cos_alpha_u)[:, 1:], np.asarray(native_cos_u))
        np.testing.assert_array_equal(
            np.asarray(grid.sin_alpha_u)[:, 1:], np.asarray(native_sin_u))


class TestPadCoversEveryGeometryField:
    """The pad must grow EVERY array field per its stagger — the tripwire.

    ``cos_lat_v`` was added to the geometry after the pad was written and the
    pad missed it; nothing failed until the eORCA025 full-card 4-GPU run,
    where the SPMD band slicer handed the north band one fewer v-face row
    than the interior bands and the per-field band stack died with "All input
    arrays must have the same shape" (job 9471878).  eORCA1's n_lat divides
    evenly, so the pad never fires there and the miss was invisible.  These
    tests are generic over ``_fields`` so the NEXT field added to the
    geometry cannot repeat this.
    """

    def _padded_pair(self, n_lat=15, n_lon=24, n_pad=5):
        from legoesm.grids.tripole import (
            create_synthetic_tripole, pad_tripole_grid_south,
        )
        g = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
        return g, pad_tripole_grid_south(g, n_pad), n_lat, n_pad

    def test_every_array_field_grows_with_its_stagger(self):
        """Each axis that measured n_lat (or n_lat+1) must grow by n_pad;
        every other axis is unchanged.  A field the pad forgot keeps its old
        shape and fails here by construction."""
        g, gp, n_lat, n_pad = self._padded_pair()
        checked = 0
        for name in g._fields:
            a = getattr(g, name)
            if not hasattr(a, "shape") or getattr(a, "ndim", 0) == 0:
                continue                      # scalars / fold descriptor
            expect = tuple(
                d + n_pad if d in (n_lat, n_lat + 1) else d
                for d in a.shape)
            got = getattr(gp, name)
            assert got is not None, f"{name} became None under the pad"
            assert tuple(got.shape) == expect, (
                f"{name}: pad missed it — {tuple(a.shape)} -> "
                f"{tuple(got.shape)}, expected {expect}")
            checked += 1
        assert checked >= 20                  # the audit actually ran

    def test_cos_lat_v_wet_entries_bit_exact_and_new_entries_sane(self):
        """Shape alone cannot catch a SHIFTED or recomputed profile (codex +
        GLM both flagged it): the original v-face entries must survive the
        pad bit-exact at offset n_pad, and the new land-row entries must be
        finite and positive."""
        import numpy as np
        g, gp, n_lat, n_pad = self._padded_pair()
        a = np.asarray(g.cos_lat_v)
        b = np.asarray(gp.cos_lat_v)
        np.testing.assert_array_equal(
            b[n_pad:], a, err_msg="cos_lat_v wet entries not preserved")
        assert bool(np.all(np.isfinite(b[:n_pad])))
        assert bool(np.all(b[:n_pad] > 0.0))

    def test_past_pole_extrapolation_warns_but_keeps_the_contract(self, capsys):
        """A past-the-pole extrapolation (the synthetic full-sphere grid does
        this legitimately) must WARN loudly, and the land-row contract —
        finite, positive metrics — must still hold via the clamp.  A hard
        error was tried first and rejected: it broke the full-sphere fixture
        the existing pad tests rely on."""
        import numpy as np
        _, gp, n_lat, n_pad = self._padded_pair()   # 12-deg rows cross -90
        out = capsys.readouterr().out
        assert "crosses the pole" in out
        c = np.asarray(gp.cos_lat_v)
        assert bool(np.all(np.isfinite(c))) and bool(np.all(c > 0.0))

    def test_padded_grid_band_stacks_are_uniform(self):
        """The exact operation that crashed at 1/4 degree: slice the padded
        grid into N bands and stack every array field across them."""
        import numpy as np
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            _geom_array_field_names, build_band_grids,
        )
        _, gp, n_lat, n_pad = self._padded_pair()   # 15 + 5 = 20
        n_dev = 4                                   # 20 % 4 == 0
        bands = build_band_grids(gp, n_dev)
        for name in _geom_array_field_names(bands[0]):
            shapes = {tuple(np.shape(getattr(b, name))) for b in bands}
            assert len(shapes) == 1, (
                f"{name}: ragged across bands {sorted(shapes)} — the "
                f"eORCA025 stack crash")
            np.stack([np.asarray(getattr(b, name)) for b in bands])

    def test_seam_wall_rows_grows_and_new_rows_are_walled(self):
        from legoesm.grids.tripole import (
            create_synthetic_tripole, pad_tripole_grid_south,
        )
        import jax.numpy as jnp
        import numpy as np
        n_lat, n_pad = 15, 5
        g = create_synthetic_tripole(n_lat=n_lat, n_lon=24)
        seam = jnp.zeros((n_lat,)).at[3:7].set(1.0)
        g = g._replace(seam_wall_rows=seam)
        gp = pad_tripole_grid_south(g, n_pad)
        got = np.asarray(gp.seam_wall_rows)
        assert got.shape == (n_lat + n_pad,)
        np.testing.assert_array_equal(got[n_pad:], np.asarray(seam))
        assert bool(np.all(got[:n_pad] == 1.0)), "new land rows must be walled"


# -------------------------------------------------------------------------
# u-point metric alignment (the NEMO C-grid index convention)
# -------------------------------------------------------------------------
def _write_mesh_with_varying_u_metrics(path, n_lat=8, n_lon=16):
    """Mesh whose e1u/e2u VARY ALONG A ROW, with a curved (unambiguous) fold.

    The shared fixture above writes uniform metrics, under which a one-column
    shift of e1u/e2u is exactly the identity -- so it cannot see this bug.
    Varying them along i is what makes the alignment observable at all, and is
    the situation on a real ORCA mesh north of ~20N.
    """
    ds = netcdf4.Dataset(path, "w")
    ds.createDimension("y", n_lat)
    ds.createDimension("x", n_lon)
    i = np.arange(n_lon)
    glamt = np.broadcast_to(
        np.linspace(0.0, 360.0, n_lon, endpoint=False)[None, :],
        (n_lat, n_lon)).astype(np.float64)
    gphit = np.array(np.broadcast_to(
        np.linspace(-80.0, 80.0, n_lat)[:, None], (n_lat, n_lon)),
        dtype=np.float64)
    gphit[-1, :] = 80.0 + 5.0 * np.cos(2.0 * np.pi * i / n_lon)
    ones = np.ones((n_lat, n_lon), dtype=np.float64)
    # distinct along-i profiles for e1u and e2u so a swap cannot pass either
    e1u = 1.0e4 * (1.0 + 0.10 * i)[None, :] * ones
    e2u = 1.0e4 * (1.0 + 0.37 * i)[None, :] * ones
    for name, arr in (
        ("glamt", glamt), ("gphit", gphit),
        ("glamu", glamt), ("gphiu", gphit),
        ("glamv", glamt), ("gphiv", gphit),
        ("e1t", ones * 1.0e4), ("e2t", ones * 1.0e4),
        ("e1u", e1u), ("e2u", e2u),
        ("e1v", ones * 1.0e4), ("e2v", ones * 1.0e4),
        ("tmask", ones), ("umask", ones), ("vmask", ones),
    ):
        v = ds.createVariable(name, "f8", ("y", "x"))
        v[:] = arr
    ds.close()
    return e1u, e2u


class TestUPointMetricAlignment:
    """NEMO's u-point ``i`` is EAST of T-cell ``i``; ours is WEST of cell ``i``.

    MEASURED on the real mesh, not taken from documentation: on the 1-degree
    part of eORCA1 ``glamu - glamt = +0.5000`` deg.  So our face ``i`` carries
    NEMO's ``e1u[i-1]`` and face 0 wraps to the LAST column.

    These tests FAIL against the previous construction, which appended
    ``e1u[:, 0:1]`` and so handed every face the metric of the face one column
    EAST -- a bug that is invisible on a uniform mesh and worth several to
    twenty percent on every zonal face north of 30N of a real ORCA grid.
    """

    def _grid(self, tmp):
        from legoesm.grids.tripole import create_tripole_grid

        path = os.path.join(tmp, "mesh_varying.nc")
        e1u, e2u = _write_mesh_with_varying_u_metrics(path)
        return create_tripole_grid(path), e1u, e2u

    def test_u_face_takes_the_metric_of_the_face_to_its_west(self):
        with tempfile.TemporaryDirectory() as tmp:
            grid, e1u, e2u = self._grid(tmp)
            dx_u = np.asarray(grid.dx_u, dtype=np.float64)
            dy_u = np.asarray(grid.dy_u, dtype=np.float64)
            # face i (i>=1) carries NEMO's u-point i-1
            np.testing.assert_allclose(dx_u[:, 1:], e1u, rtol=1e-6)
            np.testing.assert_allclose(dy_u[:, 1:], e2u, rtol=1e-6)

    def test_wrap_column_is_the_last_not_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            grid, e1u, e2u = self._grid(tmp)
            dx_u = np.asarray(grid.dx_u, dtype=np.float64)
            dy_u = np.asarray(grid.dy_u, dtype=np.float64)
            np.testing.assert_allclose(dx_u[:, 0], e1u[:, -1], rtol=1e-6)
            np.testing.assert_allclose(dy_u[:, 0], e2u[:, -1], rtol=1e-6)
            # and NOT the first column, which is what the old build used
            assert not np.allclose(dy_u[:, 0], e2u[:, 0], rtol=1e-6)

    def test_metric_and_coriolis_reference_the_same_cell_pair(self):
        """f at a u-face averages cells i-1 and i, so the metric must too.

        This is the internal consistency the bug broke: the Coriolis
        construction already used the WEST convention while the metric used the
        EAST one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            grid, _e1u, e2u = self._grid(tmp)
            f_T = np.asarray(grid.f_T, dtype=np.float64)
            f_u = np.asarray(grid.f_u, dtype=np.float64)
            expected = 0.5 * (np.roll(f_T, 1, axis=1) + f_T)
            np.testing.assert_allclose(f_u[:, :f_T.shape[1]], expected,
                                       rtol=1e-6, atol=1e-12)
            dy_u = np.asarray(grid.dy_u, dtype=np.float64)
            np.testing.assert_allclose(dy_u[:, 1:], e2u, rtol=1e-6)

    def test_u_rotation_angles_pad_like_the_u_metrics(self):
        """The U ANGLES must sit on the same faces as the U METRICS.

        ``_compute_rotation_angles`` returns the angle at NEMO's u-point
        ``i`` (east of T-cell ``i``), exactly as ``e1u`` does, so the padded
        array must carry it at face ``i+1`` and wrap the LAST column into
        face 0.  The previous build appended the FIRST column instead, which
        put the angle of the face one column EAST on every face while the
        metric beside it had already been corrected -- the two operands of
        the same rotation then referred to different faces.
        """
        from legoesm.grids.tripole import (
            _compute_rotation_angles,
            _read_nemo_mesh_mask,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh_varying.nc")
            _write_mesh_with_varying_u_metrics(path)
            from legoesm.grids.tripole import create_tripole_grid

            grid = create_tripole_grid(path)
            raw = _read_nemo_mesh_mask(path)
            c_raw, s_raw, _, _ = _compute_rotation_angles(
                raw["glamu"], raw["gphiu"], raw["glamv"], raw["gphiv"],
                int(grid.fold.cap_j))
            c_raw = np.asarray(c_raw, dtype=np.float64)
            s_raw = np.asarray(s_raw, dtype=np.float64)
            # non-vacuity: the angle must really vary along i, else any
            # padding passes
            assert np.ptp(s_raw[-1, :]) > 1.0e-6
            cos_u = np.asarray(grid.cos_alpha_u, dtype=np.float64)
            sin_u = np.asarray(grid.sin_alpha_u, dtype=np.float64)
            np.testing.assert_allclose(cos_u[:, 1:], c_raw, rtol=1e-12)
            np.testing.assert_allclose(sin_u[:, 1:], s_raw, rtol=1e-12)
            np.testing.assert_allclose(sin_u[:, 0], s_raw[:, -1], rtol=1e-12)
            # and NOT the first column, which is what the old build used
            assert not np.allclose(sin_u[:, 0], s_raw[:, 0])


class TestMeshCoriolisIsOptIn:
    """``ff_t`` from the mesh file changes ``f_T``, so a card must ask for it.

    NEMO reads ``ff_t``/``ff_f`` from ``cn_domcfg`` whenever both exist
    (``domhgr.F90:222-227``), but ORCA's stored values differ from
    ``2*Omega*sin(lat)`` in the last bits, so adopting them silently would
    move every tripole run (ORCA1 OMIP production included).
    """

    def _mesh(self, tmp, ff_t_value):
        path = os.path.join(tmp, "mesh_ff.nc")
        _write_mesh_with_varying_u_metrics(path)
        ds = netcdf4.Dataset(path, "a")
        shape = ds.variables["gphit"].shape
        for name in ("ff_t", "ff_f"):
            v = ds.createVariable(name, "f8", ("y", "x"))
            v[:] = np.full(shape, ff_t_value, dtype=np.float64)
        ds.close()
        return path

    def test_default_keeps_the_analytic_coriolis(self):
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = self._mesh(tmp, 1.2345e-4)
            grid = create_tripole_grid(path)
            f_T = np.asarray(grid.f_T, dtype=np.float64)
            assert not np.allclose(f_T, 1.2345e-4)
            np.testing.assert_allclose(
                f_T,
                2.0 * float(grid.omega) * np.sin(
                    np.asarray(grid.lat_T, dtype=np.float64)),
                rtol=1e-12)

    def test_opt_in_reads_the_mesh_field(self):
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = self._mesh(tmp, 1.2345e-4)
            grid = create_tripole_grid(path, use_mesh_coriolis=True)
            np.testing.assert_allclose(
                np.asarray(grid.f_T, dtype=np.float64), 1.2345e-4, rtol=1e-12)

    def test_ff_f_is_carried_either_way(self):
        """The F-point field is a pure addition: only literal arms read it."""
        from legoesm.grids.tripole import create_tripole_grid

        with tempfile.TemporaryDirectory() as tmp:
            path = self._mesh(tmp, 1.2345e-4)
            assert create_tripole_grid(path).ff_f is not None
            assert create_tripole_grid(
                path, use_mesh_coriolis=True).ff_f is not None
