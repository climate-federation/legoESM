"""#1455: the implicit vertical momentum solve's u/v-face control volume.

NEMO builds the face scale factor on the UNMASKED reference ladder and applies
the wet-face mask SEPARATELY, when the column is summed:

    pe3u(ji,jj,jk) = 0.50 * ( pe3t(ji,jj,jk) + pe3t(ji+1,jj,jk) )
                                   -- cfgs/DINO/MY_SRC/zgr_lib.F90:231 (ln_zco)
    e3u_0(ji,jj,jk) = MIN( e3t_0(ji,jj,jk), e3t_0(ji+1,jj,jk) )
                                   -- tools/DOMAINcfg/src/domzgr.F90:1166 (ln_zps)
    hu_0(:,:) = hu_0(:,:) + e3u_0(:,:,jk) * umask(:,:,jk)
                                   -- src/OCE/DOM/domain.F90:145

``e3t_0`` is defined (and positive) at EVERY level, including below the
seafloor, so a bathymetry-staircase face -- one where ``umask`` is zero because
only one of the two neighbouring columns is wet -- contributes exactly ZERO to
``hu_0``.

legoESM's cell-centre thickness is ``h_partial * J``, which is already ZERO
below the seafloor, so interpolating it to the face returns HALF a thickness at
a staircase level.  Summing that inflates the column divisor of the implicit
solve's barotropic split.  This module pins the masking order at the site that
runs: ``LatLonCGridOceanModel._apply_implicit_vertical_mixing``.
"""

from __future__ import annotations

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_geometry, create_latlon_grid
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)

# The staircase: lon columns 0 and 1 are wet to level 2 (three cells), lon
# columns 2 and 3 only to level 1 (two cells).  Level 2 is therefore CLOSED at
# the u-face between column 1 and column 2 (and, through the periodic wrap,
# between column 3 and column 0).
_DEEP_BOTTOM_LEVEL = 2
_SHALLOW_BOTTOM_LEVEL = 1
_N_LON = 4
_N_LAT = 4
_N_LEV = 3


@pytest.fixture(autouse=True)
def _fp64():
    """The assertions below are exact-zero / exact-sum identities; fp32
    rounding would hide a half-thickness of a few metres in a 4000 m column."""
    orig_x64 = jax.config.jax_enable_x64
    orig_policy = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_policy)
    jax.config.update("jax_enable_x64", orig_x64)


def _staircase():
    """(grid, full-step-z coordinate, rest state) with the staircase above."""
    grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
    z_ref = create_ocean_z_star(
        n_levels=_N_LEV, H_max=1000.0, dz_surface=100.0, dz_deep=500.0,
    )
    # Snap the bathymetry to a level INTERFACE so every wet cell is a FULL cell
    # (NEMO ln_zco): h_partial is then exactly dz_ref or exactly 0.
    z_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))
    H = jnp.where(
        jnp.arange(_N_LON)[None, :] < 2,
        z_half[_DEEP_BOTTOM_LEVEL + 1],
        z_half[_SHALLOW_BOTTOM_LEVEL + 1],
    )
    H = jnp.broadcast_to(H, (_N_LAT, _N_LON))
    coord = create_partial_cell_coordinate(z_ref, H)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_ref, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H,
    )
    return grid, coord, state


def _cfg():
    """Implicit vertical friction ON with the barotropic split ON -- the split
    is what feeds the face control volume to ``depth_mean``.  Everything else
    is off so nothing but the vertical solve runs."""
    return LatLonCGridOceanConfig.from_flat(
        K_h=0.0, K_bih=0.0, A_h=0.0, B_h=0.0, C_smag=0.0,
        bottom_drag_r=0.0,
        K_v=1.0e-5, A_v=1.0e-3,
        implicit_vertical_mixing=True,
        zdf_baroclinic_only=True,
        use_conservation_fixer=False,
        physics=None,
    )


def _capture_face_thicknesses(model, state, dt=600.0):
    """Run the REAL implicit vertical solve eagerly and record every
    ``h_face`` handed to ``depth_mean`` by it.

    Eager (not through ``model.step``, which jits) so the recorded arrays are
    concrete values rather than tracers.  ``depth_mean`` is patched on the
    module under test, so this captures the symbol that actually runs.
    """
    seen = []
    real = omlc.depth_mean

    def _recorder(field, h_face, min_water_column_m, **kw):
        seen.append(np.asarray(h_face))
        return real(field, h_face, min_water_column_m, **kw)

    with mock.patch.object(omlc, "depth_mean", _recorder):
        model._apply_implicit_vertical_mixing(state, dt, None)
    assert seen, "depth_mean was never called -- the split did not run"
    u_shape = (_N_LAT, _N_LON + 1, _N_LEV)
    v_shape = (_N_LAT + 1, _N_LON, _N_LEV)
    h_u = [a for a in seen if a.shape == u_shape]
    h_v = [a for a in seen if a.shape == v_shape]
    assert len(h_u) == 1 and len(h_v) == 1, (
        f"expected exactly one u-face and one v-face control volume, got "
        f"{[a.shape for a in seen]}")
    return h_u[0], h_v[0]


class TestStaircaseFaceControlVolume:

    def test_closed_face_level_carries_no_thickness(self):
        """The staircase u-face must carry ZERO thickness at the level its
        mask closes -- NEMO's ``e3u_0 * umask``.  Pre-fix this is HALF the
        reference cell, because the already-masked cell thickness (0 on the
        dry side) is averaged with the wet side's full cell."""
        grid, coord, state = _staircase()
        h_u, _ = _capture_face_thicknesses(
            LatLonCGridOceanModel(grid, coord, _cfg()), state)

        dz_ref = np.asarray(coord.dz_ref)
        # u-face index 2 sits between lon cell 1 (wet to level 2) and lon cell
        # 2 (wet to level 1).  Level 2 is closed there.
        closed = h_u[:, 2, _DEEP_BOTTOM_LEVEL]
        assert np.all(closed == 0.0), (
            f"closed staircase u-face carries {closed[0]} m of thickness; "
            f"NEMO's e3u_0*umask is 0.0 (the pre-fix value was "
            f"{0.5 * dz_ref[_DEEP_BOTTOM_LEVEL]} m = half the reference cell)")

    def test_column_divisor_matches_nemo_hu_0(self):
        """Hand-computed: ``hu_0 = sum_k e3u_0*umask``.  At the staircase face
        both columns are wet only at levels 0 and 1 (eta = 0, so the z-star
        Jacobian is 1 and e3u_0 is the reference ladder), giving
        ``dz_ref[0] + dz_ref[1]``.  Pre-fix the sum also picked up
        ``0.5*dz_ref[2]``."""
        grid, coord, state = _staircase()
        h_u, _ = _capture_face_thicknesses(
            LatLonCGridOceanModel(grid, coord, _cfg()), state)

        dz_ref = np.asarray(coord.dz_ref)
        expected = float(dz_ref[0] + dz_ref[1])
        pre_fix = expected + 0.5 * float(dz_ref[_DEEP_BOTTOM_LEVEL])
        got = float(h_u[0, 2].sum())
        assert got == pytest.approx(expected, rel=1e-12), (
            f"staircase column divisor {got} m != NEMO hu_0 {expected} m "
            f"(pre-fix value {pre_fix} m)")
        assert abs(got - pre_fix) > 1.0, (
            "the pre-fix and post-fix divisors must be separable -- pick a "
            "ladder whose level-2 cell is not vanishingly thin")

    def test_open_faces_are_untouched(self):
        """The fix must change ONLY faces the mask closes: an interior face
        with both columns wet keeps the plain cell-to-face average, which on a
        flat-eta full-step ladder is the reference thickness."""
        grid, coord, state = _staircase()
        h_u, h_v = _capture_face_thicknesses(
            LatLonCGridOceanModel(grid, coord, _cfg()), state)

        dz_ref = np.asarray(coord.dz_ref)
        # u-face 1 is between lon cells 0 and 1: both wet to level 2.
        np.testing.assert_allclose(h_u[0, 1], dz_ref, rtol=1e-12)
        # Every v-face is between two columns of the SAME depth (the staircase
        # runs in longitude only), so no v-face level is closed by the
        # staircase; interior v-faces keep the full ladder.
        np.testing.assert_allclose(h_v[1, 0], dz_ref, rtol=1e-12)

    def test_land_adjacent_face_is_closed_and_the_solve_stays_finite(self):
        """A LAND-adjacent face on a FLAT bottom is closed at every level, so
        its control volume is zero at every level and its column divisor
        collapses to the ``min_water_column_m`` floor.

        This is the case the staircase fixture cannot reach, and it is the one
        that puts a zero under a divisor: the implicit bottom-drag diagonal
        divides by a face thickness, and the mixed-precision column-mass
        restoration divides by the column sum.  Both must stay finite -- and
        the drag diagonal must keep dividing by the real thickness, because
        NEMO's own ``e3u`` in that term (dynzdf.F90:296) is never masked."""
        grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
        z_ref = create_ocean_z_star(
            n_levels=_N_LEV, H_max=1000.0, dz_surface=100.0, dz_deep=500.0)
        z_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))
        deep = float(z_half[_DEEP_BOTTOM_LEVEL + 1])
        # lon column 3 is LAND (H = 0); the rest is flat at full depth.
        H = jnp.where(jnp.arange(_N_LON)[None, :] == 3, 0.0, deep)
        H = jnp.broadcast_to(H, (_N_LAT, _N_LON))
        coord = create_partial_cell_coordinate(z_ref, H)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_ref, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H)
        model = LatLonCGridOceanModel(grid, coord, _cfg())
        h_u, _ = _capture_face_thicknesses(model, state)

        dz_ref = np.asarray(coord.dz_ref)
        # u-face 3 (cells 2 | 3) and u-face 4 == face 0 (cells 3 | 0) both
        # border the land column, so every level is closed.
        for j in (3, 4):
            np.testing.assert_allclose(h_u[0, j], 0.0, atol=0.0)
        # ... while the two faces bounded by water keep the full ladder.
        for j in (1, 2):
            np.testing.assert_allclose(h_u[0, j], dz_ref, rtol=1e-12)

        # The solve itself must remain finite everywhere, including the
        # mixed-precision path whose column-mass correction divides by the
        # column sum that just collapsed to zero.
        for f32_solve in ("0", "1"):
            os.environ["LEGOESM_VMIX_F32_SOLVE"] = f32_solve
            try:
                out = model._apply_implicit_vertical_mixing(state, 600.0, None)
            finally:
                os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
            for name, arr in (("u", out.u.data), ("v", out.v.data),
                              ("T", out.T.data), ("S", out.S.data)):
                assert bool(jnp.all(jnp.isfinite(arr))), (
                    f"{name} is non-finite with LEGOESM_VMIX_F32_SOLVE="
                    f"{f32_solve} -- a closed face's zero column divisor "
                    f"reached a division")

    def test_land_column_gradients_are_finite(self):
        """The forward value on a dead column is masked away downstream, so a
        ``0/0`` there is invisible until someone differentiates the model.
        This repo requires end-to-end ``jax.grad``, so assert the REVERSE-mode
        derivative is finite on exactly the geometry that produces the zero
        column, on the mixed-precision path that divides by it."""
        grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
        z_ref = create_ocean_z_star(
            n_levels=_N_LEV, H_max=1000.0, dz_surface=100.0, dz_deep=500.0)
        z_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))
        deep = float(z_half[_DEEP_BOTTOM_LEVEL + 1])
        H = jnp.broadcast_to(
            jnp.where(jnp.arange(_N_LON)[None, :] == 3, 0.0, deep),
            (_N_LAT, _N_LON))
        coord = create_partial_cell_coordinate(z_ref, H)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_ref, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H)
        model = LatLonCGridOceanModel(grid, coord, _cfg())

        def loss(u):
            st = state._replace(u=state.u.replace(data=u))
            out = model._apply_implicit_vertical_mixing(st, 600.0, None)
            return jnp.sum(out.u.data ** 2) + jnp.sum(out.v.data ** 2)

        u0 = jnp.full_like(state.u.data, 0.05)
        os.environ["LEGOESM_VMIX_F32_SOLVE"] = "1"
        try:
            g = jax.grad(loss)(u0)
        finally:
            os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
        n_bad = int(jnp.sum(~jnp.isfinite(g)))
        assert n_bad == 0, (
            f"{n_bad} of {g.size} gradient entries are non-finite -- the "
            f"column-mass restoration divided by a zero column sum")

    def test_drag_diagonal_divides_by_the_unmasked_thickness(self):
        """NEMO's implicit bottom-drag diagonal divides by ``e3u(iku,Kaa)``
        (dynzdf.F90:296) -- a scale factor NEMO never masks.

        The face mask has a source the CELL mask knows nothing about: a
        configured partial-periodic seam wall closes a u-face column at every
        level even where BOTH neighbouring cells are wet.  The bottom-level
        indicator is built from ``bottom_level`` and therefore still fires
        there, so feeding it the MASKED control volume would make it divide by
        the 1e-10 floor: a ~1e10 entry in the matrix diagonal, five thousand
        billion times its real value.  Pin that it divides by the real
        thickness instead."""
        grid = create_latlon_geometry(
            _N_LAT, _N_LON, radius=constants.R_earth, omega=constants.Omega)
        seam = jnp.zeros(_N_LAT, dtype=jnp.float64).at[0].set(1.0)
        grid = grid._replace(seam_wall_rows=seam)
        z_ref = create_ocean_z_star(
            n_levels=_N_LEV, H_max=1000.0, dz_surface=100.0, dz_deep=500.0)
        z_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))
        H = jnp.full((_N_LAT, _N_LON), float(z_half[_DEEP_BOTTOM_LEVEL + 1]))
        coord = create_partial_cell_coordinate(z_ref, H)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_ref, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H)
        cfg = LatLonCGridOceanConfig.from_flat(
            K_h=0.0, K_bih=0.0, A_h=0.0, B_h=0.0, C_smag=0.0,
            K_v=1.0e-5, A_v=1.0e-3,
            implicit_vertical_mixing=True,
            zdf_baroclinic_only=True,
            zdf_drag_in_matrix=True,
            barotropic_drag_substep=True,
            bottom_drag_scheme="nemo_quadratic",
            use_conservation_fixer=False,
            physics=None,
        )
        model = LatLonCGridOceanModel(grid, coord, cfg)
        # Non-zero velocity so the quadratic drag rate is non-zero: a drag
        # diagonal of exactly 0 would make this control vacuous.
        st = state._replace(
            u=state.u.replace(data=jnp.full_like(state.u.data, 0.2)),
            v=state.v.replace(data=jnp.full_like(state.v.data, 0.2)))

        # The method imports the solver from the vertical_mixing PACKAGE at
        # call time, so patch it there -- that is the symbol that runs.
        import legoesm.ocean.physics.vertical_mixing as vmix
        seen = {}
        real = vmix.implicit_vertical_diffusion_ocean

        def _recorder(field, K, dz, dz_half, dt, *, extra_diag=0.0):
            if field.shape == (_N_LAT, _N_LON + 1, _N_LEV):
                seen["u"] = np.asarray(extra_diag)
            return real(field, K, dz, dz_half, dt, extra_diag=extra_diag)

        with mock.patch.object(
                vmix, "implicit_vertical_diffusion_ocean", _recorder):
            model._apply_implicit_vertical_mixing(st, 600.0, None)
        assert "u" in seen, "the momentum solve did not run"
        e = seen["u"]
        assert np.all(np.isfinite(e))
        assert e.max() > 0.0, (
            "the drag diagonal is identically zero -- this control perturbs a "
            "zero and proves nothing; give the state a non-zero velocity")
        # dt*r/h with h a real ladder cell is O(1e-3); dividing by the 1e-10
        # floor instead would put it at O(1e9).
        assert e.max() < 1.0, (
            f"drag diagonal peaks at {e.max():.3e} -- it divided by the "
            f"1e-10 thickness floor on a face the seam wall closes, instead "
            f"of by NEMO's unmasked e3u")

    def test_flat_bottom_is_unchanged(self):
        """Control: with no staircase the fix is a no-op -- every face level is
        open, so the mask multiplies by one."""
        grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
        z_ref = create_ocean_z_star(
            n_levels=_N_LEV, H_max=1000.0, dz_surface=100.0, dz_deep=500.0)
        z_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))
        H = jnp.full((_N_LAT, _N_LON), float(z_half[_DEEP_BOTTOM_LEVEL + 1]))
        coord = create_partial_cell_coordinate(z_ref, H)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_ref, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H)
        h_u, _ = _capture_face_thicknesses(
            LatLonCGridOceanModel(grid, coord, _cfg()), state)

        dz_ref = np.asarray(coord.dz_ref)
        for j in range(_N_LON + 1):
            np.testing.assert_allclose(h_u[0, j], dz_ref, rtol=1e-12)
