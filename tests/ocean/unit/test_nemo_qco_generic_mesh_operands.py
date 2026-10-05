"""The NEMO ``wzv`` arm must be selectable on a card without NEMO's mesh file.

``nemo_qco_wzv_operands`` transcribes ``sshwzv.F90:331-336`` (the ``key_qco``
arm) together with ``div_hor`` (``divhor.F90:108-141``).  It used to hard-raise
unless the z-coordinate carried NEMO's own ``hu_0``/``e1e2*``/``e2u``/``e1v``,
which only the cards built from a real ``mesh_mask.nc`` do -- so LOCK_EXCHANGE,
OVERFLOW and ORCA1 were locked out of a NEMO routine by a mesh-file privilege
rather than by a config choice.  These tests pin the two halves of the fix:

* the operands built from a card's own grid + reference ladder are the ones
  NEMO builds (``domain.F90:145`` for ``hu_0``,
  ``tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:179-186`` for the partial-cell
  reference face), and
* when a card DOES carry the raw NEMO arrays they are used verbatim, so the
  certified DINO arithmetic is untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_qco_wzv_operands
from legoesm.ocean.vertical import (
    NemoEENBarotropicOperands,
    OceanPartialCellCoordinate,
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
    nemo_qco_card_mesh_operands,
    nemo_qco_resolved_mesh_operands,
)


NLEV = 4
NLAT, NLON = 3, 6


def _fixture():
    grid = create_beta_plane_cgrid_geometry(
        n_lat=NLAT, n_lon=NLON, dx_m=1000.0, dy_m=1000.0,
        f0=1.0e-4, beta=0.0)
    dz = np.array([2.0, 3.0, 5.0, 10.0], dtype=np.float64)
    z_ref = create_z_star_from_thicknesses(dz)
    # A ridge, NOT a monotone slope: on a monotone floor the east face's
    # shallower neighbour is always the cell that owns the face, so the
    # min-rule face would coincide with e3t_0 and every comparison below
    # would be vacuous.  The descending side makes them genuinely differ.
    bathy = np.array([[5.0, 9.0, 20.0, 20.0, 9.0, 5.0]] * NLAT)
    z_coord = create_partial_cell_coordinate(z_ref, jnp.asarray(bathy))
    active = np.asarray(z_coord.is_active).astype(np.float64)
    u_mask = np.zeros((NLAT, NLON + 1, NLEV))
    u_mask[:, 1:, :] = active * np.roll(active, -1, axis=1)
    u_mask[:, 0, :] = u_mask[:, -1, :]
    v_mask = np.zeros((NLAT + 1, NLON, NLEV))
    v_mask[1:-1, :, :] = active[:-1] * active[1:]
    return grid, z_coord, jnp.asarray(u_mask), jnp.asarray(v_mask), bathy


def test_card_operands_reproduce_nemos_domain_definitions():
    grid, z_coord, u_mask, v_mask, _ = _fixture()
    h_ref = jnp.asarray(z_coord.h_partial, dtype=jnp.float64)
    ops = nemo_qco_card_mesh_operands(
        h_ref, u_mask, v_mask, grid, jnp.float64)

    h = np.asarray(h_ref)
    # usrdef_zgr.F90:179-186 -- the reference face is the shallower neighbour.
    expected_e3u_0 = np.minimum(np.roll(h, 1, axis=1), h)
    expected_e3u_0 = np.concatenate(
        [expected_e3u_0, expected_e3u_0[:, :1]], axis=1)[:, 1:, :]
    np.testing.assert_array_equal(np.asarray(ops.e3u_0), expected_e3u_0)
    # domain.F90:145 -- hu_0 accumulates e3u_0*umask, closed faces add zero.
    np.testing.assert_array_equal(
        np.asarray(ops.hu_0),
        np.sum(expected_e3u_0 * np.asarray(u_mask)[:, 1:, :], axis=-1))
    np.testing.assert_array_equal(
        np.asarray(ops.area_t), np.asarray(grid.area_T))
    np.testing.assert_array_equal(
        np.asarray(ops.e2u), np.asarray(grid.dy_u)[:, 1:])
    np.testing.assert_array_equal(
        np.asarray(ops.e1v), np.asarray(grid.dx_v)[1:, :])
    # The slope must actually make the min-rule bite, else the row above is
    # a vacuous identity on a flat mesh.
    assert not np.array_equal(expected_e3u_0, np.asarray(h_ref)[:, 1:, :])


def _attach_raw(z_coord, ops, *, e3u_0=None, e3v_0=None):
    """A z-coordinate carrying NEMO-style raw operands equal to ``ops``.

    NEMO carries the reference FACE thicknesses separately from ``e3t_0``
    (``usrdef_zgr.F90:225,228``: the shallower neighbour's), and every card
    that carries the raw qco set carries them on the EEN barotropic bundle,
    so a fixture without them is not a NEMO mesh.  ``e3u_0``/``e3v_0``
    override them, which is how the alias this module used to apply is
    reproduced in the non-vacuity check below.
    """
    z = jnp.zeros_like(ops.e3t_0)
    z2 = jnp.zeros_like(ops.hu_0)
    bundle = NemoEENBarotropicOperands(
        ff_f=z2, e3u_0=ops.e3u_0 if e3u_0 is None else e3u_0,
        e3v_0=ops.e3v_0 if e3v_0 is None else e3v_0, e3f_0=z,
        umask=ops.umask3, vmask=ops.vmask3, fmask=z, fe3mask=z,
        hu_0=ops.hu_0, hv_0=ops.hv_0, hf_0=z2,
        e1t=z2, e2t=z2, e1u=z2, e2u=ops.e2u, e1v=ops.e1v, e2v=z2,
        e1f=z2, e2f=z2)
    return z_coord._replace(
        nemo_e3t_0=ops.e3t_0, nemo_hu_0=ops.hu_0, nemo_hv_0=ops.hv_0,
        nemo_e1e2t=ops.area_t, nemo_e1e2u=ops.area_u,
        nemo_e1e2v=ops.area_v, nemo_e2u=ops.e2u, nemo_e1v=ops.e1v,
        nemo_een_barotropic=bundle)


def _wzv(grid, z_coord, u_mask, v_mask):
    rng = np.random.default_rng(4)
    tmask = jnp.asarray(z_coord.is_active, dtype=jnp.float64)
    eta_now = jnp.asarray(rng.normal(scale=0.02, size=(NLAT, NLON)))
    eta_before = jnp.asarray(rng.normal(scale=0.02, size=(NLAT, NLON)))
    u = jnp.asarray(rng.normal(scale=0.1, size=(NLAT, NLON + 1, NLEV)))
    v = jnp.asarray(rng.normal(scale=0.1, size=(NLAT + 1, NLON, NLEV)))
    # STALE-FIXTURE FIX (round 214; this was a pre-existing red on the clean
    # tree, not a regression of this round): the call grew a required
    # `after_ssh_form` -- a card resolving this branch must STATE the NEMO
    # time-stepping scheme whose after-SSH slot the call reads rather than
    # have it inferred.  These fixtures model the RK3 program the certified
    # cards run (stprk3.f90:239-241), so they say so.  The arm under test
    # (the reference face thickness) is upstream of the after-SSH slot and
    # identical for every form.
    ww, live_u, live_v = nemo_qco_wzv_operands(
        eta_now, eta_before, u, v, grid, z_coord, u_mask, v_mask, tmask,
        120.0, after_ssh_form="rk3_extrapolated")
    return np.asarray(ww), np.asarray(live_u), np.asarray(live_v)


def test_raw_and_card_operand_sources_agree_and_the_arm_is_constructible():
    grid, z_coord, u_mask, v_mask, _ = _fixture()
    ops = nemo_qco_resolved_mesh_operands(
        z_coord, grid, u_mask, v_mask, jnp.float64, NLEV)
    # The card-built arm now runs at all -- this is the gap being closed.
    ww_card, hu_card, _ = _wzv(grid, z_coord, u_mask, v_mask)
    assert np.isfinite(ww_card).all()
    assert np.max(np.abs(ww_card)) > 0.0

    raw_coord = _attach_raw(z_coord, ops)
    resolved = nemo_qco_resolved_mesh_operands(
        raw_coord, grid, u_mask, v_mask, jnp.float64, NLEV)
    # Raw path takes the attached arrays verbatim.
    np.testing.assert_array_equal(
        np.asarray(resolved.hu_0), np.asarray(ops.hu_0))
    np.testing.assert_array_equal(
        np.asarray(resolved.e2u), np.asarray(ops.e2u))
    # ROUND 214.  The raw branch used to alias e3u_0 = e3v_0 = e3t_0, so on
    # this sloping fixture the two sources were different physics at every
    # step face.  They are now the SAME statement -- both take the shallower
    # neighbour's reference thickness (usrdef_zgr.F90:225,228) -- the card by
    # rebuilding it and the raw branch by reading the arrays NEMO built.
    np.testing.assert_array_equal(
        np.asarray(resolved.e3u_0), np.asarray(ops.e3u_0))
    np.testing.assert_array_equal(
        np.asarray(resolved.e3v_0), np.asarray(ops.e3v_0))
    ww_raw, hu_raw, _ = _wzv(grid, raw_coord, u_mask, v_mask)
    np.testing.assert_array_equal(hu_raw, hu_card)
    np.testing.assert_array_equal(ww_raw, ww_card)
    # NON-VACUITY: the fixture must have step faces, and the OLD alias must
    # be visibly wrong on them -- otherwise the two rows above pass on a mesh
    # where every rule agrees and prove nothing.
    step_face = np.any(
        np.asarray(ops.e3u_0) != np.asarray(ops.e3t_0), axis=-1)
    assert step_face.any(), "fixture has no step face; the rows above are vacuous"
    aliased = _attach_raw(z_coord, ops, e3u_0=ops.e3t_0, e3v_0=ops.e3t_0)
    ww_alias, hu_alias, _ = _wzv(grid, aliased, u_mask, v_mask)
    assert not np.array_equal(hu_alias, hu_card)
    assert not np.array_equal(ww_alias, ww_card)


def test_raw_branch_is_bit_identical_when_the_mesh_is_full_step():
    """On a full-step mesh ``e3u_0 == e3t_0``, so the two sources coincide."""
    grid = create_beta_plane_cgrid_geometry(
        n_lat=NLAT, n_lon=NLON, dx_m=1000.0, dy_m=1000.0, f0=1e-4, beta=0.0)
    dz = np.array([2.0, 3.0, 5.0, 10.0], dtype=np.float64)
    z_ref = create_z_star_from_thicknesses(dz)
    z_coord = create_partial_cell_coordinate(
        z_ref, jnp.full((NLAT, NLON), 20.0))
    active = np.asarray(z_coord.is_active).astype(np.float64)
    u_mask = np.zeros((NLAT, NLON + 1, NLEV))
    u_mask[:, 1:, :] = active * np.roll(active, -1, axis=1)
    u_mask[:, 0, :] = u_mask[:, -1, :]
    v_mask = np.zeros((NLAT + 1, NLON, NLEV))
    v_mask[1:-1, :, :] = active[:-1] * active[1:]
    u_mask, v_mask = jnp.asarray(u_mask), jnp.asarray(v_mask)
    ops = nemo_qco_resolved_mesh_operands(
        z_coord, grid, u_mask, v_mask, jnp.float64, NLEV)
    # Flat, x-uniform mesh: the min-rule east face IS e3t_0, so the two
    # operand sources coincide exactly and the comparison below is a real
    # bit-identity claim rather than a shape coincidence.
    np.testing.assert_array_equal(
        np.asarray(ops.e3u_0), np.asarray(ops.e3t_0))
    ww_card, _, _ = _wzv(grid, z_coord, u_mask, v_mask)
    ww_raw, _, _ = _wzv(grid, _attach_raw(z_coord, ops), u_mask, v_mask)
    np.testing.assert_array_equal(ww_raw, ww_card)

    # Synthetic violation: the test must be able to fail.  Perturb one raw
    # metric by 1 part in 1e6 and the vertical velocity must move.
    bad = ops._replace(e2u=ops.e2u * (1.0 + 1.0e-6))
    ww_bad, _, _ = _wzv(grid, _attach_raw(z_coord, bad), u_mask, v_mask)
    assert not np.array_equal(ww_bad, ww_card)


def test_arm_is_constructible_on_the_certified_l1_cards():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )

    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        card = build_nemo_testcase_card(case)
        z_coord = card.recipe.z_coord
        assert isinstance(z_coord, OceanPartialCellCoordinate)
        assert getattr(z_coord, "nemo_hu_0", None) is None
        u_mask, v_mask = compute_face_masks_3d(z_coord.is_active,
                                               card.recipe.grid)
        dtype = card.recipe.initial_state.eta.data.dtype
        # OVERFLOW now carries a lone raw e3t operand; since 03f0a1a07 that
        # partial provenance must fail closed.  LOCK_EXCHANGE carries none.
        if getattr(z_coord, "nemo_e3t_0", None) is not None:
            with pytest.raises(ValueError, match="some but not all"):
                nemo_qco_resolved_mesh_operands(
                    z_coord, card.recipe.grid, u_mask.astype(dtype),
                    v_mask.astype(dtype), dtype, z_coord.n_levels)
            # Removing the lone qco operand selects the all-card source.
            z_coord = z_coord._replace(nemo_e3t_0=None)
        ops = nemo_qco_resolved_mesh_operands(
            z_coord, card.recipe.grid, u_mask.astype(dtype),
            v_mask.astype(dtype), dtype, z_coord.n_levels)
        assert np.all(np.asarray(ops.hu_0) >= 0.0)
        assert np.max(np.asarray(ops.hu_0)) > 0.0
        assert np.all(np.asarray(ops.area_u) > 0.0)


def test_missing_operands_and_no_ladder_still_fail_closed():
    grid, z_coord, u_mask, v_mask, _ = _fixture()

    class _NoLadder:
        pass

    with pytest.raises(ValueError, match="raw NEMO mesh operands"):
        nemo_qco_resolved_mesh_operands(
            _NoLadder(), grid, u_mask, v_mask, jnp.float64, NLEV)

    # A PARTIAL NEMO mesh must raise rather than quietly rebuild everything
    # from the card: that would mix two operand provenances in one wzv call.
    ops = nemo_qco_resolved_mesh_operands(
        z_coord, grid, u_mask, v_mask, jnp.float64, NLEV)
    partial = _attach_raw(z_coord, ops)._replace(nemo_e2u=None)
    with pytest.raises(ValueError, match="some but not all"):
        nemo_qco_resolved_mesh_operands(
            partial, grid, u_mask, v_mask, jnp.float64, NLEV)

    # ROUND 214: a card carrying the raw qco set but NOT NEMO's reference
    # face thicknesses must fail closed, not fall back to the e3t_0 alias --
    # on a partial-cell mesh that alias is wrong on every stepped face.
    no_faces = _attach_raw(z_coord, ops)._replace(nemo_een_barotropic=None)
    with pytest.raises(ValueError, match="reference face thicknesses"):
        nemo_qco_resolved_mesh_operands(
            no_faces, grid, u_mask, v_mask, jnp.float64, NLEV)
