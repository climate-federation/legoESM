"""``create_full_step_coordinate`` — NEMO ln_zco full-step-z vertical grid.

Truth-tier structural checks for the full-step constructor used to wrap the
NEMO bridge's z* reference grid into a staircase of dry bottom cells
(``usrdef_zgr.F90`` ``zgr_zco_3d`` fixed ``pe3t_1d`` + ``zgr_msk_top_bot``
per-column ``k_bot``):

  - staircase wet/dry mask from an explicit ``bottom_level`` (= NEMO k_bot-1),
  - every wet cell is a FULL cell (``h_partial == dz_ref``, never a partial
    ln_zps thickness), dry cells below the seafloor are 0,
  - per-column ``Σ h_partial`` = the staircase depth ``gdepw(k_bot)``,
  - Jacobian consistency (``water_col / H_bathy``) so a stretched column sums
    to ``eta + H_bathy``,
  - agreement with ``create_partial_cell_coordinate`` when the continuous
    bathymetry sits exactly on a level interface (both are full-step there),
  - ``t_depth_ref`` (NEMO gdept_1d) propagation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    create_full_step_coordinate,
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture
def zref():
    # NEMO-style growing thicknesses (surface thin -> deep thick), 5 levels.
    e3t_1d = np.array([10.0, 30.0, 80.0, 200.0, 680.0])   # sum = 1000 m
    gdept_1d = np.array([5.0, 25.0, 80.0, 220.0, 660.0])  # strictly increasing
    return create_z_star_from_thicknesses(e3t_1d, t_depth_ref_m=gdept_1d)


def test_staircase_mask_and_full_cells(zref):
    dz = np.asarray(zref.dz_ref)
    # A 2x2 column field of deepest-wet-level indices (NEMO k_bot - 1).
    bl = jnp.asarray([[4, 2], [0, -1]], dtype=jnp.int32)
    coord = create_full_step_coordinate(zref, bl)

    assert isinstance(coord, OceanPartialCellCoordinate)
    ia = np.asarray(coord.is_active)
    hp = np.asarray(coord.h_partial)

    # Column (0,0): bottom_level=4 -> all 5 wet, all FULL cells.
    assert ia[0, 0].tolist() == [True] * 5
    np.testing.assert_allclose(hp[0, 0], dz)
    # Column (0,1): bottom_level=2 -> top 3 wet (full), 2 dry.
    assert ia[0, 1].tolist() == [True, True, True, False, False]
    np.testing.assert_allclose(hp[0, 1], [dz[0], dz[1], dz[2], 0.0, 0.0])
    # Column (1,0): bottom_level=0 -> only surface cell wet.
    assert ia[1, 0].tolist() == [True, False, False, False, False]
    # Column (1,1): dry column (bottom_level=-1) -> nothing wet, h=0.
    assert ia[1, 1].tolist() == [False] * 5
    np.testing.assert_allclose(hp[1, 1], 0.0)

    # FULL-step (ln_zco), never ln_zps: every wet cell equals dz_ref exactly.
    wet = ia
    ref = np.broadcast_to(dz, hp.shape)
    assert np.all(hp[wet] == ref[wet])


def test_column_sum_is_staircase_depth(zref):
    dz = np.asarray(zref.dz_ref)
    bl = jnp.asarray([4, 2, 0, -1], dtype=jnp.int32)
    coord = create_full_step_coordinate(zref, bl)
    colsum = np.asarray(jnp.sum(coord.h_partial, axis=-1))
    # gdepw(k_bot) = cumsum(dz)[bottom_level]; dry column -> 0.
    cum = np.cumsum(dz)
    expected = np.array([cum[4], cum[2], cum[0], 0.0])
    np.testing.assert_allclose(colsum, expected)


def test_jacobian_column_sums_to_water_column(zref):
    # Stretched column: sum of h_partial*J must equal eta + H_bathy.
    bl = jnp.asarray([2], dtype=jnp.int32)
    coord = create_full_step_coordinate(zref, bl)
    H_bathy = jnp.sum(coord.h_partial, axis=-1)            # staircase depth
    eta = jnp.asarray([0.7])
    J = compute_ocean_jacobian(eta, H_bathy, coord)
    dz_cell = coord.h_partial * J[..., None]
    np.testing.assert_allclose(
        np.asarray(jnp.sum(dz_cell, axis=-1)),
        np.asarray(eta + H_bathy),
        rtol=1e-12,
    )


def test_matches_partial_cell_when_bathy_on_interface(zref):
    # When H_bathy sits EXACTLY on a level interface, the ln_zps partial-cell
    # factory degenerates to full-step; the two constructors must agree.
    dz = np.asarray(zref.dz_ref)
    cum = np.cumsum(dz)                                     # interface depths
    H_iface = jnp.asarray([cum[2], cum[4], cum[0]])        # exact interfaces
    part = create_partial_cell_coordinate(zref, H_iface)
    full = create_full_step_coordinate(
        zref, jnp.asarray(part.bottom_level, dtype=jnp.int32))
    np.testing.assert_array_equal(
        np.asarray(part.is_active), np.asarray(full.is_active))
    np.testing.assert_allclose(
        np.asarray(part.h_partial), np.asarray(full.h_partial))


def test_t_depth_ref_propagated(zref):
    coord = create_full_step_coordinate(zref, jnp.asarray([4], dtype=jnp.int32))
    assert coord.t_depth_ref is not None
    np.testing.assert_allclose(
        np.asarray(coord.t_depth_ref), np.asarray(zref.t_depth_ref))

    # A reference grid without t_depth_ref -> None (no crash).
    z_noref = create_z_star_from_thicknesses(np.asarray(zref.dz_ref))
    c2 = create_full_step_coordinate(z_noref, jnp.asarray([4], dtype=jnp.int32))
    assert c2.t_depth_ref is None


def test_jacobian_forms_nemos_ratio_before_adding_one(zref):
    """The quasi-Eulerian stretch is NEMO's ``1 + ssh/ht_0``, not ``(ssh+H)/H``.

    ``dom_qco_r3c_RK3`` writes ``r3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)``
    (``domqco.F90:209``) with ``r1_ht_0 = ssmask/(ht_0 + 1 - ssmask)``
    (``domain.F90:158``), i.e. exactly ``1/ht_0`` on a wet column, and the
    thickness macro is ``e3t = e3t_0*(1 + r3t)``
    (``domzgr_substitute.h90:139``).  NEMO never rounds the SUM
    ``ssh + ht_0`` first.

    The operands below are GYRE's own uniform ``ht_0`` and four of its kt=1
    stage-3 ``ssh`` values; on every one of them the two forms disagree in the
    last bit, so this test FAILS on the pre-round-40 ``(eta + H)/H``.
    """
    ht_0 = 4300.710017215397
    ssh = np.array([-0.0013969278195402545, -0.0014360062781082884,
                    -0.0014051606031964237, -0.0013718170244874474])
    coord = create_full_step_coordinate(
        zref, jnp.full(ssh.shape, len(np.asarray(zref.dz_ref)),
                       dtype=jnp.int32))
    bathy = np.full(ssh.shape, ht_0)

    nemo = 1.0 + ssh * (1.0 / ht_0)
    sum_first = (ssh + ht_0) / ht_0
    # Non-vacuity: the two forms really do differ on these operands, so the
    # equality below cannot be satisfied by both.
    assert not np.any(nemo == sum_first)

    got = np.asarray(compute_ocean_jacobian(
        jnp.asarray(ssh), jnp.asarray(bathy), coord))
    np.testing.assert_array_equal(got, nemo)


def test_jacobian_min_column_clip_is_unchanged_by_the_ratio_form(zref):
    """legoESM's own ``min_water_column_m`` floor keeps its pre-round-40 value.

    NEMO has no such floor; expressing it on the Jacobian rather than on the
    water column leaves every clipped cell at exactly ``min_col/H`` and every
    unclipped cell bit-identical to the unclipped call.
    """
    coord = create_full_step_coordinate(
        zref, jnp.full((3,), len(np.asarray(zref.dz_ref)), dtype=jnp.int32))
    bathy = jnp.asarray([1.0, 1.0, 4300.710017215397])
    eta = jnp.asarray([-0.9, -0.2, -0.0013969278195402545])   # first is clipped
    clipped = np.asarray(compute_ocean_jacobian(
        eta, bathy, coord, min_water_column_m=0.5))
    plain = np.asarray(compute_ocean_jacobian(eta, bathy, coord))
    assert clipped[0] == 0.5 / 1.0
    np.testing.assert_array_equal(clipped[1:], plain[1:])


def test_jacobian_multiplies_the_stored_reciprocal(zref):
    """NEMO MULTIPLIES ``r1_ht_0``; it does not divide by ``ht_0``.

    ``domain.F90:158`` builds ``r1_ht_0 = ssmask/(ht_0 + 1 - ssmask)`` once and
    ``domqco.F90:209`` multiplies by it.  ``a/b`` and ``a*(1/b)`` are not the
    same double, and on GYRE's own magnitudes the difference is absorbed by the
    ``1 +``, so it cannot be caught there.  These operands are chosen so it is
    NOT absorbed.
    """
    ht_0 = np.array([16.75804935889166, 27.469830318264258,
                     11.822863395492591, 7.497700412938896])
    ssh = np.array([-1.8590317289517415, -1.7274640401377739,
                    -1.549283275814433, 1.9251583919946986])
    coord = create_full_step_coordinate(
        zref, jnp.full(ssh.shape, len(np.asarray(zref.dz_ref)),
                       dtype=jnp.int32))

    nemo = 1.0 + ssh * (1.0 / ht_0)
    divided = 1.0 + ssh / ht_0
    assert not np.any(nemo == divided)     # non-vacuity

    got = np.asarray(compute_ocean_jacobian(
        jnp.asarray(ssh), jnp.asarray(ht_0), coord))
    np.testing.assert_array_equal(got, nemo)


def test_layer_thickness_uses_the_one_shared_stretch(zref):
    """``compute_layer_thickness`` must not carry a second rounding of it.

    Its partial-cell branch used to form ``(eta+H)/H`` inline, so the model
    would have carried two different roundings of one NEMO statement, and the
    inline one feeds the momentum right-hand side.
    """
    coord = create_full_step_coordinate(
        zref, jnp.full((4,), len(np.asarray(zref.dz_ref)), dtype=jnp.int32))
    ht_0 = jnp.asarray([16.75804935889166, 27.469830318264258,
                        11.822863395492591, 7.497700412938896])
    ssh = jnp.asarray([-1.8590317289517415, -1.7274640401377739,
                       -1.549283275814433, 1.9251583919946986])
    got = np.asarray(compute_layer_thickness(ssh, ht_0, coord))
    want = np.asarray(coord.h_partial) * np.asarray(
        compute_ocean_jacobian(ssh, ht_0, coord))[..., None]
    np.testing.assert_array_equal(got, want)
    # non-vacuity: the retired inline form really does differ here
    stale = np.asarray(coord.h_partial) * (
        (np.asarray(ssh) + np.asarray(ht_0)) / np.asarray(ht_0))[..., None]
    assert np.any(stale != got)
