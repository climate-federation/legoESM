"""Tests for pad_halo_dgrid_cell_scalar_4d (cell-centre metric halo).

The helper supplies the FAITHFUL cross-seam ghost ring for cell-centre
metrics (cosa_s = cos_sg[..., 4], rsin2): quarter-turn seams negate
cos-type quantities (tensorial e_i.e_j law, phase-2B rot90 remap
certification), everything else is a plain mpp-style index copy.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.grids.dgrid_halo import (  # noqa: E402
    _SG_QUARTER_TURN,
    SG_BIG_NUMBER,
    pad_halo_dgrid_cell_scalar_4d,
)
from legoesm.grids.halo import CONNECTIVITY, EAST, NORTH, SOUTH, WEST  # noqa: E402


def _tagged_field(n, nlev=1, seed=0):
    """Unique-valued (6, n, n, nlev) field so any mis-mapping is visible."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.1, 1.0, size=(6, n, n, nlev))
    # Make every entry unique in magnitude (mapping discriminator).
    q += np.arange(q.size).reshape(q.shape) * 1e-6
    return jnp.asarray(q)


def _neighbour_line(q, nbr_face, nbr_edge, reversed_, depth):
    """The neighbour's cell line ``depth`` cells in from ``nbr_edge``."""
    if nbr_edge == WEST:
        line = q[nbr_face, depth, :, :]
    elif nbr_edge == EAST:
        line = q[nbr_face, -1 - depth, :, :]
    elif nbr_edge == SOUTH:
        line = q[nbr_face, :, depth, :]
    else:
        line = q[nbr_face, :, -1 - depth, :]
    return line[::-1, :] if reversed_ else line


def _ghost_line(q_pad, face, edge, depth, halo):
    """The ghost line ``depth`` rings out from ``edge`` (diagonals excluded)."""
    sl = slice(halo, -halo)
    if edge == WEST:
        return q_pad[face, halo - 1 - depth, sl, :]
    if edge == EAST:
        return q_pad[face, -halo + depth, sl, :]
    if edge == SOUTH:
        return q_pad[face, sl, halo - 1 - depth, :]
    return q_pad[face, sl, -halo + depth, :]


@pytest.mark.parametrize("nlev", [1, 3])
def test_physical_region_preserved(nlev):
    n, halo = 8, 2
    q = _tagged_field(n, nlev)
    p = pad_halo_dgrid_cell_scalar_4d(q, cos_type=False, halo=halo)
    assert p.shape == (6, n + 2 * halo, n + 2 * halo, nlev)
    np.testing.assert_array_equal(
        np.asarray(p[:, halo:-halo, halo:-halo, :]), np.asarray(q))


@pytest.mark.parametrize("cos_type", [False, True])
def test_all_24_directed_seams_ring1_and_ring2(cos_type):
    """Every ghost line == (signed) copy of the mapped neighbour line.

    Expectations are built directly from CONNECTIVITY in this test — the
    same independent seam-oracle style as test_dgrid_sg_slot_halo — so a
    wrong source line, missed reversal, wrong ring depth, or wrong sign on
    ANY of the 24 directed seams fails with its (face, edge, depth) named.
    """
    n, halo = 8, 2
    q = _tagged_field(n)
    p = pad_halo_dgrid_cell_scalar_4d(q, cos_type=cos_type, halo=halo)
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, reversed_ = CONNECTIVITY[face][edge]
            sign = (-1.0 if (cos_type and (face, edge) in _SG_QUARTER_TURN)
                    else 1.0)
            for depth in range(halo):
                expected = sign * np.asarray(
                    _neighbour_line(q, nbr_face, nbr_edge, reversed_, depth))
                got = np.asarray(_ghost_line(p, face, edge, depth, halo))
                np.testing.assert_allclose(
                    got, expected, rtol=0, atol=0,
                    err_msg=f"seam face={face} edge={edge} depth={depth}")


def test_cos_sign_overlay_is_not_vacuous():
    """cos_type must CHANGE the result (synthetic-violation guard).

    If the sign overlay were dropped, cos_type=True would silently equal
    cos_type=False and the quarter-turn seams would carry the wrong sign.
    """
    n, halo = 8, 2
    q = _tagged_field(n)
    p_even = pad_halo_dgrid_cell_scalar_4d(q, cos_type=False, halo=halo)
    p_cos = pad_halo_dgrid_cell_scalar_4d(q, cos_type=True, halo=halo)
    diff = np.asarray(p_even) != np.asarray(p_cos)
    assert diff.any(), "sign overlay had no effect anywhere"
    # And the difference is EXACTLY a sign flip where it occurs.
    both = np.asarray(p_even)[diff], np.asarray(p_cos)[diff]
    np.testing.assert_array_equal(both[0], -both[1])
    # Physical region identical.
    np.testing.assert_array_equal(
        np.asarray(p_even[:, halo:-halo, halo:-halo]),
        np.asarray(p_cos[:, halo:-halo, halo:-halo]))


def test_diagonal_blocks_poisoned():
    n, halo = 8, 2
    q = _tagged_field(n)
    for cos_type in (False, True):
        p = np.asarray(
            pad_halo_dgrid_cell_scalar_4d(q, cos_type=cos_type, halo=halo))
        for si in (slice(0, halo), slice(-halo, None)):
            for sj in (slice(0, halo), slice(-halo, None)):
                block = np.abs(p[:, si, sj, :])
                np.testing.assert_array_equal(block, SG_BIG_NUMBER)


def test_input_validation():
    with pytest.raises(ValueError):
        pad_halo_dgrid_cell_scalar_4d(jnp.zeros((6, 4, 4)))  # 3D
    with pytest.raises(ValueError):
        pad_halo_dgrid_cell_scalar_4d(jnp.zeros((4, 4, 4, 1)))  # not 6 faces
    with pytest.raises(ValueError):
        pad_halo_dgrid_cell_scalar_4d(jnp.zeros((6, 4, 5, 1)))  # not square
