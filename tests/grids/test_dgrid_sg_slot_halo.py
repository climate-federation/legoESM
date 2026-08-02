"""Guards for the raw sin_sg/cos_sg SLOT halo.

MUST use NON-UNIT metrics.  The exact stencil assertions used elsewhere in this
work (+8 seam, +5 vertex, +1 sparse) set sin=1 and cos=0, which makes them
provably BLIND to this layer: a wrong slot permutation, a wrong reversal or a
missing cos sign all leave them green.  Every test here therefore uses distinct,
non-degenerate values per (face, cell, slot) so that a mis-permutation cannot
coincide with the right answer.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.dgrid_halo import (
    SG_BIG_NUMBER,
    SG_TINY_NUMBER,
    _SG_HALF_TURN,
    _SG_QUARTER_TURN,
    pad_halo_dgrid_sg_slots_4d,
)
from legoesm.grids.halo import CONNECTIVITY, EAST, NORTH, SOUTH, WEST

W, S, E, N = 0, 1, 2, 3


def _tagged(n):
    """Every (face, i, j, slot) gets a unique value, so any wrong source is
    detectable rather than coincidentally equal."""
    f, i, j, s = np.meshgrid(np.arange(6), np.arange(n), np.arange(n),
                             np.arange(4), indexing="ij")
    sin_sg = 0.5 + 0.01 * (1000 * f + 100 * i + 10 * j + s)
    cos_sg = -0.3 - 0.01 * (1000 * f + 100 * i + 10 * j + s)
    return jnp.asarray(sin_sg), jnp.asarray(cos_sg)


def test_shapes_and_interior_preserved():
    n = 6
    sin_sg, cos_sg = _tagged(n)
    sin_p, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)
    assert sin_p.shape == (6, n + 2, n + 2, 4)
    assert cos_p.shape == (6, n + 2, n + 2, 4)
    np.testing.assert_array_equal(np.asarray(sin_p[:, 1:-1, 1:-1, :]),
                                  np.asarray(sin_sg))
    np.testing.assert_array_equal(np.asarray(cos_p[:, 1:-1, 1:-1, :]),
                                  np.asarray(cos_sg))


def test_rejects_malformed_input():
    n = 4
    sin_sg, cos_sg = _tagged(n)
    with pytest.raises(ValueError, match="4"):
        pad_halo_dgrid_sg_slots_4d(sin_sg[..., :3], cos_sg[..., :3])
    with pytest.raises(ValueError, match="identical shapes"):
        pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg[:, :-1])


@pytest.mark.parametrize("key", sorted(_SG_QUARTER_TURN))
def test_quarter_turn_permutes_slots_and_flips_cos(key):
    """Each quarter-turn seam: dest slot k must equal the NEIGHBOUR's perm[k],
    with cos negated.  Non-unit tagged data, so an identity copy or a wrong
    permutation cannot pass."""
    face, edge = key
    perm = _SG_QUARTER_TURN[key]
    n = 6
    sin_sg, cos_sg = _tagged(n)
    sin_p, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)

    nbr_face, nbr_edge, reversed_ = CONNECTIVITY[face][edge]
    if nbr_edge == WEST:
        strip_s, strip_c = sin_sg[nbr_face, 0, :, :], cos_sg[nbr_face, 0, :, :]
    elif nbr_edge == EAST:
        strip_s, strip_c = sin_sg[nbr_face, -1, :, :], cos_sg[nbr_face, -1, :, :]
    elif nbr_edge == SOUTH:
        strip_s, strip_c = sin_sg[nbr_face, :, 0, :], cos_sg[nbr_face, :, 0, :]
    else:
        strip_s, strip_c = sin_sg[nbr_face, :, -1, :], cos_sg[nbr_face, :, -1, :]
    if reversed_:
        strip_s, strip_c = strip_s[::-1, :], strip_c[::-1, :]

    if edge == WEST:
        got_s, got_c = sin_p[face, 0, 1:-1, :], cos_p[face, 0, 1:-1, :]
    elif edge == EAST:
        got_s, got_c = sin_p[face, -1, 1:-1, :], cos_p[face, -1, 1:-1, :]
    elif edge == SOUTH:
        got_s, got_c = sin_p[face, 1:-1, 0, :], cos_p[face, 1:-1, 0, :]
    else:
        got_s, got_c = sin_p[face, 1:-1, -1, :], cos_p[face, 1:-1, -1, :]

    for dst_slot in range(4):
        np.testing.assert_allclose(
            np.asarray(got_s[:, dst_slot]),
            np.asarray(strip_s[:, perm[dst_slot]]), rtol=0, atol=1e-12,
            err_msg=f"sin slot {dst_slot} on face {face} edge {edge}")
        np.testing.assert_allclose(
            np.asarray(got_c[:, dst_slot]),
            -np.asarray(strip_c[:, perm[dst_slot]]), rtol=0, atol=1e-12,
            err_msg=f"cos slot {dst_slot} (sign) on face {face} edge {edge}")


def test_quarter_turn_is_not_an_identity_copy():
    """Non-vacuity: the permutation must actually reorder.  If this ever passes
    trivially the seam mapping has collapsed to a plain scalar copy."""
    n = 6
    sin_sg, cos_sg = _tagged(n)
    sin_p, _ = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)
    face, edge = (1, NORTH)
    nbr_face, nbr_edge, _ = CONNECTIVITY[face][edge]
    got = np.asarray(sin_p[face, 1:-1, -1, :])
    identity = np.asarray(sin_sg[nbr_face, -1, :, :])
    assert not np.allclose(got, identity), (
        "quarter-turn seam is an identity slot copy -- permutation lost")


@pytest.mark.parametrize("key", sorted(_SG_HALF_TURN))
def test_half_turn_permutes_without_flipping_cos(key):
    """The four half-turn seams use (E,N,W,S) and do NOT negate cos."""
    face, edge = key
    n = 6
    sin_sg, cos_sg = _tagged(n)
    sin_p, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)
    nbr_face, nbr_edge, reversed_ = CONNECTIVITY[face][edge]
    strip_c = (cos_sg[nbr_face, :, 0, :] if nbr_edge == SOUTH
               else cos_sg[nbr_face, :, -1, :])
    if reversed_:
        strip_c = strip_c[::-1, :]
    got_c = (cos_p[face, 1:-1, 0, :] if edge == SOUTH
             else cos_p[face, 1:-1, -1, :])
    strip_s = (sin_sg[nbr_face, :, 0, :] if nbr_edge == SOUTH
               else sin_sg[nbr_face, :, -1, :])
    if reversed_:
        strip_s = strip_s[::-1, :]
    got_s = (sin_p[face, 1:-1, 0, :] if edge == SOUTH
             else sin_p[face, 1:-1, -1, :])
    perm = (2, 3, 0, 1)
    for dst_slot in range(4):
        np.testing.assert_allclose(
            np.asarray(got_c[:, dst_slot]),
            np.asarray(strip_c[:, perm[dst_slot]]), rtol=0, atol=1e-12,
            err_msg=f"half-turn cos slot {dst_slot} must NOT flip sign")
        np.testing.assert_allclose(
            np.asarray(got_s[:, dst_slot]),
            np.asarray(strip_s[:, perm[dst_slot]]), rtol=0, atol=1e-12,
            err_msg=f"half-turn sin slot {dst_slot}")


def test_diagonal_cells_assigned_and_poisoned():
    """FV3 assigns exactly two inward-facing slots per diagonal ghost cell and
    deliberately leaves the other two invalid (fv_grid_utils.F90:51).  Assert
    both halves: the assignments AND that the rest stay poisoned, so nobody
    silently consumes a plausible-looking wrong metric."""
    n = 6
    L = n + 1
    sin_sg, cos_sg = _tagged(n)
    sin_p, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)

    for p, poison in ((sin_p, SG_TINY_NUMBER), (cos_p, SG_BIG_NUMBER)):
        p = np.asarray(p)
        # assigned
        np.testing.assert_allclose(p[:, 0, 0, E], p[:, 0, 1, S], atol=1e-12)
        np.testing.assert_allclose(p[:, 0, 0, N], p[:, 1, 0, W], atol=1e-12)
        np.testing.assert_allclose(p[:, L, 0, W], p[:, L, 1, S], atol=1e-12)
        np.testing.assert_allclose(p[:, L, 0, N], p[:, L - 1, 0, E], atol=1e-12)
        np.testing.assert_allclose(p[:, L, L, W], p[:, L, L - 1, N], atol=1e-12)
        np.testing.assert_allclose(p[:, L, L, S], p[:, L - 1, L, E], atol=1e-12)
        np.testing.assert_allclose(p[:, 0, L, E], p[:, 0, L - 1, N], atol=1e-12)
        np.testing.assert_allclose(p[:, 0, L, S], p[:, 1, L, W], atol=1e-12)
        # poisoned (the two outward-facing slots of each diagonal)
        np.testing.assert_allclose(p[:, 0, 0, [W, S]], poison, atol=1e-12)
        np.testing.assert_allclose(p[:, L, 0, [E, S]], poison, atol=1e-12)
        np.testing.assert_allclose(p[:, L, L, [E, N]], poison, atol=1e-12)
        np.testing.assert_allclose(p[:, 0, L, [W, N]], poison, atol=1e-12)


def test_differentiable():
    n = 4
    sin_sg, cos_sg = _tagged(n)

    def loss(s):
        sp, cp = pad_halo_dgrid_sg_slots_4d(s, cos_sg)
        return jnp.sum(sp ** 2) + jnp.sum(cp ** 2)

    g = jax.grad(loss)(sin_sg)
    assert g.shape == sin_sg.shape
    assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------------------
# INDEPENDENT canonical seam table (codex review P1).
#
# Transcribed by hand from the oracle derivation and deliberately NOT imported
# from the production module.  The earlier version of this file parameterised
# from ``_SG_QUARTER_TURN`` and then used that same table as the expected
# answer, which is circular: a wrong non-identity permutation passed, and
# DELETING a row simply shrank the parametrisation and passed.  Everything below
# is checked against these literals instead, and the production tables are
# asserted to EQUAL them.
#
# row: (face, edge) -> (nbr_face, nbr_edge, reversed, perm, cos_sign)
# perm is "dst slot k takes src slot perm[k]", slots (W,S,E,N) = (0,1,2,3).
_SENW = (1, 2, 3, 0)
_NWSE = (3, 0, 1, 2)
_HALF = (2, 3, 0, 1)

CANON_QUARTER = {
    (1, NORTH): (4, EAST,  False, _SENW, -1.0),
    (4, EAST):  (1, NORTH, False, _NWSE, -1.0),
    (1, SOUTH): (5, EAST,  True,  _NWSE, -1.0),
    (5, EAST):  (1, SOUTH, True,  _SENW, -1.0),
    (3, NORTH): (4, WEST,  True,  _NWSE, -1.0),
    (4, WEST):  (3, NORTH, True,  _SENW, -1.0),
    (3, SOUTH): (5, WEST,  False, _SENW, -1.0),
    (5, WEST):  (3, SOUTH, False, _NWSE, -1.0),
}

CANON_HALF = {
    (2, SOUTH): (5, SOUTH, True, _HALF, 1.0),
    (2, NORTH): (4, NORTH, True, _HALF, 1.0),
    (4, NORTH): (2, NORTH, True, _HALF, 1.0),
    (5, SOUTH): (2, SOUTH, True, _HALF, 1.0),
}


def test_production_tables_equal_the_canonical_rows():
    """The production tables must match the hand-transcribed derivation.

    This is what makes every other seam assertion non-circular: a row silently
    edited, added or DELETED from the production dicts goes red here.
    """
    assert set(_SG_QUARTER_TURN) == set(CANON_QUARTER), (
        "quarter-turn seam key set drifted from the derivation")
    for key, (_, _, _, perm, _) in CANON_QUARTER.items():
        assert _SG_QUARTER_TURN[key] == perm, (
            f"quarter-turn permutation wrong at {key}")
    assert set(_SG_HALF_TURN) == set(CANON_HALF), (
        "half-turn seam key set drifted from the derivation")


def test_canonical_rows_agree_with_repo_connectivity():
    """The derivation's neighbour/reversal must equal this package's own
    CONNECTIVITY -- checked, not assumed."""
    for key, (nf, ne, rev, _, _) in {**CANON_QUARTER, **CANON_HALF}.items():
        face, edge = key
        got_f, got_e, got_r = CONNECTIVITY[face][edge]
        assert (got_f, got_e, got_r) == (nf, ne, rev), (
            f"CONNECTIVITY disagrees with the derivation at {key}: "
            f"got {(got_f, got_e, got_r)}, derivation says {(nf, ne, rev)}")


def _strip_from(field, nbr_face, nbr_edge, reversed_):
    if nbr_edge == WEST:
        strip = field[nbr_face, 0, :, :]
    elif nbr_edge == EAST:
        strip = field[nbr_face, -1, :, :]
    elif nbr_edge == SOUTH:
        strip = field[nbr_face, :, 0, :]
    else:
        strip = field[nbr_face, :, -1, :]
    return np.asarray(strip)[::-1, :] if reversed_ else np.asarray(strip)


def _ghost_of(padded, face, edge):
    if edge == WEST:
        return np.asarray(padded[face, 0, 1:-1, :])
    if edge == EAST:
        return np.asarray(padded[face, -1, 1:-1, :])
    if edge == SOUTH:
        return np.asarray(padded[face, 1:-1, 0, :])
    return np.asarray(padded[face, 1:-1, -1, :])


def test_all_24_directed_seams_against_the_canonical_table():
    """P2: ONE reference test covering every directed seam -- the 12 permuting
    ones AND the 12 identity ones -- for BOTH sin and cos.

    Expected values come from the canonical literals above, so this locks the
    generic pad_halo_4d baseline too: an identity seam that silently stopped
    copying, or a permuting seam that fell back to identity, both go red.
    """
    n = 6
    sin_sg, cos_sg = _tagged(n)
    sin_p, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)

    checked = 0
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            key = (face, edge)
            if key in CANON_QUARTER:
                nf, ne, rev, perm, cos_sign = CANON_QUARTER[key]
            elif key in CANON_HALF:
                nf, ne, rev, perm, cos_sign = CANON_HALF[key]
            else:
                nf, ne, rev = CONNECTIVITY[face][edge]
                perm, cos_sign = (0, 1, 2, 3), 1.0      # identity seam
            got_s, got_c = _ghost_of(sin_p, face, edge), _ghost_of(cos_p, face, edge)
            ref_s = _strip_from(sin_sg, nf, ne, rev)
            ref_c = _strip_from(cos_sg, nf, ne, rev)
            for dst in range(4):
                np.testing.assert_allclose(
                    got_s[:, dst], ref_s[:, perm[dst]], rtol=0, atol=1e-12,
                    err_msg=f"sin face {face} edge {edge} slot {dst}")
                np.testing.assert_allclose(
                    got_c[:, dst], cos_sign * ref_c[:, perm[dst]],
                    rtol=0, atol=1e-12,
                    err_msg=f"cos face {face} edge {edge} slot {dst}")
            checked += 1
    assert checked == 24, f"expected 24 directed seams, checked {checked}"


def test_eager_and_jit_agree():
    """P3: the gradient test is not a JIT test.  Compile it and compare."""
    n = 4
    sin_sg, cos_sg = _tagged(n)
    eager_s, eager_c = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)
    jitted = jax.jit(pad_halo_dgrid_sg_slots_4d)
    jit_s, jit_c = jitted(sin_sg, cos_sg)
    np.testing.assert_allclose(np.asarray(jit_s), np.asarray(eager_s),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(jit_c), np.asarray(eager_c),
                               rtol=0, atol=0)
