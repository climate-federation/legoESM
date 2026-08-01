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
    _, cos_p = pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)
    nbr_face, nbr_edge, reversed_ = CONNECTIVITY[face][edge]
    strip_c = (cos_sg[nbr_face, :, 0, :] if nbr_edge == SOUTH
               else cos_sg[nbr_face, :, -1, :])
    if reversed_:
        strip_c = strip_c[::-1, :]
    got_c = (cos_p[face, 1:-1, 0, :] if edge == SOUTH
             else cos_p[face, 1:-1, -1, :])
    perm = (2, 3, 0, 1)
    for dst_slot in range(4):
        np.testing.assert_allclose(
            np.asarray(got_c[:, dst_slot]),
            np.asarray(strip_c[:, perm[dst_slot]]), rtol=0, atol=1e-12,
            err_msg=f"half-turn cos slot {dst_slot} must NOT flip sign")


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
