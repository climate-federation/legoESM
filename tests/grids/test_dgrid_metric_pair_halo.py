"""Unit guard for the paired staggered-metric halo (codex round 3).

Goes RED if the helper is reverted to edge replication on the eight
axis-swapping seams -- the defect that left the corner divergence pairing
real cross-panel winds with wrong seam metrics.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.dgrid_halo import (
    pad_halo_dgrid_scalar_4d,
    pad_halo_dgrid_scalar_pair_4d,
)


def _pair(n, nlev=1):
    rng = np.random.default_rng(7)
    u_like = jnp.asarray(rng.uniform(1.0, 2.0, size=(6, n, n + 1, nlev)))
    v_like = jnp.asarray(rng.uniform(1.0, 2.0, size=(6, n + 1, n, nlev)))
    return u_like, v_like


def test_metric_pair_halo_shapes():
    n = 4
    u_like, v_like = _pair(n)
    u_p, v_p = pad_halo_dgrid_scalar_pair_4d(u_like, v_like)
    assert u_p.shape == (6, n + 2, n + 3, 1)
    assert v_p.shape == (6, n + 3, n + 2, 1)


def test_metric_pair_halo_differs_from_edge_replication():
    """The whole point: axis-swap seams must NOT be edge copies."""
    n = 4
    u_like, v_like = _pair(n)
    u_p, _ = pad_halo_dgrid_scalar_pair_4d(u_like, v_like)
    u_edge = pad_halo_dgrid_scalar_4d(u_like, axis_swap_fill="edge")
    assert not np.allclose(np.asarray(u_p), np.asarray(u_edge)), (
        "paired metric halo is identical to edge replication -- the "
        "axis-swap seams were not filled from the partner field")


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_metric_pair_halo_sign_convention(sign):
    """+1 for dyc/dxc and sina pairs, -1 for the cosa pair."""
    n = 4
    u_like, v_like = _pair(n)
    u_p, v_p = pad_halo_dgrid_scalar_pair_4d(
        u_like, v_like, axis_swap_sign=sign)
    u_ref, v_ref = pad_halo_dgrid_scalar_pair_4d(
        u_like, v_like, axis_swap_sign=1.0)
    if sign == 1.0:
        np.testing.assert_array_equal(np.asarray(u_p), np.asarray(u_ref))
    else:
        # the swapped strips must flip; interiors must not
        assert not np.allclose(np.asarray(u_p), np.asarray(u_ref))
        np.testing.assert_allclose(
            np.asarray(u_p[:, 1:-1, 1:-1, :]),
            np.asarray(u_ref[:, 1:-1, 1:-1, :]))


def test_metric_pair_halo_rejects_bad_input():
    n = 4
    u_like, v_like = _pair(n)
    with pytest.raises(ValueError, match="4D"):
        pad_halo_dgrid_scalar_pair_4d(u_like[..., 0], v_like)
    with pytest.raises(ValueError, match="axis_swap_sign"):
        pad_halo_dgrid_scalar_pair_4d(u_like, v_like, axis_swap_sign=2.0)
    with pytest.raises(ValueError, match="Expected"):
        pad_halo_dgrid_scalar_pair_4d(v_like, u_like)


def test_metric_pair_halo_uniform_field_is_preserved():
    """A constant metric must stay constant everywhere it is copied."""
    n = 4
    u_like = jnp.full((6, n, n + 1, 1), 3.0)
    v_like = jnp.full((6, n + 1, n, 1), 3.0)
    u_p, v_p = pad_halo_dgrid_scalar_pair_4d(u_like, v_like)
    assert np.allclose(np.abs(np.asarray(u_p)), 3.0)
    assert np.allclose(np.abs(np.asarray(v_p)), 3.0)
