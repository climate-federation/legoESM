"""``_owned_p_s_and_lat`` must hand HOST-USABLE arrays to the forcing code.

Under multi-controller SPMD (``--distributed-mode spmd``, #693/#749) the state
leaves are globally sharded across processes.  The external-forcing consumers
(``get_ozone_at_time`` & co) are host/NumPy interpolators; ``np.asarray`` on a
process-spanning ``jax.Array`` raises ``"Fetching value for jax.Array that
spans non-addressable devices"``.  First hit by the 2-node Levante receipt run
(job 26030421): the production config activates rrtmgp + CMIP ozone, a path the
CPU parity smokes (gray radiation, no external forcing) never reach.

Fix under test: the non-MPI branch of ``ModelDriver._owned_p_s_and_lat`` routes
``p_s`` through ``_gather_spmd_tree_to_host`` — a collective gather for
non-fully-addressable leaves and an exact no-op otherwise, so the serial /
single-GPU / mpi4jax lanes are byte-unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from legoesm.driver.model_driver import ModelDriver


def _fake_driver(p_s, face_ids=None):
    fake = MagicMock()
    fake.state.p_s.data = p_s
    fake._owned_face_ids = face_ids
    fake._physics_lat = jnp.zeros(4)
    fake._grid_lat = jnp.linspace(-1.0, 1.0, p_s.shape[-1])
    # Bind the REAL gather (the seam under test) instead of a MagicMock,
    # so the addressable no-op contract is exercised, not mocked away.
    fake._gather_spmd_tree_to_host = (
        lambda tree: ModelDriver._gather_spmd_tree_to_host(fake, tree))
    return fake


def test_serial_branch_routes_p_s_through_the_spmd_gather():
    """The non-MPI branch must call the gather — the seam the 2-node SPMD
    fix lives at.  (A truly non-addressable array needs >1 process; here we
    assert the routing with a spy.)"""
    p_s = jnp.ones((6, 4, 4))
    fake = _fake_driver(p_s)
    sentinel = np.zeros((6, 4, 4))
    fake._gather_spmd_tree_to_host = MagicMock(return_value=sentinel)

    out_p_s, out_lat = ModelDriver._owned_p_s_and_lat(fake)

    fake._gather_spmd_tree_to_host.assert_called_once()
    assert out_p_s is sentinel
    assert out_lat is fake._grid_lat


def test_gather_is_identity_for_fully_addressable_arrays():
    """Single-process arrays pass through UNCHANGED (same object) — the
    serial / single-GPU lanes must be byte-identical to before the fix."""
    p_s = jnp.arange(24.0).reshape(6, 2, 2)
    fake = _fake_driver(p_s)

    out_p_s, out_lat = ModelDriver._owned_p_s_and_lat(fake)

    assert out_p_s is p_s          # exact no-op, not a copy
    assert out_lat is fake._grid_lat


def test_mpi_branch_slices_and_never_gathers():
    """The mpi4jax lane (rank-local face slice) must not touch the gather —
    its arrays are already process-local."""
    p_s = jnp.arange(96.0).reshape(6, 4, 4)
    fake = _fake_driver(p_s, face_ids=jnp.array([0, 3]))
    spy = MagicMock()
    fake._gather_spmd_tree_to_host = spy

    out_p_s, out_lat = ModelDriver._owned_p_s_and_lat(fake)

    spy.assert_not_called()
    assert out_p_s.shape == (2, 4, 4)
    np.testing.assert_array_equal(np.asarray(out_p_s),
                                  np.asarray(p_s[jnp.array([0, 3])]))
    assert out_lat is fake._physics_lat
