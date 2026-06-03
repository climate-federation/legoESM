"""MPI halo exchange with duogrid `interp_offsets` — bit-for-bit
parity with the single-device local backend.

Closes a documented FV3 3D fidelity gap: prior to this iteration
``pad_halo_mpi`` / ``pad_halo_mpi_4d`` only did nearest-index copies
on the receive side, so any caller passing ``interp_offsets`` under
MPI hit a hard ``NotImplementedError``.  The PE 3D cubed-sphere
step (``primitive_eq_cdgrid.fv3_hydrostatic_tendencies``) is one
such caller — without offsets in MPI the 3-step bit-for-bit driver
test (``test_distributed_3_steps_matches_single_rank``) was
unrunnable.

Run with::

    JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest \\
        tests/distributed/test_mpi_interp_offsets.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import (
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    compute_halo_interp_offsets_h3,
    get_halo_backend,
    pad_halo,
    pad_halo_4d,
    pad_halo_vector_4d,
    set_halo_backend,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.parallel.comm import build_comm_topology


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _gaussian_field(n: int, nlev: int | None = None) -> jax.Array:
    """Smooth deterministic field — same on every rank."""
    key = jax.random.PRNGKey(20260527)
    if nlev is None:
        return jax.random.normal(key, (6, n, n), dtype=jnp.float64)
    return jax.random.normal(key, (6, n, n, nlev), dtype=jnp.float64)


def _build_face_only_topology():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip(
            f"Test designed for face-only mode (1/2/3/6 ranks); got {size}."
        )
    return build_comm_topology(rank, size), rank


def _owned_faces_match(
    result: np.ndarray,
    ref: np.ndarray,
    topology,
    msg: str,
) -> None:
    """Compare only the faces this rank owns.

    In replicated mode (every rank holds the full ``(6, ...)`` array),
    ``pad_halo_mpi`` only fills halos for ``topology.local_face_ids``;
    non-owned face halos remain at their zero-pad initial state.  That
    is the contract — comparing only owned faces against the local
    reference is the correct fidelity test.
    """
    for f in topology.local_face_ids:
        np.testing.assert_allclose(
            np.asarray(result)[f], np.asarray(ref)[f],
            atol=1e-12, rtol=1e-12,
            err_msg=f"{msg} (face {f})",
        )


class TestPadHaloMPIInterpOffsets:
    """Scalar 2D ``pad_halo`` with ``interp_offsets`` under MPI."""

    def test_h1_matches_local(self):
        topology, _ = _build_face_only_topology()
        n = 8
        data = _gaussian_field(n)
        offsets = compute_halo_interp_offsets(n)

        set_halo_backend("local")
        ref = pad_halo(data, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo(data, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo(h1) with offsets ≠ local",
        )

    def test_h2_matches_local(self):
        topology, _ = _build_face_only_topology()
        n = 8
        data = _gaussian_field(n)
        offsets = compute_halo_interp_offsets_h2(n)

        set_halo_backend("local")
        ref = pad_halo(data, halo=2, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo(data, halo=2, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo(h2) with offsets ≠ local",
        )

    def test_h3_matches_local(self):
        topology, _ = _build_face_only_topology()
        n = 8
        data = _gaussian_field(n)
        offsets = compute_halo_interp_offsets_h3(n)

        set_halo_backend("local")
        ref = pad_halo(data, halo=3, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo(data, halo=3, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo(h3) with offsets ≠ local",
        )


class TestPadHalo4DMPIInterpOffsets:
    """4D ``pad_halo_4d`` with ``interp_offsets`` under MPI."""

    def test_h1_matches_local(self):
        topology, _ = _build_face_only_topology()
        n, nlev = 8, 5
        data = _gaussian_field(n, nlev)
        offsets = compute_halo_interp_offsets(n)

        set_halo_backend("local")
        ref = pad_halo_4d(data, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo_4d(data, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo_4d(h1) with offsets ≠ local",
        )

    def test_h2_matches_local(self):
        topology, _ = _build_face_only_topology()
        n, nlev = 8, 5
        data = _gaussian_field(n, nlev)
        offsets = compute_halo_interp_offsets_h2(n)

        set_halo_backend("local")
        ref = pad_halo_4d(data, halo=2, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo_4d(data, halo=2, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo_4d(h2) with offsets ≠ local",
        )

    def test_h3_matches_local(self):
        topology, _ = _build_face_only_topology()
        n, nlev = 8, 5
        data = _gaussian_field(n, nlev)
        offsets = compute_halo_interp_offsets_h3(n)

        set_halo_backend("local")
        ref = pad_halo_4d(data, halo=3, interp_offsets=offsets)

        set_halo_backend("mpi", topology)
        result = pad_halo_4d(data, halo=3, interp_offsets=offsets)

        _owned_faces_match(
            result, ref, topology,
            "MPI pad_halo_4d(h3) with offsets ≠ local",
        )

    def test_offsets_none_unchanged_baseline(self):
        """When offsets=None the MPI path is unchanged (bit-for-bit)."""
        topology, _ = _build_face_only_topology()
        n, nlev = 8, 5
        data = _gaussian_field(n, nlev)

        set_halo_backend("local")
        ref = pad_halo_4d(data)

        set_halo_backend("mpi", topology)
        result = pad_halo_4d(data)

        for f in topology.local_face_ids:
            np.testing.assert_array_equal(
                np.asarray(result)[f], np.asarray(ref)[f],
                err_msg=f"MPI pad_halo_4d (no offsets) baseline drifted (face {f})",
            )


class TestPadHaloVector4DMPIInterpOffsets:
    """Packed (u, v) vector halo with ``interp_offsets`` under MPI.

    Exercises the ``hydrostatic_to_fv3`` cell-centre → D-grid lift used
    by every PE 3D step.
    """

    def test_packed_h1_matches_local(self):
        topology, _ = _build_face_only_topology()
        n, nlev = 8, 5
        grid = create_cubed_sphere(n)
        offsets = grid.halo_interp_offsets

        u = _gaussian_field(n, nlev)
        v = _gaussian_field(n, nlev) + 0.5  # distinct seed-derived field
        cos_a = grid.cos_angle
        sin_a = grid.sin_angle
        cos_a_pad = grid.cos_angle_padded
        sin_a_pad = grid.sin_angle_padded

        set_halo_backend("local")
        u_ref, v_ref = pad_halo_vector_4d(
            u, v, cos_a, sin_a, cos_a_pad, sin_a_pad,
            interp_offsets=offsets,
        )

        set_halo_backend("mpi", topology)
        u_mpi, v_mpi = pad_halo_vector_4d(
            u, v, cos_a, sin_a, cos_a_pad, sin_a_pad,
            interp_offsets=offsets,
        )

        _owned_faces_match(
            u_mpi, u_ref, topology,
            "MPI pad_halo_vector_4d u with offsets ≠ local",
        )
        _owned_faces_match(
            v_mpi, v_ref, topology,
            "MPI pad_halo_vector_4d v with offsets ≠ local",
        )
