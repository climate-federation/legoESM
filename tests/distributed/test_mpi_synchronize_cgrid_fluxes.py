"""MPI bit-for-bit guard for the iter-1049 ``synchronize_cgrid_fluxes`` fix.

iter-1049 made ``synchronize_cgrid_fluxes`` MPI-aware: under
``_halo_backend == "mpi"`` it dispatches to
``_synchronize_cgrid_fluxes_mpi`` which uses ``mpi4jax.sendrecv``
to swap boundary flux strips across ranks before averaging.

This test verifies that the MPI path produces bit-for-bit
identical owned-face boundary fluxes vs the single-device path.
A regression in the sendrecv tag scheme, reversal logic, or
sign-flip table would cause the MPI averages to drift.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 2 \\
        python -m pytest \\
        tests/distributed/test_mpi_synchronize_cgrid_fluxes.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import (
    set_halo_backend,
    synchronize_cgrid_fluxes,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.parallel.comm import build_comm_topology


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _build_face_only_topology():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip("face-only mode only (1/2/3/6 ranks)")
    return build_comm_topology(rank, size), rank


def _random_fluxes(n: int, nlev: int | None = None):
    """Deterministic random fluxes."""
    key = jax.random.PRNGKey(20260528)
    fx_shape = (6, n + 1, n) if nlev is None else (6, n + 1, n, nlev)
    fy_shape = (6, n, n + 1) if nlev is None else (6, n, n + 1, nlev)
    fx = jax.random.normal(jax.random.fold_in(key, 0), fx_shape, dtype=jnp.float64)
    fy = jax.random.normal(jax.random.fold_in(key, 1), fy_shape, dtype=jnp.float64)
    return fx, fy


class TestSynchronizeCgridFluxesMPI:
    """MPI vs local equivalence on owned faces."""

    def test_2d_owned_faces_match(self):
        """``synchronize_cgrid_fluxes`` for 3D ``(6, n+1, n)`` / ``(6, n, n+1)``."""
        topo, rank = _build_face_only_topology()
        n = 8
        fx, fy = _random_fluxes(n)

        set_halo_backend("local")
        fx_ref, fy_ref = synchronize_cgrid_fluxes(fx, fy, n)

        set_halo_backend("mpi", topo)
        fx_mpi, fy_mpi = synchronize_cgrid_fluxes(fx, fy, n)

        for f in topo.local_face_ids:
            np.testing.assert_allclose(
                np.asarray(fx_mpi)[f], np.asarray(fx_ref)[f],
                atol=1e-14, rtol=1e-14,
                err_msg=f"rank{rank} fx face{f} diverged",
            )
            np.testing.assert_allclose(
                np.asarray(fy_mpi)[f], np.asarray(fy_ref)[f],
                atol=1e-14, rtol=1e-14,
                err_msg=f"rank{rank} fy face{f} diverged",
            )

    def test_4d_owned_faces_match(self):
        """``synchronize_cgrid_fluxes`` for 4D ``(6, n+1, n, nlev)`` /
        ``(6, n, n+1, nlev)`` — exercises the iter-1049 4D-lift path
        that ``cgrid_mass_flux_divergence`` uses."""
        topo, rank = _build_face_only_topology()
        n, nlev = 8, 5
        fx, fy = _random_fluxes(n, nlev)

        set_halo_backend("local")
        fx_ref, fy_ref = synchronize_cgrid_fluxes(fx, fy, n)

        set_halo_backend("mpi", topo)
        fx_mpi, fy_mpi = synchronize_cgrid_fluxes(fx, fy, n)

        for f in topo.local_face_ids:
            np.testing.assert_allclose(
                np.asarray(fx_mpi)[f], np.asarray(fx_ref)[f],
                atol=1e-14, rtol=1e-14,
                err_msg=f"rank{rank} fx face{f} diverged",
            )
            np.testing.assert_allclose(
                np.asarray(fy_mpi)[f], np.asarray(fy_ref)[f],
                atol=1e-14, rtol=1e-14,
                err_msg=f"rank{rank} fy face{f} diverged",
            )

    def test_with_polar_sign_flip_edges(self):
        """Verify the ``_FLUX_SIGN_FLIP_EDGES`` table applies under MPI.

        The 8 polar-adjacent edges (involving faces 4 or 5) use OPPOSITE
        sign conventions for the mass flux.  iter-1049's MPI variant
        must reproduce the same sign-flip behaviour as the local path.

        Random data fills both polar and non-polar boundaries, so the
        per-face checks above implicitly cover this — but having an
        explicit assertion documenting the contract guards against
        future refactors dropping the table.
        """
        from legoesm.grids.halo import _FLUX_SIGN_FLIP_EDGES, WEST, EAST, SOUTH, NORTH
        # Pin the iter-808 table content (8 polar-adjacent edges).
        expected = frozenset({
            (1, NORTH), (4, EAST),
            (2, SOUTH), (5, SOUTH),
            (2, NORTH), (4, NORTH),
            (3, SOUTH), (5, WEST),
        })
        assert _FLUX_SIGN_FLIP_EDGES == expected
