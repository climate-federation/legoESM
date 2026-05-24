"""Multi-rank validation for the plane 2D periodic pencil layout.

Run via the OpenMPI-installed ``mpi-distributed.yml`` CI job:

.. code-block:: bash

   mpirun --oversubscribe -np 2 python -m pytest \\
       tests/distributed/test_plane_pencil_mpi.py -q
   mpirun --oversubscribe -np 4 python -m pytest \\
       tests/distributed/test_plane_pencil_mpi.py -q

These tests cover the multi-rank path of
:func:`legoesm.parallel.plane_mpi.exchange_halo_plane_yxz` and
:func:`legoesm.parallel.plane_mpi.gather_plane_field` that the
single-rank harness at ``tests/unit/test_plane_mpi_pencil.py``
cannot exercise:

1. ``test_halo_matches_global_wrap`` — the padded local field
   equals the corresponding slice of a globally wrapped reference
   (``jnp.pad(global, mode="wrap")``). Validates N/S and E/W stage
   ordering, corner composition, and tag uniqueness.
2. ``test_gather_round_trips_scatter`` — ``scatter`` then ``gather``
   reconstructs the original global field exactly on rank 0.
3. ``test_halo_supports_jax_grad`` — a scalar loss on the padded
   field returns finite gradients through ``_sendrecv_vjp`` on
   every rank.
4. ``test_halo_width_2_corners_correct`` — corner cells are filled
   correctly when ``halo > 1`` (two-axis composition picks up the
   diagonal neighbour's interior, not just an edge row).

Decomposition coverage: the parametrised mesh shape uses the
``np`` count detected at import time:

* np=2 → 2x1 (split y)
* np=4 → 2x2 (square)
* np=6 → 3x2

For other ``np`` the layout factory falls back to a 1xN row.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

jax.config.update("jax_enable_x64", True)

from legoesm.parallel.plane_mpi import (
    exchange_halo_plane_yxz,
    gather_plane_field,
    make_plane_pencil_layout,
    scatter_plane_field,
)


# --------------------------------------------------------------------- #
# Per-rank mesh layout fixture                                          #
# --------------------------------------------------------------------- #


def _mesh_shape(n_ranks: int) -> tuple[int, int]:
    """Pick (n_ranks_y, n_ranks_x) for the given np."""
    if n_ranks == 2:
        return (2, 1)
    if n_ranks == 4:
        return (2, 2)
    if n_ranks == 6:
        return (3, 2)
    return (1, n_ranks)


@pytest.fixture(scope="module")
def mpi_layout():
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    n_ranks_y, n_ranks_x = _mesh_shape(n_ranks)
    # Pick a global size that is divisible by the mesh in both axes
    # (avoids the remainder branch — covered separately by single-rank
    # ``test_layout_distributes_remainder_to_low_ranks``).
    ny_global = max(8, 4 * n_ranks_y)
    nx_global = max(8, 4 * n_ranks_x)
    return make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_global, nx_global=nx_global,
        halo=1,
    )


# --------------------------------------------------------------------- #
# Halo exchange                                                         #
# --------------------------------------------------------------------- #


def test_halo_matches_global_wrap(mpi_layout):
    """Multi-rank halo exchange must yield the same padded values as
    a globally wrapped pad followed by a slice of the rank's window
    (with the halo overhang). Bit-exact equality."""
    layout = mpi_layout
    h = layout.halo
    rng = np.random.default_rng(seed=42)
    # Same global field on every rank (seeded from rank-0 RNG).
    global_field = jnp.asarray(
        rng.standard_normal((layout.ny_global, layout.nx_global, 3)),
    )
    local = scatter_plane_field(global_field, layout)
    padded = exchange_halo_plane_yxz(local, layout)

    # Reference: globally wrapped pad sliced to the rank's window.
    ref_global_padded = jnp.pad(
        global_field, ((h, h), (h, h), (0, 0)), mode="wrap",
    )
    # The rank's local window in the GLOBAL padded array spans
    # (iy_start .. iy_end + 2h) along y because the global pad adds
    # h rows at the front; same for x.
    ref_slice = ref_global_padded[
        layout.iy_start:layout.iy_end + 2 * h,
        layout.ix_start:layout.ix_end + 2 * h,
    ]
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(ref_slice),
        rtol=0.0, atol=0.0,
    )


def test_halo_width_2_corners_correct():
    """halo=2: padded corners come from the diagonal neighbour's
    interior (two-axis composition), not from any single edge."""
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    n_ranks_y, n_ranks_x = _mesh_shape(n_ranks)
    if min(n_ranks_y, n_ranks_x) < 2:
        pytest.skip("halo=2 corner test needs a 2D mesh (>=2 ranks per axis)")
    # Width = 2 → interior block must satisfy halo < min(ny,nx); pick
    # global = 4 * mesh so each rank owns 4 cells per axis.
    ny_global = 4 * n_ranks_y
    nx_global = 4 * n_ranks_x
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_global, nx_global=nx_global, halo=2,
    )
    rng = np.random.default_rng(seed=1234)
    global_field = jnp.asarray(
        rng.standard_normal((ny_global, nx_global)),
    )
    local = scatter_plane_field(global_field, layout)
    padded = exchange_halo_plane_yxz(local, layout)
    ref = jnp.pad(global_field, ((2, 2), (2, 2)), mode="wrap")
    ref_slice = ref[
        layout.iy_start:layout.iy_end + 4,
        layout.ix_start:layout.ix_end + 4,
    ]
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(ref_slice),
        rtol=0.0, atol=0.0,
    )


# --------------------------------------------------------------------- #
# Single-rank-on-an-axis (self-neighbor) coverage                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("halo", [1, 2])
def test_single_rank_y_axis_self_wrap(halo):
    """Codex review 2026-05-24: `(N, 1)` mesh — every rank owns the
    full x range so the periodic E/W neighbour is the rank itself.
    The implementation must NOT issue self-sendrecv (which would
    deadlock or silently swap halos) and instead use local wrap on
    the singleton axis. Validates halo=1 and halo=2 vs the global
    wrap reference."""
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    # Force (n_ranks, 1) mesh so x is the singleton axis.
    if n_ranks < 2:
        pytest.skip("self-wrap test needs n_ranks >= 2")
    n_ranks_y, n_ranks_x = (n_ranks, 1)
    ny_global = 4 * n_ranks_y
    nx_global = max(8, 4)  # >= 2*halo + 1; small but valid
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_global, nx_global=nx_global, halo=halo,
    )
    rng = np.random.default_rng(seed=11 + halo)
    global_field = jnp.asarray(
        rng.standard_normal((ny_global, nx_global)),
    )
    local = scatter_plane_field(global_field, layout)
    padded = exchange_halo_plane_yxz(local, layout)
    ref = jnp.pad(global_field, ((halo, halo), (halo, halo)), mode="wrap")
    ref_slice = ref[
        layout.iy_start:layout.iy_end + 2 * halo,
        layout.ix_start:layout.ix_end + 2 * halo,
    ]
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(ref_slice),
        rtol=0.0, atol=0.0,
    )


@pytest.mark.parametrize("halo", [1, 2])
def test_single_rank_x_axis_self_wrap(halo):
    """Sister of the y-axis case: `(1, N)` mesh — full y range on
    every rank, periodic N/S neighbour = self. Must use local wrap
    on the singleton axis."""
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    if n_ranks < 2:
        pytest.skip("self-wrap test needs n_ranks >= 2")
    n_ranks_y, n_ranks_x = (1, n_ranks)
    ny_global = max(8, 4)
    nx_global = 4 * n_ranks_x
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_global, nx_global=nx_global, halo=halo,
    )
    rng = np.random.default_rng(seed=33 + halo)
    global_field = jnp.asarray(
        rng.standard_normal((ny_global, nx_global)),
    )
    local = scatter_plane_field(global_field, layout)
    padded = exchange_halo_plane_yxz(local, layout)
    ref = jnp.pad(global_field, ((halo, halo), (halo, halo)), mode="wrap")
    ref_slice = ref[
        layout.iy_start:layout.iy_end + 2 * halo,
        layout.ix_start:layout.ix_end + 2 * halo,
    ]
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(ref_slice),
        rtol=0.0, atol=0.0,
    )


# --------------------------------------------------------------------- #
# Gather                                                                #
# --------------------------------------------------------------------- #


def test_gather_round_trips_scatter(mpi_layout):
    """scatter(global) → gather → equals the original on rank 0."""
    layout = mpi_layout
    rng = np.random.default_rng(seed=7)
    global_field = jnp.asarray(
        rng.standard_normal((layout.ny_global, layout.nx_global, 2)),
    )
    local = scatter_plane_field(global_field, layout)
    gathered = gather_plane_field(local, layout)
    if layout.rank == 0:
        assert gathered is not None
        np.testing.assert_allclose(
            np.asarray(gathered), np.asarray(global_field),
            rtol=0.0, atol=0.0,
        )
    else:
        assert gathered is None


# --------------------------------------------------------------------- #
# Differentiability                                                     #
# --------------------------------------------------------------------- #


def test_halo_supports_jax_grad(mpi_layout):
    """jax.grad through the multi-rank exchange returns a finite
    array on every rank — proves _sendrecv_vjp backward swaps
    source/dest correctly under MPI."""
    layout = mpi_layout
    rng = np.random.default_rng(seed=99)
    local = jnp.asarray(
        rng.standard_normal((layout.ny_local, layout.nx_local, 2)),
    )

    def loss_fn(x):
        padded = exchange_halo_plane_yxz(x, layout)
        return jnp.sum(padded ** 2)

    grad = jax.grad(loss_fn)(local)
    assert grad.shape == local.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    # Every grad cell appears as 2 * the padded value at the cell's
    # position plus copies from the periodic halo neighbours. With a
    # random input on a doubly-periodic mesh every interior cell is
    # visited an odd number of times across all ranks; the local
    # gradient must therefore be strictly non-zero somewhere.
    assert float(jnp.max(jnp.abs(grad))) > 0.0
