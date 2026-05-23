"""Single-rank tests for the plane 2D periodic pencil layout (PR5).

Multi-rank validation runs under ``mpirun -np N`` and lives in
``tests/distributed/`` (separate suite, exercised by the MPI CI job).
The single-rank tests below cover:

1. Layout factory: correct decomposition, neighbour ranks, halo
   guard.
2. Halo exchange fallback (``n_ranks == 1``) is bit-exact equal to
   ``jnp.pad(mode='wrap')`` — proves the single-rank path matches
   the PR1 ``pad_halo_plane_4d`` semantics so MPI cannot regress
   the non-MPI dycore.
3. Scatter + identity gather round-trip on the single-rank case.
4. Local plane grid offset: ``xc``, ``yc`` shifted to global
   coordinates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.plane_mpi import (
    PlanePencilLayout,
    exchange_halo_plane_yxz,
    gather_plane_field,
    make_plane_pencil_grid,
    make_plane_pencil_layout,
    scatter_plane_field,
)


jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------- #
# Layout factory                                                        #
# --------------------------------------------------------------------- #


def test_single_rank_layout_owns_full_domain():
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=16, nx_global=16,
    )
    assert layout.ny_local == 16
    assert layout.nx_local == 16
    assert layout.iy_start == 0 and layout.iy_end == 16
    assert layout.ix_start == 0 and layout.ix_end == 16
    # Single-rank periodic: every neighbour is self.
    assert layout.north_rank == 0
    assert layout.south_rank == 0
    assert layout.east_rank == 0
    assert layout.west_rank == 0


def test_two_by_two_layout_decomposes_correctly():
    """4-rank 2x2 mesh: each rank owns 8x8 of a 16x16 global plane."""
    layouts = [
        make_plane_pencil_layout(
            rank=r, n_ranks=4, n_ranks_y=2, n_ranks_x=2,
            ny_global=16, nx_global=16,
        )
        for r in range(4)
    ]
    for L in layouts:
        assert L.ny_local == 8
        assert L.nx_local == 8
    # Rank 0 at (0, 0); periodic neighbours wrap.
    L0 = layouts[0]
    assert L0.ry == 0 and L0.rx == 0
    assert L0.east_rank == 1   # (0, 1)
    assert L0.west_rank == 1   # (0, 1) wraps
    assert L0.north_rank == 2  # (1, 0)
    assert L0.south_rank == 2  # (1, 0) wraps


def test_layout_rejects_mismatched_mesh_count():
    with pytest.raises(ValueError, match="n_ranks="):
        make_plane_pencil_layout(
            rank=0, n_ranks=4, n_ranks_y=2, n_ranks_x=3,
            ny_global=12, nx_global=12,
        )


def test_layout_rejects_halo_too_wide():
    """halo >= min(ny_local, nx_local) would wrap the interior."""
    with pytest.raises(ValueError, match="halo="):
        make_plane_pencil_layout(
            rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
            ny_global=4, nx_global=4, halo=4,
        )


def test_layout_rejects_over_decomposition_y():
    """Codex PR5 iter-1 finding M4: n_ranks_y > ny_global must raise."""
    with pytest.raises(ValueError, match="n_ranks_y=5 exceeds ny_global=4"):
        make_plane_pencil_layout(
            rank=0, n_ranks=5, n_ranks_y=5, n_ranks_x=1,
            ny_global=4, nx_global=8,
        )


def test_layout_rejects_over_decomposition_x():
    with pytest.raises(ValueError, match="n_ranks_x=6 exceeds nx_global=4"):
        make_plane_pencil_layout(
            rank=0, n_ranks=6, n_ranks_y=1, n_ranks_x=6,
            ny_global=8, nx_global=4,
        )


def test_layout_rejects_zero_halo():
    """Codex PR5 iter-1: halo == 0 leaves no padding for stencils."""
    with pytest.raises(ValueError, match="halo=0"):
        make_plane_pencil_layout(
            rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
            ny_global=8, nx_global=8, halo=0,
        )


def test_exchange_rejects_wrong_leading_axes():
    """Codex PR5 iter-1 M3: leading two axes must match (ny_local,
    nx_local). Catch mismatches before the silent (and wrong)
    pad/exchange runs."""
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=8, nx_global=8, halo=1,
    )
    from legoesm.parallel.plane_mpi import exchange_halo_plane_yxz
    bad = jnp.zeros((4, 8, 2))  # ny doesn't match layout.ny_local=8
    with pytest.raises(ValueError, match="leading axes"):
        exchange_halo_plane_yxz(bad, layout)


def test_layout_distributes_remainder_to_low_ranks():
    """ny_global=17 with 4 ranks_y: ranks 0 get 5 rows, rank 1-3 get
    4 (5+4+4+4 = 17). Codex-style check for the remainder branch."""
    sizes = [
        make_plane_pencil_layout(
            rank=r, n_ranks=4, n_ranks_y=4, n_ranks_x=1,
            ny_global=17, nx_global=8,
        ).ny_local
        for r in range(4)
    ]
    assert sizes == [5, 4, 4, 4]


# --------------------------------------------------------------------- #
# Halo exchange fallback                                                #
# --------------------------------------------------------------------- #


def test_single_rank_halo_exchange_matches_jnp_pad_wrap():
    """Single-rank fallback path must equal ``jnp.pad(mode='wrap')``
    bit-exact so the MPI wrapper cannot regress the non-MPI dycore."""
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=6, nx_global=8, halo=2,
    )
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((6, 8, 4)))
    padded = exchange_halo_plane_yxz(field, layout)
    expected = jnp.pad(field, ((2, 2), (2, 2), (0, 0)), mode="wrap")
    assert jnp.array_equal(padded, expected)


def test_single_rank_halo_corners_filled_by_two_axis_wrap():
    """The single-rank fallback uses ``jnp.pad(mode='wrap')``; this
    test pins that corners come from the diagonally opposite block
    (NW corner of padded = SE corner of input, etc.) per the PR1
    test_plane_halo conventions."""
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=4, nx_global=4, halo=1,
    )
    rng = np.random.default_rng(1)
    field = jnp.asarray(rng.standard_normal((4, 4, 2)))
    padded = exchange_halo_plane_yxz(field, layout)
    # NW corner of padded = SE of input.
    assert jnp.array_equal(padded[0, 0], field[-1, -1])
    # SE corner of padded = NW of input.
    assert jnp.array_equal(padded[-1, -1], field[0, 0])


# --------------------------------------------------------------------- #
# Scatter / gather                                                      #
# --------------------------------------------------------------------- #


def test_scatter_extracts_correct_subblock():
    layout = make_plane_pencil_layout(
        rank=1, n_ranks=4, n_ranks_y=2, n_ranks_x=2,
        ny_global=8, nx_global=8,
    )
    # Rank 1 = (ry=0, rx=1); owns (0:4, 4:8).
    global_field = jnp.arange(8 * 8).reshape(8, 8).astype(jnp.float64)
    local = scatter_plane_field(global_field, layout)
    assert local.shape == (4, 4)
    assert jnp.array_equal(local, global_field[0:4, 4:8])


def test_single_rank_gather_is_identity():
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=6, nx_global=6,
    )
    rng = np.random.default_rng(2)
    field = jnp.asarray(rng.standard_normal((6, 6, 3)))
    gathered = gather_plane_field(field, layout)
    assert jnp.array_equal(gathered, field)


# --------------------------------------------------------------------- #
# Pencil grid offset                                                    #
# --------------------------------------------------------------------- #


def test_pencil_grid_beta_plane_f_y_uses_global_y():
    """Codex PR5 iter-1 finding M1: ``make_plane_pencil_grid`` must
    compute ``f_y = f0 + beta*(y_global - Ly_global/2)`` from the
    GLOBAL y coordinate, not the local sub-block's y. On a 2x1 mesh
    (split in y), rank 1 owns the northern half and its
    ``f_y - f0`` should be > 0 (north of global midpoint), not
    centred around zero as a local-only formula would give."""
    layout = make_plane_pencil_layout(
        rank=1, n_ranks=2, n_ranks_y=2, n_ranks_x=1,
        ny_global=8, nx_global=4,
    )
    dx = dy = 1_000.0
    f0 = 1.0e-4
    beta = 2.0e-11
    grid = make_plane_pencil_grid(
        layout, dx=dx, dy=dy, nlev=4,
        coriolis_mode="beta_plane", f0=f0, beta=beta,
    )
    # Rank 1 (ry=1) owns global rows 4..7; y_centres = 4.5*dy..7.5*dy.
    # Global Ly = 8*dy = 8 km; midpoint at 4*dy = 4 km.
    # So f - f0 = beta * (y - 4 km) > 0 for every cell in rank 1.
    f_anomaly = grid.f_y - f0
    assert float(jnp.min(f_anomaly)) > 0.0, (
        f"Rank 1 (north half) should have f > f0 everywhere; "
        f"min(f - f0) = {float(jnp.min(f_anomaly)):.3e}"
    )
    # Also confirm Lx, Ly carry the GLOBAL extent.
    assert float(grid.Lx) == pytest.approx(layout.nx_global * dx)
    assert float(grid.Ly) == pytest.approx(layout.ny_global * dy)


def test_pencil_grid_offsets_local_coordinates_to_global():
    """Local grid's ``xc``, ``yc`` must match where the rank's
    sub-domain sits in the global plane (matters for diagnostics +
    radiation latitude)."""
    layout = make_plane_pencil_layout(
        rank=3, n_ranks=4, n_ranks_y=2, n_ranks_x=2,
        ny_global=8, nx_global=8,
    )
    # Rank 3 = (ry=1, rx=1); owns (4:8, 4:8). Offset should be
    # (iy_start * dy, ix_start * dx).
    dx = dy = 1_000.0
    grid = make_plane_pencil_grid(layout, dx=dx, dy=dy, nlev=4)
    assert grid.ny == 4 and grid.nx == 4
    # Local cell-centres start at (ix_start + 0.5) * dx, etc.
    assert float(grid.xc[0]) == pytest.approx(
        (layout.ix_start + 0.5) * dx, rel=1.0e-12,
    )
    assert float(grid.yc[0]) == pytest.approx(
        (layout.iy_start + 0.5) * dy, rel=1.0e-12,
    )
