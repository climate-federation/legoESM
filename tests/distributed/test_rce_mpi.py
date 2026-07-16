"""MPI multi-rank tests for the RCE plane stack.

Run via the OpenMPI-installed CI job:

.. code-block:: bash

   mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_rce_mpi.py

Verifies that the MPI-aware mean-wind removal + moist-mass fixer
produce the SAME result as a single-rank run on the equivalent
global state. This is the canonical "single-rank vs MPI agreement"
check from CLAUDE.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")
mpi4jax = pytest.importorskip("mpi4jax")  # noqa: F401

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.mean_wind_filter import (
    remove_horizontal_mean_wind,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    compute_total_water_mass_plane_mpi,
    fix_moist_mass_plane_mpi,
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)

NY_GLOBAL, NX_GLOBAL, NLEV = 8, 16, 4


@pytest.fixture
def mpi_layout():
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    # 1xN decomposition along x.
    return make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=1, n_ranks_x=n_ranks,
        ny_global=NY_GLOBAL, nx_global=NX_GLOBAL,
    )


def _global_state_and_pieces(layout):
    """Build a global state on every rank; return both the global
    state and the per-rank owned slab."""
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=4_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    state_global = make_rest_state(grid_global, hc, dtype=jnp.float64)

    # Reproducible non-trivial u + tracers (seed by global indices, not
    # rank, so every rank sees the same global state on input).
    rng = np.random.default_rng(0)
    u_global = jnp.asarray(
        rng.standard_normal(state_global.u.data.shape),
    )
    tracers_global = jnp.zeros(
        (NY_GLOBAL, NX_GLOBAL, NLEV, 3), dtype=jnp.float64,
    )
    tracers_global = tracers_global.at[..., 0].set(0.01)
    state_global = state_global._replace(
        u=state_global.u.replace(data=u_global),
        tracers=state_global.tracers.replace(data=tracers_global),
    )
    # Per-rank owned slab.
    ix0, ix1 = layout.ix_start, layout.ix_end
    iy0, iy1 = layout.iy_start, layout.iy_end
    local_grid = create_plane_grid(
        nx=layout.nx_local, ny=layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    local_state = make_rest_state(local_grid, hc, dtype=jnp.float64)
    local_state = local_state._replace(
        u=local_state.u.replace(data=u_global[iy0:iy1, ix0:ix1, :]),
        tracers=local_state.tracers.replace(
            data=tracers_global[iy0:iy1, ix0:ix1, :, :],
        ),
    )
    return state_global, grid_global, local_state, local_grid, hc


def test_mpi_mean_wind_matches_single_rank(mpi_layout):
    """Global mean wind from MPI variant matches serial on global state."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("multi-rank test")
    state_global, _, local_state, _, _ = _global_state_and_pieces(mpi_layout)
    serial_out = remove_horizontal_mean_wind(state_global)
    owned_mask = jnp.ones(
        (mpi_layout.ny_local, mpi_layout.nx_local), dtype=jnp.float64,
    )
    mpi_out = remove_horizontal_mean_wind_plane_mpi(
        local_state, mpi_layout, owned_mask,
    )
    ix0, ix1 = mpi_layout.ix_start, mpi_layout.ix_end
    iy0, iy1 = mpi_layout.iy_start, mpi_layout.iy_end
    np.testing.assert_allclose(
        np.asarray(mpi_out.u.data),
        np.asarray(serial_out.u.data[iy0:iy1, ix0:ix1, :]),
        rtol=1.0e-12,
    )


def test_mpi_mean_wind_owned_mask_excludes_halo_rows(mpi_layout):
    """Codex iter-2: halo rows masked to 0 must not bias the global
    mean. Set the LAST x-column of every rank's local data to a wild
    sentinel + zero those columns in owned_mask → result must match
    serial mean over (ny_global, nx_global - n_ranks) cells."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("multi-rank test")
    state_global, _, local_state, _, _ = _global_state_and_pieces(mpi_layout)
    # Inject sentinel in last local x-column of u.
    SENTINEL = 1.0e10
    u_data = local_state.u.data
    u_data = u_data.at[:, -1, :].set(SENTINEL)
    local_state = local_state._replace(
        u=local_state.u.replace(data=u_data),
    )
    # Mask zeros that last column.
    owned_mask = jnp.ones(
        (mpi_layout.ny_local, mpi_layout.nx_local), dtype=jnp.float64,
    ).at[:, -1].set(0.0)
    mpi_out = remove_horizontal_mean_wind_plane_mpi(
        local_state, mpi_layout, owned_mask,
    )
    # MPI output for unmasked cells = u_data - mean(global owned cells).
    # The mean over global owned cells = mean over global u WITHOUT the
    # sentinel columns (because we masked them).
    # Equivalent serial calc: subtract that masked mean from the global
    # u_global, slice this rank's owned columns.
    iy0, iy1 = mpi_layout.iy_start, mpi_layout.iy_end
    ix0, ix1 = mpi_layout.ix_start, mpi_layout.ix_end
    # The MPI output on the masked column will be `SENTINEL - global_mean`
    # — still finite because the mask zeroes its contribution.
    mpi_owned_u = mpi_out.u.data[:, :-1, :]
    # Compute the equivalent serial-on-owned mean.
    # Build a global mask matching the per-rank pattern.
    nx_local = mpi_layout.nx_local
    global_mask = jnp.zeros((NY_GLOBAL, NX_GLOBAL), dtype=jnp.float64)
    for rx in range(mpi_layout.n_ranks_x):
        rx_ix0 = rx * nx_local
        global_mask = global_mask.at[
            :, rx_ix0 : rx_ix0 + nx_local - 1
        ].set(1.0)
    # Inject same sentinel pattern into global u.
    u_global = state_global.u.data
    for rx in range(mpi_layout.n_ranks_x):
        rx_ix1 = (rx + 1) * nx_local - 1
        u_global = u_global.at[:, rx_ix1, :].set(SENTINEL)
    masked_mean = jnp.sum(
        u_global * global_mask[:, :, None], axis=(0, 1),
    ) / jnp.sum(global_mask)
    expected = u_global - masked_mean[None, None, :]
    np.testing.assert_allclose(
        np.asarray(mpi_owned_u),
        np.asarray(expected[iy0:iy1, ix0 : ix1 - 1, :]),
        rtol=1.0e-12,
    )


def test_mpi_total_water_matches_single_rank(mpi_layout):
    """Global moist mass via MPI reduction equals serial on global state."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("multi-rank test")
    state_global, grid_global, local_state, local_grid, hc = (
        _global_state_and_pieces(mpi_layout)
    )
    serial_mass = float(compute_total_water_mass_plane(
        state_global, hc, grid_global,
    ))
    owned_mask = jnp.ones(
        (mpi_layout.ny_local, mpi_layout.nx_local), dtype=jnp.float64,
    )
    mpi_mass = float(compute_total_water_mass_plane_mpi(
        local_state, hc, local_grid, mpi_layout, owned_mask,
    ))
    np.testing.assert_allclose(mpi_mass, serial_mass, rtol=1.0e-12)


def test_mpi_fix_moist_mass_matches_single_rank(mpi_layout):
    """Multiplicative fixer produces identical local data after MPI run."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("multi-rank test")
    state_global, grid_global, local_state, local_grid, hc = (
        _global_state_and_pieces(mpi_layout)
    )
    serial_mass = compute_total_water_mass_plane(
        state_global, hc, grid_global,
    )
    target = 0.5 * serial_mass
    serial_fixed = fix_moist_mass_plane(
        state_global, hc, grid_global, target,
    )
    owned_mask = jnp.ones(
        (mpi_layout.ny_local, mpi_layout.nx_local), dtype=jnp.float64,
    )
    mpi_fixed = fix_moist_mass_plane_mpi(
        local_state, hc, local_grid, mpi_layout, owned_mask, target,
    )
    ix0, ix1 = mpi_layout.ix_start, mpi_layout.ix_end
    iy0, iy1 = mpi_layout.iy_start, mpi_layout.iy_end
    np.testing.assert_allclose(
        np.asarray(mpi_fixed.tracers.data[..., 0]),
        np.asarray(serial_fixed.tracers.data[iy0:iy1, ix0:ix1, :, 0]),
        rtol=1.0e-12,
    )


def test_mpi_total_water_supports_jax_grad(mpi_layout):
    """allreduce(SUM) is differentiable — grad flows."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("multi-rank test")
    _, _, local_state, local_grid, hc = _global_state_and_pieces(mpi_layout)
    owned_mask = jnp.ones(
        (mpi_layout.ny_local, mpi_layout.nx_local), dtype=jnp.float64,
    )

    def loss_fn(qv_data):
        s = local_state._replace(
            tracers=local_state.tracers.replace(
                data=local_state.tracers.data.at[..., 0].set(qv_data),
            ),
        )
        return compute_total_water_mass_plane_mpi(
            s, hc, local_grid, mpi_layout, owned_mask,
        )

    qv = local_state.tracers.data[..., 0]
    g = jax.grad(loss_fn)(qv)
    assert g.shape == qv.shape
    assert bool(jnp.all(jnp.isfinite(g)))
