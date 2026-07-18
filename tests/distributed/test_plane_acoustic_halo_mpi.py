"""Multi-rank SI-horizontal acoustic substep vs single-process serial.

Splits a global state across ranks, runs the halo-aware full
Skamarock-Klemp acoustic substep locally
(:func:`plane_acoustic_substeps_si_horizontal_halo`), gathers, and
compares against the serial substep on the SAME global state. Also runs
the full ``step_halo`` with ``substep_horizontal_acoustic=True`` (the
path that previously raised ``NotImplementedError`` on multi-rank)
against the serial ``step``.

Run under mpirun:
    mpirun -np 2 .venv/bin/python -m pytest \
        tests/distributed/test_plane_acoustic_halo_mpi.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")
mpi4jax = pytest.importorskip("mpi4jax")  # noqa: F841

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (  # noqa: E402
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_acoustic_substeps_si_horizontal,
    plane_acoustic_substeps_si_horizontal_halo,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.grids.vertical import create_height_coordinate  # noqa: E402
from legoesm.parallel.plane_mpi import (  # noqa: E402
    gather_plane_field,
    make_plane_pencil_layout,
)
from legoesm.timestepping.split_explicit import SplitExplicitConfig  # noqa: E402

jax.config.update("jax_enable_x64", True)

NY_GLOBAL, NX_GLOBAL, NLEV = 8, 12, 6


def _balanced_factor(n):
    best = (1, n)
    for nry in range(1, n + 1):
        if n % nry:
            continue
        nrx = n // nry
        if NY_GLOBAL // nry >= 4 and NX_GLOBAL // nrx >= 4:
            if abs(nry - nrx) < abs(best[0] - best[1]) or best == (1, n):
                best = (nry, nrx)
    return best


@pytest.fixture
def mpi_layout():
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    nry, nrx = _balanced_factor(n_ranks)
    return make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=nry, n_ranks_x=nrx,
        ny_global=NY_GLOBAL, nx_global=NX_GLOBAL,
    )


def _build_state(grid, hc, rng_seed=0):
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    keys = jax.random.split(jax.random.PRNGKey(rng_seed), 6)
    return rest._replace(
        u=rest.u.replace(
            data=0.1 * jax.random.normal(keys[0], rest.u.data.shape)),
        v=rest.v.replace(
            data=0.1 * jax.random.normal(keys[1], rest.v.data.shape)),
        w=rest.w.replace(
            data=0.01 * jax.random.normal(keys[2], rest.w.data.shape)),
        theta_prime=rest.theta_prime.replace(
            data=0.5 * jax.random.normal(
                keys[3], rest.theta_prime.data.shape)),
        rho_prime=rest.rho_prime.replace(
            data=0.001 * jax.random.normal(
                keys[4], rest.rho_prime.data.shape)),
    )


def _slice_state_to_rank(state, layout):
    iy0, iy1 = layout.iy_start, layout.iy_end
    ix0, ix1 = layout.ix_start, layout.ix_end

    def slc(arr):
        return arr[iy0:iy1, ix0:ix1]

    return state._replace(
        u=state.u.replace(data=slc(state.u.data)),
        v=state.v.replace(data=slc(state.v.data)),
        w=state.w.replace(data=slc(state.w.data)),
        theta_prime=state.theta_prime.replace(
            data=slc(state.theta_prime.data)),
        rho_prime=state.rho_prime.replace(data=slc(state.rho_prime.data)),
        phis=state.phis.replace(data=slc(state.phis.data)),
        tracers=state.tracers.replace(data=slc(state.tracers.data)),
    )


def _cfg():
    return CompressibleEulerConfig(
        semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True,
        acoustic_off_centering=0.1,
        n_acoustic_substeps=4,
        sponge_coeff=0.05, sponge_width=5_000.0,
        smagorinsky_cs=0.0, use_coriolis=False, fix_mass=False,
    )


def test_acoustic_si_horizontal_halo_multirank(mpi_layout):
    """Gathered multi-rank halo substep == serial substep on global state."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = _cfg()
    se_cfg = SplitExplicitConfig(n_substeps=cfg.n_acoustic_substeps)

    state_global = _build_state(grid_global, hc)
    ref = plane_acoustic_substeps_si_horizontal(
        state_global, None, 0.5, cfg.n_acoustic_substeps, se_cfg,
        hc, tm_global, cfg, grid_global,
    )

    local_grid = create_plane_grid(
        nx=mpi_layout.nx_local, ny=mpi_layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    local_state = _slice_state_to_rank(state_global, mpi_layout)

    local_out = plane_acoustic_substeps_si_horizontal_halo(
        local_state, None, 0.5, cfg.n_acoustic_substeps, se_cfg,
        hc, local_tm, cfg, local_grid, mpi_layout,
    )

    for fld in ("u", "v", "w", "theta_prime", "rho_prime"):
        gathered = gather_plane_field(
            getattr(local_out, fld).data, mpi_layout,
        )
        if mpi_layout.rank == 0:
            np.testing.assert_allclose(
                np.asarray(gathered),
                np.asarray(getattr(ref, fld).data),
                rtol=1.0e-12, atol=1.0e-12,
                err_msg=f"{fld} multi-rank mismatch",
            )


def test_acoustic_si_horizontal_halo_yslab(mpi_layout):
    """Forced (n_ranks × 1) Y-SLAB decomposition: exercises the NORTH/SOUTH
    per-substep exchange, which the balanced factoring skips at np=2 (it
    picks 1×2 → x-only)."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    comm = MPI.COMM_WORLD
    n_ranks, rank = comm.Get_size(), comm.Get_rank()
    if NY_GLOBAL // n_ranks < 4:
        # create_plane_grid needs ny >= 4 per rank; at np>=4 the balanced
        # fixture already decomposes y (2x2), so the dedicated y-slab
        # variant only needs to run where it adds coverage (np=2).
        pytest.skip("y-slab too thin for create_plane_grid at this np")
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks, n_ranks_y=n_ranks, n_ranks_x=1,
        ny_global=NY_GLOBAL, nx_global=NX_GLOBAL,
    )
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = _cfg()
    se_cfg = SplitExplicitConfig(n_substeps=cfg.n_acoustic_substeps)
    state_global = _build_state(grid_global, hc)
    ref = plane_acoustic_substeps_si_horizontal(
        state_global, None, 0.5, cfg.n_acoustic_substeps, se_cfg,
        hc, tm_global, cfg, grid_global,
    )
    local_grid = create_plane_grid(
        nx=layout.nx_local, ny=layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    local_out = plane_acoustic_substeps_si_horizontal_halo(
        _slice_state_to_rank(state_global, layout), None, 0.5,
        cfg.n_acoustic_substeps, se_cfg, hc, local_tm, cfg,
        local_grid, layout,
    )
    for fld in ("u", "v", "w", "theta_prime", "rho_prime"):
        gathered = gather_plane_field(getattr(local_out, fld).data, layout)
        if layout.rank == 0:
            np.testing.assert_allclose(
                np.asarray(gathered),
                np.asarray(getattr(ref, fld).data),
                rtol=1.0e-12, atol=1.0e-12,
                err_msg=f"{fld} y-slab mismatch",
            )


def test_step_halo_substep_horizontal_multirank(mpi_layout):
    """Full step_halo with the Skamarock-Klemp split on multi-rank
    (previously NotImplementedError) matches the serial step."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = _cfg()

    state_global = _build_state(grid_global, hc)
    model_global = PlaneCompressibleEulerModel(
        grid_global, hc, tm_global, cfg,
    )
    ref = state_global
    for _ in range(3):
        ref = model_global.step(ref, dt=2.0)

    local_grid = create_plane_grid(
        nx=mpi_layout.nx_local, ny=mpi_layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    model_local = PlaneCompressibleEulerModel(local_grid, hc, local_tm, cfg)
    local = _slice_state_to_rank(state_global, mpi_layout)
    for _ in range(3):
        local = model_local.step_halo(local, dt=2.0, layout=mpi_layout)

    for fld in ("u", "v", "w", "theta_prime", "rho_prime"):
        gathered = gather_plane_field(getattr(local, fld).data, mpi_layout)
        if mpi_layout.rank == 0:
            np.testing.assert_allclose(
                np.asarray(gathered),
                np.asarray(getattr(ref, fld).data),
                rtol=1.0e-11, atol=1.0e-11,
                err_msg=f"{fld} full-step multi-rank mismatch",
            )
