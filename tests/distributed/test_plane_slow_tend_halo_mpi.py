"""Multi-rank halo slow tendency vs single-process original.

Splits a global state across ranks, runs halo-aware slow tendency
locally, gathers, compares to single-process original on the SAME
global state. Catches halo-edge bugs (off-by-one slicing, wrong
direction in sendrecv, etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")
mpi4jax = pytest.importorskip("mpi4jax")  # noqa: F401

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric, make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
    plane_compressible_euler_slow_tendencies_halo,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import (
    gather_plane_field, make_plane_pencil_layout,
)

jax.config.update("jax_enable_x64", True)

NY_GLOBAL, NX_GLOBAL, NLEV = 8, 12, 6


def _balanced_factor(n):
    """Factor ``n`` into ``(nry, nrx)`` keeping BOTH local dims ≥ 4 (the plane
    dycore's del4 biharmonic needs nx,ny ≥ 4). A 1×n slab of a 12-wide domain
    gives nx_local=3 at n=4 — too thin — so prefer the most balanced factoring
    whose local extents both clear the stencil floor."""
    best = (1, n)
    for nry in range(1, n + 1):
        if n % nry:
            continue
        nrx = n // nry
        if NY_GLOBAL // nry >= 4 and NX_GLOBAL // nrx >= 4:
            # pick the pair closest to square
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
    rng = jax.random.PRNGKey(rng_seed)
    keys = jax.random.split(rng, 6)
    u = 0.1 * jax.random.normal(keys[0], rest.u.data.shape)
    v = 0.1 * jax.random.normal(keys[1], rest.v.data.shape)
    w = 0.01 * jax.random.normal(keys[2], rest.w.data.shape)
    th = 0.5 * jax.random.normal(keys[3], rest.theta_prime.data.shape)
    rho_p = 0.001 * jax.random.normal(keys[4], rest.rho_prime.data.shape)
    tr = 0.01 * jax.random.normal(
        keys[5], (NY_GLOBAL, NX_GLOBAL, NLEV, 3),
    )
    return rest._replace(
        u=rest.u.replace(data=u), v=rest.v.replace(data=v),
        w=rest.w.replace(data=w),
        theta_prime=rest.theta_prime.replace(data=th),
        rho_prime=rest.rho_prime.replace(data=rho_p),
        tracers=rest.tracers.replace(data=tr),
    )


def _slice_state_to_rank(state, layout):
    iy0, iy1 = layout.iy_start, layout.iy_end
    ix0, ix1 = layout.ix_start, layout.ix_end

    def slc(arr, two_d_axes=(0, 1)):
        return arr[iy0:iy1, ix0:ix1]

    return state._replace(
        u=state.u.replace(data=slc(state.u.data)),
        v=state.v.replace(data=slc(state.v.data)),
        w=state.w.replace(data=slc(state.w.data)),
        theta_prime=state.theta_prime.replace(data=slc(state.theta_prime.data)),
        rho_prime=state.rho_prime.replace(data=slc(state.rho_prime.data)),
        phis=state.phis.replace(data=slc(state.phis.data)),
        tracers=state.tracers.replace(data=slc(state.tracers.data)),
    )


def test_halo_slow_tend_multirank_matches_single_process(mpi_layout):
    """Local halo tendency gathered across ranks == single-process tendency."""
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e6,
        hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
        smagorinsky_cs=0.0,
        use_coriolis=False,
        n_acoustic_substeps=12,
    )

    state_global = _build_state(grid_global, hc)
    # Reference: single-process original on global state.
    ref_tend = plane_compressible_euler_slow_tendencies(
        state_global, grid_global, hc, tm_global, cfg,
    )

    # Each rank gets a local slab.
    local_grid = create_plane_grid(
        nx=mpi_layout.nx_local, ny=mpi_layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    local_state = _slice_state_to_rank(state_global, mpi_layout)

    local_tend = plane_compressible_euler_slow_tendencies_halo(
        local_state, local_grid, hc, local_tm, cfg, mpi_layout,
    )

    # Gather each tendency component to rank 0 and compare.
    for fld in (
        "du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt",
        "drho_prime_dt", "dtracers_dt",
    ):
        local_arr = getattr(local_tend, fld).data
        gathered = gather_plane_field(local_arr, mpi_layout)
        if mpi_layout.rank == 0:
            ref_arr = getattr(ref_tend, fld).data
            np.testing.assert_allclose(
                np.asarray(gathered),
                np.asarray(ref_arr),
                rtol=1.0e-12, atol=1.0e-12,
                err_msg=f"{fld} multi-rank mismatch",
            )


def test_full_step_halo_matches_single_process(mpi_layout):
    """Full multi-step ``step_halo`` (acoustic substeps + Smagorinsky)
    gathered across ranks == single-rank, to machine precision.

    Guards the iter-9 halo-tag fix (the exchange silently swapped halos
    at any 2-rank-per-axis decomposition) AND the iter-8 JIT of the
    multi-rank ``step_halo``.  Slow-tendency alone is covered above; this
    also exercises the column-local acoustic substeps + the (no-op here,
    fix_mass=False) mass path over several steps.
    """
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6, smagorinsky_cs=0.2,
        use_coriolis=False, n_acoustic_substeps=12, fix_mass=False,
    )
    state_global = _build_state(grid_global, hc)
    # Zero the tracers: this test validates the DYNAMICS halo path (the
    # iter-9 tag fix + iter-8 JIT), which is bit-identical multi-rank vs
    # single-rank (~1e-15).  With NON-zero moisture tracers the full step
    # currently diverges ~6e-4 over 3 steps — a SEPARATE multi-rank
    # moist-coupling discrepancy (tracer-derived ``b_moist`` in the acoustic
    # substep), unrelated to the halo-exchange fix and tracked separately.
    state_global = state_global._replace(
        tracers=state_global.tracers.replace(
            data=jnp.zeros_like(state_global.tracers.data)),
    )

    # Single-rank reference (n_ranks=1 routes step_halo -> jit'd step()).
    lay1 = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=NY_GLOBAL, nx_global=NX_GLOBAL,
    )
    m_ref = PlaneCompressibleEulerModel(grid_global, hc, tm_global, cfg)
    ref = state_global
    for _ in range(3):
        ref = m_ref.step_halo(ref, dt=0.5, layout=lay1)

    # Multi-rank on local slabs.
    local_grid = create_plane_grid(
        nx=mpi_layout.nx_local, ny=mpi_layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    m_loc = PlaneCompressibleEulerModel(local_grid, hc, local_tm, cfg)
    local_state = _slice_state_to_rank(state_global, mpi_layout)
    for _ in range(3):
        local_state = m_loc.step_halo(local_state, dt=0.5, layout=mpi_layout)

    for fld in ("u", "v", "w", "theta_prime", "rho_prime"):
        gathered = gather_plane_field(getattr(local_state, fld).data, mpi_layout)
        if mpi_layout.rank == 0:
            ref_arr = getattr(ref, fld).data
            np.testing.assert_allclose(
                np.asarray(gathered), np.asarray(ref_arr),
                rtol=1.0e-10, atol=1.0e-10,
                err_msg=f"{fld} full-step multi-rank mismatch",
            )


def test_moist_global_mean_exact_serial_parity(mpi_layout):
    """``acoustic_moist_global_mean=True`` restores EXACT single-rank parity
    for the moist (SAM b_moist) path under a pencil decomposition.

    Default (rank-local mean) is the scalable choice and diverges ~1e-4 from
    single-rank; the opt-in global-mean flag (one allreduce/field) makes it
    bit-identical, for oracle / validation runs.
    """
    if mpi_layout.n_ranks == 1:
        pytest.skip("requires multi-rank mpirun")
    grid_global = create_plane_grid(
        nx=NX_GLOBAL, ny=NY_GLOBAL, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm_global = make_flat_plane_terrain_metric(grid_global, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6, smagorinsky_cs=0.2,
        use_coriolis=False, n_acoustic_substeps=12, fix_mass=False,
        moist_buoyancy=True, acoustic_moist_global_mean=True,
    )
    state_global = _build_state(grid_global, hc)
    # Positive (physical) moisture so b_moist is active.
    state_global = state_global._replace(
        tracers=state_global.tracers.replace(
            data=jnp.abs(state_global.tracers.data)),
    )

    lay1 = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=NY_GLOBAL, nx_global=NX_GLOBAL,
    )
    m_ref = PlaneCompressibleEulerModel(grid_global, hc, tm_global, cfg)
    ref = state_global
    for _ in range(3):
        ref = m_ref.step_halo(ref, dt=0.5, layout=lay1)

    local_grid = create_plane_grid(
        nx=mpi_layout.nx_local, ny=mpi_layout.ny_local, nlev=NLEV,
        dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    m_loc = PlaneCompressibleEulerModel(local_grid, hc, local_tm, cfg)
    local_state = _slice_state_to_rank(state_global, mpi_layout)
    for _ in range(3):
        local_state = m_loc.step_halo(local_state, dt=0.5, layout=mpi_layout)

    for fld in ("u", "w", "theta_prime"):
        gathered = gather_plane_field(getattr(local_state, fld).data, mpi_layout)
        if mpi_layout.rank == 0:
            np.testing.assert_allclose(
                np.asarray(gathered), np.asarray(getattr(ref, fld).data),
                rtol=1.0e-9, atol=1.0e-9,
                err_msg=f"{fld} moist global-mean must match single-rank",
            )
