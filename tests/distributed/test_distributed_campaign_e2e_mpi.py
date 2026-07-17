"""End-to-end distributed-MPAS correction campaign (iter 89).

Run: ``mpirun -np 2 .venv/bin/python -m pytest <this file>``

The capstone of iters 86–89: ``build_distributed_correction_campaign`` composes the
three distributed hooks (owned ``valid_mask`` + global top-k ``manifest_reducer`` +
collective ``global_reduce``) from a REAL Voronoi partition layout and runs the FULL
campaign across ranks.  A global mesh (``create_voronoi_mesh(2)``, 162 cells) is
partitioned with ``make_voronoi_partition_layout``; one global cell is cold-biased
in the GLOBAL ERA5 reference, so it is the single global-worst column.  Verifies:
the campaign COMPLETES on every rank (no deadlock); the globally-worst cell is
diagnosed + corrected on EXACTLY its owning rank (no double-count); every other
rank corrects nothing yet still runs in lockstep; and the bias verdict is identical
across ranks.  np=1 degenerates to a single owner of every cell (the real local
mesh + LES path), so it also runs serially.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")
from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.grids.voronoi import create_voronoi_mesh  # noqa: E402
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402
from legoesm.parallel.voronoi_mpi import make_voronoi_partition_layout  # noqa: E402
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.distributed_campaign import (  # noqa: E402
    assemble_global_campaign_result,
    assemble_global_field,
)
from mpi4py import MPI  # noqa: E402

from scripts.run.run_correction_campaign import (  # noqa: E402
    build_distributed_correction_campaign,
    build_distributed_mpas_campaign,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_REGIME = LESRegimeConfig(shallow=_RES, deep=_RES)


def _mock_run_les_sheared(setup):
    """A mock plane-LES with synthetic w/θ'/tracers + a mean-wind shear → a VALID
    clubb_coefficient diagnosis (mirrors the single-process MPAS capstone mock)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import make_rest_state
    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    z = jnp.asarray(hc.z_full)
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))   # constant shear
    return state._replace(
        u=state.u.replace(data=u),
        w=state.w.replace(data=w),
        theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr))


def _mpas_base_config(nlev):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic", discretization="mpas"),
        radiation="gray", turbulence="clubb_lite")


def _local_model_state(local_mesh, nlev):
    """A rank-local MPAS comparison ColumnState (uniform; the bias lives in the
    reference) carrying the native edge velocity for the Voronoi forcing extract."""
    ncell = local_mesh.nCells
    u_edge = 6.0 * np.cos(np.asarray(local_mesh.angleEdge))[:, None] * np.ones((1, nlev))
    return ColumnState(
        T=jnp.full((ncell, nlev), 285.0), q_v=jnp.full((ncell, nlev), 6e-3),
        u=jnp.zeros((ncell, nlev)), v=jnp.zeros((ncell, nlev)),
        p_s=jnp.full((ncell,), 1.0e5), sst_K=jnp.full((ncell,), 290.0),
        u_edge=jnp.asarray(u_edge))


def _global_reference(global_mesh, nlev, bias_cell, bias_dt):
    temp = np.full((global_mesh.nCells, nlev), 285.0)
    temp[bias_cell] += bias_dt                       # one cold-biased global cell
    ncell = global_mesh.nCells
    return ColumnState(
        T=jnp.asarray(temp), q_v=jnp.full((ncell, nlev), 6e-3),
        u=jnp.zeros((ncell, nlev)), v=jnp.zeros((ncell, nlev)),
        p_s=jnp.full((ncell,), 1.0e5), sst_K=jnp.full((ncell,), 290.0))


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):       # noqa: ARG002
        segment_callback(self, 0.0, 1.0)


@pytest.mark.timeout(300)
def test_distributed_campaign_corrects_global_worst_on_its_owner():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    nlev = 5
    bias_cell = 100                                  # the single global-worst cell
    bg = float(CLUBBLiteConfig().C_K)
    global_mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(nlev)
    layout = make_voronoi_partition_layout(global_mesh, RANK, NPROC)
    local_mesh = layout.local_mesh
    local_cells = np.asarray(layout.partition.local_cells)
    n_owned = int(layout.partition.n_owned_cells)
    owns_bias = bias_cell in local_cells[:n_owned]   # is THIS rank the owner?

    model_state = _local_model_state(local_mesh, nlev)
    reference = _global_reference(global_mesh, nlev, bias_cell, bias_dt=-12.0)

    def build_base(cfg):
        return _FakeDriver(model_state)

    def extract(driver, day, dt):                    # noqa: ARG001
        return driver.state

    result = build_distributed_correction_campaign(
        layout=layout, reference=reference,
        area_weights=jnp.asarray(global_mesh.grid_area), n_worst=1,
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, sigma=sigma, grid=local_mesh, n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, accept_only_if_improved=False)

    # Reaching here on every rank ⇒ no deadlock in the collective loop.
    it = result.iterations[0]
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (local_mesh.nCells,)

    if owns_bias:
        assert it.n_diagnosed == 1                   # the owner spun off the LES
        assert it.n_diagnoses_valid == 1             # ...and the sheared LES was VALID
        li = int(np.where(local_cells == bias_cell)[0][0])
        assert not np.isclose(ck[li], bg), "owner did not correct the global-worst cell"
    else:
        assert it.n_diagnosed == 0                   # owns none of the global top-1
        assert it.n_diagnoses_valid == 0
        np.testing.assert_allclose(ck, bg)           # nothing corrected, but ran lockstep

    # Every rank must agree on the GLOBAL improvement verdict (collective bias).
    verdicts = COMM.allgather(bool(it.bias.improved))
    assert all(v == verdicts[0] for v in verdicts), f"ranks disagree: {verdicts}"
    # Exactly ONE rank owns + corrects the global-worst cell (no double-count).
    n_owners = COMM.allreduce(1 if owns_bias else 0, op=MPI.SUM)
    assert n_owners == 1

    # PERSIST step (iter 253): assemble the rank-local C_K into the GLOBAL field under
    # the REAL allreduce. The global-worst cell is corrected at its GLOBAL id (on
    # whichever rank owns it); every other global cell keeps the background bg. This
    # proves assemble_global_field is the correct forward-inverse of the reference
    # slice under the actual collective, not just single-process.
    global_ck = assemble_global_field(layout, ck, global_reduce=global_sum_mpi)
    assert global_ck.shape == (global_mesh.nCells,)
    assert not np.isclose(global_ck[bias_cell], bg), "global-worst cell not corrected"
    np.testing.assert_allclose(np.delete(global_ck, bias_cell), bg)  # all others bg

    # RESULT-LEVEL persist (iter 256): the same global field, but reached through
    # assemble_global_campaign_result on the REAL CampaignResult NamedTuple — proving
    # _replace works on the actual type and the global result feeds the EXISTING
    # build_campaign_output_dict (the deployable on-disk JSON) unchanged.
    gres = assemble_global_campaign_result(
        result, layout, corrected_field="C_K", global_reduce=global_sum_mpi)
    np.testing.assert_allclose(np.asarray(gres.final_field), global_ck)
    np.testing.assert_allclose(np.asarray(gres.final_config.C_K), global_ck)
    if RANK == 0:                                    # the rank-0-only persist an MPI driver does
        from legoesm.training.campaign_summary import campaign_health, summarize_campaign

        from scripts.run.run_correction_campaign import build_campaign_output_dict
        summary = summarize_campaign(gres, promotion_key="clubb_lite_C_K")
        out = build_campaign_output_dict(
            gres, grid_provenance={"grid_type": "mpas", "ncol": int(global_mesh.nCells)},
            summary=summary, health=campaign_health(summary), corrected_field="C_K")
        assert len(out["C_K"]) == global_mesh.nCells  # GLOBAL field in the deployable JSON
    COMM.Barrier()


@pytest.mark.timeout(300)
def test_build_distributed_mpas_campaign_entry_point():
    """The one-call entry point (iter 96) partitions the GLOBAL mesh internally and
    wires the rank-local mesh as the grid + the driver builder — same end-to-end
    result as the manual layout setup above (the global-worst cell corrected on its
    owner), but the HPC user supplies only the global mesh + a
    build_local_driver(cfg, local_mesh)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    nlev = 5
    bias_cell = 100
    bg = float(CLUBBLiteConfig().C_K)
    global_mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(nlev)
    reference = _global_reference(global_mesh, nlev, bias_cell, bias_dt=-12.0)

    # the entry point passes layout.local_mesh here — building the model state on it
    # PROVES the rank-local (not global) mesh is wired as the grid.
    seen = {}

    def build_local_driver(cfg, local_mesh):         # noqa: ARG001
        # record the FULL per-cell latitude (not just nCells) so the mesh-identity
        # check below cannot pass on a coincidentally same-size WRONG mesh.
        seen["latCell"] = np.asarray(local_mesh.latCell)
        return _FakeDriver(_local_model_state(local_mesh, nlev))

    def extract(driver, day, dt):                    # noqa: ARG001
        return driver.state

    # NOTE: rank/n_ranks OMITTED on purpose → exercises the advertised MPI.COMM_WORLD
    # default path (the deferred mpi4py import) instead of explicit args.
    # return_layout=True (iter 263) hands back the partition the entry point built
    # INTERNALLY, so a turnkey driver can feed it straight into the persist step
    # (assemble_global_campaign_result) without rebuilding make_voronoi_partition_layout.
    result, layout = build_distributed_mpas_campaign(
        global_mesh=global_mesh, reference=reference,
        area_weights=jnp.asarray(global_mesh.grid_area), n_worst=1,
        build_local_driver=build_local_driver,
        base_atm_config=_mpas_base_config(nlev), extract_column_state=extract,
        sigma=sigma, n_iterations=1, return_layout=True,
        les_config=ColumnLESConfig(
            regime=_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, accept_only_if_improved=False)

    # the returned layout IS the deterministic partition (identical to a manual build).
    np.testing.assert_array_equal(
        np.asarray(layout.partition.local_cells),
        np.asarray(make_voronoi_partition_layout(global_mesh, RANK, NPROC).partition.local_cells))
    local_cells = np.asarray(layout.partition.local_cells)
    n_owned = int(layout.partition.n_owned_cells)
    owns_bias = bias_cell in local_cells[:n_owned]
    # the LOCAL mesh (exact per-cell lats) was wired — not the global mesh.
    np.testing.assert_array_equal(seen["latCell"], np.asarray(layout.local_mesh.latCell))
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (layout.local_mesh.nCells,)
    it = result.iterations[0]
    if owns_bias:
        assert it.n_diagnosed == 1
        li = int(np.where(local_cells == bias_cell)[0][0])
        assert not np.isclose(ck[li], bg)
    else:
        assert it.n_diagnosed == 0
        np.testing.assert_allclose(ck, bg)
    # Every rank agrees on the GLOBAL improvement verdict (collective bias) — the
    # entry point's loop is lockstep, same as the manual capstone.
    verdicts = COMM.allgather(bool(it.bias.improved))
    assert all(v == verdicts[0] for v in verdicts), f"ranks disagree: {verdicts}"
    # With np>=2 this proves no double-count; np=1 degenerates to the serial smoke
    # path (single owner of every cell), matching the sibling test's contract.
    n_owners = COMM.allreduce(1 if owns_bias else 0, op=MPI.SUM)
    assert n_owners == 1
    COMM.Barrier()


@pytest.mark.timeout(120)
def test_assert_partition_covers_global_on_a_real_partition():
    """The iter-98 pre-flight passes on a REAL multi-rank Voronoi partition (the owned
    sets tile the global mesh exactly once) and FAILS LOUDLY — on EVERY rank, no
    deadlock — when a cell is dropped from its owner so it becomes a global gap."""
    from types import SimpleNamespace

    from legoesm.training.distributed_campaign import assert_partition_covers_global

    global_mesh = create_voronoi_mesh(2)
    layout = make_voronoi_partition_layout(global_mesh, RANK, NPROC)
    # the real partition is clean — collective check returns on all ranks.
    assert_partition_covers_global(layout) is None
    COMM.Barrier()

    # Corrupt it: rank 0 disowns its FIRST owned cell (a partition is disjoint, so
    # that global cell is then owned by NO rank). Every rank runs the same collective
    # allreduce and sees the gap ⇒ every rank raises (no rank-divergent control flow).
    owned = np.asarray(layout.owned_mask_cells).copy()
    if RANK == 0:
        first_owned = int(np.nonzero(owned)[0][0])
        owned[first_owned] = False
    corrupt = SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned), partition=layout.partition)
    with pytest.raises(ValueError, match="does NOT cover the global mesh"):
        assert_partition_covers_global(corrupt)
    COMM.Barrier()


@pytest.mark.timeout(120)
def test_assert_partition_covers_global_synchronizes_malformed_layout():
    """A malformation on ONE rank only (here rank 0's owned mask is the wrong length)
    is SYNCHRONIZED through the collective so EVERY rank raises — the structural check
    must not raise pre-allreduce and hang the healthy ranks inside it (Codex iter 98,
    HIGH). Both ranks reaching pytest.raises without timeout proves no deadlock."""
    if NPROC < 2:
        pytest.skip("cross-rank malformed-layout sync needs >=2 ranks (vacuous at np=1)")
    from types import SimpleNamespace

    from legoesm.training.distributed_campaign import assert_partition_covers_global

    global_mesh = create_voronoi_mesh(2)
    layout = make_voronoi_partition_layout(global_mesh, RANK, NPROC)
    owned = np.asarray(layout.owned_mask_cells)
    if RANK == 0:
        owned = owned[:-1]                           # wrong length on rank 0 ONLY
    bad = SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned), partition=layout.partition)
    with pytest.raises(ValueError, match="same length"):
        assert_partition_covers_global(bad)
    COMM.Barrier()
