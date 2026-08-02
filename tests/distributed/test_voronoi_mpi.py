"""MPI correctness tests for Voronoi (MPAS) domain decomposition.

Run with::

    mpirun -np 2 python -m pytest tests/distributed/test_voronoi_mpi.py -v
    mpirun -np 4 python -m pytest tests/distributed/test_voronoi_mpi.py -v

Verifies:
- Halo exchange correctness (cell and edge fields)
- Scattered state round-trips through gather
- N-rank MPI step matches single-rank reference
- Mass conservation with global MPI fixer
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    MPASPrimitiveEquationModel,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_mpi import (
    gather_state_voronoi,
    make_voronoi_mpi_step,
    make_voronoi_partition_layout,
    scatter_state_voronoi,
)

from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas


def _skip_if_no_mpi():
    """Skip test if not running under MPI."""
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() < 2:
            pytest.skip("Need at least 2 MPI ranks")
    except ImportError:
        pytest.skip("mpi4py not available")


def _get_mpi_info():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


# Serial-vs-MPI equivalence + mass conservation go through the GLOBAL ALLREDUCE
# (mass fixer / global_sum_mpi), which drifts ~1e-9 vs serial under a jax/mpi4jax
# stack outside legoESM's tested range (the point-to-point halo tests below stay
# bit-correct).  xfail those two on an incompatible stack so they are not spurious
# reds, while still REQUIRING a pass once a tested stack (the FFI generation:
# mpi4jax >= 0.9 paired with jax >= 0.10) is installed -- the condition flips off
# automatically then.  VERIFIED green on jax 0.10.1 + mpi4jax 0.9.0.post1
# (2026-07-13): these run as ordinary passes, not xfail/xpass.
from legoesm.parallel.reductions import mpi_stack_outside_tested_range

_xfail_mpi_stack = pytest.mark.xfail(
    mpi_stack_outside_tested_range(),
    reason="jax/mpi4jax outside legoESM's tested MPI range: the global-allreduce "
           "path drifts ~1e-9 vs serial under the incompatible custom-call ABI "
           "(halo exchange stays bit-correct). Install mpi4jax >= 0.9 (the FFI rewrite).",
    strict=False,
    run=True,
)


# Small mesh for fast tests
SUBDIVISION_LEVEL = 2  # 162 cells
NLEV = 5
DT = 600.0


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=SUBDIVISION_LEVEL, lloyd_iterations=5)


@pytest.fixture
def sigma():
    return create_sigma_coordinate(NLEV)


@pytest.fixture(params=["ssp_rk3", "ssp_rk54_scan"])
def config(request):
    """MPAS PE config, parametrized over the integrators that ship under MPI.

    ``ssp_rk54_scan`` is the production library default (what direct MPAS
    construction selects), and it evaluates the TRiSK tendency — including the
    mpi4jax ``sendrecv`` halo exchange — INSIDE ``lax.scan``.  That ordered-
    effect-through-scan path is structurally different from the inline
    ``ssp_rk3``, so the serial-vs-MPI equivalence and mass-conservation tests
    run under BOTH.  (Both pass on the verified FFI stack jax 0.10.x +
    mpi4jax 0.9.x; the ``_xfail_mpi_stack`` guard above only fires on a
    genuinely-incompatible CROSS pairing, not on this shipped production path.)
    """
    return MPASPrimitiveEquationConfig(
        nu_del4=0.0,
        nu_del4_ps=0.0,
        fix_mass=True,
        time_integrator=request.param,
    )


class TestHaloExchange:
    """Verify MPI halo exchange vs simulated exchange."""

    def test_cell_halo_matches_global(self, mesh):
        """Exchanged cell halo values should match global field."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import partition_cells_geometric
        cell_owner = partition_cells_geometric(mesh, n_ranks)

        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        # Global field: cell index as value
        global_field = jnp.arange(mesh.nCells, dtype=jnp.float64)

        # Scatter to local
        from legoesm.parallel.voronoi_partition import scatter_to_local
        local_field = scatter_to_local(global_field, layout.partition, "cell")

        # Exchange halos
        exchanged = layout.halo_exchange.exchange_cell_field(local_field)

        # Verify: all local values (owned + halo) should match global
        expected = global_field[layout.partition.local_cells]
        np.testing.assert_allclose(
            exchanged, expected, atol=1e-14,
            err_msg=f"Rank {rank}: cell halo mismatch")

    def test_edge_halo_matches_global(self, mesh):
        """Exchanged edge halo values should match global field."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import partition_cells_geometric
        cell_owner = partition_cells_geometric(mesh, n_ranks)

        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        global_field = jnp.arange(mesh.nEdges, dtype=jnp.float64)
        from legoesm.parallel.voronoi_partition import scatter_to_local
        local_field = scatter_to_local(global_field, layout.partition, "edge")

        exchanged = layout.halo_exchange.exchange_edge_field(local_field)
        expected = global_field[layout.partition.local_edges]

        np.testing.assert_allclose(
            exchanged, expected, atol=1e-14,
            err_msg=f"Rank {rank}: edge halo mismatch")

    def test_repeated_same_entity_exchange_no_cross_match(self, mesh):
        """Two cell exchanges between the same pair must not cross-match.

        The rank-free halo tags (cell == tag 0 for BOTH calls) rely on order,
        not the tag, to separate two same-entity exchanges between one pair:
        SPMD program order + mpi4jax's ordered effect + MPI non-overtaking.
        This pins that the production MPAS step's double cell exchange (T+p_s,
        then tracers) cannot silently receive the wrong buffer.  The two fields
        share the SAME shape, so a tag cross-match would be SILENT (no size
        error) -- only the value check below would catch it.
        """
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import (
            partition_cells_geometric,
            scatter_to_local,
        )
        cell_owner = partition_cells_geometric(mesh, n_ranks)
        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        # Two DISTINCT global cell fields, identical shape (disjoint value
        # ranges so a cross-match cannot accidentally look correct).
        field_a = jnp.arange(mesh.nCells, dtype=jnp.float64)
        field_b = -(jnp.arange(mesh.nCells, dtype=jnp.float64) + 1.0)
        a_local = scatter_to_local(field_a, layout.partition, "cell")
        b_local = scatter_to_local(field_b, layout.partition, "cell")

        # BOTH exchanges in ONE compiled function, same order on every rank --
        # the composed-step case the rank-free tag scheme depends on.
        @jax.jit
        def _two_cell_exchanges(a, b):
            a_ex = layout.halo_exchange.exchange_cell_field(a)
            b_ex = layout.halo_exchange.exchange_cell_field(b)
            return a_ex, b_ex

        a_ex, b_ex = _two_cell_exchanges(a_local, b_local)

        np.testing.assert_allclose(
            a_ex, field_a[layout.partition.local_cells], atol=1e-14,
            err_msg=f"Rank {rank}: first cell exchange cross-matched")
        np.testing.assert_allclose(
            b_ex, field_b[layout.partition.local_cells], atol=1e-14,
            err_msg=f"Rank {rank}: second cell exchange cross-matched")


class TestScatterGather:
    """Verify scatter/gather round-trip."""

    def test_cell_roundtrip(self, mesh):
        """scatter -> gather should recover the global cell field."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import partition_cells_geometric
        cell_owner = partition_cells_geometric(mesh, n_ranks)

        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        global_field = jnp.sin(mesh.latCell) * 1000.0
        from legoesm.parallel.voronoi_partition import scatter_to_local
        local_field = scatter_to_local(global_field, layout.partition, "cell")

        from legoesm.parallel.voronoi_mpi import gather_voronoi_field
        recovered = gather_voronoi_field(local_field, layout.partition, "cell")

        if rank == 0:
            np.testing.assert_allclose(
                recovered, global_field, atol=1e-12,
                err_msg="Cell scatter/gather round-trip failed")


class TestMPIStep:
    """Verify MPI step matches serial reference."""

    @_xfail_mpi_stack
    def test_step_matches_serial(self, mesh, sigma, config):
        """N-rank MPI step should match single-rank serial step."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import partition_cells_geometric
        cell_owner = partition_cells_geometric(mesh, n_ranks)

        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        # Build global state
        global_state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)

        # --- Serial reference (all ranks compute for comparison) ---
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        serial_state = model.step(global_state, DT)

        # --- MPI step ---
        local_state = scatter_state_voronoi(global_state, layout.partition)
        mpi_step = make_voronoi_mpi_step(model, layout, sigma, config)
        local_result = mpi_step(local_state, DT)

        # Gather MPI result
        mpi_global = gather_state_voronoi(local_result, layout.partition)

        if rank == 0:
            # MPI allreduce sums partial results in tree order (not
            # serial order), so FP non-associativity produces O(eps)
            # differences per reduction.  The mass fixer amplifies this
            # through a global-mean correction.  ``rtol=1e-8`` handles the
            # O(1)-magnitude elements; ``atol=1e-8`` is the matching
            # absolute FLOOR for near-zero elements — the velocity field
            # crosses zero (sign reversals across the jet), where a tiny
            # |desired| makes ``rtol*|desired|`` vanish and a single step's
            # reduction-order difference (measured ~3e-9 on ``u``, x64)
            # otherwise trips the default atol=1e-10.  1e-8 is still ~1e-8
            # of the field scale, so a REAL halo/stencil bug (O(1e-2)+) is
            # caught; this only absorbs FP non-associativity on cells whose
            # value happens to be ~0.  Consistent with the documented
            # ~1% MPI-vs-serial envelope in
            # ``test_voronoi_sharded_equivalence``.
            np.testing.assert_allclose(
                mpi_global.T.data, serial_state.T.data,
                rtol=1e-8, atol=1e-8,
                err_msg="MPI T mismatch vs serial")
            np.testing.assert_allclose(
                mpi_global.u.data, serial_state.u.data,
                rtol=1e-8, atol=1e-8,
                err_msg="MPI u mismatch vs serial")
            np.testing.assert_allclose(
                mpi_global.p_s.data, serial_state.p_s.data,
                rtol=1e-8, atol=1e-8,
                err_msg="MPI p_s mismatch vs serial")


class TestMassConservation:
    """Verify global mass is conserved under MPI."""

    @_xfail_mpi_stack
    def test_mass_conserved(self, mesh, sigma, config):
        """Global dry mass should be conserved after MPI stepping."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import partition_cells_geometric
        cell_owner = partition_cells_geometric(mesh, n_ranks)

        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        global_state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
        local_state = scatter_state_voronoi(global_state, layout.partition)

        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        mpi_step = make_voronoi_mpi_step(model, layout, sigma, config)

        # Step 10 times
        state = local_state
        for _ in range(10):
            state = mpi_step(state, DT)

        # Compute local mass contribution (owned cells only)
        owned_area = jnp.where(
            layout.owned_mask_cells,
            layout.local_mesh.areaCell,
            0.0,
        )
        local_mass = jnp.sum(state.p_s.data * owned_area)

        # Global mass via MPI reduction
        from legoesm.parallel.reductions import global_sum_mpi
        global_mass = global_sum_mpi(local_mass)

        # Initial global mass
        initial_mass = jnp.sum(global_state.p_s.data * mesh.areaCell)

        if rank == 0:
            # rtol floor: ``global_mass`` is a rank-partitioned
            # allreduce while ``initial_mass`` is a serial jnp.sum —
            # different summation ORDERS, so 1e-12 agreement is not
            # achievable.  Observed floor 7.5e-9 at total mass ~5e19
            # (ssp_rk3, np=2); pre-existing at HEAD, NOT a halo-batching
            # regression (bisect job 8456934, 2026-06-10).  3e-8 = 4x
            # margin; a real conservation bug sits orders above.
            np.testing.assert_allclose(
                float(global_mass), float(initial_mass), rtol=3e-8,
                err_msg="Mass not conserved under MPI")


class TestConservativeClampMPIStep:
    """Clamp-ON floors under MPI: the per-tracer collectives must execute
    with matching counts on every rank and reproduce the serial result.

    NOTE the iteration-order angle is guarded by the ``sorted(...)`` loop +
    its source pin in test_conservative_positive_clip.py: JAX canonicalizes
    dict-pytree keys at flatten time, so a rank-dependent INSERTION order
    cannot reach the traced loop anyway (codex 2026-07-28 round 3) — this
    test's value is the end-to-end clamp-ON MPI step (collectives, factors,
    serial equivalence).  Compares LOCAL owned cells against the scattered
    serial reference because ``gather_state_voronoi`` does not carry
    ``tracers`` (pre-existing gap, flagged)."""

    @_xfail_mpi_stack
    def test_clamp_on_step_matches_serial(self, mesh, sigma, config):
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import (
            partition_cells_geometric,
            scatter_to_local,
        )
        cell_owner = partition_cells_geometric(mesh, n_ranks)
        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        cfg = config._replace(conservative_tracer_clamp=True)
        global_state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
        ncell, nlev = global_state.T.data.shape
        rng = np.random.default_rng(9)
        tracers = {}
        for k in ("q_v", "q_i", "N_i"):
            data = jnp.asarray(rng.uniform(0.0, 1e-3, (ncell, nlev)))
            data = data.at[:, 1].add(-2e-4)   # guarantee clamp work
            tracers[k] = global_state.p_s.replace(data=data)
        global_state = global_state._replace(tracers=tracers)

        model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
        serial_state = model.step(global_state, DT)

        local_state = scatter_state_voronoi(global_state, layout.partition)
        mpi_step = make_voronoi_mpi_step(model, layout, sigma, cfg)
        local_result = mpi_step(local_state, DT)

        owned = np.asarray(layout.owned_mask_cells)
        for k in ("q_v", "q_i", "N_i"):
            serial_local = np.asarray(scatter_to_local(
                serial_state.tracers[k].data, layout.partition, "cell"))
            got = np.asarray(local_result.tracers[k].data)
            np.testing.assert_allclose(
                got[owned], serial_local[owned], rtol=1e-7, atol=1e-10,
                err_msg=f"clamp-ON MPI tracer {k} mismatch vs serial "
                        f"on rank {rank}")


class TestFluxFormTracerMPIStep:
    """#1354: the mass-CONSISTENT flux-form tracer transport under MPI.

    The claim being tested is that flux form needs NO new communication.  Its
    stencil footprint is identical to the advective operator it replaces —
    ``cell_to_edge_avg_3d(q)`` and ``divergence_cell_3d(u·q_e·δp_e)`` read the
    same one ring of neighbour CELLS (``q`` and ``p_s``, both already exchanged
    every RK stage by ``_exchange_mpas_state``) and the same edges of the owned
    cell.  ``div_dp`` and the half-level mass flux ``F`` are cell-local
    reductions of quantities the dycore already forms for its own continuity.
    If that claim were wrong, boundary-owned cells would read stale halo values
    and the owned-cell result would differ from serial — which is exactly what
    this asserts does not happen.

    Compares LOCAL owned cells against the scattered serial reference because
    ``gather_state_voronoi`` does not carry ``tracers`` (pre-existing gap,
    mirrored from ``TestConservativeClampMPIStep``).
    """

    @_xfail_mpi_stack
    def test_flux_form_step_matches_serial(self, mesh, sigma, config):
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        from legoesm.parallel.voronoi_partition import (
            partition_cells_geometric,
            scatter_to_local,
        )
        cell_owner = partition_cells_geometric(mesh, n_ranks)
        layout = make_voronoi_partition_layout(
            mesh, rank, n_ranks, cell_owner=cell_owner)

        cfg = config._replace(moisture_flux_form=True)
        global_state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
        ncell, nlev = global_state.T.data.shape
        rng = np.random.default_rng(11)
        # A per-MASS tracer (flux form) AND a per-VOLUME one (kept advective):
        # both lanes must reproduce serial under the same exchange.
        tracers = {}
        for k in ("q_v", "N_c"):
            scale = 1e-3 if k == "q_v" else 1e8
            tracers[k] = global_state.p_s.replace(
                data=jnp.asarray(rng.uniform(0.1, 1.0, (ncell, nlev)) * scale))
        global_state = global_state._replace(tracers=tracers)

        model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
        serial_state = model.step(global_state, DT)

        local_state = scatter_state_voronoi(global_state, layout.partition)
        mpi_step = make_voronoi_mpi_step(model, layout, sigma, cfg)
        local_result = mpi_step(local_state, DT)

        owned = np.asarray(layout.owned_mask_cells)
        for k in ("q_v", "N_c"):
            serial_local = np.asarray(scatter_to_local(
                serial_state.tracers[k].data, layout.partition, "cell"))
            got = np.asarray(local_result.tracers[k].data)
            np.testing.assert_allclose(
                got[owned], serial_local[owned], rtol=1e-7, atol=1e-12,
                err_msg=f"flux-form MPI tracer {k} mismatch vs serial "
                        f"on rank {rank} (stale halo in the new operator?)")
