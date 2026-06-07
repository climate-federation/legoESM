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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
    MPASPrimitiveEquationModel,
    MPASPrimitiveEquationConfig,
)
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
from legoesm.parallel.voronoi_mpi import (
    make_voronoi_partition_layout,
    scatter_state_voronoi,
    gather_state_voronoi,
    make_voronoi_mpi_step,
    _fix_mass_mpi,
)


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
    run under BOTH.  (Both currently fail on the out-of-range mpi4jax-0.9 /
    jax-0.10 stack — see the mpi-stack-version memory — so this guard turns
    green only once that env is repaired; it is the committed pin for the
    shipped production path until then.)
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
            # through a global-mean correction.  rtol=1e-8 accommodates
            # the worst-case accumulation while still catching real bugs.
            np.testing.assert_allclose(
                mpi_global.T.data, serial_state.T.data,
                rtol=1e-8, atol=1e-10,
                err_msg="MPI T mismatch vs serial")
            np.testing.assert_allclose(
                mpi_global.u.data, serial_state.u.data,
                rtol=1e-8, atol=1e-10,
                err_msg="MPI u mismatch vs serial")
            np.testing.assert_allclose(
                mpi_global.p_s.data, serial_state.p_s.data,
                rtol=1e-8, atol=1e-10,
                err_msg="MPI p_s mismatch vs serial")


class TestMassConservation:
    """Verify global mass is conserved under MPI."""

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
            np.testing.assert_allclose(
                float(global_mass), float(initial_mass), rtol=1e-12,
                err_msg="Mass not conserved under MPI")
