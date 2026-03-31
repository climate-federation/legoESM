"""MPI correctness tests for lat-lon band decomposition.

Run with::

    mpirun -np 2 python -m pytest tests/distributed/test_latlon_mpi.py -v
    mpirun -np 4 python -m pytest tests/distributed/test_latlon_mpi.py -v

Verifies:
- Halo exchange produces correct boundary values
- Scattered state round-trips through gather
- N-rank step matches single-rank reference (within fp roundoff)
- Mass conservation under MPI
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
    FVLatLonPrimitiveEquationModel,
    FVLatLonPrimitiveEquationConfig,
)
from legoesm.parallel.latlon_mpi import (
    make_latlon_band_layout,
    build_local_grid,
    exchange_latlon_halos,
    scatter_latlon,
    gather_latlon,
    scatter_state_latlon,
    make_latlon_mpi_step,
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


# Small grid for fast tests
N_LAT = 32
N_LON = 64
NLEV = 5
DT = 300.0


@pytest.fixture
def grid():
    return create_latlon_grid(N_LAT, N_LON)


@pytest.fixture
def sigma():
    return create_sigma_coordinate(NLEV)


@pytest.fixture
def config():
    return FVLatLonPrimitiveEquationConfig(
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
        use_polar_filter=False,
    )


def _make_test_state(grid, nlev):
    """Create a simple test state with smooth fields."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    lat2d = grid.lat2d
    lon2d = grid.lon2d

    u = jnp.cos(lat2d)[:, :, None] * jnp.ones((1, 1, nlev)) * 10.0
    v = jnp.sin(2 * lon2d)[:, :, None] * jnp.ones((1, 1, nlev)) * 5.0
    T = 280.0 + 30.0 * jnp.cos(lat2d)[:, :, None] * jnp.ones((1, 1, nlev))
    p_s = 1e5 + 500.0 * jnp.cos(lat2d)
    phis = jnp.zeros_like(p_s)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    return HydrostaticState(
        u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


class TestHaloExchange:
    """Verify halo exchange correctness."""

    def test_scalar_halo_matches_neighbors(self, grid):
        """Exchanged halo rows should match neighbor's boundary rows."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        layout = make_latlon_band_layout(rank, n_ranks, N_LAT, N_LON, halo=2)

        # Global field with distinct values per row
        global_T = (jnp.arange(N_LAT)[:, None] * jnp.ones((1, N_LON))).astype(jnp.float64)
        local_T = scatter_latlon(global_T, layout)

        # Exchange halos
        padded = exchange_latlon_halos(local_T, layout, sign_flip=False)

        # Verify south halo
        if layout.south_rank >= 0:
            expected_south = global_T[layout.lat_start - 2:layout.lat_start]
            np.testing.assert_allclose(
                padded[:2], expected_south, atol=1e-14,
                err_msg=f"Rank {rank}: south halo mismatch")

        # Verify north halo
        if layout.north_rank >= 0:
            expected_north = global_T[layout.lat_end:layout.lat_end + 2]
            np.testing.assert_allclose(
                padded[-2:], expected_north, atol=1e-14,
                err_msg=f"Rank {rank}: north halo mismatch")

        # Verify owned portion unchanged
        np.testing.assert_allclose(
            padded[2:-2], local_T, atol=0.0,
            err_msg=f"Rank {rank}: owned data changed")


class TestScatterGather:
    """Verify scatter/gather round-trip."""

    def test_roundtrip(self, grid):
        """scatter -> gather should recover the global field."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        layout = make_latlon_band_layout(rank, n_ranks, N_LAT, N_LON, halo=2)

        global_field = jnp.arange(N_LAT * N_LON, dtype=jnp.float64).reshape(N_LAT, N_LON)
        local_field = scatter_latlon(global_field, layout)
        recovered = gather_latlon(local_field, layout)

        if rank == 0:
            np.testing.assert_allclose(
                recovered, global_field, atol=0.0,
                err_msg="Scatter/gather round-trip failed")


class TestMPIStep:
    """Verify MPI step matches serial reference."""

    def test_step_matches_serial(self, grid, sigma, config):
        """N-rank MPI step should match single-rank serial step."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        layout = make_latlon_band_layout(rank, n_ranks, N_LAT, N_LON, halo=2)

        # Build global state
        global_state = _make_test_state(grid, NLEV)

        # --- Serial reference (all ranks compute for comparison) ---
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config)
        serial_state = model.step(global_state, DT)

        # --- MPI step ---
        local_state = scatter_state_latlon(global_state, layout)
        mpi_step = make_latlon_mpi_step(model, grid, layout, sigma, config)
        local_result = mpi_step(local_state, DT)

        # Gather MPI result
        from legoesm.parallel.latlon_mpi import gather_latlon
        u_global = gather_latlon(local_result.u.data, layout)
        T_global = gather_latlon(local_result.T.data, layout)
        ps_global = gather_latlon(local_result.p_s.data, layout)

        if rank == 0:
            # Allow some tolerance for associative reduction differences
            np.testing.assert_allclose(
                u_global, serial_state.u.data, rtol=1e-10, atol=1e-10,
                err_msg="MPI u mismatch vs serial")
            np.testing.assert_allclose(
                T_global, serial_state.T.data, rtol=1e-10, atol=1e-10,
                err_msg="MPI T mismatch vs serial")
            np.testing.assert_allclose(
                ps_global, serial_state.p_s.data, rtol=1e-10, atol=1e-10,
                err_msg="MPI p_s mismatch vs serial")


class TestMassConservation:
    """Verify global mass is conserved under MPI."""

    def test_mass_conserved(self, grid, sigma, config):
        """Global dry mass should be conserved after MPI stepping."""
        _skip_if_no_mpi()
        rank, n_ranks = _get_mpi_info()

        layout = make_latlon_band_layout(rank, n_ranks, N_LAT, N_LON, halo=2)

        global_state = _make_test_state(grid, NLEV)
        local_state = scatter_state_latlon(global_state, layout)

        model = FVLatLonPrimitiveEquationModel(grid, sigma, config)
        mpi_step = make_latlon_mpi_step(model, grid, layout, sigma, config)

        # Step 10 times
        state = local_state
        for _ in range(10):
            state = mpi_step(state, DT)

        # Compute local mass contribution (owned cells only)
        owned_area = grid.area[layout.lat_start:layout.lat_end]
        local_mass = jnp.sum(state.p_s.data * owned_area)

        # Global mass via MPI reduction
        from legoesm.parallel.reductions import global_sum_mpi
        global_mass = global_sum_mpi(local_mass)

        # Initial global mass
        initial_mass = jnp.sum(global_state.p_s.data * grid.area)

        if rank == 0:
            np.testing.assert_allclose(
                float(global_mass), float(initial_mass), rtol=1e-12,
                err_msg="Mass not conserved under MPI")
