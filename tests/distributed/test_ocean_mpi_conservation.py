"""Long-run MPI ocean conservation regression.

Run with:
    mpirun -np 3 python -m pytest tests/distributed/test_ocean_mpi_conservation.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Guard: skip all tests if mpi4jax/mpi4py are not installed.
mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import set_halo_backend
from legoesm.ocean.conservation import ocean_conservation_fixer
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.distributed import gather_state, partition_state
from legoesm.parallel.reductions import global_sum_mpi


@pytest.fixture(autouse=True)
def reset_halo_backend():
    """Reset halo backend to local after each test."""
    yield
    set_halo_backend("local")


@pytest.fixture
def topology():
    """Build topology for this MPI rank."""
    rank = MPI.COMM_WORLD.Get_rank()
    n_processes = MPI.COMM_WORLD.Get_size()
    return build_comm_topology(rank, n_processes)


def _global_ocean_invariants(state, grid, z_coord, min_water_column_m: float) -> tuple[float, float, float, float]:
    """Return (ocean_area, eta_integral, heat_integral, salt_integral) globally."""
    mask = state.land_mask.data
    h_k = compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        z_coord,
        min_water_column_m=min_water_column_m,
    )
    weighted_area = mask * grid.area

    local_terms = jnp.stack(
        [
            jnp.sum(weighted_area),
            jnp.sum(state.eta.data * weighted_area),
            jnp.sum(jnp.sum(state.T.data * h_k, axis=-1) * weighted_area),
            jnp.sum(jnp.sum(state.S.data * h_k, axis=-1) * weighted_area),
        ],
    )
    global_terms = global_sum_mpi(local_terms)
    return tuple(float(x) for x in global_terms)


class TestMPIOceanConservationLongrun:
    """Nightly long-run MPI conservation regression."""

    def test_longrun_mpi_ocean_conservation(self, topology):
        """Conservation fixers should stay stable over many MPI cycles."""
        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)

        state0 = rest_state_ocean(
            grid,
            z_coord,
            T_surface=20.0,
            T_deep=2.0,
            S_uniform=35.0,
            H_max=4000.0,
            land_lat_threshold=70.0,
        )

        mask = state0.land_mask.data
        mask_3d = mask[..., jnp.newaxis]
        # Deterministic perturbation to excite non-trivial dynamics.
        eta_pert = (
            state0.eta.data
            + 0.05 * mask * jnp.sin(3.0 * grid.lon) * jnp.cos(2.0 * grid.lat)
        )
        u_pert = state0.u.data + 0.02 * mask_3d * jnp.sin(grid.lat)[..., jnp.newaxis]
        v_pert = state0.v.data + 0.015 * mask_3d * jnp.cos(grid.lon)[..., jnp.newaxis]

        state0 = state0._replace(
            eta=state0.eta.replace(data=eta_pert),
            u=state0.u.replace(data=u_pert),
            v=state0.v.replace(data=v_pert),
        )

        config = OceanConfig(
            use_conservation_fixer=True,
            fix_volume=True,
            fix_heat=True,
            fix_salt=True,
            min_water_column_m=0.5,
        )

        set_halo_backend("mpi", topology)
        state = partition_state(state0, topology)

        ocean_area0, eta_int0, heat0, salt0 = _global_ocean_invariants(
            state,
            grid,
            z_coord,
            config.min_water_column_m,
        )

        mask = state.land_mask.data
        mask_3d = mask[..., jnp.newaxis]
        z_phase = (z_coord.z_full_ref / 500.0)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

        n_steps = 120
        for _ in range(n_steps):
            phase = float(_ + 1)
            eta_drift = 0.01 * mask * (
                jnp.sin(grid.lon + 0.1 * phase) * jnp.cos(2.0 * grid.lat)
            )
            T_drift = 0.05 * mask_3d * jnp.sin(z_phase + 0.03 * phase)
            S_drift = -0.02 * mask_3d * jnp.cos(z_phase + 0.05 * phase)

            state_drifted = state._replace(
                eta=state.eta.replace(data=state.eta.data + eta_drift),
                T=state.T.replace(data=state.T.data + T_drift),
                S=state.S.replace(data=state.S.data + S_drift),
            )
            state = ocean_conservation_fixer(
                state_drifted,
                state,
                grid,
                z_coord,
                config,
            )

        ocean_area1, eta_int1, heat1, salt1 = _global_ocean_invariants(
            state,
            grid,
            z_coord,
            config.min_water_column_m,
        )

        # Area should remain exactly unchanged under static geometry/mask.
        area_rel = abs(ocean_area1 - ocean_area0) / max(abs(ocean_area0), 1.0)
        mean_eta_drift = abs(eta_int1 - eta_int0) / max(abs(ocean_area0), 1.0)
        heat_rel_drift = abs(heat1 - heat0) / max(abs(heat0), 1.0)
        salt_rel_drift = abs(salt1 - salt0) / max(abs(salt0), 1.0)

        # Collective gather validates full-state reconstruction after long run.
        state_global = gather_state(state, topology)

        if topology.rank == 0:
            assert area_rel < 1.0e-12
            assert mean_eta_drift < 5.0e-4
            assert heat_rel_drift < 5.0e-4
            assert salt_rel_drift < 5.0e-4
            assert jnp.all(jnp.isfinite(state_global.u.data))
            assert jnp.all(jnp.isfinite(state_global.T.data))
