"""FULL-MODEL tripole (ORCA-fold) MPI-vs-serial step parity (audit item 4).

The operator-level fold-aware band-MPI machinery was already validated
(``tests/distributed/test_latlon_mpi_tripole.py``); what was missing — and
what ``bench_ocean_mpi_scaling --tripole`` refused over — is a validated
FULL ``LatLonCGridOceanModel.step`` on a tripole grid under band MPI:

    synthetic tripole -> serial step (reference, every rank, pre-arming)
    same grid        -> fold-aware band layout -> scatter -> band step
                        -> gather -> compare T/S/eta/u/v on rank 0

The north band exercises the REAL fold branches (``fold_is_local`` True
there under MPI), interior cuts the fold-aware exchange.

Run:
    mpirun -np 1 python -m pytest tests/ocean/distributed/test_ocean_mpi_tripole_step_parity.py -v
    mpirun -np 2 python -m pytest tests/ocean/distributed/test_ocean_mpi_tripole_step_parity.py -v
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy


@pytest.fixture(autouse=True, scope="module")
def _fp64_policy():
    # JAX_ENABLE_X64 alone leaves the state at the policy default (f32);
    # force fp64 for the tight parity floor and RESTORE afterwards.
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


N_LAT, N_LON, N_LEV = 16, 24, 4
DT = 300.0
N_STEPS = 2
# Same-extent band ops with exchanged rows: the fold-aware exchange
# injects exact values, so the floor is the re-association of the
# split-explicit subcycle at f64 (measured regular-grid analogue ~1e-10;
# a real fold/cut defect is O(1e-3+) at the seam).
RTOL = 1e-8
ATOL = 1e-8
PARITY_FIELDS = ("T", "S", "eta", "u", "v")

_MIN_ROWS_PER_RANK = 4
_n_procs = MPI.COMM_WORLD.Get_size()
if _n_procs > 1 and (N_LAT // _n_procs) < _MIN_ROWS_PER_RANK:
    pytest.skip(
        f"MPI size {_n_procs} gives {N_LAT // _n_procs} lat rows/rank "
        f"(need >={_MIN_ROWS_PER_RANK}).",
        allow_module_level=True,
    )

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _reset_halo_backend():
    yield
    set_halo_backend("local")


def _perturbed_state(grid, z_coord):
    """Mask-aware perturbation exercising every fold-touching term:
    nonzero meridional transport up to the cap + zonal-wavenumber eta
    (a fold-antisymmetric field crosses the seam with real signal)."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=2000.0)
    rng = np.random.default_rng(4)
    u = 0.02 * rng.standard_normal(state.u.data.shape)
    v = 0.02 * rng.standard_normal(state.v.data.shape)
    eta = 0.01 * rng.standard_normal(state.eta.data.shape)
    T = 0.3 * rng.standard_normal(state.T.data.shape)
    u = u * np.asarray(state.u_mask.data)[..., None]
    v = v * np.asarray(state.v_mask.data)[..., None]
    eta = eta * np.asarray(state.land_mask.data)
    T = state.T.data + T * np.asarray(state.land_mask.data)[..., None]
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


class TestOceanMPITripoleStepParity:

    def test_full_model_step_parity_gathered_vs_serial(self):
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()

        grid = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        assert grid.fold.is_active
        z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=2000.0)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="explicit_substep",
        )
        state_global = _perturbed_state(grid, z_coord)

        # Serial reference on EVERY rank, BEFORE arming MPI (deadlock
        # discipline of the band-MPI parity twins).
        serial_model = LatLonCGridOceanModel(grid, z_coord, config)
        ref = state_global
        for _ in range(N_STEPS):
            ref = serial_model.step(ref, DT)
        ref_np = {
            name: np.asarray(getattr(ref, name).data)
            for name in PARITY_FIELDS
        }

        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            gather_state_latlon_cgrid_ocean,
            scatter_state_latlon_cgrid_ocean,
            slice_cgrid_geometry_to_band,
            slice_zcoord_to_band,
        )

        layout = initialize_distributed_latlon(
            global_n_lat=N_LAT, global_n_lon=N_LON, fold=grid.fold,
        )
        band_geom = slice_cgrid_geometry_to_band(grid, layout)
        z_band = slice_zcoord_to_band(z_coord, layout)
        band_model = LatLonCGridOceanModel(band_geom, z_band, config)

        local = scatter_state_latlon_cgrid_ocean(state_global, layout)
        for _ in range(N_STEPS):
            local = band_model.step(local, DT)
        jax.block_until_ready(jax.tree.leaves(local))

        gathered = gather_state_latlon_cgrid_ocean(local, layout)

        if rank != 0:
            assert gathered is None
            return

        assert gathered is not None
        for name in PARITY_FIELDS:
            got = np.asarray(getattr(gathered, name).data)
            np.testing.assert_allclose(
                got, ref_np[name], rtol=RTOL, atol=ATOL,
                err_msg=(
                    f"np={n_ranks} tripole gathered '{name}' diverged from "
                    f"the serial reference after {N_STEPS} step(s): a fold "
                    f"seam / partition-cut defect."
                ),
            )
