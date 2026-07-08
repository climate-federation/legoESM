"""Gathered-field MPI-vs-serial parity for the WIDE-HALO barotropic path.

Twin of ``test_ocean_mpi_tvd_parity.py`` with
``barotropic_wide_halo=True``: the wide path's SINGLE fused wide exchange +
communication-free extended-band substeps must reproduce the serial
wide-halo step on the gathered fields.  What this catches that the serial
and SPMD gates cannot: the mpi4jax leg of the wide exchange
(``pad_with_pole_bc_lat_multi`` fused sendrecv at width W, the staggered
``v[:-1]``/``halo=W+1`` trick across a REAL rank cut, and the band-local
chunk auto-sizing, which differs per rank count).

Deadlock discipline (module docstring of the twin): the serial reference
runs on EVERY rank BEFORE ``initialize_distributed_latlon`` arms the MPI
halo backend.

Run:
    mpirun -np 1 python -m pytest tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py -v
    mpirun -np 2 python -m pytest tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py -v
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

# JAX_ENABLE_X64 alone is NOT enough: the ocean state dtype comes from the
# legoESM precision POLICY (its f32 default would leave the whole subcycle
# in float32 — 2026-06-29 full-suite-health gotcha).  Force fp64 for this
# module and RESTORE afterwards (a bare module-level set_policy would leak
# into every later-collected test in the process).
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy


@pytest.fixture(autouse=True, scope="module")
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)

N_LAT, N_LON, N_LEV = 16, 32, 4
DT = 600.0
N_STEPS = 2
# The wide path runs the SAME per-row op sequence serial-vs-band, but on
# different array EXTENTS (global+2W_serial vs band+2W_np) with different
# auto-chunk boundaries, so per-element rounding (fma/fusion grouping)
# differs by ~1 ULP — the std path compares bit-identically only because
# its band arrays see the SAME extents every substep.  Diagnosed at f32:
# every field diff was an exact power of two (1-2 f32 ULPs of the velocity
# scale), GLOBAL, not cut-localized — the localization signature of a real
# halo bug is O(1e-3+) at the band cut.  At fp64 the same mechanism sits at
# the 1e-13 scale over 2 steps; 1e-10 is ~3 orders above that floor and
# ~7 below a real cut defect.
RTOL = 1e-10
ATOL = 1e-10
PARITY_FIELDS = ("T", "S", "eta", "u", "v")

_MIN_ROWS_PER_RANK = 4  # wide exchange needs a usable band (chunk >= 1)
_n_procs = MPI.COMM_WORLD.Get_size()
if _n_procs > 1 and (N_LAT // _n_procs) < _MIN_ROWS_PER_RANK:
    pytest.skip(
        f"MPI size {_n_procs} gives {N_LAT // _n_procs} lat rows/rank on "
        f"the n_lat={N_LAT} parity grid (need >={_MIN_ROWS_PER_RANK} for "
        f"a usable wide-halo chunk). Use np in 1..{N_LAT // 4}.",
        allow_module_level=True,
    )

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
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


def _perturbed_global_state(grid, z_coord):
    """Deterministic, mask-aware perturbation (same recipe as the TVD
    parity twin): eta gravity-wave exciter + nonzero meridional v maximal
    at mid-band, so the barotropic subcycle carries real signal across the
    np=2 partition cut."""
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    mask = state.land_mask.data
    eta_pert = state.eta.data + 0.05 * mask * (
        jnp.sin(3.0 * grid.lon2d) * jnp.cos(2.0 * grid.lat2d)
    )
    T_pert = state.T.data + 0.5 * mask[..., jnp.newaxis] * (
        jnp.cos(grid.lon2d) * jnp.sin(2.0 * grid.lat2d)
    )[..., jnp.newaxis]
    v_data = state.v.data
    n_faces, n_lon = v_data.shape[0], v_data.shape[1]
    i_face = jnp.arange(n_faces, dtype=v_data.dtype)[:, None, None]
    j_cell = jnp.arange(n_lon, dtype=v_data.dtype)[None, :, None]
    v_pert = v_data + 0.05 * state.v_mask.data[..., jnp.newaxis] * (
        jnp.sin(jnp.pi * i_face / (n_faces - 1))
        * jnp.cos(2.0 * jnp.pi * j_cell / n_lon)
    )
    return state._replace(
        eta=state.eta.replace(data=eta_pert),
        T=state.T.replace(data=T_pert),
        v=state.v.replace(data=v_pert),
    )


class TestOceanMPIWideHaloParity:

    def test_wide_halo_step_parity_gathered_vs_serial(self):
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()

        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
        z_coord = create_ocean_z_star(n_levels=N_LEV)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="explicit_substep",
            barotropic_wide_halo=True,
            barotropic_local_subcycle_clamp=True,
        )
        state_global = _perturbed_global_state(grid, z_coord)

        # Serial reference on EVERY rank, BEFORE arming MPI (deadlock
        # discipline of the twin module).
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
            global_n_lat=N_LAT, global_n_lon=N_LON,
        )
        band_geom = slice_cgrid_geometry_to_band(ensure_geometry(grid), layout)
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
                    f"np={n_ranks} gathered '{name}' diverged from the "
                    f"serial wide-halo reference after {N_STEPS} step(s): "
                    f"a wide-exchange / staggered-v / chunk-budget defect "
                    f"at the partition cut."
                ),
            )
