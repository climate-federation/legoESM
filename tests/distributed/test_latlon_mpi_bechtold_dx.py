"""Bechtold's grid spacing must not depend on the MPI decomposition (#1968).

Run under MPI::

    mpirun -n 4 python -m pytest tests/distributed/test_latlon_mpi_bechtold_dx.py

Under lat-lon band MPI ``self.grid`` is this rank's latitude band, so a
spacing filled from it differs per rank (polar bands have smaller cells).
Every rank must use the global value a single-process run uses.
"""
from __future__ import annotations

import jax  # noqa: F401  (force x64 config import order)
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.driver.physics_pipeline import mean_grid_spacing_m  # noqa: E402
from legoesm.grids.factory import create_grid  # noqa: E402

N_LAT = 16
N_LON = 32


@pytest.fixture
def _latlon_band_backend():
    """Point the halo backend at this rank's lat-lon band layout (see
    ``test_latlon_mpi_amip._latlon_band_backend`` for why)."""
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    layout = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(), n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT, n_lon=N_LON)
    set_halo_backend("mpi", layout)
    yield
    set_halo_backend("local")


def test_bechtold_dx_is_global_on_every_rank(_latlon_band_backend):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    if n_ranks < 4:
        # Two hemispheric bands have the global mean cell area, so the
        # band-local fill cannot be told apart from the global one.
        pytest.skip("needs >= 4 MPI ranks")
    if N_LAT % n_ranks != 0:
        pytest.skip(f"N_LAT={N_LAT} not divisible by n_ranks={n_ranks}")
    import tempfile
    out_dir = comm.bcast(
        tempfile.mkdtemp(prefix="legoesm_bdx_mpi_")
        if comm.Get_rank() == 0 else None, root=0)

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=N_LAT, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(energy_fixer=False,
                            discretization="finite_volume", dt=600.0,
                            time_integrator="ssp_rk3_scan"),
        output=OutputConfig(output_dir=out_dir, diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        convection="bechtold", turbulence="none", precision="fp64",
        distributed=True,
    )
    driver = ModelDriver(cfg, output_dir=out_dir)
    driver.setup()
    assert driver.grid.grid_lat.shape[0] == N_LAT // n_ranks  # band-local

    dx_global = mean_grid_spacing_m(create_grid("latlon", N_LAT))
    local_off = comm.allgather(
        abs(mean_grid_spacing_m(driver.grid) - dx_global) > 1.0e3)
    assert any(local_off), "bands do not discriminate; test is vacuous"

    dx = driver.physics.convection_config.dx_m
    all_dx = comm.allgather(dx)
    assert dx == dx_global, all_dx
    assert len(set(all_dx)) == 1, all_dx
