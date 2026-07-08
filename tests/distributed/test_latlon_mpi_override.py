"""Real lat-lon band-MPI validation of the per-column turbulence override slice.

Run under MPI::

    mpirun --oversubscribe -n 2 python -m pytest tests/distributed/test_latlon_mpi_override.py

The LES-informed correction loop deploys a GLOBAL ``(ncol,)`` per-column
``clubb_lite`` override.  Under lat-lon band MPI each rank owns a latitude band, so
the physics runs on rank-local ``(ncol_local, nlev)`` shapes and a global override
would mismatch ``broadcast_column_param``.  ``turbulence_config_for`` slices the
global override to the rank's band at dycore-build time
(:func:`legoesm.atmosphere.physics.turbulence.override_sharding.localize_turbulence_override`).

This test proves it END-TO-END on >=2 real ranks:

1. **Run completes.**  With a per-column override and turbulence="clubb_lite", the
   2-rank AMIP step runs to completion.  WITHOUT the slice, ``broadcast_column_param``
   raises on the global-(512)-vs-local-(256) length mismatch — so completion is the
   regression guard.
2. **Non-vacuous slice.**  Each rank's resolved ``turbulence_config_for`` yields a
   per-column ``C_K`` of shape ``(ncol_local,)`` whose values equal the GLOBAL
   override's band slice ``[lat_start:lat_end, :]`` — the coefficients landed on the
   rank's own columns, not some other band.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # fp64 run → the override stays float64

mpi4py = pytest.importorskip("mpi4py")
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.driver.physics_pipeline import turbulence_config_for  # noqa: E402
from mpi4py import MPI  # noqa: E402

N_LAT = 16
N_LON = 32


def _global_ck(n_lat=N_LAT, n_lon=N_LON):
    """A spatially-varying GLOBAL per-column C_K (so the band slice is non-trivial)."""
    lat = np.linspace(-1.0, 1.0, n_lat)[:, None]
    lon = np.linspace(0.0, 1.0, n_lon)[None, :]
    return (0.4 + 0.2 * np.sin(3.0 * lat) + 0.05 * lon).reshape(-1)


def _broadcast_tmpdir() -> str:
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        import tempfile
        d = tempfile.mkdtemp(prefix="legoesm_override_mpi_")
    else:
        d = None
    return comm.bcast(d, root=0)


@pytest.fixture
def _latlon_band_backend():
    """Activate the lat-lon band MPI topology for the driver (mirror of the AMIP test)."""
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    layout = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(), n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT, n_lon=N_LON)
    set_halo_backend("mpi", layout)
    yield layout
    set_halo_backend("local")


def test_latlon_mpi_per_column_override(_latlon_band_backend):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    if N_LAT % n_ranks != 0:
        pytest.skip(f"N_LAT={N_LAT} not divisible by n_ranks={n_ranks}")

    layout = _latlon_band_backend
    out_dir = _broadcast_tmpdir()
    global_ck = _global_ck()

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=N_LAT, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="finite_volume", dt=600.0,
                            time_integrator="ssp_rk3_scan"),
        output=OutputConfig(output_dir=out_dir, diag_days=1),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite",
            clubb_lite=CLUBBLiteConfig(C_K=np.asarray(global_ck))),
        precision="fp64", distributed=(n_ranks > 1),
    )

    # (2) Non-vacuous: the resolved override is THIS rank's band slice — the SAME
    # config path the dycore build below consumes.
    tc = turbulence_config_for(cfg)
    local_ck = np.asarray(tc.clubb_lite.C_K)
    expected = global_ck.reshape(N_LAT, N_LON)[
        layout.lat_start:layout.lat_end, :].reshape(-1)
    assert local_ck.shape == (layout.n_lat_local * N_LON,), (
        f"rank {comm.Get_rank()}: local C_K shape {local_ck.shape}, expected "
        f"({layout.n_lat_local * N_LON},)")
    np.testing.assert_allclose(local_ck, expected, rtol=1e-6, atol=1e-7)

    # (1) Regression guard: the full 2-rank AMIP step completes — without the slice
    # broadcast_column_param raises on the global-vs-local length mismatch.
    driver = ModelDriver(cfg, output_dir=out_dir)
    driver.setup()
    driver.run()
    comm.Barrier()


if __name__ == "__main__":
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    _lay = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(), n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT, n_lon=N_LON)
    set_halo_backend("mpi", _lay)
    try:
        test_latlon_mpi_per_column_override(_lay)
    finally:
        set_halo_backend("local")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
