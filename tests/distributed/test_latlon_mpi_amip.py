"""End-to-end lat-lon band-MPI AMIP smoke + diagnostics-gather regression.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_latlon_mpi_amip.py

Companion to the per-step correctness test
(:mod:`tests.distributed.test_latlon_mpi_step`, which checks the bare
dycore step) and the gather/scatter round-trip
(:mod:`tests.distributed.test_latlon_mpi_checkpoint`).  Those never
exercise the full ``ModelDriver`` AMIP path, where two regressions hid:

1. **Double-sliced SST/SIC forcing.**  ``_create_grid`` band-slices
   ``self.grid``/``self._grid_lat`` to the rank's latitude band; if the
   SST forcing function is built on that already-local grid it is sliced
   a *second* time by ``_band_get_sst_sic`` in ``_setup_parallel``, so
   every non-zero rank gets an empty ``(0, n_lon)`` band and
   ``compute_radiation_core`` crashes (size-0 reshape).  The run here
   completing on >=2 ranks is the regression guard.

2. **Rank-local diagnostics.**  The full-collect gather keyed on the
   cubed-sphere ``_owned_face_ids`` (``None`` for lat-lon), so lat-lon
   runs collected on the rank-local band — wrong global means and a
   snapshot holding a single band.  We assert the gathered snapshot has
   the *global* latitude extent, not ``n_lat // n_ranks``.

Kept tiny (16x32, 1 day) so it runs in a few seconds under mpirun.
"""
from __future__ import annotations

import jax  # noqa: F401  (force x64 config import order)
import numpy as np
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


N_LAT = 16
N_LON = 32


def _broadcast_tmpdir() -> str:
    """Rank 0 picks a shared output dir under the repo tmp area; bcast it.

    All ranks must agree on the path (the driver writes from rank 0, and
    rank 0 reads back to assert).  ``tmp_path`` is per-process under
    mpirun, so derive a single deterministic dir on rank 0 and broadcast.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        import tempfile
        d = tempfile.mkdtemp(prefix="legoesm_amip_mpi_")
    else:
        d = None
    return comm.bcast(d, root=0)


@pytest.fixture
def _latlon_band_backend():
    """Activate the lat-lon band MPI topology for the driver.

    The session-autouse fixture in ``conftest.py`` initializes a
    *cubed-sphere* ``DistributedLayout`` and sets the halo backend to its
    topology, so ``get_mpi_topology()`` would hand the driver a 6-face
    layout and the lat-lon grid would never take its band-MPI branch
    (manifesting as a ``(16,32) vs (6,1)`` broadcast error).
    ``initialize_distributed_latlon`` alone does not fix it: its
    "called more than once" guard returns the existing cubed-sphere
    topology without re-arming the halo backend.  So point the backend
    directly at this rank's ``LatLonBandLayout`` — exactly what
    ``get_mpi_topology()`` feeds the driver in a fresh process — then
    restore afterwards so later tests keep the cubed-sphere backend.
    """
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    layout = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(),
        n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT,
        n_lon=N_LON,
    )
    set_halo_backend("mpi", layout)
    yield
    set_halo_backend("local")


def test_latlon_mpi_amip_runs_and_gathers_global(_latlon_band_backend):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    if N_LAT % n_ranks != 0:
        pytest.skip(f"N_LAT={N_LAT} not divisible by n_ranks={n_ranks}")

    out_dir = _broadcast_tmpdir()

    cfg = ExperimentConfig(
        grid=GridConfig(
            grid_type="latlon", resolution=N_LAT, nlev=20,
            vertical_coord="hybrid",
        ),
        dycore=DycoreConfig(
            discretization="finite_volume", dt=600.0,
            time_integrator="ssp_rk3_scan",
        ),
        output=OutputConfig(output_dir=out_dir, diag_days=1),
        days=1,
        dataset="analytical",
        radiation="gray",
        convection="none",
        turbulence="none",
        precision="fp64",
        distributed=(n_ranks > 1),
    )

    driver = ModelDriver(cfg, output_dir=out_dir)
    driver.setup()
    # Regression guard #1: under the double-slice bug this raised inside
    # compute_radiation_core on every non-zero rank.
    driver.run()

    comm.Barrier()

    # Only rank 0 writes / owns the gathered diagnostics.
    if comm.Get_rank() != 0:
        return

    snap = np.load(f"{out_dir}/snapshots.npz", allow_pickle=True)
    key = "day001_T_low"
    assert key in snap.files, f"missing {key}; keys={list(snap.files)}"
    t_low = np.asarray(snap[key])

    # Regression guard #2: the gathered field must span the GLOBAL latitude
    # extent, not this rank's band (n_lat // n_ranks).
    assert t_low.shape == (N_LAT, N_LON), (
        f"snapshot is {t_low.shape}, expected global ({N_LAT}, {N_LON}); "
        f"diagnostics collected on the rank-local band instead of gathering."
    )

    # Physical sanity: finite, no NaNs, sane near-surface temperatures.
    assert np.all(np.isfinite(t_low)), "non-finite T_low in gathered snapshot"
    assert 200.0 < float(np.mean(t_low)) < 330.0, (
        f"mean low-level T {float(np.mean(t_low)):.1f} K out of physical range"
    )

    ts = np.load(f"{out_dir}/timeseries.npz")
    # Global vertical profile must be present (skipped under the broken
    # perf-mode-on-MPI path) and span all levels.
    assert ts["profiles_T"].shape == (1, 20), (
        f"profiles_T {ts['profiles_T'].shape}; profiles were skipped "
        f"(perf-mode lightweight path) instead of gathered."
    )
    assert np.all(np.isfinite(ts["profiles_T"]))


if __name__ == "__main__":
    test_latlon_mpi_amip_runs_and_gathers_global()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
