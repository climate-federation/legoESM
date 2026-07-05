"""End-to-end production deploy path under real lat-lon band MPI.

Run under MPI::

    mpirun --oversubscribe -n 2 python -m pytest tests/distributed/test_deploy_mpi_e2e.py

The full deploy workflow was built piecewise across iters 57-64 (the JSON loader,
the grid-identity guard, the per-rank MPI slice).  This validates them as ONE
flow, from the actual campaign-output JSON ARTIFACT, MULTI-coefficient
(C_K+Pr_t+C_eps), under real 2-rank MPI:

    campaign JSON (multi "fields" + grid provenance)
      -> corrected_turbulence_override(path, grid=GLOBAL grid)   [loader + grid guard]
      -> ExperimentConfig.turbulence_override                     [the deploy vehicle]
      -> turbulence_config_for (per rank)                         [MPI slice, all 3 coeffs]

and asserts each rank's resolved C_K / Pr_t / C_eps equals the GLOBAL override's
own latitude-band slice — the deployed coefficients land on the rank's columns.
"""
from __future__ import annotations

import json
import tempfile

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

mpi4py = pytest.importorskip("mpi4py")
from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import turbulence_config_for  # noqa: E402
from legoesm.training.deploy_correction import (  # noqa: E402
    corrected_turbulence_override,
    grid_fingerprint,
)
from mpi4py import MPI  # noqa: E402

N_LAT = 16
N_LON = 32
_COEFFS = {"clubb_lite_C_K": 0.4, "clubb_lite_Pr_t": 0.5, "clubb_lite_C_eps": 0.1}
_FIELD = {"clubb_lite_C_K": "C_K", "clubb_lite_Pr_t": "Pr_t",
          "clubb_lite_C_eps": "C_eps"}


def _global_field(base, n_lat=N_LAT, n_lon=N_LON):
    lat = np.linspace(-1.0, 1.0, n_lat)[:, None]
    lon = np.linspace(0.0, 1.0, n_lon)[None, :]
    return (base + 0.1 * np.sin(3.0 * lat) + 0.02 * lon).reshape(-1)


@pytest.fixture
def _latlon_band_backend():
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    layout = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(), n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT, n_lon=N_LON)
    set_halo_backend("mpi", layout)
    yield layout
    set_halo_backend("local")


def _shared_campaign_json(global_grid) -> str:
    """Rank 0 writes ONE campaign-output JSON to a shared dir; bcast the path so
    every rank reads the SAME artifact (faithful to a production shared FS)."""
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        fields = {key: _global_field(base) for key, base in _COEFFS.items()}
        campaign_out = {
            "coefficients": ["C_K", "Pr_t", "C_eps"],
            "fields": {k: v.tolist() for k, v in fields.items()},
            "grid": grid_fingerprint(global_grid),
        }
        d = tempfile.mkdtemp(prefix="legoesm_deploy_e2e_")
        path = f"{d}/campaign_out.json"
        with open(path, "w") as f:
            json.dump(campaign_out, f)
    else:
        path = None
    return comm.bcast(path, root=0)


def test_deploy_json_multicoeff_distributed(_latlon_band_backend):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    if N_LAT % n_ranks != 0:
        pytest.skip(f"N_LAT={N_LAT} not divisible by n_ranks={n_ranks}")
    layout = _latlon_band_backend

    from legoesm.grids.latlon import create_latlon_grid
    global_grid = create_latlon_grid(N_LAT, N_LON, dtype=jax.numpy.float64)
    path = _shared_campaign_json(global_grid)
    comm.Barrier()  # every rank sees the written file before reading
    try:
        # Deploy: load + grid-guard against the GLOBAL grid → the global override.
        override = corrected_turbulence_override(path, grid=global_grid)

        # The grid guard is LIVE in this distributed deploy: a DIFFERENT grid (the
        # JSON's provenance is for 16x32) must be rejected, not silently accepted.
        with pytest.raises(ValueError, match="deploy grid mismatch|spans"):
            corrected_turbulence_override(
                path, grid=create_latlon_grid(8, 16, dtype=jax.numpy.float64))

        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=N_LAT, nlev=5),
            dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                                discretization="finite_volume"),
            radiation="gray", turbulence="clubb_lite",
            turbulence_override=override, distributed=(n_ranks > 1))

        # The driver's single source of truth slices ALL THREE coefficients to this
        # rank's latitude band.
        tc = turbulence_config_for(cfg)
        for key, base in _COEFFS.items():
            field = _FIELD[key]
            local = np.asarray(getattr(tc.clubb_lite, field))
            expected = _global_field(base).reshape(N_LAT, N_LON)[
                layout.lat_start:layout.lat_end, :].reshape(-1)
            assert local.shape == (layout.n_lat_local * N_LON,), (
                f"rank {comm.Get_rank()} {field}: shape {local.shape}")
            np.testing.assert_allclose(local, expected, rtol=1e-6, atol=1e-7)
    finally:
        comm.Barrier()  # all ranks done reading before rank 0 removes the dir
        if comm.Get_rank() == 0 and path is not None:
            import os
            import shutil
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)


if __name__ == "__main__":
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    _lay = make_latlon_band_layout(
        rank=MPI.COMM_WORLD.Get_rank(), n_ranks=MPI.COMM_WORLD.Get_size(),
        n_lat=N_LAT, n_lon=N_LON)
    set_halo_backend("mpi", _lay)
    try:
        test_deploy_json_multicoeff_distributed(_lay)
    finally:
        set_halo_backend("local")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
