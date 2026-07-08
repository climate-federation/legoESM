"""Real MPAS/Voronoi MPI validation of the per-column turbulence override slice.

Run under MPI::

    mpirun --oversubscribe -n 2 python -m pytest tests/distributed/test_mpas_mpi_override.py

Completes the MPI override-localization across all three grid families (lat-lon +
cubed-sphere + MPAS).  Unlike lat-lon (halo topology) and cubed-sphere (CommTopology
+ DistributedLayout), MPAS registers ONLY the active VoronoiPartitionLayout (no halo
topology), so ``active_column_layout()`` falls back to ``get_active_voronoi_layout``.
Under a REAL Voronoi partition (built by the driver's ``setup()``),
``turbulence_config_for`` must gather a GLOBAL ``(nCells_global,)`` per-column
override at THIS rank's owned+halo ``local_cells``.
"""
from __future__ import annotations

import tempfile

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

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

RES, NLEV = 3, 20  # level-3 icosahedral mesh = 642 cells


def test_mpas_mpi_per_column_override_localizes():
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()

    # A global per-column C_K sized to the level-3 mesh; the exact nCells_global is
    # read back from the partition the driver builds.
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir=tempfile.mkdtemp(), diag_days=0),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="clubb_lite", precision="fp64", distributed=(n_ranks > 1),
    )
    # Build the partition first (a no-override driver) to learn nCells_global.
    probe = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    probe.setup()
    if n_ranks > 1 and probe._voronoi_layout is None:
        pytest.skip("MPAS distributed partition not built")
    n_global = (int(probe._voronoi_layout.partition.nCells_global)
                if probe._voronoi_layout is not None
                else int(probe.grid.nCells))
    global_ck = np.arange(n_global, dtype=np.float64) * 0.001 + 0.3

    cfg = cfg._replace(turbulence_override=TurbulenceConfig(
        scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=global_ck)))

    # MPAS sets only the active voronoi layout, NOT a halo topology — clear any
    # leftover (cubed) halo topology from the session conftest so get_mpi_topology()
    # is None and active_column_layout() falls back to the voronoi layout, exactly
    # as in a fresh MPAS process.  (The MPAS dynamics halo uses the voronoi layout's
    # own exchange, not the global halo backend.)
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")

    # The driver's single source of truth slices the override to this rank's cells.
    tc = turbulence_config_for(cfg)
    local_ck = np.asarray(tc.clubb_lite.C_K)
    if n_ranks > 1:
        local_cells = np.asarray(probe._voronoi_layout.partition.local_cells)
        expected = global_ck[local_cells]
    else:
        expected = global_ck  # serial / single-rank: identity (no voronoi layout)
    assert local_ck.shape == expected.shape, (
        f"rank {comm.Get_rank()}: local C_K {local_ck.shape}, "
        f"expected {expected.shape}")
    np.testing.assert_allclose(local_ck, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    test_mpas_mpi_per_column_override_localizes()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
