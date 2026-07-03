"""Real cubed-sphere MPI validation of the per-column turbulence override slice.

Run under MPI::

    mpirun --oversubscribe -n 2 python -m pytest tests/distributed/test_cubed_mpi_override.py
    mpirun --oversubscribe -n 3 python -m pytest tests/distributed/test_cubed_mpi_override.py

Companion to ``test_latlon_mpi_override.py`` (lat-lon band): under a REAL cubed
``DistributedLayout`` (from ``initialize_distributed``), ``turbulence_config_for``
must localize a GLOBAL ``(6*n*n,)`` per-column ``clubb_lite`` override to THIS
rank's owned faces — exactly the model's own ``scatter`` partition.
"""
from __future__ import annotations

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
)
from legoesm.driver.physics_pipeline import turbulence_config_for  # noqa: E402
from mpi4py import MPI  # noqa: E402


@pytest.fixture
def _cubed_backend():
    """Activate the cubed-sphere MPI topology + DistributedLayout (face-only)."""
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.distributed import get_active_layout, initialize_distributed

    n_procs = MPI.COMM_WORLD.Get_size()
    initialize_distributed(global_n=max(n_procs, 2))  # registers the DistributedLayout
    yield get_active_layout()
    set_halo_backend("local")


def test_cubed_mpi_per_column_override_localizes(_cubed_backend):
    layout = _cubed_backend
    if layout is None or not hasattr(layout, "global_n"):
        pytest.skip("no cubed DistributedLayout active")

    from legoesm.parallel.layout import scatter

    n = int(layout.global_n)
    ncol = 6 * n * n
    global_ck = np.arange(ncol, dtype=np.float64) * 0.001 + 0.3

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=n, nlev=5),
        dycore=DycoreConfig(discretization="finite_volume", dt=600.0),
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=global_ck)),
    )

    # The driver's single source of truth slices the override to this rank's faces.
    tc = turbulence_config_for(cfg)
    local_ck = np.asarray(tc.clubb_lite.C_K)
    expected = np.asarray(
        scatter(np.asarray(global_ck).reshape(6, n, n), layout)).reshape(-1)
    assert local_ck.shape == expected.shape, (
        f"rank {MPI.COMM_WORLD.Get_rank()}: local C_K {local_ck.shape}, "
        f"expected {expected.shape}")
    np.testing.assert_allclose(local_ck, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    from legoesm.grids.halo import get_mpi_topology, set_halo_backend
    from legoesm.parallel.distributed import initialize_distributed

    initialize_distributed()
    try:
        test_cubed_mpi_per_column_override_localizes(get_mpi_topology())
    finally:
        set_halo_backend("local")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
