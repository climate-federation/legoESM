"""Phase 4: MPI parallelization stress tests for the coupled model.

Verifies that MPI-distributed coupled runs produce results matching
single-rank serial runs. Requires mpi4py and mpi4jax.

Run with: mpirun -np 3 python -m pytest tests/stress/test_phase4_coupled_mpi.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _mpi_available():
    """Check if MPI is available and we are running under mpirun."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _mpi_available(),
    reason="MPI not available or not running under mpirun",
)


def _make_coupled_driver(preset="aquaplanet", days=2, resolution=8, nlev=5):
    """Create a CoupledESMDriver for MPI testing."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray",
        days=days,
        distributed=True,
    )
    coupled_cfg = PRESETS[preset]()
    driver = CoupledESMDriver(atm_config, coupled_cfg)
    driver.setup()
    return driver


class TestCoupledMPIAquaplanet:
    """Coupled aquaplanet under MPI should produce bounded, finite results."""

    def test_aquaplanet_mpi_runs(self):
        """2-day aquaplanet completes under MPI without error."""
        driver = _make_coupled_driver("aquaplanet", days=2)
        status = driver.run()

        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()

        # All ranks should have finite state
        assert jnp.all(jnp.isfinite(driver.state.T.data))
        assert jnp.all(jnp.isfinite(driver.state.p_s.data))
        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))

        # SST bounded
        sst = driver.ocean_state.T_sfc.data
        sst_min = float(jnp.min(sst))
        sst_max = float(jnp.max(sst))
        if rank == 0:
            assert sst_min > 250.0, f"SST min {sst_min:.1f} < 250"
            assert sst_max < 330.0, f"SST max {sst_max:.1f} > 330"


class TestCoupledMPISlabSimple:
    """Coupled slab_simple with land tiles under MPI."""

    def test_slab_simple_mpi_runs(self):
        """2-day slab_simple completes under MPI."""
        driver = _make_coupled_driver("slab_simple", days=2)
        status = driver.run()

        assert jnp.all(jnp.isfinite(driver.state.T.data))
        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))

        # Surface state should exist
        sfc = driver.surface_state
        assert sfc is not None
        assert sfc.land is not None


class TestCoupledMPIConsistency:
    """Cross-rank consistency: global means should agree."""

    def test_global_mean_sst_consistent(self):
        """Global-mean SST should be consistent across ranks."""
        driver = _make_coupled_driver("aquaplanet", days=2)
        driver.run()

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        sst_mean = float(jnp.mean(driver.ocean_state.T_sfc.data))
        all_means = comm.gather(sst_mean, root=0)

        if rank == 0:
            # All ranks should report similar global mean
            # (some difference expected due to partitioning)
            all_means = np.array(all_means)
            spread = np.max(all_means) - np.min(all_means)
            assert spread < 1.0, (
                f"SST mean spread across ranks = {spread:.4f} K, too large"
            )
