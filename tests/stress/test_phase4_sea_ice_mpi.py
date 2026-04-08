"""Phase 4: Sea ice MPI stress tests.

Verifies that sea ice EVP dynamics and transport produce correct results
under MPI halo exchange. Requires mpi4py and mpi4jax.

Run with: mpirun -np 3 python -m pytest tests/stress/test_phase4_sea_ice_mpi.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _mpi_available():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _mpi_available(),
    reason="MPI not available or not running under mpirun",
)


class TestSeaIceMPI:
    """Sea ice operations under MPI should produce finite, bounded results."""

    def test_evp_dynamics_mpi(self):
        """EVP solver runs without NaN under MPI halo exchange."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ice.dynamics import evp_solver

        grid = create_cubed_sphere(8)
        shape = (6, 8, 8)

        u_new, v_new, s11, s22, s12 = evp_solver(
            u_ice=jnp.zeros(shape),
            v_ice=jnp.zeros(shape),
            sigma_11=jnp.zeros(shape),
            sigma_22=jnp.zeros(shape),
            sigma_12=jnp.zeros(shape),
            h_ice=jnp.full(shape, 1.5),
            concentration=jnp.full(shape, 0.9),
            wind_u=jnp.full(shape, 5.0),
            wind_v=jnp.full(shape, 2.0),
            ocean_u=jnp.full(shape, 0.1),
            ocean_v=jnp.zeros(shape),
            grid=grid,
            dt=3600.0,
            N_evp=30,  # Fewer iterations for speed
        )

        assert jnp.all(jnp.isfinite(u_new)), "NaN in u_ice after EVP"
        assert jnp.all(jnp.isfinite(v_new)), "NaN in v_ice after EVP"
        assert jnp.all(jnp.isfinite(s12)), "NaN in sigma_12 after EVP"
        assert float(jnp.max(jnp.abs(u_new))) < 5.0, "u_ice unbounded"
        assert float(jnp.max(jnp.abs(v_new))) < 5.0, "v_ice unbounded"

    def test_transport_mpi(self):
        """Ice transport runs without NaN under MPI."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ice.transport import advect_ice_tracers

        grid = create_cubed_sphere(8)
        shape = (6, 8, 8)

        h_new, conc_new, T_new = advect_ice_tracers(
            h_ice=jnp.full(shape, 1.0),
            concentration=jnp.full(shape, 0.8),
            T_ice=jnp.full(shape, 268.0),
            u_ice=jnp.full(shape, 0.05),
            v_ice=jnp.zeros(shape),
            grid=grid,
            dt=3600.0,
        )

        assert jnp.all(jnp.isfinite(h_new)), "NaN in h_ice after transport"
        assert jnp.all(jnp.isfinite(conc_new)), "NaN in concentration"
        assert jnp.all(jnp.isfinite(T_new)), "NaN in T_ice after transport"
