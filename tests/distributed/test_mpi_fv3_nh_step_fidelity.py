"""FV3 NH compressible-Euler 3D step bit-for-bit fidelity under MPI.

NH counterpart to ``test_mpi_fv3_step_fidelity.py`` (PE).  The NH
hot path uses the same iter-1040/iter-1041 plumbing — including
the ``packed_pad_halo_mpi_4d`` (K, pi_prime) exchange in step 6
of the split-explicit acoustic loop — so this test guards against
regressions on the NH side.

Run with::

    JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest \\
        tests/distributed/test_mpi_fv3_nh_step_fidelity.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import (
    get_halo_backend,
    set_halo_backend,
)
from legoesm.parallel.distributed import initialize_distributed


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _build_nh_model_and_state(n: int = 8, nlev: int = 5):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )
    from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState

    grid = create_cubed_sphere(n)
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    config = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0,
        n_acoustic_substeps=4,
        fix_mass=False,
    )

    # Two distinct instances → independent JIT caches (static_argnums=0).
    ref_model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config,
    )
    dist_model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config,
    )

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    # Small zonal-wind perturbation IC (mirrors iter-1045 fixture).
    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return ref_model, dist_model, state


class TestFV3NHStepMPIFidelity:
    """NH compressible-Euler MPI step matches single-rank reference."""

    def test_nh_3_step_owned_faces_match_single_rank(self):
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        dt = 10.0  # NH needs smaller dt (acoustic CFL)
        n_steps = 3

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state()

        ref_state = state
        for _ in range(n_steps):
            ref_state = ref_model.step(ref_state, dt)
        jax.block_until_ready(ref_state.theta_prime.data)

        initialize_distributed(global_n=state.u.data.shape[1])
        assert get_halo_backend() == "mpi"

        dist_state = state
        for _ in range(n_steps):
            dist_state = dist_model.step(dist_state, dt)
        jax.block_until_ready(dist_state.theta_prime.data)

        from legoesm.parallel.comm import build_comm_topology
        topology = build_comm_topology(rank, size)

        # FV3_3D iter-1042 (codex review claim-6): assert on every rank's
        # owned faces, not just rank 0.  A bug that diverges on rank 1's
        # face 3 must not pass undetected because rank 0 didn't compare it.
        for field_name in ("u", "v", "w", "theta_prime", "rho_prime"):
            ref_arr = np.asarray(getattr(ref_state, field_name).data)
            dist_arr = np.asarray(getattr(dist_state, field_name).data)
            for f in topology.local_face_ids:
                np.testing.assert_allclose(
                    dist_arr[f], ref_arr[f],
                    atol=1e-10, rtol=1e-10,
                    err_msg=(
                        f"FV3 NH 3D step diverges from single-rank "
                        f"reference on rank {rank} owned face {f}, "
                        f"field '{field_name}'.  Likely the MPI halo "
                        f"path (packed_pad_halo_mpi_4d or "
                        f"pad_halo_mpi_4d with interp_offsets) "
                        f"regressed."
                    ),
                )
