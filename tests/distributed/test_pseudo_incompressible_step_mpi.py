"""serial == MPI parity for the FULL pseudo-incompressible distributed time step.

Exercises every distributed path (weno5 advection y-halo, Vreman SGS, MOST most_cooling
surface with a GLOBAL planar mean, Coriolis, distributed pressure projection). The gathered
MPI step must match the single-process serial step to BiCGSTAB tolerance.

    PATH=$HOME/.local/mpich/bin:$PATH LD_LIBRARY_PATH=$HOME/.local/mpich/lib \
      mpirun -np 2 .venv-mpi/bin/python tests/distributed/test_pseudo_incompressible_step_mpi.py
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as ser  # noqa: E402
from legoesm.atmosphere._future import pseudo_incompressible_plane_mpi as dmpi  # noqa: E402

NY, NX, NZ = 16, 8, 32
LX, LY, LZ = 400.0, 400.0, 400.0


def _cfg():
    return ser.PseudoIncompressibleConfig(
        nx=NX, ny=NY, nz=NZ, Lx=LX, Ly=LY, Lz=LZ, theta_ref0=265.0, scheme="weno5",
        sgs="vreman", surface="most_cooling", z0=0.1, f_cor=1.39e-4, ug=8.0,
        poisson_tol=1e-11, poisson_atol=1e-13, poisson_maxiter=500)


def _global_state(g):
    sh = (NY, NX, NZ)
    z = g.z_c
    th = jnp.broadcast_to((265.0 + 0.01 * jnp.maximum(z - 100.0, 0.0))[None, None, :], sh)
    th = th + 0.05 * jax.random.normal(jax.random.PRNGKey(0), sh)
    u = jnp.full(sh, 8.0) + 0.1 * jax.random.normal(jax.random.PRNGKey(1), sh)
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(2), sh)
    w = jnp.zeros((NY, NX, NZ + 1))
    u, v, w, pi0 = ser.project(u, v, w, th, None, jnp.zeros(sh), 1.0, g)
    return ser.PseudoIncompressibleState(u=u, v=v, w=w, theta=th, pi_prev=jnp.zeros(sh))


def _run(comm):
    rank, nranks = comm.Get_rank(), comm.Get_size()
    assert NY % nranks == 0
    g = ser.make_grid(_cfg())
    gs = _global_state(g)
    forcing = ser.PseudoIncompressibleForcing(t_sfc=jnp.asarray(263.0))
    dt = 1.0
    # serial reference (full domain)
    ss = ser.step(gs, g, dt, forcing)
    # distributed: slice this rank's y-slab, step, gather
    nyl = NY // nranks
    sl = slice(rank * nyl, (rank + 1) * nyl)
    ls = ser.PseudoIncompressibleState(
        u=gs.u[sl], v=gs.v[sl], w=gs.w[sl], theta=gs.theta[sl],
        pi_prev=gs.pi_prev[sl])
    ms = dmpi.step_mpi(ls, g, dt, comm, NY, forcing)

    def gather(local):
        return np.concatenate(comm.allgather(np.asarray(local)), axis=0)

    errs = {}
    for name in ("u", "v", "w", "theta"):
        mg = gather(getattr(ms, name))
        sv = np.asarray(getattr(ss, name))
        errs[name] = float(np.max(np.abs(mg - sv)))
    return rank, errs


def test_step_serial_mpi_parity():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs >=2 MPI ranks")
    rank, errs = _run(comm)
    if rank == 0:
        print("\nfull-step serial-vs-MPI max-err:",
              {k: f"{v:.2e}" for k, v in errs.items()})
    for k, v in errs.items():
        assert v < 1e-5, f"{k} serial!=MPI: {v}"


if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, errs = _run(comm)
    if rank == 0:
        ok = all(v < 1e-5 for v in errs.values())
        print("full-step max-err:", {k: f"{v:.2e}" for k, v in errs.items()},
              "PASS" if ok else "FAIL")


# Parked module: see its docstring.
pytestmark = pytest.mark.skip(
    reason="parked in _future/: not wired into production (ponytail #23)")
