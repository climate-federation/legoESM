"""serial == MPI parity for the distributed pseudo-incompressible pressure Poisson.

Run under MPI (needs >=2 ranks), .venv-mpi (jax 0.9.2 + mpi4jax):

    PATH=$HOME/.local/mpich/bin:$PATH mpirun -np 2 .venv-mpi/bin/python -m pytest \
        tests/distributed/test_pseudo_incompressible_poisson_mpi.py -q

Each rank owns a y-slab; the gathered distributed Laplacian and the gathered distributed
solve must match the single-process (serial) result to BiCGSTAB tolerance. This is the
correctness gate for the nearest-neighbour, mesh-scalable elliptic solve.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.dynamics.les import pseudo_incompressible_poisson as ser  # noqa: E402
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_poisson_mpi as mpi  # noqa: E402

NY, NX, NZ = 16, 8, 12
LX, LY, LZ = 1600.0, 800.0, 2400.0
DX, DY, DZ = LX / NX, LY / NY, LZ / NZ


def _global_fields():
    """Deterministic global (NY,NX,NZ) coefficient + rhs (same on every rank)."""
    x = (jnp.arange(NX) + 0.5) * DX
    y = (jnp.arange(NY) + 0.5) * DY
    z = (jnp.arange(NZ) + 0.5) * DZ
    Y, X, Z = jnp.meshgrid(y, x, z, indexing="ij")
    c = 1.0 + 0.4 * jnp.cos(2 * np.pi * X / LX) * jnp.cos(np.pi * Z / LZ)   # >0
    rhs = jax.random.normal(jax.random.PRNGKey(0), (NY, NX, NZ))
    return c, rhs


def _run_parity(comm):
    rank, nranks = comm.Get_rank(), comm.Get_size()
    assert NY % nranks == 0, "NY must divide by rank count"
    nyl = NY // nranks
    sl = slice(rank * nyl, (rank + 1) * nyl)
    c, rhs = _global_fields()
    c_loc, rhs_loc = c[sl], rhs[sl]
    n_global = NY * NX * NZ

    # (1) Laplacian parity: gathered MPI matvec == serial matvec
    lap_loc = mpi.laplace_pi_mpi(rhs_loc, c_loc, DX, DY, DZ, comm)
    lap_gathered = np.concatenate(comm.allgather(np.asarray(lap_loc)), axis=0)
    lap_serial = np.asarray(ser.laplace_pi(rhs, c, DX, DY, DZ))
    lap_err = np.max(np.abs(lap_gathered - lap_serial))

    # (2) solve parity: gathered MPI solve == serial solve (both zero-mean)
    pil = mpi.solve_pressure_mpi(rhs_loc, c_loc, DX, DY, DZ, comm, n_global,
                                 tol=1e-10, atol=1e-12, maxiter=400)
    pi_gathered = np.concatenate(comm.allgather(np.asarray(pil)), axis=0)
    pi_serial = np.asarray(ser.solve_pressure(rhs, c, DX, DY, DZ,
                                              tol=1e-10, atol=1e-12, maxiter=400)[0])
    pi_gathered = pi_gathered - pi_gathered.mean()
    pi_serial = pi_serial - pi_serial.mean()
    solve_err = np.max(np.abs(pi_gathered - pi_serial))
    return rank, lap_err, solve_err


def test_serial_mpi_parity():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs >=2 MPI ranks (run under mpirun)")
    rank, lap_err, solve_err = _run_parity(comm)
    if rank == 0:
        print(f"\nlaplacian max-err={lap_err:.2e}  solve max-err={solve_err:.2e}")
    assert lap_err < 1e-9, f"Laplacian serial!=MPI: {lap_err}"
    assert solve_err < 1e-6, f"solve serial!=MPI: {solve_err}"


if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, lap_err, solve_err = _run_parity(comm)
    if rank == 0:
        ok = lap_err < 1e-9 and solve_err < 1e-6
        print(f"laplacian max-err={lap_err:.2e}  solve max-err={solve_err:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
