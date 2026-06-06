# Running legoESM on MPI (local, user-space MPICH)

MPI works on this machine via a **user-space MPICH** (no system OpenMPI, no sudo)
and a **dedicated venv** with the pinned stack — the main `.venv` (JAX 0.10) can't
host `mpi4jax` (JAX 0.10 removed the `custom_call` API mpi4jax uses; see
`requirements_mpi.txt`).

## One-time setup
```bash
bash scripts/experiment/setup_mpi_local.sh
```
This builds `.venv-mpi` (JAX 0.9.2 + jaxlib 0.9.2 + `mpi4py` + `mpi4jax==0.8.1.post2`
compiled against MPICH + legoESM editable + pytest), then smoke-tests
`mpi4jax.sendrecv` and the bootstrap/halo distributed tests.

MPICH is expected at `~/.local/mpich` (override with `MPICH_HOME`). If absent,
build it user-space: `./configure --prefix=$HOME/.local/mpich && make -j && make install`.

## Running
```bash
source scripts/experiment/setup_mpi_local.sh env     # PATH/LD_LIBRARY_PATH/MPICC + JAX_PLATFORMS=cpu, x64
mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/ -q
# or a model under MPI:
mpirun -np 2 .venv-mpi/bin/python scripts/run/run_amip.py ...
```

## Verified (Ubuntu 24.04, MPICH 4.2.3, JAX 0.9.2)
- `mpi4jax.sendrecv` under `mpirun -np 2` exchanges correctly (rank 0 ↔ rank 1).
- **31 distributed tests pass** (`test_mpi_bootstrap.py` + `test_halo_mpi.py`).
- `test_latlon_mpi_step.py` (serial==MPI dycore consistency): **1 passed, 2 skipped**.
- A harmless deprecation warning per `sendrecv` call
  (`API_VERSION_STATUS_RETURNING not supported by XLA:CPU`) — JAX 0.9 slow path,
  exactly as documented in `requirements_mpi.txt`; results are correct.

## Known issue
`tests/distributed/test_coupler_mpi.py` HANGS under `np=2` (no progress, killed by
timeout) — a collective-mismatch / rank-count requirement in that test, NOT a
runtime problem (the halo, reductions and dycore-step paths all work). To
investigate: run it under `np=4`, or trace which collective deadlocks. The MPI
HALO/DYCORE consistency (the "MPI == single-CPU" guarantee) is verified by the
passing step/halo tests.
