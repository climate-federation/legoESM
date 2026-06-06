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

## Running the WHOLE distributed suite
Run each file in its OWN `mpirun` (a fresh MPI_Init/Finalize) — do NOT pass the
whole `tests/distributed/` directory to a single `mpirun pytest`:
```bash
bash scripts/experiment/run_mpi_tests.sh 2 300   # np=2, 300 s hard timeout/file
```
**Why:** different files use different comm topologies (latlon partition vs
cubed-sphere face split) and `mpi4jax.sendrecv`. In ONE session a leftover /
unmatched request from one file desyncs the next → MPICH `dtp_ != NULL`
assertion (ch3u_request.c) or a deadlock. Each file in ISOLATION passes, so the
runner launches a fresh `mpirun` per file, guarded by a hard wall-clock
`timeout` (pytest's signal-timeout can't interrupt a blocked MPI C call).

Per-file results (np=2): coupler 2✓, halo 10✓, latlon_checkpoint 8✓,
latlon_halo 11✓, latlon_polar_filter 5✓, ... (the earlier "coupler hangs"
report was a cross-file-session artefact — the coupler passes in isolation).

## Known flaky test
`tests/distributed/test_latlon_mpi_step.py::...[False]` INTERMITTENTLY deadlocks
under MPI (passes most runs in ~2 s; occasionally hangs until the timeout). It is
a collective-ordering race between the mass-fixer `allreduce` and the halo
`sendrecv` on the JAX-0.9 `mpi4jax` slow XLA:CPU path (the `[True]` and
`mass_conserved` cases are already `skip`-ped as unimplemented for MPI). NOT the
coupler, NOT a barrier issue (a per-test `COMM_WORLD.Barrier` was tried and did
not help — reverted). Fixing it needs explicit mpi4jax token-dependency ordering
so XLA cannot reorder the two collectives; tracked as a separate issue. The
runner's hard timeout bounds it.
