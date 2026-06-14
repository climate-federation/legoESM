# Portability sweep: GPU / MPI / fp32-fp64 vs single-CPU-fp64

Verification that the legoESM model components run on **GPU** and **MPI** in
**float32 and float64**, with results consistent with the single-CPU-float64
reference (the oracle). Machine: Ubuntu 24.04, RTX 5090 (Blackwell), MPICH 4.2.3.

## Results

| Component | Driver / matrix | CPU-fp64 | GPU | Consistency |
|---|---|---|---|---|
| 3D dycore — shallow water | `run_atmosphere_test_matrix --only sw` | 4/4 PASS | 4/4 PASS | **GPU==CPU** (Williamson2 L2=1.76e-4 identical; mass drift ~1e-16) |
| 3D dycore — nonhydrostatic | `--only nh` (DCMIP TC1/2/3) | — | 3/3 PASS | mass drift 0–1e-16 |
| 3D dycore — hydrostatic | `--only hydro` | (slow; SW+NH already prove the C-grid dycore is GPU-portable) | | |
| Ocean | `run_ocean_test_matrix --grid latlon` | 16/16 PASS | 16/16 PASS | **GPU==CPU** (overturning 0.3449, neverworld 0.646, dino 0.201 identical) |
| SCM (single column) | `run_scm_test_matrix {gabls1,ekman}` | — | run OK | column physics |
| LES (neutral ABL) | `run_spectral_les --dynamic` | — | **4/4 MOST** (fp32) | log law, φ_m≈1, σ_w/u*≈1.1 |
| CRM (GATE plane) | `run_gate_plane --microphysics morrison` | stable, d(mass)=1.95e-16 | stable, d(mass)=0.00 | both mass-conserving |
| Slab / coupler | `tests/distributed/test_coupler_mpi` | — | (CPU) | 2/2 PASS (land/ice/lake/ocean tiles) |
| Precision ops/grids | `tests/unit/test_precision*`, `test_backend_precision` | 147 pass | 47 pass | fp32/fp64 consistency |

GPU enabled via `jax-cuda12-plugin` + `JAX_PLATFORMS=cuda`; **float32** is the
production GPU mode (`--f32`; RTX-50xx fp64 is ~1/64 of fp32). The fp64 GPU runs
reproduce the CPU-fp64 norms to round-off; fp32 runs agree to single-precision
tolerance (per the 147+47 precision tests).

## MPI

Works via a user-space MPICH + a dedicated `.venv-mpi` (JAX 0.9.2 + mpi4jax;
the main JAX 0.10 venv can't host mpi4jax). See `docs/md_files/mpi_local_setup.md`.
Run the suite with `bash scripts/experiment/run_mpi_tests.sh 2` (each file in its
own `mpirun`). Verified serial==MPI for the halo exchange and the latlon dycore
step; coupler 2/2. One known intermittent flake (`test_latlon_mpi_step`, a
collective-ordering race on the deprecated JAX-0.9 mpi4jax path) — diagnosed +
bounded in `docs/md_files/mpi_local_setup.md`.

## How to reproduce
```bash
# GPU portability (atmosphere / ocean):
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py --only sw --quick
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py --grid latlon --quick
# fp32/fp64 consistency:
JAX_PLATFORMS=cuda .venv/bin/python -m pytest tests/unit/test_backend_precision.py tests/unit/test_precision_modes.py
# CRM / LES on GPU:
JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_gate_plane.py --steps 60 --microphysics morrison
JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_spectral_les.py --dynamic --f32
# MPI:
bash scripts/experiment/run_mpi_tests.sh 2
```

## Nondeterminism sources

"Same config, same machine" does **not** guarantee bit-identical output in every
configuration. Known sources, and what bounds them:

- **GPU reduction order.** Floating-point addition is non-associative; XLA on
  GPU may reorder reductions (and pick different kernels after autotuning), so
  global sums / norms can differ in the last bits between runs and between
  driver/XLA versions. CPU runs are deterministic. For strict GPU determinism
  set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (slower).
- **XLA autotuning + persistent JIT cache.** A warm compilation cache
  (`LEGOESM_JAX_CACHE_DIR`) replays previously-selected kernels; a cold cache
  may autotune to different ones. Disable with `LEGOESM_JAX_CACHE_DISABLE=1`
  when bisecting numeric differences.
- **fp32 vs x64.** The unit tier and Metal paths run fp32; scientific runs use
  `JAX_ENABLE_X64=1`. Comparing across precision modes is a model change, not
  nondeterminism — see `runtime/precision.py` (`PrecisionPolicy`).
- **MPI reductions.** `global_sum_mpi` uses `allreduce(SUM)`; MPI libraries
  guarantee the same reduction order for a fixed rank count/topology, so a
  fixed `-np` is reproducible, but results differ across rank counts (different
  partial-sum trees). Serial-vs-MPI equality is therefore tested to tolerance,
  not bitwise (`tests/distributed/`).
- **Metal CPU fallback.** On Apple Silicon spectral work routes to CPU
  (`parallel/metal.py`); a run that silently fell back executes different
  kernels than a GPU run — check `get_backend()` when comparing machines.
- **What IS deterministic.** All model randomness descends from
  `config.seed` via `runtime/rng.py` (SHA-256 named subkeys, order-independent),
  and the run manifest records seeds + `state_digest` so
  `legoesm reproduce <manifest> --check` verifies bit-identical replay on the
  same platform/precision/backend.
