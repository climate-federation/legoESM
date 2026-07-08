# Real MPI and Multi-GPU Scaling Tests (Atmosphere + Ocean)

> **2026-07 — cluster job sets + ocean multi-GPU lane.** Ready-made
> weak/strong job sets now exist for **NCAR Derecho**
> (`scripts/cluster/scaling_derecho/`, incl. multi-node NCCL + ocean GPU/CPU
> jobs) and **DKRZ Levante** (`scripts/cluster/scaling_levante/`). The ocean
> gained a full-step multi-GPU driver:
> `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (the ocean twin of the
> atm SPMD bench; single-process multi-device AND `--multicontroller`
> multi-node, with `--parity-gate`/`--check-conservation` fail-fast armor) —
> superseding §4's "no single ocean script" caveat for the GPU-node case. SOTA comparison + readiness review:
> `docs/performance/scaling/derecho_levante_sota_review_2026-07.md`.

This runbook describes how to run real-hardware scaling tests for legoESM
atmosphere and ocean components on CPUs/GPUs and MPI clusters.

It is focused on commands that already exist in this repository:
- `scripts/bench/run_levante_gpu_scaling.py` (atmosphere scaling + MPI validation)
- `scripts/matrix/run_ocean_test_matrix.py` (ocean runtime/regression matrix)
- `tests/distributed/test_halo_mpi.py` and `tests/ocean/distributed/test_ocean_mpi_conservation.py` (MPI ocean/distributed checks)

## 0. What's new (2026-05)

- **Persistent JAX JIT cache** (issue #273) is enabled by default — the
  first segment compile is cached to disk and reused. Override with
  `LEGOESM_JAX_CACHE_DIR=/path`, disable with `LEGOESM_JAX_CACHE_DISABLE=1`.
  Large multi-GPU runs benefit most.
- **SPMD halo backend** (issue #275) is activated in the AMIP
  production profile for multi-device runs.
- **OMIP centennial driver** (`scripts/run/run_omip.py`) supports
  tripolar (eORCA1), MPAS Voronoi (ico5 / ico6), lat-lon, and
  cubed-sphere grids with auto-restart and `jra55_3way` run sets.
- **mpi4jax compatibility guardrails** in `legoesm.parallel.reductions`
  hard-error for `mpi4jax<0.8`; warn for JAX / mpi4jax outside the
  tested range; promote to error with
  `LEGOESM_MPI_STRICT_COMPAT=1`.

## 1. Prerequisites

From repo root:

```bash
cd /Users/pierregentine/legoESM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install "mpi4py>=4.1,<5" "mpi4jax>=0.9,<0.10"
```

MPI runtime:
- OpenMPI 4.x or 5.x recommended (`mpirun` or `mpiexec` in `PATH`)

Quick checks:

```bash
which mpirun || which mpiexec
.venv/bin/python -c "import jax; print('backend=', jax.default_backend(), 'devices=', len(jax.devices()))"
.venv/bin/python -c "import mpi4py, mpi4jax; print('mpi4py ok, mpi4jax', mpi4jax.__version__)"
```

## 2. Output Convention

Use timestamped output folders to compare runs:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/hardware_scaling_${STAMP}"
mkdir -p "$OUTDIR"
```

## 3. Atmosphere Scaling (Multi-GPU + MPI)

Use `scripts/bench/run_levante_gpu_scaling.py` for atmosphere scaling benchmarks.
This includes:
- compile time
- steady-state time per step
- global/per-device/per-rank throughput
- strong and weak scaling checks with thresholds

### 3.1 Single-node multi-GPU (or multi-CPU) scaling

```bash
JAX_PLATFORMS=gpu .venv/bin/python scripts/bench/run_levante_gpu_scaling.py \
  --grid cubed-sphere \
  --mode strong \
  --precision float32 \
  --n-levels 26 \
  --n-warmup 3 \
  --n-timing 100 \
  --output-dir "$OUTDIR/atm_gpu"
```

CPU comparison:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/bench/run_levante_gpu_scaling.py \
  --grid cubed-sphere \
  --mode strong \
  --precision float32 \
  --n-levels 26 \
  --output-dir "$OUTDIR/atm_cpu"
```

### 3.2 MPI rank scaling (real cluster)

The benchmark uses `initialize_distributed(global_n=...)` to build the MPI
topology and set up MPI halo exchange.  Each rank keeps the full
`(6, n, n, ...)` state (matching the production driver), and `pad_halo_mpi`
handles inter-rank communication during each step.

```bash
mpirun -np 6 .venv/bin/python scripts/bench/run_levante_gpu_scaling.py \
  --grid cubed-sphere \
  --mode strong \
  --precision float32 \
  --n-levels 26 \
  --n-gpus 6 \
  --output-dir "$OUTDIR/atm_mpi"
```

For pure communication-scaling measurement (no global sync from
conservation fixers):

```bash
mpirun -np 6 .venv/bin/python scripts/bench/run_levante_gpu_scaling.py \
  --grid cubed-sphere \
  --mode strong \
  --precision float32 \
  --n-levels 26 \
  --n-gpus 6 \
  --no-conservation \
  --output-dir "$OUTDIR/atm_mpi_nofix"
```

### 3.3 Atmosphere outputs to inspect

Each run writes:
- `results.json`
- `report.md`
- `logs/*.log`

Key metrics are in `scaling_validation` and `mpi_validation.scaling`:
- `compile_time_s`
- `steady_ms_per_step`
- `throughput_global_mcells_s`
- `throughput_per_device_mcells_s`
- `throughput_per_rank_mcells_s`
- `derived.speedup_vs_baseline`
- `derived.efficiency_vs_baseline`

## 4. Ocean Scaling and MPI Hardware Tests

> **Superseded caveat.** The earlier "no single ocean script equivalent to
> `run_levante_gpu_scaling.py`" note is out of date. Two automated ocean
> scaling drivers now exist with parity + conservation fail-fast gates:
> `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (full-step multi-GPU /
> SPMD, single-process **and** `--multicontroller`) and
> `scripts/bench/bench_ocean_mpi_scaling.py` (CPU-MPI latitude-band, with
> `--wet-balance` and the distributed fixed-M PCG options). The
> matrix-driven workflow below (§4.1–4.3) remains a valid single-process
> runtime-trend + MPI-correctness cross-check. Grid coverage and status per
> ocean grid: `scaling/SCALING_STATUS_AUDIT.md`.

### 4.1 Ocean scaling matrix (single-process runtime/perf trend)

```bash
.venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --output "$OUTDIR/ocean_matrix_cpu" \
  --std-resolutions 8,16 \
  --std-levels 10,20 \
  --std-dt 3600 \
  --std-days 5 \
  --include-realistic \
  --realistic-resolutions 8 \
  --realistic-levels 10 \
  --realistic-dt 1800 \
  --realistic-days 10 \
  --runtime-checks
```

GPU backend variant:

```bash
JAX_PLATFORMS=gpu .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --output "$OUTDIR/ocean_matrix_gpu" \
  --std-resolutions 8,16 \
  --std-levels 10,20 \
  --std-dt 3600 \
  --std-days 5 \
  --runtime-checks
```

This produces per-run wall time in:
- `results.json` -> `runs[*].wall_time_s`

Use `summary.json` from each run for `n_steps` and compute step throughput:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("results")
for p in root.glob("hardware_scaling_*/ocean_matrix_*/results.json"):
    data = json.loads(p.read_text())
    print("\n", p)
    for run in data["runs"]:
        s = run.get("summary") or {}
        n_steps = ((s.get("meta") or {}).get("n_steps") or 0)
        wall = float(run.get("wall_time_s", 0.0))
        tput = (n_steps / wall) if wall > 0 and n_steps > 0 else 0.0
        print(f"  {run['id']}: wall={wall:.2f}s, n_steps={n_steps}, steps/s={tput:.3f}")
PY
```

### 4.2 Ocean MPI distributed correctness and long-run conservation

Halo/reduction + ocean fixer behavior:

```bash
for np in 2 3 6; do
  mpirun -np "$np" .venv/bin/python -m pytest -q tests/distributed/test_halo_mpi.py || exit 1
done
```

Long-run MPI conservation regression:

```bash
mpirun -np 3 .venv/bin/python -m pytest -q tests/ocean/distributed/test_ocean_mpi_conservation.py
```

### 4.3 Ocean MPI timing sweep (manual scaling proxy)

For a quick wall-time scaling proxy on real MPI hardware:

```bash
for np in 1 2 3 6; do
  echo "=== ocean mpi np=${np} ==="
  /usr/bin/time -p mpirun -np "$np" .venv/bin/python -m pytest -q \
    tests/ocean/distributed/test_ocean_mpi_conservation.py
done
```

Record `real` time and compare speedup/efficiency manually.

## 5. MPI Distributed Correctness and Differentiability

### 5.1 Distributed correctness regression

Verifies that multi-rank execution (scatter → step → gather) matches a
single-rank reference within machine precision:

```bash
for np in 2 3 6; do
  mpirun -np "$np" .venv/bin/python -m pytest -q \
    tests/distributed/test_mpi_driver.py::TestMPIDriverPath::test_distributed_3_steps_matches_single_rank
done
```

### 5.2 MPI reverse-mode AD (gradient) tests

Verifies that `jax.grad` flows correctly through MPI halo exchange
(`sendrecv` with `custom_vjp`) and global reductions (`allreduce SUM`):

```bash
for np in 2 3 6; do
  mpirun -np "$np" .venv/bin/python -m pytest -q \
    tests/distributed/test_mpi_differentiability.py
done
```

Key tests:
- `test_global_sum_mpi_grad` — gradient through `allreduce(SUM)`
- `test_pad_halo_mpi_grad_matches_local` — MPI halo gradient matches local
- `test_pad_halo_4d_mpi_grad_matches_local` — 4D variant
- `test_fix_mass_mpi_grad` — conservation fixer gradient through MPI

### 5.3 AD-safety reference

| MPI Primitive | AD Status | Use in Differentiable Path |
|---------------|-----------|---------------------------|
| `allreduce(SUM)` | Differentiable (JVP + VJP) | Conservation fixers, global integrals |
| `sendrecv` | Differentiable via `_sendrecv_vjp` | Halo exchange |
| `allreduce(MAX/MIN)` | NOT differentiable | CFL diagnostics only |
| `allgather` | NOT differentiable | I/O gathering only |
| `bcast` | NOT differentiable | Initialization only |

## 6. Pass/Fail Interpretation

Atmosphere (`run_levante_gpu_scaling.py` defaults):
- compile time threshold: `--scaling-compile-time-max-s` (default 30s)
- strong efficiency threshold: `--strong-min-efficiency` (default 0.12)
- weak step growth threshold: `--weak-max-step-growth` (default 2.5)
- weak throughput threshold: `--weak-min-per-device-throughput-ratio` (default 0.35)

Ocean (`run_ocean_test_matrix.py` defaults):
- heat drift: `<= 1e-3`
- salt drift: `<= 1e-3`
- mean eta drift: `<= 1e-3 m`
- max speed/SSH checks with optional case-specific profiles

Recommended practice for real hardware reports:
- publish raw `results.json` and `report.md`
- keep one run with cold compile and one with warm cache
- include hardware metadata (GPU model, driver/CUDA, MPI version, node count)

## 7. Common Issues

- MPI runs marked `skipped` due launcher/socket policy:
  - use `--mpi-interface`, `--mpi-mca`, and `--mpi-extra-args`
  - verify network interface and container privileges
- `mpi4jax` compatibility warning:
  - tested range is enforced in `legoesm.parallel.reductions`
  - set `LEGOESM_MPI_STRICT_COMPAT=1` to make out-of-range versions fail fast
- No GPU detected:
  - check `JAX_PLATFORMS`, device visibility, and runtime install (`jaxlib`)

## 8. Suggested Report Template

For each machine/cluster:
- software stack: Python/JAX/jaxlib/mpi4py/mpi4jax/OpenMPI versions
- atmosphere scaling: strong/weak tables (1,2,3,6 devices/ranks)
- ocean matrix: per-case wall time and steps/s
- ocean MPI: halo test status and long-run conservation status
- notes on skipped cases and launcher settings used
