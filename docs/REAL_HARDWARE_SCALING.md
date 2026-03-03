# Real MPI and Multi-GPU Scaling Tests (Atmosphere + Ocean)

This runbook describes how to run real-hardware scaling tests for legoESM
atmosphere and ocean components on CPUs/GPUs and MPI clusters.

It is focused on commands that already exist in this repository:
- `scripts/run_parallel_validation.py` (atmosphere scaling + MPI validation)
- `scripts/run_ocean_regression_matrix.py` (ocean runtime/regression matrix)
- `tests/distributed/test_halo_mpi.py` and `tests/distributed/test_ocean_mpi_conservation.py` (MPI ocean/distributed checks)

## 1. Prerequisites

From repo root:

```bash
cd /Users/pierregentine/legoESM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install mpi4py mpi4jax
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

Use `scripts/run_parallel_validation.py` with `--scaling-workload atmosphere_sw`.
This includes:
- compile time
- steady-state time per step
- global/per-device/per-rank throughput
- strong and weak scaling checks with thresholds

### 3.1 Single-node multi-GPU (or multi-CPU) scaling

```bash
JAX_PLATFORMS=gpu .venv/bin/python scripts/run_parallel_validation.py \
  --output "$OUTDIR/atm_gpu" \
  --scaling-workload atmosphere_sw \
  --scaling-backends gpu \
  --scaling-gpu-devices 1,2,3,6 \
  --scaling-strong-grid 256 \
  --scaling-weak-base-grid 256 \
  --scaling-iterations 20 \
  --scaling-warmup 3 \
  --scaling-dt 300 \
  --timeout-sec 3600
```

CPU comparison:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/run_parallel_validation.py \
  --output "$OUTDIR/atm_cpu" \
  --scaling-workload atmosphere_sw \
  --scaling-backends cpu \
  --scaling-cpu-devices 1,2,3,6 \
  --scaling-strong-grid 256 \
  --scaling-weak-base-grid 256
```

### 3.2 MPI rank scaling (real cluster)

```bash
.venv/bin/python scripts/run_parallel_validation.py \
  --output "$OUTDIR/atm_mpi" \
  --mpi-ranks 2,3,6 \
  --scaling-workload atmosphere_sw \
  --mpi-scaling \
  --mpi-scaling-strong-grid 96 \
  --mpi-scaling-weak-base-grid 96 \
  --mpi-scaling-iterations 12 \
  --mpi-scaling-warmup 2
```

Portability knobs for restricted environments:

```bash
.venv/bin/python scripts/run_parallel_validation.py \
  --output "$OUTDIR/atm_mpi_portable" \
  --mpi-launcher mpirun \
  --mpi-ranks 2,3,6 \
  --mpi-interface eth0 \
  --mpi-mca "btl=self,tcp;pml=ob1" \
  --mpi-env OMPI_MCA_btl_tcp_if_include=eth0 \
  --mpi-extra-args "--bind-to none --map-by slot"
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

There is currently no single ocean script equivalent to atmosphere
`run_parallel_validation.py` for automated strong/weak MPI+multi-GPU gating.
Use the workflow below to cover real ocean hardware behavior.

### 4.1 Ocean scaling matrix (single-process runtime/perf trend)

```bash
.venv/bin/python scripts/run_ocean_regression_matrix.py \
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
JAX_PLATFORMS=gpu .venv/bin/python scripts/run_ocean_regression_matrix.py \
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
mpirun -np 3 .venv/bin/python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py
```

### 4.3 Ocean MPI timing sweep (manual scaling proxy)

For a quick wall-time scaling proxy on real MPI hardware:

```bash
for np in 1 2 3 6; do
  echo "=== ocean mpi np=${np} ==="
  /usr/bin/time -p mpirun -np "$np" .venv/bin/python -m pytest -q \
    tests/distributed/test_ocean_mpi_conservation.py
done
```

Record `real` time and compare speedup/efficiency manually.

## 5. Pass/Fail Interpretation

Atmosphere (`run_parallel_validation.py` defaults):
- compile time threshold: `--scaling-compile-time-max-s` (default 30s)
- strong efficiency threshold: `--strong-min-efficiency` (default 0.12)
- weak step growth threshold: `--weak-max-step-growth` (default 2.5)
- weak throughput threshold: `--weak-min-per-device-throughput-ratio` (default 0.35)

Ocean (`run_ocean_regression_matrix.py` defaults):
- heat drift: `<= 1e-3`
- salt drift: `<= 1e-3`
- mean eta drift: `<= 1e-3 m`
- max speed/SSH checks with optional case-specific profiles

Recommended practice for real hardware reports:
- publish raw `results.json` and `report.md`
- keep one run with cold compile and one with warm cache
- include hardware metadata (GPU model, driver/CUDA, MPI version, node count)

## 6. Common Issues

- MPI runs marked `skipped` due launcher/socket policy:
  - use `--mpi-interface`, `--mpi-mca`, and `--mpi-extra-args`
  - verify network interface and container privileges
- `mpi4jax` compatibility warning:
  - tested range is enforced in `legoesm.parallel.reductions`
  - set `LEGOESM_MPI_STRICT_COMPAT=1` to make out-of-range versions fail fast
- No GPU detected:
  - check `JAX_PLATFORMS`, device visibility, and runtime install (`jaxlib`)

## 7. Suggested Report Template

For each machine/cluster:
- software stack: Python/JAX/jaxlib/mpi4py/mpi4jax/OpenMPI versions
- atmosphere scaling: strong/weak tables (1,2,3,6 devices/ranks)
- ocean matrix: per-case wall time and steps/s
- ocean MPI: halo test status and long-run conservation status
- notes on skipped cases and launcher settings used
