# Derecho weak/strong scaling jobs

AMIP-representative weak+strong scaling of legoESM on **NCAR Derecho**, for both
GPU (`deg*` nodes: 4× A100-40GB) and CPU-MPI (`main` queue: 128-core EPYC nodes).

```
scaling_derecho/
├── _env.sh              # shared env, sourced by all jobs (edit 2 values, see below)
├── gpu_scaling.pbs      # GPU weak+strong (gpu nodes; single-process multi-GPU sharding; cubed-sphere AMIP physics)
├── gpu_moist_scaling.pbs # GPU weak+strong, route-A mpi4jax (1 GPU/rank; latlon/icosahedral + --physics moist Kessler)
└── cpu_scaling.pbs      # CPU weak+strong over MPI ranks (queue main; latlon/icosahedral)
```

`gpu_moist_scaling.pbs` is the GPU twin of `cpu_scaling.pbs` (same
`run_cpu_mpi_scaling.py --physics moist` Kessler path), not of `gpu_scaling.pbs`.
It runs the new all-grids moist baroclinic wave on the two genuinely
GPU-decomposed grids — `latlon` and `icosahedral` — with one MPI rank per A100
over the mpi4jax halo. It needs the **`legoesm-gpu`** env to also carry a
CUDA-aware mpi4jax (the route-A overlay), and it pins `CUDA_VISIBLE_DEVICES` per
rank from the Cray-PALS local rank id (Derecho's `mpiexec` does not export the
OpenMPI/SLURM vars the auto-pin looks for). Cubed-sphere/spectral are rejected
up front (cube moist on GPU is covered by `gpu_scaling.pbs`; spectral has no MPI
path).

The two jobs need **two different conda envs** — GPU JAX and CPU+MPI JAX are
incompatible builds. Build both once (Steps 1–2), edit `_env.sh` (Step 0), then
submit (Step 4).

---

## Step 0 — Edit `_env.sh`

Set the two marked values; the account and `$SCRATCH` are preset:

- `LEGOESM_REPO` → your clone path, e.g. `/glade/work/$USER/legoESM`
- `LEGOESM_CONDA_ENV` → leave default; each job overrides it (`legoesm-gpu` / `legoesm-mpi`).

---

## Installing legoESM — NOT `pip install -e .`

legoESM is a **uv workspace** of federated members (`legoesm-core`,
`legoesm-atmosphere`, …). A bare `pip install -e ".[dev]"` fails with
`No matching distribution found for legoesm-atmosphere~=0.1.0` because the
members are not on PyPI. Use one of:

```bash
# plain pip (works inside a conda env — resolves the member DAG locally):
python scripts/experiment/install_federation.py --all --extras dev

# or, uv (creates its own .venv, skips conda):
python -m venv .venv && source .venv/bin/activate && uv sync --extra dev
```

The steps below use the `install_federation.py` path so everything lands in the
active conda env.

---

## Step 1 — GPU env (`legoesm-gpu`) for the `deg*` nodes

Build it in an interactive GPU session so the CUDA build sees a real A100:

```bash
qsub -I -A $PROJECT -q main -l job_priority=premium \
     -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB -l walltime=00:40:00
# on the deg* node:
module load conda cuda
conda create -n legoesm-gpu python=3.11 -y && conda activate legoesm-gpu
cd /glade/work/$USER/legoESM
python scripts/experiment/install_federation.py --all --extras dev
pip install "jax[cuda12]==0.9.2"          # CUDA wheels, pinned to the tested envelope

# sanity — backend must be 'gpu':
python -c "import jax; print(jax.default_backend(), jax.devices())"
#   -> gpu [CudaDevice(id=0), ...]   (JAX never prints the model name; this is correct)
python -c "import jax; print(jax.devices()[0].device_kind)"   # -> 'NVIDIA A100-...'
```

> **Why JAX 0.9.2?** Keeps both envs in the tested `jax 0.8–0.9` envelope and
> aligned with the MPI env (see the mpi4jax constraint below).

> **Two different GPU jobs, two different requirements.** `gpu_scaling.pbs`
> (cubed-sphere AMIP physics) shards a *single process* across the node's GPUs
> via JAX's device mesh — it does **not** use mpi4jax, so the steps above are
> enough for it. `gpu_moist_scaling.pbs` (latlon/icosahedral moist) is
> **multi-process, one rank per GPU over mpi4jax** (route-A) and needs the
> overlay below. Skip Step 1b if you only run `gpu_scaling.pbs`.

### Step 1b — route-A overlay (only for `gpu_moist_scaling.pbs`)

The route-A GPU path launches `mpiexec -n N` (one rank per A100) and exchanges
halos over mpi4jax — so `legoesm-gpu` ALSO needs a Cray-MPICH-built `mpi4py`
plus a **CUDA-built** `mpi4jax`, exactly like the CPU env in Step 2. The
`jax[cuda12]` wheel does NOT bring these. If a generic/conda `mpi4py` is present
it ignores Derecho's Cray PALS launcher, so `mpiexec -n 2` silently runs **two
independent size-1 jobs** (each prints `Ranks: 1`, both overwrite the same
`_n1_` result) — you get rank-1 points only, never a scaling curve. Build the
overlay INTO the GPU env. The order below is battle-tested; the footguns
(annotated) are real and each cost a debugging round.

```bash
conda activate legoesm-gpu
module load gcc cray-mpich cuda    # cuda is REQUIRED so mpi4jax finds nvcc & builds its GPU ext
cc --version                       # ncarcompilers wrapper around gcc 14.x (NOT Intel icx) -- correct on Derecho

# (1) PURGE any generic/conda mpi4py first. A conda-forge mpi4py ships dual
#     'MPI.mpich.*' + 'MPI.openmpi.*' .so files that want a stock libmpi.so.12 /
#     libmpi.so.40 (absent on Derecho); with it present, `pip --no-binary` sees
#     "already satisfied" and NEVER rebuilds -> import fails "libmpi.so.12: cannot
#     open shared object file". Remove from BOTH managers:
pip uninstall -y mpi4py mpi4jax
conda remove -y --force mpi4py mpich openmpi 2>/dev/null || true

# (2) source-build mpi4py against Cray MPICH -> ONE 'MPI.cpython-311-*.so'
MPICC=cc pip install --no-cache-dir --no-binary mpi4py "mpi4py>=4.1,<5"

# (3) build mpi4jax WITH CUDA and --no-deps. --no-deps is CRITICAL: without it
#     pip pulls jax 0.10.x, which (a) orphans the jax_cuda12_plugin 0.9.2 -> GPU
#     silently DISABLED, and (b) breaks mpi4jax 0.8.x ("cannot import get_aval").
MPICC=cc CUDA_ROOT="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}" \
    pip install --no-deps --no-cache-dir --no-binary mpi4jax "mpi4jax==0.9.0"

# (4) PIN jax back into the tested envelope (matches legoesm-mpi: jax/jaxlib
#     0.9.2). This realigns jaxlib + jax_cuda12_plugin to 0.9.2 (re-enables the
#     GPU) and keeps mpi4jax importable (get_aval exists in 0.9.2, gone in 0.10).
pip install --force-reinstall "jax[cuda12]==0.9.2"

# (5) loader path: the Cray module exports CRAY_LD_LIBRARY_PATH, NOT
#     LD_LIBRARY_PATH, so mpi4py can't find Cray's libmpi without this bridge.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

# (6) verify import (single .so, resolves, CRAY MPICH, gpu backend, mpi4jax OK):
ls $CONDA_PREFIX/lib/python3.11/site-packages/mpi4py/MPI*.so      # exactly ONE, no .mpich/.openmpi
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # -> CRAY MPICH ...
python -c "import jax; print(jax.default_backend(), jax.__version__)"  # -> gpu 0.9.2
python -c "import mpi4jax; print('mpi4jax OK')"                        # no get_aval error
```

Final check — federation AND GPU together, **inside a PBS allocation** (a
login-node `mpiexec` errors with "No host list provided"):

```bash
module load craype-accel-nvidia80          # CUDA GTL for GPU-aware sends
export MPICH_GPU_SUPPORT_ENABLED=1
mpiexec -n 2 python -c "from mpi4py import MPI; import jax; \
    print('rank', MPI.COMM_WORLD.Get_rank(), 'of', MPI.COMM_WORLD.Get_size(), jax.default_backend())"
#   want: two lines, size 2, backend gpu  ==  route-A overlay fully working
```

`gpu_moist_scaling.pbs` sets `MPICH_GPU_SUPPORT_ENABLED=1`, the
`craype-accel-nvidia80` module, and the `LD_LIBRARY_PATH` bridge itself at
runtime, so once the env is built the submitted job carries the right
environment without any of the manual exports above.

### Slingshot fabric + GPU-direct halo (why a job "runs but doesn't scale")

Two settings the halo exchange needs on Derecho's Slingshot 11 fabric, both
now wired into the job scripts:

1. **`MPI4JAX_USE_CUDA_MPI=1`** (`gpu_moist_scaling.pbs`). mpi4jax's *default*
   is to copy each `sendrecv` buffer device→host→device. On the GPU route-A
   path that host round-trip per exchange **erases multi-GPU scaling** even
   though the run is numerically correct — the classic "halo works but doesn't
   speed up" symptom. Setting this hands the on-device buffer straight to
   GPU-aware cray-mpich. It requires an mpi4jax built **with** CUDA support
   (Step 1b); if that's missing, `require_mpi_stack()` now raises a clear
   `ImportError` at startup instead of segfaulting on the first exchange. With
   a CUDA-capable mpi4jax but the var unset, you get a one-time `RuntimeWarning`
   telling you the halo is silently host-staging.

2. **`FI_CXI_RX_MATCH_MODE=hybrid`** + `FI_CXI_DEFAULT_CQ_SIZE=131072` +
   `FI_CXI_DISABLE_HOST_REGISTER=1` (`_env.sh`, so CPU and GPU jobs alike). The
   nearest-neighbour halo posts many small messages; Slingshot's Cassini NIC
   offloads tag matching to hardware and, at scale, **aborts** when the match
   (LE) pool fills — `"LE resources not recovered during flow control.
   FI_CXI_RX_MATCH_MODE=[hybrid|software] is required"`. `hybrid` falls back to
   software matching instead of killing the job; the larger CQ absorbs the
   many concurrent completions; `FI_CXI_DISABLE_HOST_REGISTER=1` keeps the
   CUDA-aware path from deadlocking on the libfabric MR cache. All are
   `${VAR:-default}` so you can override at `qsub -v`.

---

## Step 2 — CPU + MPI env (`legoesm-mpi`) for the `main` queue

`mpi4py`/`mpi4jax` **must be built from source against Cray MPICH**, and **with
the GNU compiler, not Intel**. Derecho's default `cc` is the Intel oneAPI
compiler (`icx`); mpi4jax then tries to build its Intel-GPU (SYCL/XPU) backend
and fails with `fatal error: 'sycl/CL/sycl.hpp' file not found`. GCC has no SYCL
path, so it builds only the CPU bridge we want.

```bash
module load conda
conda create -n legoesm-mpi python=3.11 -y && conda activate legoesm-mpi
cd /glade/work/$USER/legoESM
python scripts/experiment/install_federation.py --all --extras dev

# pin JAX into the mpi4jax-compatible envelope (CPU jaxlib):
pip install jax==0.9.2 jaxlib==0.9.2 "numpy>=2.1,<2.3"

# --- the critical part: build MPI bindings under GCC, against Cray MPICH ---
module load gcc           # swap the compiler wrapper from intel -> gcc
module load cray-mpich    # re-point cray-mpich to the GNU build
cc --version              # MUST show gcc, NOT 'Intel ... icx'
module list               # confirm: gcc + cray-mpich loaded
# (if cc still says icx:  module unload intel-oneapi-compilers  then reload gcc cray-mpich)

MPICC=cc pip install --no-cache-dir --force-reinstall --no-binary mpi4py  "mpi4py>=4.1,<5"
MPICC=cc pip install --no-cache-dir                  --no-binary mpi4jax "mpi4jax==0.9.0"

python -c "import mpi4py, mpi4jax; print('mpi4jax', mpi4jax.__version__)"
```

Notes:
- `cc --version` showing **gcc** is the whole fix. If it still says `icx`, the
  module swap didn't take — unload Intel first (`module unload intel-oneapi-compilers`).
- Rebuild `mpi4py` too (`--force-reinstall`) so both extensions share one
  compiler/MPI ABI; if you built `mpi4py` earlier under Intel, redo it under gcc.
- If `gcc` isn't the module name, run `module avail gcc` and use what it lists
  (e.g. `gcc/12.2.0`).

### ⚠ mpi4jax CPU slow-path caveat
Per `requirements_mpi.txt`, mpi4jax 0.8.x on JAX 0.8–0.9 routes every
`sendrecv`/`allreduce` through a legacy XLA custom-call slow path
(`STATUS_RETURNING-not-supported`), with little/no speedup measured at np=2 —
pending the mpi4jax FFI rewrite. Run the np=2 smoke test below and **watch for
that warning**: if it fires, the CPU-MPI scaling curve reflects mpi4jax overhead
rather than true comms cost. The GPU sweep is unaffected — run it first.

---

## Step 3 — Smoke-test each path interactively (before batch)

> When sourcing `_env.sh` interactively, set `JAX_PLATFORMS` yourself first —
> `_env.sh` keeps an already-set value, so a stale `cpu`/`cuda` from earlier in
> the same shell would leak in (and a GPU run on `cpu` fails silently). The
> batch `.pbs` jobs pin it unconditionally, so this only matters interactively.
> Always confirm with `python -c "import jax; print(jax.default_backend())"`.

**GPU** (on a `deg*` node, env `legoesm-gpu`):
```bash
export JAX_PLATFORMS=cuda
export LEGOESM_CONDA_ENV=legoesm-gpu
source scripts/cluster/scaling_derecho/_env.sh
python -c "import jax; print(jax.default_backend())"      # MUST print: gpu
python scripts/bench/run_levante_gpu_scaling.py \
    --grid cubed-sphere --physics gray_sbm --mode strong --precision float32 \
    --n-gpus 1 --n-timing 10
```

**CPU-MPI** (interactive `main` node, env `legoesm-mpi`):
```bash
qsub -I -A $PROJECT -q main -l job_priority=premium \
     -l select=1:ncpus=128:mpiprocs=128 -l walltime=00:30:00
export JAX_PLATFORMS=cpu
export LEGOESM_CONDA_ENV=legoesm-mpi
source scripts/cluster/scaling_derecho/_env.sh
mpiexec -n 2 python scripts/bench/run_cpu_mpi_scaling.py \
    --grid latlon --resolution 64 --mode single --physics held_suarez --n-timing 10
# ^ watch for the STATUS_RETURNING slow-path warning
```

---

## Step 4 — Submit the full sweeps

```bash
qsub scripts/cluster/scaling_derecho/gpu_scaling.pbs        # GPU weak+strong (1->2->4 A100; cubed-sphere AMIP)
qsub scripts/cluster/scaling_derecho/gpu_moist_scaling.pbs   # GPU weak+strong moist (latlon/icosahedral; 1 GPU/rank)
qsub scripts/cluster/scaling_derecho/cpu_scaling.pbs         # CPU weak+strong (ranks 1,2,4,...)
```

Override knobs without editing files (`qsub -v NAME=value,...`):

| Var | GPU job default | CPU job default | Notes |
|-----|-----------------|-----------------|-------|
| `GRID` | `cubed-sphere` | `latlon` | CPU also: `icosahedral`. cubed-sphere/spectral are rank-1 only on CPU. |
| `PHYSICS` | `gray_sbm` | `held_suarez` | GPU: `rrtmg_full` (needs RRTMGP data). CPU: `moist` (AMIP-like; moisture+Kessler). |
| `MODE` | `both` | `both` | `weak` \| `strong` \| `both` |
| `PRECISION` | `float32` | `float64` | |
| `MAX_RANKS` | — | `64` | CPU only; powers of 2, ≤128 per node. >128 ⇒ multi-node `select=`. |
| `N_GPUS` | `0` (auto) | — | GPU only; 0 = use all PBS-allocated GPUs. |
| `STRONG_RES` | `48,96,192,384` | — | GPU only; cubed-sphere strong-scaling resolutions (cN/face-edge). Drop `48` for a cleaner curve. |

Examples:
```bash
qsub -v PHYSICS=rrtmg_full,MODE=strong,PRECISION=both scripts/cluster/scaling_derecho/gpu_scaling.pbs
qsub -v GRID=icosahedral,PRECISION=float64 scripts/cluster/scaling_derecho/gpu_moist_scaling.pbs
qsub -v PHYSICS=moist,MAX_RANKS=128 scripts/cluster/scaling_derecho/cpu_scaling.pbs
qsub -v GRID=icosahedral scripts/cluster/scaling_derecho/cpu_scaling.pbs
```

---

## Results

Each run writes a timestamped dir under `$SCRATCH/legoesm_scaling/` containing
per-case JSON (with `backend`, `n_cores`, `sypd`, `time_per_step_ms`, …) plus
the GPU job's weak/strong PNG plots.

```bash
qstat -u $USER
ls $SCRATCH/legoesm_scaling/
```

---

## Verify-before-submit checklist
- [ ] `_env.sh`: `LEGOESM_REPO` set; conda envs named `legoesm-gpu` / `legoesm-mpi`.
- [ ] GPU env: `jax.default_backend()` == `gpu`, `device_kind` shows A100.
- [ ] MPI env: `import mpi4jax` succeeds (built under **gcc**, against cray-mpich).
- [ ] GPU header tokens (`gpu_type=a100`, `job_priority=premium`) schedule on your
      allocation — confirm with the interactive `qsub -I` in Step 1, adjust the
      `select=`/`gpu_type` line if not.
