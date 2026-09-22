# Derecho scaling: CPU node vs A100, strong scaling

> **For the Nature-figure sweep (strong + weak, f32 + f64, all five grids),
> use `nature_ladder.pbs`** — the PBS port of the Levante campaign ladder
> (`scripts/cluster/scaling_levante/nature_ladder.sbatch`). Same arm table,
> same benches, same receipt schema, so Derecho and Levante rows feed the one
> plotter (`scripts/plot/plot_nature_scaling.py`). Submit one job per matrix
> and device count with the wrapper, which builds the right select string for
> the backend so the long qsub line cannot be mistyped:
> ```bash
> ./scripts/cluster/scaling_derecho/submit_nature_ladder.sh ocean_gpu 1024
> ./scripts/cluster/scaling_derecho/submit_nature_ladder.sh ocean_cpu 1024 12:00:00
> ./scripts/cluster/scaling_derecho/submit_nature_ladder.sh atm_gpu    512
> ```
> Four ranks per node either way, so the node count is the device count over
> four and 1024 devices is 256 nodes. `DRY_RUN=1` prints the qsub line without
> submitting, and `OUTDIR=...` resumes into an existing directory, since a
> valid receipt is never re-run and arms needing more nodes than the
> allocation are skipped.
>
> The processor pool is far larger than the accelerator pool, so a thousand
> ranks is routine while a thousand accelerators may not be grantable at all.
> Check what the queue actually gives rather than assuming the top rung will
> run, and submit the largest allocation you can get.
>
> The raw form still works:
> ```bash
> qsub -l select=32:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB \
>      -v MATRIX=atm_gpu scripts/cluster/scaling_derecho/nature_ladder.pbs
> ```
> The scripts below are the ORIGINAL per-grid CPU-vs-GPU comparison and are
> unchanged.


**The single supported way to test legoESM scaling on NCAR Derecho.** It runs a
strong-scaling sweep on BOTH backends — CPU MPI ranks 1→128 (128-core EPYC,
`main` queue) and GPU A100 — single node (1→4) or multi-node via `NODES`
(latlon/icosahedral, up to 1→2→4→8→16…) — at each resolution, one grid at a
time, then compares the curves.

`submit_scaling.sh` is the only entry point. It fans each resolution out into
its own CPU job + GPU job and dispatches to the right backend scripts; you never
call the building blocks directly.

```
scaling_derecho/
├── _env.sh              # shared env, sourced by every job (edit 2 values, Step 0)
├── submit_scaling.sh   # ►ENTRY POINT◄  submit_scaling.sh <outdir> <grid> [res...]
├── scaling_cpu.sh      # CPU scaling sweep (latlon/ico = ranks 1..128; spectral = 1 x threads)
├── scaling_gpu.sh      # GPU scaling sweep (latlon + ico = 1->2->4 A100 single node, ->8->16… multi-node via NODES; spectral = 1)
├── cube_scaling_cpu.sh # cube CPU scaling, route A (faces 1,2,3,6; mpi4jax scatter)
├── cube_scaling_cpu_routeb.sh # cube CPU scaling, route B (jax.distributed/gloo; --cs-spmd; extends past 6 to 6*kt^2) — #764
├── cube_scaling_gpu.sh   # cube GPU scaling (1->2->3 A100; mpi4jax face-scatter)
└── finalize_scaling.sh # after jobs finish: aggregate <outdir> + per-grid CPU-vs-GPU plots
```

**Why cubed-sphere has its own pair.** Cube has only 6 faces, so MPI caps at 6
ranks — it sweeps the face decomposition (GPU 1→2→3 A100; CPU faces 1,2,3,6 with
each rank multithreaded to fill the node) via the mpi4jax face-scatter path
(`run_levante_gpu_scaling.py --cs-mpi-scatter`). latlon and icosahedral are
genuinely domain-decomposed, so they sweep the full rank/GPU ladder via
`run_cpu_mpi_scaling.py` (`--device cpu`/`--device gpu`); spectral has no MPI
path (single device). `submit_scaling.sh` hides all of this.

---

## Quick start

Already built the two conda envs and edited `_env.sh`? Then a full latlon
comparison is four commands:

```bash
cd /glade/work/$USER/legoESM                       # the repo root (qsub from here)
OUT=$SCRATCH/legoesm_scaling/cmp01                 # pick a results dir

# 1. preview what would be submitted (no jobs created):
DRYRUN=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT latlon

# 2. submit the CPU-sweep + GPU-sweep jobs (one pair per resolution):
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT latlon

# 3. wait for the queue to drain:
watch -n 60 qstat -u $USER

# 4. aggregate every job under $OUT and plot the CPU-vs-GPU curves:
scripts/cluster/scaling_derecho/finalize_scaling.sh $OUT
#    -> $OUT/all_tidy.csv  +  $OUT/plots/cpu_vs_gpu_scaling_latlon.png
```

Repeat step 2 for the other grids into the SAME `$OUT` (then one `finalize`
covers them all):

```bash
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT cubed-sphere
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT icosahedral
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT spectral
```

First time on this machine? Do the **one-time setup** below before the Quick
start, and run the **smoke tests** once to catch a broken env before it burns
batch walltime.

---

# Part A — one-time environment setup

The CPU and GPU sides need **two different conda envs** — GPU JAX and CPU+MPI JAX
are incompatible builds. Build both once, edit `_env.sh` once.

## Step 0 — Edit `_env.sh`

Set the two marked values; the account and `$SCRATCH` are preset:

- `LEGOESM_REPO` → your clone path, e.g. `/glade/work/$USER/legoESM`
- `LEGOESM_CONDA_ENV` → leave default; each job overrides it (`legoesm-gpu` / `legoesm-mpi`).

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

> **The GPU sweeps need Step 1b.** `scaling_gpu.sh` (1→2→4 A100) and
> `cube_scaling_gpu.sh` (1→2→3 A100) exchange halos over mpi4jax across GPUs, so
> the `legoesm-gpu` env MUST carry the route-A overlay (a CUDA-built mpi4jax).
> Build it before submitting any GPU job. (A 1-rank run would not need it, but
> the sweeps always go past 1 GPU.)

### Step 1b — route-A overlay (REQUIRED for the GPU scaling sweeps)

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

`scaling_gpu.sh` and `cube_scaling_gpu.sh` set `MPICH_GPU_SUPPORT_ENABLED=1`,
the `craype-accel-nvidia80` module, and the `LD_LIBRARY_PATH` bridge themselves
at runtime, so once the overlay env is built the jobs carry the right
environment without the manual exports above.

### Slingshot fabric + GPU-direct halo (why a job "runs but doesn't scale")

Two settings the halo exchange needs on Derecho's Slingshot 11 fabric, both
now wired into the job scripts (#681):

1. **`MPI4JAX_USE_CUDA_MPI=1`** (`scaling_gpu.sh`). mpi4jax's *default*
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
   CUDA-aware path from deadlocking on the libfabric MR cache. This is the
   intended fix for the cross-node `cxil_map: write error` / OFI `injectdata`
   abort previously documented in `docs/performance/multinode_gpu_direct_cxi.md`.
   All are `${VAR:-default}` so you can override at `qsub -v`.

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
rather than true comms cost. The GPU sweep is unaffected.

---

# Part B — running a scaling test

## Step 3 — Smoke-test each path (once, before any batch job)

Catches a broken env in 30 s instead of after a 2 h job records garbage.

> When sourcing `_env.sh` interactively, set `JAX_PLATFORMS` yourself first —
> `_env.sh` keeps an already-set value, so a stale `cpu`/`cuda` from earlier in
> the same shell would leak in (and a GPU run on `cpu` fails silently). The
> batch jobs pin it unconditionally, so this only matters interactively. Always
> confirm with `python -c "import jax; print(jax.default_backend())"`.

**GPU** (interactive `deg*` node, env `legoesm-gpu`) — verify multi-GPU route-A
actually spreads across devices (the #1 failure mode):
```bash
qsub -I -A $PROJECT -q main -l job_priority=premium \
     -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB -l walltime=00:30:00
cd /glade/work/$USER/legoESM
export JAX_PLATFORMS=cuda LEGOESM_CONDA_ENV=legoesm-gpu
source scripts/cluster/scaling_derecho/_env.sh
module load craype-accel-nvidia80; export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
PIN='export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"'
mpiexec -n 2 bash -c "$PIN" _ python scripts/bench/run_cpu_mpi_scaling.py \
    --grid latlon --resolution 128 --mode strong --device gpu --latlon-2d --n-timing 10
#   want: "Ranks: 2" and two DISTINCT GPUs used (not both rank-1 / both GPU 0)
```

**CPU-MPI** (interactive `main` node, env `legoesm-mpi`):
```bash
qsub -I -A $PROJECT -q main -l job_priority=premium \
     -l select=1:ncpus=128:mpiprocs=128 -l walltime=00:30:00
cd /glade/work/$USER/legoESM
export JAX_PLATFORMS=cpu LEGOESM_CONDA_ENV=legoesm-mpi
source scripts/cluster/scaling_derecho/_env.sh
module load gcc cray-mpich
mpiexec -n 2 python scripts/bench/run_cpu_mpi_scaling.py \
    --grid latlon --resolution 64 --mode strong --device cpu --n-timing 10
# ^ watch for the STATUS_RETURNING slow-path warning
```

## Step 4 — Submit a campaign (one grid at a time)

`submit_scaling.sh` is the entry point. Run it from the repo root (so each
job's `PBS_O_WORKDIR` resolves). The **outdir is required**; every job writes a
unique subdir under it, so multiple grids can share one `$OUT`.

```bash
submit_scaling.sh <outdir> <grid> [res ...]
#   <outdir>  REQUIRED scratch dir for all results (created if missing)
#   <grid>    cubed-sphere | latlon | icosahedral | spectral
#   [res]     resolutions to cover (default per-grid list if omitted)
```

Always preview first, then submit:

```bash
cd /glade/work/$USER/legoESM
OUT=$SCRATCH/legoesm_scaling/cmp01

DRYRUN=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT latlon   # prints the qsub lines
scripts/cluster/scaling_derecho/submit_scaling.sh        $OUT latlon   # actually submits
```

For each resolution this submits **one CPU-sweep job + one GPU-sweep job** (each
sweeps its device ladder internally). latlon's default `128 256` → 4 jobs.

Cover the other grids into the SAME `$OUT`:
```bash
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT cubed-sphere
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT icosahedral
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT spectral
```

Override resolutions by listing them; pass knobs as environment variables:

```bash
scripts/cluster/scaling_derecho/submit_scaling.sh $OUT cubed-sphere 96 192   # only C96, C192
PHYSICS=moist PRECISION=float64 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT latlon
GPU_ONLY=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT icosahedral # one tier only
```

| Var | Default | Notes |
|-----|---------|-------|
| `PHYSICS` | `none` | `none` (dycore-only) \| `held_suarez` \| `moist` (moisture+Kessler) |
| `PRECISION` | `float32` | `float32` \| `float64` |
| `DRYRUN` | `0` | `1` = print the `qsub` lines without submitting |
| `CPU_ONLY` / `GPU_ONLY` | `0` | submit just one side |
| `STRONG_ONLY` | `0` | `1` = skip the weak-scaling jobs (the CPU-vs-GPU plot uses strong only) |
| `NODES` | `1` | **icosahedral + latlon GPU**: `>1` overrides the GPU job's `select=` to span N nodes (4 A100/node), so the A100 curve goes multi-node (1→2→4→8→16… GPUs, self-capped to `NODES*4`). Rejected for cubed-sphere (≤6-GPU face shard) and spectral (no MPI); see #641/#660. |
| `PBS_ACCOUNT` | `P08010000` | charge account, passed to every `qsub` via `-A` (set once instead of editing each `#PBS -A` header). |

Per-grid default resolutions: cubed-sphere `48 96 192`, latlon `128 256`,
icosahedral `6 7 8` (L8 = 655,362 cells, ~25 km; needs several A100s — pair with
`NODES>1`), spectral `85 170`.

**Multi-node GPU** (latlon / icosahedral) spans nodes via `NODES`. Inter-node
halos are **host-staged**: cross-node GPU-direct currently aborts on Derecho's
CXI fabric (`cxil_map` / OFI `injectdata`; `scaling_gpu.sh` auto-selects
`MPI4JAX_USE_CUDA_MPI=0` for `NODES>1`), so multi-node points are correct but a
**lower bound** on inter-node scaling — footnote them. See
`docs/performance/multinode_gpu_direct_cxi.md`.

> **The mpi4jax-free alternative (cubed-sphere, 2026-07):** the production
> driver now runs true multi-node cubed-sphere via
> `run_amip.py --distributed --distributed-mode spmd` — multi-controller
> `jax.distributed` + NCCL collectives, no mpi4jax anywhere, so the CXI
> GPU-direct abort does not apply (NCCL has its own Slingshot path via
> `aws-ofi-nccl`). Parity receipt: 2-process bit-exact vs single-controller
> over a full day incl. checkpoint writes (jobs 8686550/8687224; gate
> `scripts/validate/validate_driver_cs_spmd_parity.py`). Launch with
> `mpiexec --ppn 4 -n <NODES*4>` (any launcher that sets PMI env);
> diagnostics writer must stay off (`diag_days=0`, milestone). This is the
> path to benchmark AGAINST the host-staged mpi4jax lower bound.

**Run a 2-node canary first** so
any problem surfaces on one cheap job, not the whole sweep:

```bash
# canary: confirm n=8 completes (host-staged) before launching the suite
NODES=2 GPU_ONLY=1 STRONG_ONLY=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT/canary latlon 256

# then the real multi-node sweeps (4 nodes = 16 A100):
NODES=4 GPU_ONLY=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT latlon      128 256 512 720
NODES=4 GPU_ONLY=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT icosahedral 6 7 8
# cubed-sphere + spectral are single-node only (no NODES):
GPU_ONLY=1 scripts/cluster/scaling_derecho/submit_scaling.sh $OUT cubed-sphere 48 96 192 384
```

## Step 5 — Monitor the jobs

```bash
qstat -u $USER                         # your queue (R = running, Q = queued)
qstat -f <jobid> | grep -i comment     # why a job is still queued
ls $OUT                                # per-job result subdirs appear as jobs start
tail -f $OUT/latlon_cpu_res128/run.log # live console of one job (rank counts, SYPD/case)
```

Each job mirrors its console to `<subdir>/run.log`. A job is done when its
`run.log` ends with `=== DONE rc=0 ===` and writes its own `*_tidy.csv`. A
non-zero `rc` means at least one (resolution, device-count) point failed — open
`run.log` to see which (often a resolution too small for the high rank counts;
the rest of the curve is still valid).

## Step 6 — Finalize: aggregate + plot

Once `qstat -u $USER` is empty:

```bash
scripts/cluster/scaling_derecho/finalize_scaling.sh $OUT
```

This runs `aggregate_bcw_scaling.py` over the whole `$OUT` into
`$OUT/all_tidy.csv`, then `plot_cpu_vs_gpu_scaling.py` to render, **for each
grid**, the CPU and GPU strong-scaling curves (SYPD and Mcells/s vs device
count, one panel per resolution) → `$OUT/plots/cpu_vs_gpu_scaling_<grid>.png`,
and prints a peak GPU/CPU speedup table:

```
grid              res   CPU peak   GPU peak  GPU/CPU   (peak SYPD across the scaling curve)
latlon            128       6.96       34.8    5.00x
```

Any conda env with legoESM installed works for this step (no GPU/MPI env needed).
Re-run it any time to refresh after more jobs land. To inspect the raw numbers:

```bash
column -s, -t < $OUT/all_tidy.csv | less -S    # grid, backend, n_resource, resolution, sypd, mcells_per_s, ...
```

---

## Troubleshooting

- **GPU "scaling" curve is flat / every point says `Ranks: 1`** — the route-A
  overlay isn't active (a generic `mpi4py` shadows it). Rebuild Step 1b; confirm
  with the Step 3 GPU smoke test (must show `Ranks: 2`, two distinct GPUs).
- **`mpiexec ... No host list provided`** — you ran `mpiexec` on a login node.
  GPU/MPI runs must be inside a PBS allocation (`qsub -I ...` or a batch job).
- **A CPU `(res, ranks)` point fails** with a "needs ≥2 lat rows" / partition
  error — that resolution is too small for that rank count. Expected; that point
  drops out and the rest of the curve stands. Raise the resolution or cap ranks
  (`RANKS="1 2 4 8 16 32"`).
- **`cube_scaling_cpu.sh` rejects `--cpu-bind depth --depth N`** — a PALS
  version quirk; the fallback is `--cpu-bind depth -d N` (edit the `mpiexec`
  line).
- **Empty plot / `no CPU/GPU rows ...`** — `finalize` ran before any job
  finished, or `$OUT` is wrong. Wait for `run.log` `DONE` lines, re-run finalize.

## Verify-before-submit checklist
- [ ] `_env.sh`: `LEGOESM_REPO` set; conda envs named `legoesm-gpu` / `legoesm-mpi`.
- [ ] GPU env: `jax.default_backend()` == `gpu`, `device_kind` shows A100, and the
      Step 3 GPU smoke test shows **2 distinct GPUs** at `-n 2`.
- [ ] MPI env: `import mpi4jax` succeeds (built under **gcc**, against cray-mpich).
- [ ] GPU header tokens (`gpu_type=a100`, `job_priority`) schedule on your
      allocation — confirm with the interactive `qsub -I` in Step 1; adjust the
      `select=`/`gpu_type` line in `scaling_gpu.sh` / `cube_scaling_gpu.sh` if not.
- [ ] `DRYRUN=1` preview looks right before the real submit.

---

## Ocean + multi-node GPU additions (2026-07)

Alongside the cube pair above, three further jobs + a build script:

| File | What |
|---|---|
| `ocean_gpu_scaling.pbs` | OCEAN weak+strong on one GPU node via `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (full lat-lon C-grid step sharded over 1/2/4 A100; fail-fast `--parity-gate` + `--check-conservation` smoke first). Plain GPU env — no mpi4jax. |
| `ocean_cpu_scaling.pbs` | OCEAN weak+strong CPU-MPI rank ladder (`bench_ocean_mpi_scaling.py`, `legoesm-mpi` env), with a 2-rank parity+conservation smoke. |
| `gpu_multinode_scaling.pbs` | MULTI-NODE GPU lanes over jax.distributed + NCCL: A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU); plus the optional route-A CUDA-aware mpi4jax lane (`RUN_ROUTEA=1`, needs the overlay env) and the comm-tuning A/B ladder (`RUN_TUNE=1`, lane T below). |
| `build_nccl_ofi.sh` | Login-node build of **aws-ofi-nccl** against Derecho's Cray libfabric (no NCCL build dep — the plugin vendors the net-API headers and is dlopen'd by the jax-wheel NCCL). |
| `diagnosis.pbs` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) that the throughput jobs above do NOT capture. Climbs the cube face-shard `1 2 3` ladder (must divide 6) so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `qsub -v MODE=census` for counts only. |

NCCL on Slingshot-11 has NO native CXI support: without the plugin the
multi-node lanes fall back to TCP sockets over `hsn` (correct, 2-3x slower
comm — loud warning, fine for shakeout). For production numbers:

```bash
bash scripts/cluster/scaling_derecho/build_nccl_ofi.sh
qsub -v LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib \
     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
```

Verify the first run's `NCCL_DEBUG=INFO` log prints
`Using network AWS Libfabric` (not `Socket`). The NCCL lanes keep
`MPICH_GPU_SUPPORT_ENABLED=0` (mixing GPU-aware cray-mpich and NCCL in one
program risks deadlock); the route-A lane sets it to 1 — the two transports
never share a process.


## 2026-07 lane E: icosahedral/MPAS multicontroller

`gpu_multinode_scaling.pbs` gained lane E (`RUN_MPAS=1`, default on): the
icosahedral MPAS PE dycore over `jax.distributed` + NCCL via
`scripts/bench/bench_mpas_spmd_scaling.py` — cell-partition reorder
(`reorder_voronoi_for_sharding`, Hilbert-SFC pinned for cross-process
determinism) + `make_voronoi_sharded_step` ppermute halos. 6 processes
(2 nodes x 3 GPUs): `nCells = 10*4^L + 2` admits 1/2/3/6 even splits at
every level. A subdiv-4 smoke with `--parity-gate --check-conservation`
runs before the timed `ICO_LEVEL` (default L7 = 163842 cells, ~27k
cells/GPU at np=6) case. 2-process CPU federation gate:
`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.


## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)

Once the route-B lanes are green on this machine, the remaining strong-
scaling headroom at small tiles is **per-step message count × latency**
(census: cube 46 collective-permutes/step at 6 devices with field packing
already at floor; atm latlon 41 → 29 behind the fused-halo flag — see
`docs/performance/scaling/spmd_message_census_2026-07-08.md`). Lane T runs
the ranked rungs as same-allocation A/B arms (a fresh `base` control arm is
re-run in the same job — never compare against an earlier job's numbers):

1. `fused` — `LEGOESM_LATLON_SPMD_FUSED_HALO=1`.  **MEASURED 2026-07-09
   (8×A100): latlon −12 %, ocean +1 % (noise) — the default stays OFF**
   (bigger messages + pack/unpack copies outweigh the −29 % message count
   at LL512/np8; re-A/B at higher rank counts before discarding).
2. `xla` — CP-combine 32 MiB + pipelined p2p.  **MEASURED: −10 % — not
   recommended combined; split the two flags in a follow-up arm.**
3. `pgle` — `JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3`.  **MEASURED:
   +8.5 % (5.97 vs 6.48 ms/step) — the winner; recommend per-run on
   route-B latlon lanes.**  Stays per-run opt-in (recompiles after the
   profiling runs — AOT-incompatible, never a `backend.py` default).
4. If still send/recv-bound: sweep `NCCL_NCHANNELS_PER_NET_PEER` 4→8/16.
   Full numbers: `docs/performance/scaling/spmd_message_census_2026-07-08.md`.

```bash
qsub -v RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0 \
     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
```

Outputs land under `$OUTDIR/_ab_tuning/` — a path the tidy-CSV aggregator
deliberately skips, so A/B receipts never contaminate the scaling curves;
read the per-arm `steady_median_ms` / `sypd` straight from the JSONL rows.
