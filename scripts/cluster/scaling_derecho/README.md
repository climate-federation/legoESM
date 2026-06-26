# Derecho scaling: CPU node vs A100, strong scaling

**The single supported way to test legoESM scaling on NCAR Derecho.** It runs a
strong-scaling sweep on BOTH backends — CPU MPI ranks 1→128 (128-core EPYC,
`main` queue) and GPU 1→4 A100 (`deg*` node) — at each resolution, one grid at a
time, then compares the curves (headline = the full-node point on each side).

`submit_fullnode.sh` is the only entry point — it fans each resolution out into
its own CPU job + GPU job and dispatches to the right backend scripts; you never
call the building blocks directly.

```
scaling_derecho/
├── _env.sh              # shared env, sourced by every job (edit 2 values, Step 0)
├── submit_fullnode.sh   # ►ENTRY POINT◄  submit_fullnode.sh <outdir> <grid> [res...]
├── fullnode_cpu.sh      # CPU scaling sweep (latlon/ico = ranks 1..128; spectral = 1 x threads)
├── fullnode_gpu.sh      # GPU scaling sweep (latlon/ico = 1->2->4 A100, route-A; spectral = 1)
├── cube_fullnode_cpu.sh # cube CPU scaling (faces 1,2,3,6 x node-filling threads)
├── cube_strong_gpu.sh   # cube GPU scaling (1->2->3 A100; mpi4jax face-scatter)
└── finalize_fullnode.sh # after jobs finish: aggregate <outdir> + per-grid CPU-vs-GPU plots
```

**Why cubed-sphere has its own pair.** Cube has only 6 faces, so MPI caps at 6
ranks — it sweeps the face decomposition (GPU 1→2→3 A100; CPU faces 1,2,3,6 with
each rank multithreaded to fill the node) via the mpi4jax face-scatter path
(`run_levante_gpu_scaling.py --cs-mpi-scatter`). latlon and icosahedral are
genuinely domain-decomposed, so they sweep the full rank/GPU ladder via
`run_cpu_mpi_scaling.py` (`--device cpu`/`--device gpu`); spectral has no MPI
path (single device). `submit_fullnode.sh` hides all of this.

The CPU and GPU sides need **two different conda envs** — GPU JAX and CPU+MPI JAX
are incompatible builds. Build both once (Steps 1–2), edit `_env.sh` (Step 0),
then submit (Step 4).

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

> **The GPU sweeps need Step 1b.** `fullnode_gpu.sh` (1→2→4 A100) and
> `cube_strong_gpu.sh` (1→2→3 A100) exchange halos over mpi4jax across GPUs, so
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
    pip install --no-deps --no-cache-dir --no-binary mpi4jax "mpi4jax==0.8.1.post2"

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

`fullnode_gpu.sh` and `cube_strong_gpu.sh` set `MPICH_GPU_SUPPORT_ENABLED=1`,
the `craype-accel-nvidia80` module, and the `LD_LIBRARY_PATH` bridge themselves
at runtime, so once the overlay env is built the jobs carry the right
environment without the manual exports above.

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
MPICC=cc pip install --no-cache-dir                  --no-binary mpi4jax "mpi4jax==0.8.1.post2"

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

## Step 4 — Submit (one grid at a time)

`submit_fullnode.sh` is the only entry point. Run it from anywhere — it `cd`s to
the repo root so each job's `PBS_O_WORKDIR` resolves. The **outdir is required**;
each job writes a unique subdir under it.

```bash
submit_fullnode.sh <outdir> <grid> [res ...]
#   <outdir>  REQUIRED scratch dir for all results (created if missing)
#   <grid>    cubed-sphere | latlon | icosahedral | spectral
#   [res]     resolutions to cover (default per-grid list if omitted)

OUT=$SCRATCH/legoesm_scaling/cmp01
scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT latlon            # default res (128 256)
scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT cubed-sphere 48 96 192
scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT icosahedral
scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT spectral
```

For each resolution it submits **one CPU-sweep job + one GPU-sweep job** (each
sweeps its device ladder internally), so a slow high-res case gets its own
walltime and runs in parallel.

Knobs (environment, passed through to the jobs):

| Var | Default | Notes |
|-----|---------|-------|
| `PHYSICS` | `none` | `none` (dycore-only) \| `held_suarez` \| `moist` (moisture+Kessler) |
| `PRECISION` | `float32` | `float32` \| `float64` |
| `DRYRUN` | `0` | `1` = print the `qsub` lines without submitting |
| `CPU_ONLY` / `GPU_ONLY` | `0` | submit just one side |

```bash
DRYRUN=1 scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT latlon   # preview first
PHYSICS=moist PRECISION=float64 scripts/cluster/scaling_derecho/submit_fullnode.sh $OUT icosahedral
```

Per-grid default resolutions: cubed-sphere `48 96 192`, latlon `128 256`,
icosahedral `6 7`, spectral `85 170`. Override by listing resolutions as args.

---

## Results — aggregate + plot

Each job writes its subdir under `<outdir>` (`<grid>_cpu_res<R>/`,
`<grid>_a100_res<R>/`) with per-case JSON and its own tidy CSV. Once the queue
is empty, finalize the whole campaign in one step:

```bash
qstat -u $USER                                              # wait until empty
scripts/cluster/scaling_derecho/finalize_fullnode.sh $OUT   # aggregate + plot
```

This runs `aggregate_bcw_scaling.py` over the whole `<outdir>` into
`$OUT/all_tidy.csv`, then `plot_fullnode_cpu_vs_gpu.py` to render, **for each
grid**, the CPU and GPU strong-scaling curves (SYPD and Mcells/s vs device
count) one panel per resolution — `$OUT/plots/fullnode_cpu_vs_gpu_<grid>.png` —
and prints a peak (full-node) GPU/CPU speedup table. Any conda env with legoESM
installed works (no GPU/MPI env needed for this step).

---

## Verify-before-submit checklist
- [ ] `_env.sh`: `LEGOESM_REPO` set; conda envs named `legoesm-gpu` / `legoesm-mpi`.
- [ ] GPU env: `jax.default_backend()` == `gpu`, `device_kind` shows A100.
- [ ] MPI env: `import mpi4jax` succeeds (built under **gcc**, against cray-mpich).
- [ ] GPU header tokens (`gpu_type=a100`, `job_priority=premium`) schedule on your
      allocation — confirm with the interactive `qsub -I` in Step 1, adjust the
      `select=`/`gpu_type` line if not.
