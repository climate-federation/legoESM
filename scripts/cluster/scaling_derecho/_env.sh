# Shared environment for NCAR Derecho scaling jobs (sourced by PBS scripts).
# ---------------------------------------------------------------------------
# EDIT the three marked values for your account/paths before first submit.
# Everything here can also be overridden from the qsub environment, e.g.
#   qsub -v LEGOESM_CONDA_ENV=my-jax-env scaling_derecho/scaling_gpu.sh
# ---------------------------------------------------------------------------

# --- (1) Project allocation (matches the PBS -A directive in the job scripts) -
export PBS_ACCOUNT="${PBS_ACCOUNT:-P08010000}"

# --- (2) Repo location on GLADE -- EDIT to where you cloned legoESM ----------
REPO="${LEGOESM_REPO:-/glade/work/$USER/legoESM}"
export REPO

# NCAR normally exports $SCRATCH; default it defensively so the job scripts'
# `set -u` can't abort on an unset $SCRATCH (used below + for OUTDIR).
export SCRATCH="${SCRATCH:-/glade/derecho/scratch/$USER}"

# --- (3) Conda env that has JAX with CUDA support -- EDIT to your env name ---
CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"

# --- Federation PYTHONPATH (belt-and-braces; an editable `pip install -e .`
#     in the conda env makes this redundant but harmless) -------------------
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# --- Modules + conda activation ---------------------------------------------
module load conda 2>/dev/null || true
module load cuda  2>/dev/null || true
conda activate "$CONDA_ENV"
PY="$(command -v python)"
export PY

# --- JAX / runtime knobs ----------------------------------------------------
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
# Persistent JIT cache reuses compiles across runs; set empty to force a cold
# compile (true compile_time_s). Lives on SCRATCH so it survives between jobs.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"

# --- HPE Slingshot 11 / libfabric (CXI provider) fabric tuning --------------
# legoESM's MPI halo exchange (mpi4jax `sendrecv`: route-A latlon latitude-band
# + icosahedral cell-partition, and the MPI cubed-sphere face-scatter) is a
# nearest-neighbour, MANY-small-message pattern. On Slingshot 11 the NIC
# (Cassini) offloads MPI tag matching to hardware; a halo-heavy or high-rank
# run can exhaust the hardware match entries (LE pool) and ABORT mid-exchange:
#   "LE resources not recovered during flow control.
#    FI_CXI_RX_MATCH_MODE=[hybrid|software] is required"
# `hybrid` keeps fast hardware matching but transparently falls back to software
# matching once the NIC queues fill, instead of killing the job. This is NCAR's
# recommended Derecho setting; it is the primary fix for halo-exchange aborts.
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"

# Grow the libfabric completion queue. The halo path posts many concurrent
# send/recv completions per step; the small default CQ overflows at scale.
# 131072 is the value NCAR uses in its Derecho distributed examples.
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"

# CUDA-aware (GPU route-A) halo only: disable libfabric host-memory
# registration of the MR cache, which can DEADLOCK CUDA-aware transfers on
# Slingshot. A no-op on CPU-only runs (no device buffers are registered), so it
# is safe to set here in the shared env rather than only in the GPU job script.
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"

# libfabric memory-registration cache monitor. The default monitor can miss
# CUDA/host frees under the CUDA-aware path; userfaultfd is the setting the
# NCAR/ALCF GPU runtime configs ship for Slingshot-11
# (benkirk/derecho-pytorch-mpi profile.d config; ALCF Polaris NCCL docs).
# Harmless for CPU-only jobs.
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"

# Scratch tmp (mirrors the user's reference Casper job script).
export TMPDIR="${TMPDIR:-$SCRATCH/temp}"
mkdir -p "$TMPDIR"

# #1361 memory preflight: the target device whose HBM the benches gate against
# (`--device-hbm`, keys of scaling_preflight.DEVICE_HBM_BYTES). Set here rather
# than per-job so the gate is ON by default — a flag that every launcher forgot
# to pass is a gate that never fires (codex High, PR #1376). Derecho's GPU
# nodes are `gpu_type=a100`; the 40 GB entry is the CONSERVATIVE choice — if
# the queue hands out 80 GB parts, override with LEGOESM_DEVICE_HBM=a100-80.
export LEGOESM_DEVICE_HBM="${LEGOESM_DEVICE_HBM:-a100-40}"
