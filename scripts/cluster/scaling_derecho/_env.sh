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

# --- Shared NCCL-over-Slingshot env for every jax.distributed (route-B) lane --
# Sourced by gpu_multinode_scaling.pbs and routeb_sweep.pbs.  Call it inside the
# subshell that launches an mpiexec route-B (NCCL multi-controller) job.  This is
# a function DEFINITION only -- it sets nothing until called, so sourcing it in
# CPU / route-A jobs is a no-op.  NCCL owns the GPU traffic, so GPU-aware
# cray-mpich stays OFF (mixing GPU-aware MPICH + NCCL in one app risks deadlock --
# CSCS/ALCF guidance); mpi4py here is bootstrap-only (rank discovery for the PALS
# fallback in init_jax_distributed_with_fallback).  aws-ofi-nccl (build via
# build_nccl_ofi.sh -> LEGOESM_NCCL_OFI_LIB) drives the Cassini NICs
# ("Using network AWS Libfabric"); without it NCCL loudly falls back to TCP
# sockets over hsn (2-3x slower comm).
nccl_env() {
    export MPICH_GPU_SUPPORT_ENABLED=0
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    # send/recv (= ppermute halo) throughput knob on multi-NIC nodes.
    export NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER:-4}"
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"   # first run: check the net line
    if [ -n "${LEGOESM_NCCL_OFI_LIB:-}" ]; then
        export LD_LIBRARY_PATH="${LEGOESM_NCCL_OFI_LIB}:${LD_LIBRARY_PATH:-}"
        export NCCL_NET="AWS Libfabric"
        echo "NCCL transport: aws-ofi-nccl from $LEGOESM_NCCL_OFI_LIB"
    else
        # No plugin: TCP-sockets-over-hsn fallback. Loud, not silent.
        export NCCL_IB_DISABLE=1
        export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"  # socket-path hang workaround
        echo "WARNING: LEGOESM_NCCL_OFI_LIB unset -> NCCL on TCP sockets"
        echo "         over hsn (2-3x slower comm). Build it with"
        echo "         scripts/cluster/scaling_derecho/build_nccl_ofi.sh."
    fi
}
