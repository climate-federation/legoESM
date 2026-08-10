# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
# (which is PBS/Cray-MPICH).
# ---------------------------------------------------------------------------
# EDIT the marked values (account / repo / conda env / module versions) before
# the first submit.  Everything is overridable from the sbatch environment, e.g.
#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
#
# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
# ---------------------------------------------------------------------------

# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
#     source of truth).  This default is only for interactive sourcing.
export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"

# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
# The default is a GUESS at a per-user clone path. When it is wrong the job
# does not fail here — it fails ~60 lines later with a bare
# "cd: <path>: No such file or directory" plus "_chain_body.sh: No such file",
# 7 seconds in, which reads like a broken launcher rather than an unset
# variable (three U-Cast arms lost this way, 2026-08-01). Say it plainly.
REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
if [ ! -d "$REPO" ]; then
  echo "[_env.sh] REPO='$REPO' does not exist." >&2
  if [ -z "${LEGOESM_REPO:-}" ]; then
    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
    echo "[_env.sh] guess, not a configured path. Submit with" >&2
    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
    echo "[_env.sh] (--export=ALL alone does NOT carry it if your shell" >&2
    echo "[_env.sh]  never exported it)." >&2
  fi
  exit 1
fi
export REPO

# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"

# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
#     redundant but harmless) ------------------------------------------------
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# --- Modules + conda -- EDIT the module versions to the Levante stack you built
#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
#     the runtime libmpi ABI must match the build ABI ------------------------
module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
# Prefer the repo's own uv venv when it exists — that is the interpreter every
# dev/test workflow uses, and the bare `python` on a Levante compute node has
# no jax (three U-Cast arms died at `import jax` inside 7 s, 2026-08-02).
if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
  LEGOESM_PYTHON="$REPO/.venv/bin/python"
fi
PY="${LEGOESM_PYTHON:-$(command -v python)}"
export PY
# Fail at source time, not 4 GPU-hours in: the launcher's first real work is
# `$PY scripts/run/run_aimip.py`, which imports jax immediately.
if ! "$PY" -c "import jax" >/dev/null 2>&1; then
  echo "[_env.sh] PY='$PY' cannot import jax." >&2
  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
  echo "[_env.sh] LEGOESM_CONDA_ENV=<env with a CUDA jaxlib>." >&2
  exit 1
fi

# --- JAX / runtime knobs -----------------------------------------------------
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
# Persistent JIT cache reuses compiles across runs; set empty to force a cold
# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"

# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
# no device->host->device staging, which would erase multi-GPU scaling).  On
# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
# memtype-cache hang that CUDA-aware sendrecv is prone to.
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"

# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
# multi-node NCCL bootstrap timeouts on IB clusters).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

# --- XLA overlap defaults for the lat-lon SPMD lanes (2026-08-04) --------
# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
# 26677602) and -8.3% at @128 (26677668), A/A2 drift 0.3-0.4% — twice-
# reproduced, parity suites green with flags on. MPAS lane: null (0.0%,
# 26677669 — its edge-coloured schedule does not benefit; harmless).
# Below the pre-registered 10% bar AND other lanes are unvalidated
# (cube_tiled_step.sbatch force-disables latency hiding for a known
# comm-init sensitivity; MPAS is null) — so this is strictly OPT-IN
# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
# launchers; never a shared default, and A/B control arms must keep
# REPLACING XLA_FLAGS, not appending.
if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
fi

export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# #1361 memory preflight: target device whose HBM the benches gate against
# (`--device-hbm`). Set in the SHARED env so the gate is on for every launcher
# that sources this file — codex found the Derecho-only export left every
# Levante bench ungated. Levante's GPU jobs request `--constraint=a100_80`.
export LEGOESM_DEVICE_HBM="${LEGOESM_DEVICE_HBM:-a100-80}"
