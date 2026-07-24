#!/bin/bash -l
#SBATCH --job-name=legoesm_gpu_moist_scaling
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# ===========================================================================
# MOIST baroclinic-wave MULTI-NODE route-A GPU scaling on DKRZ Levante.
# SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/gpu_moist_scaling.pbs.
#
# Runs scripts/bench/run_cpu_mpi_scaling.py --device gpu --physics moist (the
# all-grids Kessler warm-rain case) on the two grids that are GENUINELY
# domain-decomposed across GPUs via the route-A mpi4jax halo path:
#
#   latlon        -- C-grid FV PE dycore, latitude-band MPI (1 GPU per rank)
#   icosahedral   -- MPAS Voronoi TRiSK PE dycore, cell-partition MPI
#
# ONE MPI rank per GPU; the mpi4jax sendrecv halo runs on the cuda devices over
# UCX -- cuda_ipc intra-node, gdr_copy over InfiniBand HDR *inter-node*.  Unlike
# the Derecho wrapper (single node), this scales across NODES: request
# --nodes>1 and the rank ladder extends to nodes*gpus_per_node (the roadmap's
# "generalize route-A to multi-node" + "Levante equivalent").
#
# NOT for cubed-sphere (served by run_levante_gpu_scaling.py single-process
# sharding; Kessler-moist cube needs the --cs-spmd coordinator path) or spectral
# (single device, no MPI).
#
# PREREQS (read scripts/cluster/scaling_levante/README.md):
#   * A CUDA jaxlib AND a CUDA-aware mpi4jax built against Levante's OpenMPI.
#     A CPU-only jaxlib trips the GPU-backend assertion in
#     run_cpu_mpi_scaling.py (--device gpu refuses CPU-fallback numbers) AND the
#     preflight below -- the desired loud failure, not a silent CPU run.
#
# SUBMIT (single node, both weak+strong, latlon):
#   sbatch scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
# MULTI-NODE (e.g. 4 nodes = 16 GPUs) + icosahedral strong scaling:
#   sbatch --nodes=4 --export=ALL,GRID=icosahedral,MODE=strong \
#          scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
# ===========================================================================

set -uo pipefail

# SLURM copies the script to a spool dir, so $0 does not point at the repo.
# Locate _env.sh from the submit dir instead.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    echo "       sbatch from the repo root or from scripts/cluster/scaling_levante/." >&2
    exit 1
fi

# Force the CUDA backend + GPU conda env BEFORE sourcing _env.sh so a stale
# JAX_PLATFORMS=cpu inherited from the submit shell can never silently run this
# GPU benchmark on the CPU.
export JAX_PLATFORMS=cuda
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# GPU-DIRECT halo: tell mpi4jax to hand the on-device sendrecv buffer straight
# to CUDA-aware OpenMPI/UCX instead of a device->host->device copy.  Without it
# route-A silently HOST-STAGES every exchange -- erasing multi-GPU scaling.
# FORCED to 1 (not overridable): host-staging is the exact "fake scaling" this
# script exists to avoid, and =0 would still pass the preflight when mpi4jax has
# CUDA support (the driver only warns).  The require_mpi_stack() preflight in
# the driver (and the shell preflight below) turn a mpi4jax-without-CUDA build
# into a clear failure rather than a segfault.
export MPI4JAX_USE_CUDA_MPI=1

# --- Benchmark configuration (overridable via --export=ALL,NAME=value) -------
GRID="${GRID:-latlon}"              # latlon | icosahedral (route-A, 1 GPU/rank)
PHYSICS="${PHYSICS:-moist}"         # moist (Kessler) | held_suarez | none
MODE="${MODE:-both}"                # weak | strong | both
PRECISION="${PRECISION:-float32}"   # float32 (A100-fast, FV-safe) | float64
NLEV="${NLEV:-26}"
N_WARMUP="${N_WARMUP:-3}"
N_TIMING="${N_TIMING:-100}"

# GPUS_PER_NODE: SLURM_GPUS_PER_NODE can be NON-numeric on a typed GPU request
# (e.g. "a100:4"), which would abort the arithmetic below under `set -u`.  Strip
# any "type:" prefix to the trailing count; prefer the always-numeric
# SLURM_GPUS_ON_NODE, then nvidia-smi, then a defensive default of 4.
_gpn="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}}"
_gpn="${_gpn##*:}"
if ! [ "${_gpn:-x}" -ge 1 ] 2>/dev/null; then
    _gpn="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
fi
if ! [ "${_gpn:-x}" -ge 1 ] 2>/dev/null; then _gpn=4; fi
GPUS_PER_NODE="$_gpn"
NNODES="${SLURM_NNODES:-1}"
# One rank per GPU across all allocated nodes.
MAX_RANKS="${MAX_RANKS:-$(( NNODES * GPUS_PER_NODE ))}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/gpu_moist_levante_${STAMP}_j${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"
exec > >(tee -a "$OUTDIR/run.log") 2>&1
CASE_FILE="$OUTDIR/cases.jsonl"

# --- Reject out-of-scope grids up front (clear message, not a crash) ---------
case "$GRID" in
    latlon|icosahedral) : ;;
    cubed-sphere)
        echo "ERROR: cubed-sphere is out of scope for gpu_moist_scaling.slurm." >&2
        echo "       Cube on GPU is served by run_levante_gpu_scaling.py (single-" >&2
        echo "       process sharding); Kessler-moist cube needs --cs-spmd." >&2
        exit 2 ;;
    spectral)
        echo "ERROR: spectral has no MPI path (single device) -- not a scaling case." >&2
        exit 2 ;;
    *)
        echo "ERROR: unknown GRID=$GRID (use latlon | icosahedral)." >&2
        exit 2 ;;
esac

echo "=== HOST ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -8 || true
echo "=== CONFIG ==="
echo "grid=$GRID physics=$PHYSICS mode=$MODE precision=$PRECISION nodes=$NNODES gpus/node=$GPUS_PER_NODE max_ranks=$MAX_RANKS nlev=$NLEV"
echo "python=$PY"; echo "outdir=$OUTDIR"

# --- PREFLIGHT: fail LOUD before the sweep if the GPU/CUDA-MPI stack is wrong -
# (roadmap: no silent CPU fallback / host-staged "fake scaling").  Runs on the
# batch node (a GPU node); the driver re-checks per rank via require_mpi_stack.
echo "=== PREFLIGHT ==="
"$PY" - <<'PY'
import sys
try:
    import jax
    be = jax.default_backend()
except Exception as e:
    print("PREFLIGHT FAIL: jax import/backend:", e); sys.exit(3)
if be not in ("gpu", "cuda", "rocm"):
    print(f"PREFLIGHT FAIL: JAX backend is {be!r}, not GPU -- refusing to record "
          "CPU-fallback scaling numbers."); sys.exit(3)
try:
    import mpi4jax
    cuda = bool(mpi4jax.has_cuda_support())
except Exception as e:
    print("PREFLIGHT FAIL: mpi4jax import:", e); sys.exit(3)
if not cuda:
    print("PREFLIGHT FAIL: mpi4jax lacks CUDA support but MPI4JAX_USE_CUDA_MPI=1 "
          "-- route-A would host-stage (fake scaling). Rebuild mpi4jax with CUDA "
          "(README Step 1)."); sys.exit(3)
print(f"PREFLIGHT OK: backend={be} mpi4jax_cuda={cuda}")
PY
if [ $? -ne 0 ]; then
    echo "Aborting: GPU / CUDA-aware-MPI preflight failed." >&2
    exit 3
fi

# --- 1) Enumerate cases (single process, no MPI) -----------------------------
"$PY" scripts/bench/run_cpu_mpi_scaling.py --sweep \
    --grid "$GRID" \
    --mode "$MODE" \
    --physics "$PHYSICS" \
    --precision "$PRECISION" \
    --n-levels "$NLEV" \
    --max-ranks "$MAX_RANKS" > "$CASE_FILE"

N_CASES=$(grep -c . "$CASE_FILE" || true)
echo "=== SWEEP: $N_CASES cases (rank ladder capped at MAX_RANKS=$MAX_RANKS) ==="
cat "$CASE_FILE"

# --- 2) Run each case under srun, ONE GPU PER RANK, across all nodes ----------
# srun places n_ranks tasks across the allocated nodes; SLURM binds one GPU per
# task and sets SLURM_LOCALID, which run_cpu_mpi_scaling.py::_configure_jax_gpu
# reads to pin rank->GPU (it only auto-pins when CUDA_VISIBLE_DEVICES is unset).
# Read the case list on FD 9 (not stdin): srun consumes stdin, which would
# otherwise swallow the rest of the file and end the loop after case 0.
rc_all=0
i=0
while IFS= read -r -u 9 CASE; do
    [ -z "$CASE" ] && continue
    N_RANKS=$("$PY" -c 'import sys,json; print(json.loads(sys.argv[1])["n_ranks"])' "$CASE")
    if [ "$N_RANKS" -gt "$MAX_RANKS" ]; then
        echo "--- case $i: n_ranks=$N_RANKS > MAX_RANKS=$MAX_RANKS -- SKIP ---"
        i=$((i + 1)); continue
    fi
    echo "--- case $i: n_ranks=$N_RANKS (1 GPU/rank, ${NNODES} node(s)) ---"
    echo "$CASE"
    srun --ntasks="$N_RANKS" --gpus-per-task=1 --gpu-bind=single:1 \
        --cpu-bind=cores \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py \
        --case "$CASE" \
        --device gpu \
        --n-warmup "$N_WARMUP" \
        --n-timing "$N_TIMING" \
        --output-dir "$OUTDIR" < /dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "case $i FAILED rc=$rc"; rc_all=$rc
    fi
    i=$((i + 1))
done 9< "$CASE_FILE"

echo "=== DONE rc=$rc_all ==="
echo "results under: $OUTDIR"
exit $rc_all
