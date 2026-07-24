#!/bin/bash
#SBATCH --job-name=legoesm_cube_tiled
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=01:30:00
#SBATCH --output=cube_tiled_%j.log
# ===========================================================================
# Sub-face TILED cube hydrostatic step on Levante: 24 GPUs (6 x 4 A100),
# kt=2 -> (6,2,2) mesh — the >6-GPU strong-scaling lane (audit item 5).
# Stage A: CPU-virtual parity gate; Stage B: multicontroller timed run.
# GPU binding: all 4 node GPUs visible per task (--gpu-bind=none); bare
# jax.distributed.initialize() under SLURM derives
# local_device_ids=[SLURM_LOCALID], so each task binds the LOCALID-th GPU.
# Do NOT pin one GPU per task (--gpu-bind=single:1 / CVD shims): the pinned
# device renumbers to index 0 while jax still asks for index LOCALID ->
# CUDA_ERROR_INVALID_DEVICE on every localid!=0 task (observed 26446699;
# same double-pin failure as gpu_multinode_scaling.sbatch job 26447612).
# See scripts/cluster/scaling_derecho/cube_tiled_step.pbs for the lane's
# contract (dynamics-only envelope; HLO anti-replication refusals).
# ===========================================================================
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi

export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

OUT="${OUT:-results/a1/cube_tiled_scaling_levante.jsonl}"
RES="${RES:-192}"
NLEV="${NLEV:-60}"
KT="${KT:-2}"
# This job reserves EXACTLY 24 GPUs; a different kt needs its own
# --nodes/--ntasks allocation (see the PBS twin's refusal note).
if [ "${KT}" != "2" ]; then
    echo "ERROR: this 24-GPU template is kt=2 only; clone it with a" \
         "matching allocation for KT=${KT}." >&2
    exit 1
fi

echo "== Stage A: CPU-virtual parity gates =="
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
    --parity-gate --out "${OUT%.jsonl}_parity.jsonl"
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
    --closed-loop --parity-gate --out "${OUT%.jsonl}_loop_parity.jsonl"

echo "== Stage B: 24-GPU multicontroller timed runs =="
export JAX_PLATFORMS=cuda
# #921 (same defect as the Derecho twin): the closed-loop lane adds the
# dry-mass fixer's GLOBAL all-reduce (_tile_fix_ps_mass_delta psum over
# ("face","tile_i","tile_j")) on top of the halo ppermute cliques. Under XLA
# GPU defaults those clique kinds init concurrently and can be ordered
# differently per rank -> NCCL comm-init deadlock. Force serial comm init (see
# cube_tiled_step.pbs for the full rationale + bisection).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
timeout 1800 srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" \
    --steps 12 --warmup 2 --multicontroller --out "${OUT}"
echo "== single-shot done"
# Closed-loop production lane (state resident tile-sharded across steps).
echo "== closed-loop start"
timeout 1800 srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" \
    --steps 12 --warmup 2 --multicontroller --closed-loop --out "${OUT}"

echo "done: ${OUT}"
